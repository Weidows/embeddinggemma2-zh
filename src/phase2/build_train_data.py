"""Phase 2 / step 2b: build the Chinese training pairs for the LoRA fine-tune.

Targets (fixed by the phase-1 measurements, both ~0.10 behind bge-m3):
  - 医疗领域: CmedqaRetrieval / MedicalRetrieval / CMedQAv1&2-reranking
  - 中文 STS: AFQMC / ATEC / BQ

Sources and why each is legitimate
  - STS: the *train* splits of AFQMC / ATEC / BQ / LCQMC / PAWSX / STSB. Evaluation uses
    validation/test, so no overlap.
  - Medical: `wangrongsheng/cMedQA-V2.0` train (226,266 question/answer pairs), because the
    medical *retrieval* tasks ship only corpus+queries+qrels — using the qrels would be
    training on the exam paper.
  - **Document-level decontamination**: cMedQA train answers are dropped when they occur
    verbatim in the CmedqaRetrieval / MedicalRetrieval corpora (measured ~6.8% before
    filtering). That leaves zero overlap on queries *and* documents, so any gain can only
    come from learning Chinese semantics.

Prefixes are baked into the text, matching what the evaluation harness prepends
(`task: sentence similarity | query: ` for STS, `task: search result | query: ` for queries
and `title: none | text: ` for documents).

Usage:
    env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
      uv run python src/phase2/build_train_data.py [--max-per-source N] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATA = Path("data")
STS_SOURCES = ["C-MTEB/AFQMC", "C-MTEB/ATEC", "C-MTEB/BQ", "C-MTEB/LCQMC", "C-MTEB/PAWSX", "C-MTEB/STSB"]
STS_PREFIX = "task: sentence similarity | query: "
Q_PREFIX = "task: search result | query: "
D_PREFIX = "title: none | text: "
MEDICAL_SOURCES = ["C-MTEB/CmedqaRetrieval", "C-MTEB/MedicalRetrieval"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-per-source", type=int, default=60000, help="cap pairs per STS source")
    p.add_argument("--max-medical", type=int, default=120000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower()


def to_float(x) -> float | None:
    try:
        return float(str(x).strip())
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    DATA.mkdir(exist_ok=True)
    pairs: list[dict] = []
    stats: dict[str, dict] = {}

    # ---------- 1. STS train splits ------------------------------------------------
    # Pair-level decontamination against the splits mteb actually scores (see
    # check_leakage.py [5]: LCQMC/STSB have a handful of identical pairs across the split).
    EVAL_SPLITS = {
        "C-MTEB/AFQMC": ["validation"], "C-MTEB/ATEC": ["validation", "test"],
        "C-MTEB/BQ": ["validation", "test"], "C-MTEB/LCQMC": ["test"],
        "C-MTEB/PAWSX": ["test"], "C-MTEB/STSB": ["validation", "test"],
    }
    for ds in STS_SOURCES:
        eval_pairs: set[tuple[str, str]] = set()
        for sp in EVAL_SPLITS.get(ds, []):
            try:
                ev = load_dataset(ds, split=sp)
            except Exception:  # noqa: BLE001
                continue
            for a, b in zip(ev["sentence1"], ev["sentence2"]):
                eval_pairs.add((norm(a), norm(b)))
                eval_pairs.add((norm(b), norm(a)))

        tr = load_dataset(ds, split="train")
        scores = [to_float(s) for s in tr["score"]]
        scores = [s for s in scores if s is not None]
        hi = max(scores) if scores else 1.0
        # label scale differs per task (0/1, 0-1 continuous, 0-5)
        thr = 3.0 if hi > 2 else (0.999 if hi > 1 else 0.6)
        dist = Counter()
        kept = []
        dropped_leak = 0
        for s1, s2, sc in zip(tr["sentence1"], tr["sentence2"], tr["score"]):
            v = to_float(sc)
            if v is None or not str(s1).strip() or not str(s2).strip():
                continue
            dist["pos" if v >= thr else ("neg" if v <= (0.2 if hi > 1 else 1e-9) else "mid")] += 1
            if v < thr:
                continue
            if (norm(s1), norm(s2)) in eval_pairs:
                dropped_leak += 1
                continue
            kept.append((str(s1).strip(), str(s2).strip()))
        rng.shuffle(kept)
        kept = kept[: args.max_per_source]
        for a, p in kept:
            pairs.append({
                "anchor": STS_PREFIX + a, "positive": STS_PREFIX + p,
                "source": ds.split("/")[-1], "kind": "sts",
            })
        stats[ds] = {"rows": tr.num_rows, "pos_threshold": thr, "positives_kept": len(kept),
                     "dropped_as_eval_pair": dropped_leak, "label_split": dict(dist)}

    # ---------- 2. medical, decontaminated on BOTH queries and documents ------------
    # The eval queries turned out to be *inside* cMedQA-V2.0 (7592 rows matched the 3999 eval
    # queries when compared with the correct `instruction` column), so filtering the answers
    # alone is not enough: a training row whose question is an eval query IS the exam paper.
    corpus_norm: set[str] = set()
    query_norm: set[str] = set()
    for ds in MEDICAL_SOURCES:
        c = load_dataset(ds, split="corpus")
        col = "text" if "text" in c.column_names else c.column_names[-1]
        for x in c[col]:
            corpus_norm.add(norm(x))
        q = load_dataset(ds, split="queries")
        qc = "text" if "text" in q.column_names else q.column_names[-1]
        for x in q[qc]:
            query_norm.add(norm(x))
        print(f"[corpus] {ds}: {c.num_rows} passages, {q.num_rows} queries")

    med = load_dataset("wangrongsheng/cMedQA-V2.0", split="train")
    # NOTE: `input` is empty in this dataset — the question is in `instruction`.
    def pick_col(cands: list[str]) -> str:
        for c in cands:
            if c in med.column_names and sum(1 for x in med[c][:200] if str(x).strip()) > 100:
                return c
        return cands[0]

    qcol = pick_col(["input", "instruction", "question"])
    acol = pick_col(["output", "answer", "response"])
    print(f"[cMedQA] question column={qcol!r} answer column={acol!r}")
    dropped_q = dropped_doc = 0
    cand = []
    for q, a in zip(med[qcol], med[acol]):
        q, a = str(q).strip(), str(a).strip()
        if not q or not a:
            continue
        if norm(q) in query_norm:  # query-level decontamination
            dropped_q += 1
            continue
        if norm(a) in corpus_norm:  # document-level decontamination
            dropped_doc += 1
            continue
        cand.append((q, a))
    rng.shuffle(cand)
    cand = cand[: args.max_medical]
    for q, a in cand:
        pairs.append({"anchor": Q_PREFIX + q, "positive": D_PREFIX + a,
                      "source": "cMedQA-V2.0", "kind": "medical"})
    stats["cMedQA-V2.0"] = {"rows": med.num_rows, "dropped_as_eval_query": dropped_q,
                            "dropped_as_eval_corpus": dropped_doc, "kept": len(cand)}

    # ---------- 3. write ----------------------------------------------------------
    rng.shuffle(pairs)
    out = DATA / "train_pairs.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in pairs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    sample = DATA / "train_pairs_sample.txt"
    with sample.open("w", encoding="utf-8") as fh:
        for r in pairs[:40]:
            fh.write(f"[{r['kind']}/{r['source']}]\n  A: {r['anchor']}\n  P: {r['positive'][:200]}\n\n")

    lens = [len(r["anchor"]) + len(r["positive"]) for r in pairs]
    summary = {
        "total_pairs": len(pairs),
        "by_kind": dict(Counter(r["kind"] for r in pairs)),
        "by_source": dict(Counter(r["source"] for r in pairs)),
        "char_len_mean": int(statistics.mean(lens)),
        "char_len_p99": int(sorted(lens)[int(len(lens) * 0.99)]),
        "per_source_stats": stats,
    }
    (DATA / "train_pairs_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[out] {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"[out] {sample}")


if __name__ == "__main__":
    main()
