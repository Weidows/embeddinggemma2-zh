"""Phase 2 / step 1: check that the training sources do NOT overlap the evaluation sets.

The whole fine-tune is worthless (worse: misleading) if the medical training pairs come from
the same questions that mteb scores us on. This verifies that explicitly before building any
triplets, because it decides which sources are usable.

Checks
  1. CmedqaRetrieval (eval) queries  vs  cMedQA-V2.0 train questions   -> exact + fuzzy overlap
  2. MedicalRetrieval (eval) queries vs  CmedqaRetrieval (eval) queries -> do the two medical
     tasks share queries? (they are separate mteb tasks and may well share a source)
  3. STS train/eval split sizes for the AFQMC / ATEC / BQ targets

Usage:
    env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
      uv run python src/phase2/check_leakage.py
"""

from __future__ import annotations

import re
from collections import Counter

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower()


def load_first(ds_id: str, split: str | None = None):
    """corpus/queries are *splits* in the C-MTEB datasets, not configs."""
    try:
        if split is not None:
            return load_dataset(ds_id, split=split)
        ds = load_dataset(ds_id)
        return ds[list(ds.keys())[0]]
    except Exception as exc:  # noqa: BLE001
        try:
            cfgs = get_dataset_config_names(ds_id)
        except Exception:  # noqa: BLE001
            cfgs = "?"
        print(f"  !! {ds_id} (split={split}) failed: {exc} | configs={cfgs}")
        return None


def compare(name_a: str, a: list[str], name_b: str, b: list[str]) -> None:
    set_b = set(b)
    exact = sum(1 for x in a if x in set_b)
    print(f"  {name_a} ({len(a)}) vs {name_b} ({len(b)}): exact overlap = {exact}")
    if exact:
        c = Counter(x for x in a if x in set_b)
        print(f"     e.g. {[k[:40] for k, _ in c.most_common(3)]}")


def main() -> None:
    print("[1] dataset configs / splits inventory")
    for ds_id in ["C-MTEB/CmedqaRetrieval", "C-MTEB/MedicalRetrieval", "C-MTEB/CmedqaRetrieval-qrels",
                  "wangrongsheng/cMedQA-V2.0"]:
        try:
            cfgs = get_dataset_config_names(ds_id)
        except Exception as exc:  # noqa: BLE001
            cfgs = f"ERR {exc}"
        print(f"  {ds_id}: configs={cfgs}")
        if isinstance(cfgs, list):
            for c in cfgs:
                try:
                    print(f"      {c}: splits={get_dataset_split_names(ds_id, c)}")
                except Exception as exc:  # noqa: BLE001
                    print(f"      {c}: splits ERR {exc}")

    print("\n[2] eval query sets")
    cmedqa_q = load_first("C-MTEB/CmedqaRetrieval", "queries")
    medical_q = load_first("C-MTEB/MedicalRetrieval", "queries")
    if cmedqa_q is not None:
        print("  CmedqaRetrieval queries cols:", cmedqa_q.column_names, "rows:", cmedqa_q.num_rows)
    if medical_q is not None:
        print("  MedicalRetrieval queries cols:", medical_q.column_names, "rows:", medical_q.num_rows)

    print("\n[3] medical training source vs eval")
    cm_train = load_first("wangrongsheng/cMedQA-V2.0")
    if cm_train is not None:
        print("  cMedQA-V2.0 cols:", cm_train.column_names, "rows:", cm_train.num_rows)
        # NOTE: in this dataset `input` is EMPTY and the real question lives in `instruction`.
        # Picking `input` would compare empty strings and report a meaningless overlap of 0.
        col = None
        for c in ("instruction", "question", "questions", "q", "input"):
            if c in cm_train.column_names and sum(1 for x in cm_train[c][:200] if str(x).strip()) > 100:
                col = c
                break
        print(f"  question column resolved to: {col!r}")
        if col and cmedqa_q is not None:
            qcol = "text" if "text" in cmedqa_q.column_names else cmedqa_q.column_names[-1]
            train_qs = [norm(x) for x in cm_train[col]]
            eval_qs = [norm(x) for x in cmedqa_q[qcol]]
            compare("cMedQA-V2.0 train", train_qs, "CmedqaRetrieval eval queries", eval_qs)

    print("\n[4] corpus overlap (training answers vs eval corpus)")
    cmedqa_c = load_first("C-MTEB/CmedqaRetrieval", "corpus")
    if cmedqa_c is not None and cm_train is not None:
        ccol = "text" if "text" in cmedqa_c.column_names else cmedqa_c.column_names[-1]
        corpus = {norm(x) for x in cmedqa_c[ccol]}
        print(f"  CmedqaRetrieval corpus rows: {cmedqa_c.num_rows}")
        out_col = "output" if "output" in cm_train.column_names else None
        if out_col:
            sample = [norm(x) for x in cm_train[out_col][:20000]]
            hits = sum(1 for x in sample if x in corpus)
            print(f"  first 20k cMedQA train answers appearing verbatim in the eval corpus: {hits}")

    if cmedqa_q is not None and medical_q is not None:
        qcol = "text" if "text" in cmedqa_q.column_names else cmedqa_q.column_names[-1]
        mcol = "text" if "text" in medical_q.column_names else medical_q.column_names[-1]
        print()
        compare("CmedqaRetrieval queries", [norm(x) for x in cmedqa_q[qcol]],
                "MedicalRetrieval queries", [norm(x) for x in medical_q[mcol]])

    # ---------- STS: are the train splits disjoint from the evaluated splits? ----------
    print("\n[5] STS train vs evaluated split (mteb evaluates these splits)")
    EVAL_SPLITS = {
        "C-MTEB/AFQMC": ["validation"],
        "C-MTEB/ATEC": ["validation", "test"],
        "C-MTEB/BQ": ["validation", "test"],
        "C-MTEB/LCQMC": ["test"],
        "C-MTEB/PAWSX": ["test"],
        "C-MTEB/STSB": ["validation", "test"],
    }
    for ds, splits in EVAL_SPLITS.items():
        try:
            tr = load_dataset(ds, split="train")
        except Exception as exc:  # noqa: BLE001
            print(f"  {ds}: train load ERR {str(exc)[:80]}")
            continue
        tr_pairs = {(norm(a), norm(b)) for a, b in zip(tr["sentence1"], tr["sentence2"])}
        tr_s1 = {norm(a) for a in tr["sentence1"]}
        for sp in splits:
            try:
                ev = load_dataset(ds, split=sp)
            except Exception as exc:  # noqa: BLE001
                print(f"  {ds}[{sp}]: ERR {str(exc)[:60]}")
                continue
            ev_pairs = {(norm(a), norm(b)) for a, b in zip(ev["sentence1"], ev["sentence2"])}
            pair_overlap = len(tr_pairs & ev_pairs)
            s1_overlap = sum(1 for a in {norm(x) for x in ev["sentence1"]} if a in tr_s1)
            print(f"  {ds}[{sp}]: eval_pairs={len(ev_pairs)}  pair_overlap={pair_overlap}  "
                  f"eval_sentence1_seen_in_train={s1_overlap}")


if __name__ == "__main__":
    main()
