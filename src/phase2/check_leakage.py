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


def load_first(ds_id: str, config: str | None = None):
    cfgs = None
    try:
        if config is not None:
            ds = load_dataset(ds_id, config)
        else:
            ds = load_dataset(ds_id)
        split = list(ds.keys())[0]
        return ds[split]
    except Exception as exc:  # noqa: BLE001
        try:
            cfgs = get_dataset_config_names(ds_id)
        except Exception:  # noqa: BLE001
            cfgs = "?"
        print(f"  !! {ds_id} (config={config}) failed: {exc} | configs={cfgs}")
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
        col = None
        for c in ("question", "questions", "q", "input"):
            if c in cm_train.column_names:
                col = c
                break
        if col and cmedqa_q is not None:
            qcol = "text" if "text" in cmedqa_q.column_names else cmedqa_q.column_names[-1]
            train_qs = [norm(x) for x in cm_train[col]]
            eval_qs = [norm(x) for x in cmedqa_q[qcol]]
            compare("cMedQA-V2.0 train", train_qs, "CmedqaRetrieval eval queries", eval_qs)

    if cmedqa_q is not None and medical_q is not None:
        qcol = "text" if "text" in cmedqa_q.column_names else cmedqa_q.column_names[-1]
        mcol = "text" if "text" in medical_q.column_names else medical_q.column_names[-1]
        print()
        compare("CmedqaRetrieval queries", [norm(x) for x in cmedqa_q[qcol]],
                "MedicalRetrieval queries", [norm(x) for x in medical_q[mcol]])


if __name__ == "__main__":
    main()
