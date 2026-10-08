"""Phase 2 / step 2a: introspect the train splits we intend to use.

Prints rows / columns / label range / a sample for each STS train split, so the triplet
builder can be written against the real schema instead of a guess.

Usage:
    env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
      uv run python src/phase2/inspect_sources.py
"""

from __future__ import annotations

from datasets import load_dataset

for ds in ["C-MTEB/AFQMC", "C-MTEB/ATEC", "C-MTEB/BQ", "C-MTEB/LCQMC", "C-MTEB/PAWSX", "C-MTEB/STSB"]:
    try:
        tr = load_dataset(ds, split="train")
        labels = tr["label"] if "label" in tr.column_names else None
        rng = (min(labels), max(labels)) if labels else None
        print(f"{ds:<16} rows={tr.num_rows:<8} cols={tr.column_names} label_range={rng}")
        print("    sample:", {k: str(tr[0][k])[:70] for k in tr.column_names})
    except Exception as exc:  # noqa: BLE001
        print(f"{ds}: ERR {str(exc)[:120]}")
