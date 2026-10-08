"""Build the C-MTEB comparison table: our run vs published same-harness results.

Our numbers come from results/mteb_base_*.json (this machine, MTEB(cmn, v1), 31 tasks).
Reference numbers are the published mteb/results entries for the same task names, fetched
into results/reference/<org__model>/<Task>.json — same harness, so directly comparable
(no re-running needed).

Usage: uv run python src/make_comparison.py
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

RESULTS = Path("results")
REF = RESULTS / "reference"

TASK_TYPES = {
    "T2Retrieval": "Retrieval", "MMarcoRetrieval": "Retrieval", "DuRetrieval": "Retrieval",
    "CovidRetrieval": "Retrieval", "CmedqaRetrieval": "Retrieval", "EcomRetrieval": "Retrieval",
    "MedicalRetrieval": "Retrieval", "VideoRetrieval": "Retrieval",
    "T2Reranking": "Reranking", "MMarcoReranking": "Reranking",
    "CMedQAv1-reranking": "Reranking", "CMedQAv2-reranking": "Reranking",
    "Ocnli": "PairClassification", "Cmnli": "PairClassification",
    "CLSClusteringS2S": "Clustering", "CLSClusteringP2P": "Clustering",
    "ThuNewsClusteringS2S": "Clustering", "ThuNewsClusteringP2P": "Clustering",
    "LCQMC": "STS", "PAWSX": "STS", "AFQMC": "STS", "QBQTC": "STS", "ATEC": "STS", "BQ": "STS", "STSB": "STS",
    "TNews": "Classification", "IFlyTek": "Classification", "Waimai": "Classification",
    "OnlineShopping": "Classification", "JDReview": "Classification", "MultilingualSentiment": "Classification",
}
TASK_ORDER = list(TASK_TYPES)


def avg_main_score(d: dict) -> float | None:
    vals = []
    for _split, m in d.get("scores", {}).items():
        for e in (m if isinstance(m, list) else [m]):
            if isinstance(e, dict) and isinstance(e.get("main_score"), (int, float)):
                vals.append(e["main_score"])
    return statistics.mean(vals) if vals else None


def load_ours() -> dict[str, float]:
    f = sorted(glob.glob(str(RESULTS / "mteb_base_*.json")))[-1]
    d = json.load(open(f, encoding="utf-8"))
    per_task: dict[str, list[float]] = {}
    for r in d["rows"]:
        if isinstance(r["main_value"], (int, float)):
            per_task.setdefault(r["task"], []).append(r["main_value"])
    return {t: statistics.mean(v) for t, v in per_task.items()}


def load_refs() -> dict[str, dict[str, float]]:
    out = {}
    for model_dir in sorted(REF.iterdir()) if REF.exists() else []:
        if not model_dir.is_dir():
            continue
        scores = {}
        for f in model_dir.glob("*.json"):
            try:
                v = avg_main_score(json.load(open(f, encoding="utf-8")))
            except Exception:  # noqa: BLE001
                v = None
            if v is not None:
                scores[f.stem] = v
        out[model_dir.name] = scores
    return out


def aggregate(scores: dict[str, float], only: set[str] | None = None) -> tuple[float, dict[str, float]]:
    by_type: dict[str, list[float]] = {}
    for t, v in scores.items():
        if t in TASK_TYPES and (only is None or t in only):
            by_type.setdefault(TASK_TYPES[t], []).append(v)
    type_means = {k: statistics.mean(v) for k, v in by_type.items()}
    overall = statistics.mean(type_means.values()) if type_means else float("nan")
    return overall, type_means


LOCAL_LABELS = {"base": "embeddinggemma-2 (local run)", "bgem3": "bge-m3 (local run)"}
TYPE_ORDER = ["Retrieval", "Reranking", "PairClassification", "Clustering", "STS", "Classification"]


def load_local_runs() -> dict[str, dict[str, float]]:
    """Each results/mteb_<tag>_*.json is a full local run of the same benchmark."""
    out: dict[str, dict[str, float]] = {}
    for f in sorted(glob.glob(str(RESULTS / "mteb_*_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        per: dict[str, list[float]] = {}
        for r in d["rows"]:
            if isinstance(r["main_value"], (int, float)):
                per.setdefault(r["task"], []).append(r["main_value"])
        label = LOCAL_LABELS.get(d["tag"], f"{d['tag']} (local run)")
        out[label] = {t: statistics.mean(v) for t, v in per.items()}
    return out


def _count(scores: dict[str, float]) -> int:
    return len([t for t in scores if t in TASK_TYPES])


def main() -> None:
    cols = list(load_local_runs().items()) + list(load_refs().items())
    full_cols = [(n, s) for n, s in cols if _count(s) == len(TASK_ORDER)]
    sparse_cols = [(n, s) for n, s in cols if _count(s) != len(TASK_ORDER)]

    lines = [
        "# C-MTEB (MTEB(cmn, v1)) — embeddinggemma-2 vs bge-m3 vs Qwen3-Embedding-0.6B",
        "",
        "Every column uses **the same harness** (mteb 2.24.0, `MTEB(cmn, v1)`, 31 tasks).",
        "`local run` = executed on this machine (RTX 5070 Ti, bf16, truncate 2048, batch 16);",
        "other columns are the published `mteb/results` entries for the same task names.",
        "",
        "| column | tasks covered | source |",
        "|---|---|---|",
    ]
    for name, sc in cols:
        src = "local run" if "local run" in name else "mteb/results (public)"
        lines.append(f"| {name} | {_count(sc)}/{len(TASK_ORDER)} | {src} |")

    # ---- main table: full-coverage columns only, so every cell is comparable
    lines += ["", f"## Full {len(TASK_ORDER)}-task table", "",
              "| task | type | " + " | ".join(n for n, _ in full_cols) + " |",
              "|---|---|" + "---|" * len(full_cols)]
    for t in TASK_ORDER:
        cells = []
        for _n, sc in full_cols:
            v = sc.get(t)
            cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
        lines.append(f"| {t} | {TASK_TYPES[t]} | " + " | ".join(cells) + " |")

    agg = {n: aggregate(s) for n, s in full_cols}
    lines += ["", "## Aggregate", "",
              "| | " + " | ".join(n for n, _ in full_cols) + " |", "|---|" + "---|" * len(full_cols)]
    for ty in TYPE_ORDER:
        cells = [f"{agg[n][1][ty]:.4f}" if ty in agg[n][1] else "—" for n, _ in full_cols]
        lines.append(f"| {ty} | " + " | ".join(cells) + " |")
    lines.append("| **Overall (mean of type means)** | " + " | ".join(f"**{agg[n][0]:.4f}**" for n, _ in full_cols) + " |")

    # ---- delta table: ours minus each other model (only meaningful on full coverage)
    if full_cols:
        base_name, base = full_cols[0]
        lines += ["", f"## Delta vs `{base_name}` (positive = ours better)", "",
                  "| task | type | " + " | ".join(n for n, _ in full_cols[1:]) + " |",
                  "|---|---|" + "---|" * max(0, len(full_cols) - 1)]
        for t in TASK_ORDER:
            if not isinstance(base.get(t), (int, float)):
                continue
            cells = []
            for _n, sc in full_cols[1:]:
                v = sc.get(t)
                cells.append(f"{base[t] - v:+.4f}" if isinstance(v, (int, float)) else "—")
            lines.append(f"| {t} | {TASK_TYPES[t]} | " + " | ".join(cells) + " |")
        cells = []
        for n, _s in full_cols[1:]:
            cells.append(f"**{agg[base_name][0] - agg[n][0]:+.4f}**")
        lines.append("| **Overall** | | " + " | ".join(cells) + " |")

    # ---- sparse public entries: only comparable on their own intersection
    if sparse_cols:
        common = set(TASK_ORDER)
        for _n, sc in sparse_cols + [full_cols[0]] if full_cols else sparse_cols:
            common &= set(sc)
        common = {t for t in common if t in TASK_TYPES}
        if common:
            sub = [(full_cols[0][0], full_cols[0][1])] + sparse_cols if full_cols else sparse_cols
            lines += ["", f"## Sparse public entries — {len(common)}-task intersection only", "",
                      "These models have published results for only part of the benchmark, so they",
                      "are compared on the intersection, not on their own averages.", "",
                      "| task | type | " + " | ".join(n for n, _ in sub) + " |",
                      "|---|---|" + "---|" * len(sub)]
            for t in TASK_ORDER:
                if t not in common:
                    continue
                cells = []
                for _n, sc in sub:
                    v = sc.get(t)
                    cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
                lines.append(f"| {t} | {TASK_TYPES[t]} | " + " | ".join(cells) + " |")
            sub_agg = {n: aggregate(s, only=common) for n, s in sub}
            lines.append("| **Overall (on intersection)** | | " + " | ".join(
                f"**{sub_agg[n][0]:.4f}**" for n, _ in sub) + " |")

    out = RESULTS / "cmteb_comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
