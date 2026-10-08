"""Run a mteb benchmark against a local google/embeddinggemma-2 checkout.

Defaults to MTEB(cmn, v1) == C-MTEB (31 Chinese tasks) and writes both a JSON of raw
scores and a markdown table into results/.

mteb registers only embeddinggemma-300m (v1) as of mteb 2.24.0 (v2 is PR #5588, still
open), so this script wraps the local SentenceTransformer directly with
SentenceTransformerEncoderWrapper — the model ships its own mteb task-type -> prompt
mapping in config_sentence_transformers.json, which the wrapper picks up automatically.

Usage:
    uv run python src/eval_mteb.py                       # C-MTEB, all 31 tasks
    uv run python src/eval_mteb.py --limit 3             # smoke test, first 3 tasks
    uv run python src/eval_mteb.py --tasks T2Retrieval DuRetrieval
    uv run python src/eval_mteb.py --benchmark "MTEB(eng, v2)"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

import mteb

DEFAULT_MODEL = os.environ.get("MODEL_PATH", "models/embeddinggemma-2")
RESULTS_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--benchmark", default="MTEB(cmn, v1)")
    p.add_argument("--tasks", nargs="*", default=None, help="explicit task names; overrides --benchmark")
    p.add_argument("--limit", type=int, default=None, help="only run the first N tasks (smoke test)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="truncation length (model supports 8192; C-MTEB p99 doc is ~1.3k tokens)",
    )
    p.add_argument(
        "--vram-fraction",
        type=float,
        default=0.7,
        help="hard cap on this process's VRAM share (0-1) so a run can never take the whole card",
    )
    p.add_argument("--tag", default=None, help="output file tag, e.g. 'base' or 'finetuned'")
    p.add_argument("--revision", default="local", help="revision hash written into the result metadata")
    p.add_argument(
        "--no-text-pin",
        action="store_true",
        help="keep ST's default chat-template/multimodal path (slower, crashes on media-looking URLs)",
    )
    p.add_argument(
        "--overwrite",
        default="always",
        choices=["always", "only-missing", "never"],
        help="mteb cache strategy; use only-missing to resume an interrupted run",
    )
    return p.parse_args()


def pin_text_modality(st: SentenceTransformer) -> None:
    """Pin sentence-transformers to the plain ``text`` modality.

    Why this is needed for text benchmarks on this model:

    1. **It crashes on media-looking URLs.** ST's default path for embeddinggemma-2 routes
       text through the chat template + multimodal processor, which treats an input that is
       *exactly* a media URL as media to decode. MMarcoRetrieval doc #98275 is a bare
       ``http://www.youtube.com/ehowatHomeChannel`` -> the run died 38 minutes in with
       ``ImportError: To load a video from YouTube url you have to install yt_dlp first``.
       (``https://example.com/clip.mp4`` fails the same way, with a libtorchcodec error.)
    2. **It is ~4x slower.** 92 docs/s (message path) vs 376 docs/s (text path) on 200
       short Chinese texts, batch_size=64.

    Verified equivalent: on 200 Chinese texts the two paths agree to |delta| < 1e-4
    (bf16 noise), i.e. ``np.allclose(a, b, atol=1e-4) is True``.
    """
    module = st[0]
    module.modality_config = {"text": {"method": "forward", "method_output_name": "last_hidden_state"}}
    module.module_output_name = "token_embeddings"
    module.input_formatter.supported_modalities = ["text"]


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
        print(f"[guard] VRAM capped at {args.vram_fraction:.0%} of the card")

    if args.tasks:
        tasks = list(mteb.get_tasks(tasks=args.tasks))
        bench_name = "custom"
    else:
        benchmark = mteb.get_benchmark(args.benchmark)
        tasks = list(benchmark.tasks)
        bench_name = benchmark.name

    if args.limit:
        tasks = tasks[: args.limit]

    print(f"[setup] benchmark={bench_name} tasks={len(tasks)} model={args.model}")
    print(f"[setup] device={device} dtype={dtype} batch_size={args.batch_size}")

    st = SentenceTransformer(
        args.model,
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"vision_config": None, "audio_config": None},  # text-only (270M)
        device=device,
    )
    # NOTE: ST resolves max_seq_length from max_position_embeddings (262144) and overflows
    # to ~1e60, so nothing gets truncated. Cap it explicitly (model card: 8k context).
    st.max_seq_length = args.max_seq_length
    if not args.no_text_pin:
        pin_text_modality(st)
        print("[pin] ST pinned to the plain text modality (identical embeddings, ~4x faster)")
    model = mteb.SentenceTransformerEncoderWrapper(st, embed_dim=st.get_sentence_embedding_dimension())

    t0 = time.perf_counter()
    result = mteb.evaluate(
        model,
        tasks,
        encode_kwargs={"batch_size": args.batch_size},
        overwrite_strategy=args.overwrite,
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - t0

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = args.tag or "base"
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    rows = []
    for tr in result.task_results:
        for split, metrics in tr.scores.items():
            # mteb returns a list of per-subset dicts for some task types, a dict for others
            entries = metrics if isinstance(metrics, list) else [metrics]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                main_score = tr.task.metadata.main_score if hasattr(tr, "task") else None
                rows.append(
                    {
                        "task": tr.task_name,
                        "type": tr.task.metadata.type,
                        "split": split,
                        "hf_subset": entry.get("hf_subset"),
                        "main_score": main_score,
                        "main_value": entry.get(main_score),
                        "metrics": {k: v for k, v in entry.items() if isinstance(v, (int, float))},
                    }
                )

    payload = {
        "model": args.model,
        "tag": tag,
        "benchmark": bench_name,
        "tasks": [t.metadata.name for t in tasks],
        "n_tasks": len(tasks),
        "elapsed_seconds": elapsed,
        "device": device,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "mteb_version": mteb.__version__,
        "rows": rows,
    }
    out_json = RESULTS_DIR / f"mteb_{tag}_{stamp}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# MTEB run — {args.model} ({tag})",
        "",
        f"- benchmark: `{bench_name}` ({len(tasks)} tasks)",
        f"- device: {device} / {dtype} / batch_size={args.batch_size}",
        f"- wall time: {elapsed/60:.1f} min",
        f"- mteb: {mteb.__version__}",
        "",
        "| task | type | split | main metric | score |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        val = r["main_value"]
        lines.append(
            f"| {r['task']} | {r['type']} | {r['split']} | {r['main_score']} | "
            f"{val:.4f} |" if isinstance(val, (int, float)) else f"| {r['task']} | {r['type']} | {r['split']} | {r['main_score']} | n/a |"
        )
    out_md = RESULTS_DIR / f"mteb_{tag}_{stamp}.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] {len(rows)} task results in {elapsed/60:.1f} min")
    print(f"[out]  {out_json}")
    print(f"[out]  {out_md}")
    for r in rows:
        val = r["main_value"]
        print(f"  {r['task']:<28} {r['type']:<24} {r['main_score']:<14} {val if val is None else round(val,4)}")


if __name__ == "__main__":
    main()
