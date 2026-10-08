"""Pick a batch size for the real corpus: measures docs/s and peak VRAM.

The first full-benchmark run managed only ~61 docs/s (T2Retrieval: 141k encodes in 2303 s)
at batch_size=16 — far below the 1201 docs/s measured on short synthetic text, because the
per-batch processor overhead dominates at small batches. This measures the real trade-off.

Usage: uv run python src/throughput_test.py [--slice 20000]
"""

from __future__ import annotations

import argparse
import time

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--slice", type=int, default=20000)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--vram-fraction", type=float, default=0.7)
    p.add_argument("--batch-sizes", type=int, nargs="*", default=[16, 64, 128])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset("mteb/CovidRetrieval", "corpus")["dev"]
    texts = [t for t in ds["text"]][: args.slice]
    print(f"slice: {len(texts)} docs (mean {sum(len(t) for t in texts)/len(texts):.0f} chars)")

    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    m = SentenceTransformer(
        "models/embeddinggemma-2",
        model_kwargs={"torch_dtype": torch.bfloat16},
        config_kwargs={"vision_config": None, "audio_config": None},
        device="cuda",
    )
    m.max_seq_length = args.max_seq_length
    # same pin as eval_mteb.py: the default chat-template path is ~4x slower and treats a
    # whole-string media URL as media to decode
    module = m[0]
    module.modality_config = {"text": {"method": "forward", "method_output_name": "last_hidden_state"}}
    module.module_output_name = "token_embeddings"
    module.input_formatter.supported_modalities = ["text"]

    for bs in args.batch_sizes:
        try:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            m.encode(
                texts,
                batch_size=bs,
                prompt="task: search result | query: ",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            dt = time.perf_counter() - t0
            print(
                f"bs={bs:<4} {dt:6.1f}s  {len(texts)/dt:7.0f} docs/s  "
                f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB"
            )
        except torch.OutOfMemoryError as exc:
            print(f"bs={bs:<4} OOM under the {args.vram_fraction:.0%} cap: {str(exc)[:80]}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
