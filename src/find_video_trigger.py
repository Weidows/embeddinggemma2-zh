"""Find the exact input that makes the multimodal processor demand a video decoder.

mteb's MMarcoRetrieval run died with:
    ImportError: `torchcodec` is not installed ... to decode the video
inside transformers' `_process_videos`, which is only reached when `videos is not None`.
So some text in the corpus/queries is being parsed as media by the chat template.
This script bisects to find it.

Usage: uv run python src/find_video_trigger.py [--split queries|corpus]
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="queries", choices=["queries", "corpus"])
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset("mteb/MMarcoRetrieval", args.split)
    key = list(ds.keys())[0]
    texts = [t for t in ds[key]["text"]]
    if args.limit:
        texts = texts[: args.limit]
    print(f"{args.split}: {len(texts)} texts")

    m = SentenceTransformer(
        "models/embeddinggemma-2",
        model_kwargs={"torch_dtype": torch.bfloat16},
        config_kwargs={"vision_config": None, "audio_config": None},
        device="cuda",
    )
    m.max_seq_length = 2048

    def encode(chunk: list[str]) -> None:
        m.encode(chunk, prompt="task: search result | query: ", normalize_embeddings=True, show_progress_bar=False)

    # 1. whole set
    try:
        encode(texts)
        print("whole set: OK — no offending text")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"whole set FAILED: {type(exc).__name__}")

    # 2. chunk scan: find the first failing chunk (early-exit, one pass)
    chunk = 2000
    bad_range = None
    for start in range(0, len(texts), chunk):
        try:
            encode(texts[start : start + chunk])
        except Exception:  # noqa: BLE001
            bad_range = (start, min(start + chunk, len(texts)))
            print(f"  failing chunk: [{bad_range[0]}, {bad_range[1]})")
            break
    if bad_range is None:
        print("no failing chunk found — the failure needs a bigger batch")
        return

    # 3. bisect inside the failing chunk
    lo, hi = bad_range
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            encode(texts[lo:mid])
            lo = mid
        except Exception:  # noqa: BLE001
            hi = mid
        print(f"  narrowing: [{lo}, {hi})")
    bad = texts[lo]
    print(f"\noffending index {lo}, len={len(bad)} chars")
    print("repr:", repr(bad[:400]))
    for marker in ("<|video|>", "<|image|>", "<|audio|>", "<video>", "<image>"):
        if marker in bad:
            print(f"  contains marker: {marker}")


if __name__ == "__main__":
    main()
