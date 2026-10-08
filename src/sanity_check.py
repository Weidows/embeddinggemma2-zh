"""Sanity check + throughput probe for google/embeddinggemma-2 (text-only).

Verifies (1) the model loads with vision/audio encoders disabled, (2) the output is
768-dim and the model-card example gives a sane similarity, (3) task-instruction
prefixes actually change the embedding, (4) measures encode throughput so the MTEB
runtime can be projected from real numbers instead of guesses.

Usage:
    uv run python src/sanity_check.py
"""

from __future__ import annotations

import os
import time

import torch
from sentence_transformers import SentenceTransformer

MODEL_PATH = os.environ.get("MODEL_PATH", "models/embeddinggemma-2")


def main() -> None:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch={torch.__version__} device={device} dtype={dtype}")

    model = SentenceTransformer(
        MODEL_PATH,
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"vision_config": None, "audio_config": None},  # text-only: 270M
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] loaded, params={n_params/1e6:.1f}M, max_seq={model.max_seq_length}")
    print(f"[model] prompts available: {sorted((model.prompts or {}).keys())}")

    # --- 1. model-card example -------------------------------------------------
    query = "What causes the northern lights?"
    document = "The northern lights are caused by charged particles from the sun."
    q_emb = model.encode(query, prompt_name="SearchQuery", normalize_embeddings=True)
    d_emb = model.encode(document, prompt_name="Document", normalize_embeddings=True)
    print(f"[card] query dim={q_emb.shape} doc dim={d_emb.shape}")
    print(f"[card] cos(query, document) = {float(model.similarity(q_emb, d_emb)):.4f}")

    # --- 2. prompt prefix really matters ---------------------------------------
    zh = "今天天气怎么样"
    a = model.encode(zh, prompt_name="SearchQuery", normalize_embeddings=True)
    b = model.encode(zh, prompt_name="Clustering", normalize_embeddings=True)
    c = model.encode(zh, normalize_embeddings=True)
    print(f"[prompt] cos(SearchQuery, Clustering) = {float(model.similarity(a, b)):.4f} (must be < 1.0)")
    print(f"[prompt] cos(SearchQuery, no-prefix)  = {float(model.similarity(a, c)):.4f} (must be < 1.0)")

    # --- 3. matryoshka truncation + re-normalisation ---------------------------
    t256 = model.encode(query, prompt_name="SearchQuery", truncate_dim=256, normalize_embeddings=True)
    print(f"[mrl] truncate_dim=256 -> dim={t256.shape}, |v|={float((t256**2).sum()**0.5):.4f}")

    # --- 4. throughput probe (Chinese, ~64 subwords/doc) -----------------------
    doc = (
        "上海市钢铁供应链公司主要从事服务器滑轨用冷轧钢带的对外供货业务，"
        "上游货源来自河北钢厂，下游客户覆盖数据中心机柜与服务器整机制造企业。"
    )
    docs = [f"{doc}（编号 {i}）" for i in range(2048)]
    for bs in (64, 128):
        model.encode(docs[:256], batch_size=bs, normalize_embeddings=True)  # warmup
        t0 = time.perf_counter()
        model.encode(docs, batch_size=bs, normalize_embeddings=True)
        dt = time.perf_counter() - t0
        print(f"[speed] batch_size={bs}: {len(docs)/dt:.0f} docs/s ({dt:.2f}s for {len(docs)} docs)")


if __name__ == "__main__":
    main()
