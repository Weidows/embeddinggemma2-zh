"""Phase 2 / step 3: LoRA fine-tune of the text tower on the Chinese pairs.

Design notes (each one is a deliberate choice, not a default):

- **Text tower only** (271M). The deliverable is a Chinese *text* embedding model; loading the
  vision/audio encoders would triple the memory for no benefit. Stated on the model card.
- **bf16, never fp16** — the model card warns activations exceed fp16's range and degrade
  silently.
- **sentence-transformers' text modality pin** — the same pin used for evaluation, so training
  and evaluation see identical tokenisation (and we avoid the media-URL crash on corpus text).
- **Prefixes are already baked into the training text** by build_train_data.py, matching what
  the eval harness prepends.
- **CachedMultipleNegativesRankingLoss** — in-batch negatives with a small device mini-batch,
  which is what lets a 16GB card use a large effective batch.
- **VRAM hard cap** via `set_per_process_memory_fraction` (default 0.7).

Usage:
    env -u HF_ENDPOINT PYTORCH_ALLOC_CONF=expandable_segments:True \
      uv run python src/phase2/train_lora.py --smoke          # 20 steps, validates the pipeline
    env -u HF_ENDPOINT PYTORCH_ALLOC_CONF=expandable_segments:True \
      uv run python src/phase2/train_lora.py --epochs 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

BASE = "models/embeddinggemma-2"
PAIRS = Path("data/train_pairs.jsonl")
OUT_ROOT = Path("models")

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embedding_projection", "per_layer_projection",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=BASE)
    p.add_argument("--pairs", default=str(PAIRS))
    p.add_argument("--out", default=None, help="output dir (default models/<tag>)")
    p.add_argument("--tag", default="embeddinggemma-2-zh-lora")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mini-batch", type=int, default=32, help="device mini-batch for cached MNRL")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=192)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--vram-fraction", type=float, default=0.7)
    p.add_argument("--smoke", action="store_true", help="20 steps only, to validate the pipeline")
    p.add_argument("--val-ratio", type=float, default=0.01)
    return p.parse_args()


def pin_text_modality(st: SentenceTransformer) -> None:
    """Same pin as src/eval_mteb.py — keeps train/eval tokenisation identical."""
    mod = st[0]
    mod.modality_config = {"text": {"method": "forward", "method_output_name": "last_hidden_state"}}
    mod.module_output_name = "token_embeddings"
    mod.input_formatter.supported_modalities = ["text"]


def merge_and_unwrap(model: SentenceTransformer) -> tuple[object, int]:
    """Fold in-place LoRA layers into their base weights and drop the wrappers.

    sentence-transformers injects adapters **in place** (peft's `inject_adapter_in_model`), so
    the top-level class stays `EmbeddingGemma2Model`. Consequences, both verified on a smoke run:

    - `auto_model.merge_and_unload()` does not exist (not a PeftModel);
    - `save_pretrained()` keeps writing ONLY adapter files (`adapter_config.json` +
      `adapter_model.safetensors`), i.e. no standalone artifact;
    - `mod.merge()` alone is not enough — the tuner wrapper stays in the tree (25 of them here),
      so the wrappers must also be replaced by their base layer.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    auto = model[0].auto_model
    n = 0
    for parent in auto.modules():
        for attr, child in list(parent.named_children()):
            if isinstance(child, BaseTunerLayer):
                child.merge()
                base = child.get_base_layer() if hasattr(child, "get_base_layer") else child.base_layer
                setattr(parent, attr, base)
                n += 1
    return auto, n


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out) if args.out else OUT_ROOT / (f"{args.tag}-smoke" if args.smoke else args.tag)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
        print(f"[guard] VRAM capped at {args.vram_fraction:.0%}")

    model = SentenceTransformer(
        args.base,
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"vision_config": None, "audio_config": None},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    n_base = sum(p.numel() for p in model.parameters())
    model.max_seq_length = args.max_seq_length

    # add_adapter FIRST, then pin: PEFT walks the module tree at injection time, and running it
    # on the untouched tree keeps it away from the multimodal sub-modules (a first attempt with
    # the pin applied first hit `Gemma4ClippableLinear` from the vision tower, which PEFT cannot
    # wrap, even though the encoders are disabled via config_kwargs).
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=TARGET_MODULES,
    )
    model.add_adapter(lora)
    pin_text_modality(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] base={n_base/1e6:.1f}M  trainable={n_train/1e6:.2f}M ({n_train/n_base:.2%})")

    ds = load_dataset("json", data_files=args.pairs, split="train")
    ds = ds.select_columns(["anchor", "positive"])
    split = ds.train_test_split(test_size=args.val_ratio, seed=42)
    train_ds, val_ds = split["train"], split["test"]
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=20 if args.smoke else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        bf16=dtype is torch.bfloat16,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps" if not args.smoke else "no",
        eval_steps=200,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        seed=42,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
    )
    t0 = time.perf_counter()
    trainer.train()
    print(f"[train] done in {(time.perf_counter()-t0)/60:.1f} min")

    # ---- package a standalone model ------------------------------------------------
    auto, n_merged = merge_and_unwrap(model)
    print(f"[merge] folded {n_merged} LoRA layers into the base weights")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config + weights by hand instead of auto.save_pretrained(): transformers still sees
    # peft bookkeeping left by ST and takes the adapter branch, which crashes with
    # `UnboundLocalError: cannot access local variable 'active_adapters'`.
    from safetensors.torch import save_file

    auto.config.vision_config = None  # text-tower-only build; keeps the 1.5GB multimodal weights out
    auto.config.audio_config = None
    auto.config.save_pretrained(str(out_dir))
    sd = {k: v.detach().cpu().contiguous() for k, v in auto.state_dict().items() if "lora_" not in k}
    save_file(sd, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    n_params = sum(v.numel() for v in sd.values())
    print(f"[package] config.json + model.safetensors ({n_params/1e6:.1f}M params, {len(sd)} tensors)")

    import shutil

    for item in Path(args.base).iterdir():
        name = item.name
        if name in {"model.safetensors", "config.json", "README.md", ".cache", ".gitattributes"}:
            continue
        if (out_dir / name).exists():
            continue
        if item.is_dir():
            shutil.copytree(item, out_dir / name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, out_dir / name)

    meta = {
        "base_model": args.base,
        "pairs": args.pairs,
        "n_pairs": len(train_ds),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "targets": TARGET_MODULES},
        "trainable_params": n_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "max_seq_length": args.max_seq_length,
        "dtype": str(dtype),
        "smoke": args.smoke,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out_dir}")

    # verify the saved artifact actually loads and differs from the base model.
    # NOTE: load with the same config_kwargs as everything else — reloading without them
    # rebuilds the vision/audio towers, which is both pointless and (for PEFT) fatal.
    rt = SentenceTransformer(
        str(out_dir),
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"vision_config": None, "audio_config": None},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    pin_text_modality(rt)
    rt.max_seq_length = args.max_seq_length
    probe = ["蚂蚁借呗怎么关闭", "花呗额度怎么提升"]
    import numpy as np

    new = rt.encode(probe, normalize_embeddings=True)
    base = model.encode(probe, normalize_embeddings=True)
    import numpy as np

    delta = float(np.abs(new - base).max())
    print(f"[verify] reload ok, dim={new.shape[1]}, cos={float((new * base).sum(1).mean()):.6f}, max|delta|={delta:.2e}")
    if delta == 0.0:
        print("[verify] WARNING: embeddings identical to the base model — the merge may not have applied")


if __name__ == "__main__":
    main()
