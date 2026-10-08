"""Publish the model + evaluation to the HF repo (card, weights, raw scores).

Uses HfApi.upload_folder rather than `hf upload` — the CLI silently uploads nothing in
this setup (see huggingface-hub skill notes).

The card is the single source of truth: hf_card/README.md is copied to the repo root, and
the packaged model (models/embeddinggemma-2-zh-lora, produced by src/phase2/train_lora.py)
is copied next to it so the repo is loadable with a plain SentenceTransformer(repo_id).

Run (direct endpoint + local proxy; the HF mirror breaks uploads here):

    env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
      uv run python src/publish_hf.py

Usage: uv run python src/publish_hf.py [--repo Weidows/embeddinggemma-2-zh] [--dry-run]
                                       [--no-weights]
"""

from __future__ import annotations

import argparse
import glob
import re
import shutil
from pathlib import Path

import yaml

STAGE = Path("hf_upload")
MODEL_DIR = Path("models/embeddinggemma-2-zh-lora")
# checkpoints/ holds trainer state (empty after our runs) and must never be uploaded
WEIGHT_SKIP = {"checkpoints", ".cache", ".gitattributes", "README.md", "train_meta.json"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="Weidows/embeddinggemma-2-zh")
    p.add_argument("--dry-run", action="store_true", help="build and validate the staging dir only")
    p.add_argument("--no-weights", action="store_true", help="card + eval only, skip the 542MB model")
    return p.parse_args()


def validate_card(text: str) -> None:
    """The card is parsed as YAML frontmatter; an unbalanced fence silently drops it."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "missing or unbalanced --- fences in the card (must be at the very top)"
    meta = yaml.safe_load(m.group(1))
    assert isinstance(meta, dict), "frontmatter did not parse to a mapping"
    print(f"[card] frontmatter ok: {sorted(meta)}")


def stage_weights() -> int:
    """Copy the packaged model in. config.json must be the text-only one (vision/audio null)."""
    assert MODEL_DIR.exists(), f"{MODEL_DIR} missing — run src/phase2/train_lora.py first"
    assert (MODEL_DIR / "model.safetensors").exists(), "no model.safetensors in the package"
    cfg = (MODEL_DIR / "config.json").read_text(encoding="utf-8")
    assert '"vision_config": null' in cfg.replace("'", '"'), "package config.json still declares a vision tower"
    assert '"audio_config": null' in cfg.replace("'", '"'), "package config.json still declares an audio tower"

    n = 0
    total = 0
    for item in sorted(MODEL_DIR.iterdir()):
        if item.name in WEIGHT_SKIP:
            continue
        dst = STAGE / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
            total += item.stat().st_size
        n += 1
    print(f"[weights] staged {n} entries, {total/1e6:.0f} MB of files")
    return n


def main() -> None:
    args = parse_args()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "eval" / "raw").mkdir(parents=True)

    card_src = Path("hf_card/README.md")
    card = card_src.read_text(encoding="utf-8")
    validate_card(card)
    (STAGE / "README.md").write_text(card, encoding="utf-8")

    report = sorted(glob.glob("report/*.md"))
    if report:
        shutil.copy(report[-1], STAGE / "eval" / "report.md")
    cmp_file = Path("results/cmteb_comparison.md")
    if cmp_file.exists():
        shutil.copy(cmp_file, STAGE / "eval" / "comparison.md")
    n = 0
    for f in sorted(glob.glob("results/mteb_*.json")) + sorted(glob.glob("results/mteb_*.md")):
        shutil.copy(f, STAGE / "eval" / "raw" / Path(f).name)
        n += 1

    if not args.no_weights:
        stage_weights()

    files = sorted(p.relative_to(STAGE).as_posix() for p in STAGE.rglob("*") if p.is_file())
    print(f"[stage] {STAGE} -> {len(files)} files, {sum((STAGE / f).stat().st_size for f in files)/1e6:.0f} MB")
    for f in files:
        if not f.endswith((".safetensors", ".model", ".json")) or f.count("/") == 0:
            print("   ", f, f"({(STAGE / f).stat().st_size} B)")

    if args.dry_run:
        print("[dry-run] not uploading")
        return

    from huggingface_hub import HfApi

    api = HfApi(endpoint="https://huggingface.co")
    info = api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(STAGE),
        commit_message="feat: 中文 LoRA 微调权重 + 31 任务微调前后对比（如实公开零和结论与全部回归数据）",
    )
    print(f"[upload] {args.repo} -> {info}")

    # read back: a successful call is not proof of a committed upload
    files_on_hub = api.list_repo_files(args.repo, repo_type="model")
    print(f"[verify] {len(files_on_hub)} files on the Hub:")
    for f in sorted(files_on_hub):
        print("   ", f)


if __name__ == "__main__":
    main()
