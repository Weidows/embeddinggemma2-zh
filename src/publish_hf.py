"""Publish the evaluation to the HF model repo (card + raw scores).

Uses HfApi.upload_folder rather than `hf upload` — the CLI silently uploads nothing in
this setup (see huggingface-hub skill notes).

The repo is created ahead of the fine-tune so the name is reserved and the evaluation is
citable from day one; the card states plainly that no fine-tuned weights exist yet.

Run (direct endpoint + local proxy; the HF mirror breaks uploads here):

    env -u HF_ENDPOINT HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
      uv run python src/publish_hf.py

Usage: uv run python src/publish_hf.py [--repo Weidows/embeddinggemma-2-zh] [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import re
import shutil
from pathlib import Path

import yaml

STAGE = Path("hf_upload")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="Weidows/embeddinggemma-2-zh")
    p.add_argument("--dry-run", action="store_true", help="build and validate the staging dir only")
    return p.parse_args()


def validate_card(text: str) -> None:
    """The card is parsed as YAML frontmatter; an unbalanced fence silently drops it."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "missing or unbalanced --- fences in the card (must be at the very top)"
    meta = yaml.safe_load(m.group(1))
    assert isinstance(meta, dict), "frontmatter did not parse to a mapping"
    print(f"[card] frontmatter ok: {sorted(meta)}")


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

    files = sorted(p.relative_to(STAGE).as_posix() for p in STAGE.rglob("*") if p.is_file())
    print(f"[stage] {STAGE} -> {len(files)} files")
    for f in files:
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
        commit_message="docs: C-MTEB 全量中文实测（31 任务）+ 原始分数；微调权重待补",
    )
    print(f"[upload] {args.repo} -> {info}")

    # read back: a successful call is not proof of a committed upload
    files_on_hub = api.list_repo_files(args.repo, repo_type="model")
    print(f"[verify] {len(files_on_hub)} files on the Hub:")
    for f in sorted(files_on_hub):
        print("   ", f)


if __name__ == "__main__":
    main()
