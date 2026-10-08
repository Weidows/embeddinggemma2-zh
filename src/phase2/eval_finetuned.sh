#!/usr/bin/env bash
# Evaluate a fine-tuned / packaged model with the same harness as the base run.
#
#   ./src/phase2/eval_finetuned.sh                 # quick signal on the weak tasks (~15 min)
#   ./src/phase2/eval_finetuned.sh --full          # full C-MTEB, 31 tasks (~174 min)
#   ./src/phase2/eval_finetuned.sh --model models/<other>
#
# The weak list is the set where the base model trails the locally-run bge-m3 baseline by ~0.10
# (see results/cmteb_comparison.md): AFQMC / ATEC / BQ / MedicalRetrieval / CmedqaRetrieval /
# CMedQAv1-reranking / CMedQAv2-reranking. LCQMC is included as a control — it was NOT weak
# (0.7253), so a drop there means the fine-tune is trading away general performance.
set -euo pipefail

cd "$(dirname "$0")/../.."
unset HF_ENDPOINT
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL="models/embeddinggemma-2-zh-lora"
FULL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -e "$MODEL/config.json" ]]; then
  echo "!! $MODEL/config.json not found — did training finish?" >&2
  exit 1
fi

if [[ "$FULL" == "1" ]]; then
  echo "== full C-MTEB (31 tasks) on $MODEL =="
  uv run python src/eval_mteb.py --model "$MODEL" --tag finetuned
else
  echo "== quick signal (weak tasks + LCQMC control) on $MODEL =="
  uv run python src/eval_mteb.py --model "$MODEL" --tag finetuned_weak \
    --tasks AFQMC ATEC BQ LCQMC MedicalRetrieval CmedqaRetrieval CMedQAv1-reranking CMedQAv2-reranking
fi
