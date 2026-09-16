#!/usr/bin/env bash
set -euo pipefail

QUESTIONS="${QUESTIONS:-data/plant_mcq/mobiplant_expert_all.jsonl}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL is required}"
PRUNED_MODEL="${PRUNED_MODEL:?PRUNED_MODEL is required}"
OUTPUT="${OUTPUT:-owl/plant_mcq_eval/results.json}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python scripts/eval_plant_mcq.py \
  --model "base=$BASE_MODEL" \
  --model "pruned=$PRUNED_MODEL" \
  --questions "$QUESTIONS" \
  --output "$OUTPUT" \
  --device-map auto
