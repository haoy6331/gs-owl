#!/usr/bin/env bash
set -euo pipefail

MODEL_KEYS="${MODEL_KEYS:-llama1_7b llama2_7b}"
GPU_LIST="${GPU_LIST:-0 1}"
SEEDS="${SEEDS:-0 1 2}"
SPARSITY="${SPARSITY:-0.6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/stability_wanda_owl_v9}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/run_robustness_experiments.py \
  --experiment stability \
  --model-keys $MODEL_KEYS \
  --methods wanda-owl-v9 \
  --sparsity "$SPARSITY" \
  --seeds $SEEDS \
  --gpus $GPU_LIST \
  --output-root "$OUTPUT_ROOT" \
  --resume
