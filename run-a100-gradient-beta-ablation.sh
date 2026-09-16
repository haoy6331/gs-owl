#!/usr/bin/env bash
set -euo pipefail

MODEL_KEYS="${MODEL_KEYS:-llama1_7b llama2_7b}"
BETAS="${BETAS:-0 0.25 0.5 0.75 1}"
SPARSITY="${SPARSITY:-0.6}"
GPU_LIST="${GPU_LIST:-0 1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/gradient_beta_ablation}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/run_gradient_beta_ablation.py \
  --model-keys $MODEL_KEYS \
  --betas $BETAS \
  --sparsity "$SPARSITY" \
  --gpus $GPU_LIST \
  --output-root "$OUTPUT_ROOT" \
  --resume
