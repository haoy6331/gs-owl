#!/usr/bin/env bash
set -euo pipefail

MODEL_KEYS="${MODEL_KEYS:-llama1_7b llama2_7b}"
SPARSITIES="${SPARSITIES:-0.6 0.7}"
COMPONENTS="${COMPONENTS:-count anchored core full}"
GPU_LIST="${GPU_LIST:-0 1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/v9_component_ablation}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/run_v9_component_ablation.py \
  --model-keys $MODEL_KEYS \
  --sparsities $SPARSITIES \
  --components $COMPONENTS \
  --gpus $GPU_LIST \
  --output-root "$OUTPUT_ROOT" \
  --resume
