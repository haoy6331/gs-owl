#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${MODEL_KEY:-llama2_7b}"
GPU_LIST="${GPU_LIST:-0 1}"
LOCAL_SPARSITY="${LOCAL_SPARSITY:-0.5}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-31}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/layer_sensitivity_wsqrtg}"
DENSE_PPL="${DENSE_PPL:-}"
V9_LOG="${V9_LOG:-}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

command=(
  python scripts/run_layer_sensitivity.py
  --model-key "$MODEL_KEY"
  --layer-start "$LAYER_START"
  --layer-end "$LAYER_END"
  --local-sparsity "$LOCAL_SPARSITY"
  --output-root "$OUTPUT_ROOT"
  --gpus $GPU_LIST
  --resume
)

if [[ -n "$DENSE_PPL" ]]; then
  command+=(--dense-ppl "$DENSE_PPL")
fi
if [[ -n "$V9_LOG" ]]; then
  command+=(--v9-log "$V9_LOG")
fi

"${command[@]}"
