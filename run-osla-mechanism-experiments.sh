#!/usr/bin/env bash
set -euo pipefail

SUITE="${SUITE:-projection}"
MODEL_KEYS="${MODEL_KEYS:-llama1_7b llama2_7b}"
SPARSITIES="${SPARSITIES:-0.7}"
VARIANTS="${VARIANTS:-}"
GPU_LIST="${GPU_LIST:-0 1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/osla_mechanism_ablation}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

variant_args=()
if [[ -n "$VARIANTS" ]]; then
  read -r -a variant_values <<< "$VARIANTS"
  variant_args=(--variants "${variant_values[@]}")
fi

read -r -a model_values <<< "$MODEL_KEYS"
read -r -a sparsity_values <<< "$SPARSITIES"
read -r -a gpu_values <<< "$GPU_LIST"

python scripts/run_osla_mechanism_experiments.py \
  --suite "$SUITE" \
  --model-keys "${model_values[@]}" \
  --sparsities "${sparsity_values[@]}" \
  "${variant_args[@]}" \
  --gpus "${gpu_values[@]}" \
  --output-root "$OUTPUT_ROOT" \
  --resume
