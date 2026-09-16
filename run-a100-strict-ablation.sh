#!/usr/bin/env bash
set -euo pipefail

MODEL_KEYS="${MODEL_KEYS:-llama1_7b llama2_7b}"
METHODS="${METHODS:-wanda wanda-owl wanda-owl-v9 wsqrtg owl-wsqrtg owl-v9-wsqrtg}"
SPARSITIES="${SPARSITIES:-0.5 0.6 0.7}"
GPU_LIST="${GPU_LIST:-0 1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/strict_ablation_wsqrtg_owl_v9}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUTPUT_ROOT" log

echo "============================================================"
echo "Strict ablation: 2 metrics x 3 allocation methods"
echo "Models: $MODEL_KEYS"
echo "Methods: $METHODS"
echo "Sparsities: $SPARSITIES"
echo "GPU slots: $GPU_LIST"
echo "Output: $OUTPUT_ROOT"
echo "============================================================"

command=(
  python scripts/run_strict_ablation.py
  --model-keys $MODEL_KEYS
  --methods $METHODS
  --sparsities $SPARSITIES
  --output-root "$OUTPUT_ROOT"
  --nsamples "$NSAMPLES"
  --seed "$SEED"
  --gpus $GPU_LIST
  --resume
)

if [[ -n "$EXTRA_ARGS" ]]; then
  command+=(--extra-args "$EXTRA_ARGS")
fi

"${command[@]}"
