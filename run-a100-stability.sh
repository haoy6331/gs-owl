#!/usr/bin/env bash
set -euo pipefail

MODEL_KEYS="${MODEL_KEYS:-llama1_7b llama2_7b}"
METHODS="${METHODS:-owl-v9-wsqrtg}"
GPU_LIST="${GPU_LIST:-0 1}"
SPARSITY="${SPARSITY:-0.6}"
SEEDS="${SEEDS:-0 1 2}"
NSAMPLES="${NSAMPLES:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/stability_wsqrtg_v9}"

mkdir -p "$OUTPUT_ROOT" log
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "A100 stability experiment"
echo "Models: $MODEL_KEYS"
echo "Methods: $METHODS"
echo "GPUs: $GPU_LIST"
echo "Sparsity: $SPARSITY"
echo "Seeds: $SEEDS"
echo "Nsamples: $NSAMPLES"
echo "Output: $OUTPUT_ROOT"
echo "============================================================"

python scripts/run_robustness_experiments.py \
  --experiment stability \
  --model-keys $MODEL_KEYS \
  --methods $METHODS \
  --sparsity "$SPARSITY" \
  --seeds $SEEDS \
  --default-nsamples "$NSAMPLES" \
  --output-root "$OUTPUT_ROOT" \
  --gpus $GPU_LIST \
  --resume
