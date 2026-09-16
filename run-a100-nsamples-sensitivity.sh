#!/usr/bin/env bash
set -euo pipefail

MODEL_KEYS="${MODEL_KEYS:-llama1_7b llama2_7b}"
METHODS="${METHODS:-owl-v9-wsqrtg}"
GPU_LIST="${GPU_LIST:-0 1}"
SPARSITY="${SPARSITY:-0.6}"
NSAMPLES_LIST="${NSAMPLES_LIST:-32 64 128 256}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/nsamples_sensitivity_wsqrtg_v9}"

mkdir -p "$OUTPUT_ROOT" log
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "A100 calibration-sample sensitivity experiment"
echo "Models: $MODEL_KEYS"
echo "Methods: $METHODS"
echo "GPUs: $GPU_LIST"
echo "Sparsity: $SPARSITY"
echo "Nsamples list: $NSAMPLES_LIST"
echo "Output: $OUTPUT_ROOT"
echo "============================================================"

python scripts/run_robustness_experiments.py \
  --experiment nsamples \
  --model-keys $MODEL_KEYS \
  --methods $METHODS \
  --sparsity "$SPARSITY" \
  --nsamples-list $NSAMPLES_LIST \
  --output-root "$OUTPUT_ROOT" \
  --gpus $GPU_LIST \
  --resume
