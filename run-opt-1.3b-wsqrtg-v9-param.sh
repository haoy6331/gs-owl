#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/opt-1.3b}"
GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/opt/gradients_l2_opt-1.3b_128_0.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_opt_1_3b}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$GRADIENT_PATH" ]]; then
  echo "Gradient file does not exist: $GRADIENT_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" log
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "OPT-1.3B terminal run: OWL-V9 + wsqrtg parameter sweep"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Gradient: $GRADIENT_PATH"
echo "Output: $OUTPUT_ROOT"
echo "Total: 3 sparsities x 3 Hyper_m x 10 Lamda x 4 Owl_alpha = 360"
echo "============================================================"

python scripts/run_v9_param_sweep.py \
  --main main_opt.py \
  --model "$MODEL_PATH" \
  --method owl-v9-wsqrtg \
  --output-root "$OUTPUT_ROOT" \
  --sparsities 0.5 0.6 0.7 \
  --hyper-ms 5 6 7 \
  --lamdas 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 \
  --owl-alphas 0.10 0.15 0.20 0.25 \
  --sparsity-type unstructured \
  --nsamples 128 \
  --seed 0 \
  --extra-args "--gradient_path $GRADIENT_PATH" \
  --resume

if [[ "$RUN_SUMMARY" == "1" ]]; then
  python scripts/summarize_v9_param_sweep.py --output-root "$OUTPUT_ROOT"
fi

echo "Done."
echo "Summary: $OUTPUT_ROOT/summary_all.csv"
echo "Best: $OUTPUT_ROOT/best_by_sparsity.csv"
