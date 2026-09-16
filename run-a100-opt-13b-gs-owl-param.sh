#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/opt-13b}"
GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/opt/gradients_l2_opt-13b_128_0.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/gs_owl_param_opt_13b_a100_2gpu}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-359}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
EXPECTED_GPUS="${EXPECTED_GPUS:-2}"
DEFER_SUMMARY="${DEFER_SUMMARY:-0}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$GRADIENT_PATH" ]]; then
  echo "Gradient file does not exist: $GRADIENT_PATH" >&2
  exit 1
fi

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT < EXPECTED_GPUS )); then
  echo "Expected at least $EXPECTED_GPUS visible A100 GPUs, but PyTorch sees $GPU_COUNT." >&2
  echo "Current CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

export MODEL_DEVICE_MAP="balanced"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT_ROOT"

runner_extra_args=()
if [[ "$DEFER_SUMMARY" == "1" ]]; then
  runner_extra_args+=(--defer-summary)
fi

echo "============================================================"
echo "OPT-13B GS-OWL parameter sweep on $GPU_COUNT visible A100 GPUs"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-inherited}"
echo "Device map: $MODEL_DEVICE_MAP"
echo "Task range: $START_INDEX-$END_INDEX / 0-359"
echo "Model: $MODEL_PATH"
echo "Gradient: $GRADIENT_PATH"
echo "Output: $OUTPUT_ROOT"
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
  --start-index "$START_INDEX" \
  --end-index "$END_INDEX" \
  --extra-args "--gradient_path $GRADIENT_PATH" \
  "${runner_extra_args[@]}" \
  --resume

if [[ "$RUN_SUMMARY" == "1" ]]; then
  python scripts/summarize_v9_param_sweep.py --output-root "$OUTPUT_ROOT"
fi
