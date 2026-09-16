#!/usr/bin/env bash
set -euo pipefail

GPU_GROUPS="${GPU_GROUPS:-0 1}"
MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/opt-2.7b}"
GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/opt/gradients_l2_opt-2.7b_128_0.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/gs_owl_param_opt_2_7b_a100}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-359}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$GRADIENT_PATH" ]]; then
  echo "Gradient file does not exist: $GRADIENT_PATH" >&2
  exit 1
fi

read -r -a GPU_GROUP_ARGS <<< "$GPU_GROUPS"
export MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-auto}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT_ROOT"

echo "============================================================"
echo "OPT-2.7B GS-OWL parameter sweep"
echo "GPU groups: $GPU_GROUPS"
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
  --gpu-groups "${GPU_GROUP_ARGS[@]}" \
  --start-index "$START_INDEX" \
  --end-index "$END_INDEX" \
  --extra-args "--gradient_path $GRADIENT_PATH" \
  --resume

if [[ "$RUN_SUMMARY" == "1" ]]; then
  python scripts/summarize_v9_param_sweep.py --output-root "$OUTPUT_ROOT"
fi
