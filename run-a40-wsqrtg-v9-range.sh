#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0,1}"
MODEL_KEY="${MODEL_KEY:-llama2_13b}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-359}"
RUN_SUMMARY="${RUN_SUMMARY:-0}"

case "$MODEL_KEY" in
  llama2_13b)
    MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/Llama-2-13b-hf}"
    GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-13b-hf_128_0.pth}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama2_13b_a40_2gpu_fixed}"
    ;;
  *)
    echo "Unknown MODEL_KEY: $MODEL_KEY" >&2
    echo "Supported: llama2_13b" >&2
    exit 1
    ;;
esac

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
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export DEBUG_CUDA_SYNC="${DEBUG_CUDA_SYNC:-1}"
export DEBUG_DEVICE_MAP="${DEBUG_DEVICE_MAP:-0}"
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "A40 terminal range run: OWL-V9 + wsqrtg"
echo "MODEL_KEY: $MODEL_KEY"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "DEBUG_CUDA_SYNC: $DEBUG_CUDA_SYNC"
echo "DEBUG_DEVICE_MAP: $DEBUG_DEVICE_MAP"
echo "SKIP_CUDA_EMPTY_CACHE: $SKIP_CUDA_EMPTY_CACHE"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "Task range: $START_INDEX-$END_INDEX / 0-359"
echo "Model: $MODEL_PATH"
echo "Gradient: $GRADIENT_PATH"
echo "Output: $OUTPUT_ROOT"
echo "============================================================"

python scripts/run_v9_param_sweep.py \
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
  --resume

if [[ "$RUN_SUMMARY" == "1" ]]; then
  python scripts/summarize_v9_param_sweep.py --output-root "$OUTPUT_ROOT"
fi

echo "Done range $START_INDEX-$END_INDEX."
