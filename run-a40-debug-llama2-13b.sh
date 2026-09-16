#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0,1}"
MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/Llama-2-13b-hf}"
GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-13b-hf_128_0.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/debug_llama2_13b_a40_2gpu}"

SPARSITY="${SPARSITY:-0.5}"
HYPER_M="${HYPER_M:-5}"
LAMDA="${LAMDA:-0.08}"
OWL_ALPHA="${OWL_ALPHA:-0.15}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"

RUN_TAG="s_${SPARSITY//./p}_hm_${HYPER_M//./p}_la_${LAMDA//./p}_alpha_${OWL_ALPHA//./p}_n_${NSAMPLES}"
SAVE_DIR="${SAVE_DIR:-$OUTPUT_ROOT/$RUN_TAG}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$GRADIENT_PATH" ]]; then
  echo "Gradient file does not exist: $GRADIENT_PATH" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR" log

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export DEBUG_CUDA_SYNC="${DEBUG_CUDA_SYNC:-1}"
export DEBUG_DEVICE_MAP="${DEBUG_DEVICE_MAP:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "Debug LLaMA-2-13B OWL-V9 + wsqrtg on A40"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "DEBUG_CUDA_SYNC: $DEBUG_CUDA_SYNC"
echo "DEBUG_DEVICE_MAP: $DEBUG_DEVICE_MAP"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "Model: $MODEL_PATH"
echo "Gradient: $GRADIENT_PATH"
echo "Save: $SAVE_DIR"
echo "Params: sparsity=$SPARSITY Hyper_m=$HYPER_M Lamda=$LAMDA Owl_alpha=$OWL_ALPHA nsamples=$NSAMPLES seed=$SEED"
echo "============================================================"

python main.py \
  --model "$MODEL_PATH" \
  --cache_dir llm_weights \
  --prune_method owl-v9-wsqrtg \
  --sparsity_ratio "$SPARSITY" \
  --sparsity_type unstructured \
  --nsamples "$NSAMPLES" \
  --seed "$SEED" \
  --Hyper_m "$HYPER_M" \
  --Lamda "$LAMDA" \
  --Owl_alpha "$OWL_ALPHA" \
  --save "$SAVE_DIR" \
  --gradient_path "$GRADIENT_PATH" \
  2>&1 | tee "$SAVE_DIR/debug_stdout.log"
