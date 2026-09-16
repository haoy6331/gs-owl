#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/opt-13b}"
GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/opt/gradients_l2_opt-13b_128_0.pth}"
JSON_TREE="${JSON_TREE:-data/pruner-zero_tree.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/opt_13b_baselines_a100_2gpu}"
METHODS="${METHODS:-dense magnitude wanda sparsegpt pruner-zero}"
EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-0}"
LOCAL_DATASETS_PATH="${LOCAL_DATASETS_PATH:-/home/yh114/workdir/Pruner-Zero/data_hf}"
EXPECTED_GPUS="${EXPECTED_GPUS:-2}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if [[ "$METHODS" == *"pruner-zero"* && ! -f "$GRADIENT_PATH" ]]; then
  echo "Gradient file does not exist: $GRADIENT_PATH" >&2
  exit 1
fi
if [[ "$METHODS" == *"pruner-zero"* && ! -f "$JSON_TREE" ]]; then
  echo "Pruner-Zero tree does not exist: $JSON_TREE" >&2
  exit 1
fi

read -r -a METHOD_ARGS <<< "$METHODS"
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

echo "============================================================"
echo "OPT-13B baseline experiments on $GPU_COUNT visible A100 GPUs"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-inherited}"
echo "Device map: $MODEL_DEVICE_MAP"
echo "Methods: $METHODS"
echo "Zero-shot: $EVAL_ZERO_SHOT"
echo "Output: $OUTPUT_ROOT"
echo "============================================================"

command=(
  python scripts/run_opt_baselines.py
  --main main_opt.py
  --model "$MODEL_PATH"
  --gradient-path "$GRADIENT_PATH"
  --json-tree "$JSON_TREE"
  --output-root "$OUTPUT_ROOT"
  --methods "${METHOD_ARGS[@]}"
  --sparsities 0.5 0.6 0.7
  --sparsity-type unstructured
  --nsamples 128
  --seed 0
  --resume
)

if [[ "$EVAL_ZERO_SHOT" == "1" ]]; then
  command+=(--eval-zero-shot --offline --local-datasets-path "$LOCAL_DATASETS_PATH")
fi

"${command[@]}"
python scripts/summarize_opt_baselines.py --output-root "$OUTPUT_ROOT"
