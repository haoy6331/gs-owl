#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${GPU_LIST:-0}"
MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/opt-1.3b}"
GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/opt/gradients_l2_opt-1.3b_128_0.pth}"
JSON_TREE="${JSON_TREE:-data/pruner-zero_tree.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/opt_1_3b_baselines_a100}"
METHODS="${METHODS:-dense magnitude wanda sparsegpt pruner-zero}"
EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-0}"
LOCAL_DATASETS_PATH="${LOCAL_DATASETS_PATH:-/home/yh114/workdir/Pruner-Zero/data_hf}"

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

read -r -a GPUS <<< "$GPU_LIST"
read -r -a METHOD_ARGS <<< "$METHODS"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT_ROOT"

echo "============================================================"
echo "OPT-1.3B baseline experiments"
echo "GPUs: $GPU_LIST"
echo "Methods: $METHODS"
echo "Sparsities: 0.5 0.6 0.7"
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
  --gpus "${GPUS[@]}"
  --resume
)

if [[ "$EVAL_ZERO_SHOT" == "1" ]]; then
  command+=(
    --eval-zero-shot
    --offline
    --local-datasets-path "$LOCAL_DATASETS_PATH"
  )
fi

"${command[@]}"

python scripts/summarize_opt_baselines.py --output-root "$OUTPUT_ROOT"
