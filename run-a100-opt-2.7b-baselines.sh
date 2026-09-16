#!/usr/bin/env bash
set -euo pipefail

GPU_GROUPS="${GPU_GROUPS:-0 1}"
MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/opt-2.7b}"
GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/opt/gradients_l2_opt-2.7b_128_0.pth}"
JSON_TREE="${JSON_TREE:-data/pruner-zero_tree.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/opt_2_7b_baselines_a100}"
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

read -r -a GPU_GROUP_ARGS <<< "$GPU_GROUPS"
read -r -a METHOD_ARGS <<< "$METHODS"
export MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-auto}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT_ROOT"

echo "============================================================"
echo "OPT-2.7B baseline experiments"
echo "GPU groups: $GPU_GROUPS"
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
  --gpu-groups "${GPU_GROUP_ARGS[@]}"
  --resume
)

if [[ "$EVAL_ZERO_SHOT" == "1" ]]; then
  command+=(--eval-zero-shot --offline --local-datasets-path "$LOCAL_DATASETS_PATH")
fi

"${command[@]}"
python scripts/summarize_opt_baselines.py --output-root "$OUTPUT_ROOT"
