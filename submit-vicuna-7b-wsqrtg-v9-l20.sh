#!/usr/bin/env bash
set -euo pipefail

pick_existing_path() {
  local path_type="$1"
  local explicit_path="$2"
  shift 2
  if [[ -n "$explicit_path" ]]; then
    echo "$explicit_path"
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if [[ "$path_type" == "dir" && -d "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
    if [[ "$path_type" == "file" && -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "$1"
}

MODEL_PATH="$(pick_existing_path dir "${MODEL_PATH:-${VICUNA_7B_MODEL_PATH:-/home/yh114/workdir/DSnoT/models/Vicuna-7B}}" \
  "/home/yh114/workdir/DSnoT/models/Vicuna-7B" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.5" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.3" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.1" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b" \
  "/home/yh114/workdir/models/vicuna-7b-v1.5" \
  "/home/yh114/workdir/models/vicuna-7b")"
GRADIENT_PATH="$(pick_existing_path file "${GRADIENT_PATH:-${VICUNA_7B_GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/gradients_l2_Vicuna-7B_128_0.pth}}" \
  "/home/yh114/workdir/Pruner-Zero/gradients/gradients_l2_Vicuna-7B_128_0.pth" \
  "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_vicuna-7b-v1.5_128_0.pth" \
  "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_vicuna-7b-v1.3_128_0.pth" \
  "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_vicuna-7b-v1.1_128_0.pth" \
  "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_vicuna-7b_128_0.pth" \
  "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_Vicuna-7B_128_0.pth" \
  "/home/yh114/workdir/Pruner-Zero/gradients/vicuna/gradients_l2_vicuna-7b-v1.5_128_0.pth" \
  "/home/yh114/workdir/Pruner-Zero/gradients/vicuna/gradients_l2_vicuna-7b_128_0.pth")"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_vicuna_7b_l20_2gpu}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Vicuna model directory does not exist: $MODEL_PATH" >&2
  echo "Set VICUNA_7B_MODEL_PATH to the merged Hugging Face Vicuna-7B directory." >&2
  exit 1
fi
if [[ ! -f "$GRADIENT_PATH" ]]; then
  echo "Vicuna gradient file does not exist: $GRADIENT_PATH" >&2
  echo "Generate it first with: bash submit-vicuna-7b-gradient-l20.sh" >&2
  exit 1
fi

export MODEL_KEY=vicuna_7b
export OUTPUT_ROOT
export VICUNA_7B_MODEL_PATH="$MODEL_PATH"
export VICUNA_7B_GRADIENT_PATH="$GRADIENT_PATH"

bash submit-wsqrtg-v9-param-two-gpu-l20.sh
