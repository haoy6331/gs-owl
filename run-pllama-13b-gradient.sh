#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-}"
MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/PLLaMa-13b-base}"
GRADIENT_OUTPUT="${GRADIENT_OUTPUT:-/home/yh114/workdir/Pruner-Zero/gradients/pllama/gradients_l2_PLLaMa-13b-base_128_0.pth}"
EXPECTED_GPUS="${EXPECTED_GPUS:-2}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

export MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-balanced}"
export GRADIENT_GPU_RESERVE_GIB="${GRADIENT_GPU_RESERVE_GIB:-6}"
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -d "$MODEL_PATH" ]] || {
  echo "PLLaMA-13B model directory does not exist: $MODEL_PATH" >&2
  exit 1
}

GPU_COUNT="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT != EXPECTED_GPUS )); then
  echo "Expected exactly $EXPECTED_GPUS visible GPUs, but PyTorch sees $GPU_COUNT." >&2
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
  echo "Override EXPECTED_GPUS only when you intentionally change the allocation." >&2
  exit 1
fi

mkdir -p "$(dirname "$GRADIENT_OUTPUT")" log

echo "============================================================"
echo "PLLaMA-13B gradient computation"
echo "Visible GPUs: $GPU_COUNT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-inherited}"
echo "Device map: $MODEL_DEVICE_MAP"
echo "Model: $MODEL_PATH"
echo "Output: $GRADIENT_OUTPUT"
echo "Calibration: WikiText-2, nsamples=128, seqlen=1024, seed=0"
echo "============================================================"

"$PYTHON_BIN" lib/gradient_computation.py \
  --model "$MODEL_PATH" \
  --llama_version 2 \
  --nsamples 128 \
  --seqlen 1024 \
  --seed 0 \
  --scale 100 \
  --accumulation_steps 2 \
  --output_path "$GRADIENT_OUTPUT" \
  --enable_checkpointing

"$PYTHON_BIN" - "$GRADIENT_OUTPUT" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
if not path.is_file():
    raise FileNotFoundError(f"Gradient file was not created: {path}")

gradients = torch.load(path, map_location="cpu")
if not isinstance(gradients, dict) or not gradients:
    raise RuntimeError(f"Invalid gradient file: {path}")
if not all(torch.is_tensor(value) for value in gradients.values()):
    raise RuntimeError(f"Gradient file contains non-tensor values: {path}")

print("Gradient verification passed")
print("Path:", path)
print("Tensor count:", len(gradients))
print("First keys:", list(gradients)[:5])
print("Size (GiB):", f"{path.stat().st_size / (1024 ** 3):.3f}")
PY
