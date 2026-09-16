#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/PLLaMa-7b-base}"
GRADIENT_OUTPUT="${GRADIENT_OUTPUT:-/home/yh114/workdir/Pruner-Zero/gradients/pllama/gradients_l2_PLLaMa-7b-base_128_0.pth}"
EXPECTED_GPUS="${EXPECTED_GPUS:-2}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "PLLaMA model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT != EXPECTED_GPUS )); then
  echo "Expected exactly $EXPECTED_GPUS visible A40 GPUs, but PyTorch sees $GPU_COUNT." >&2
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

mkdir -p "$(dirname "$GRADIENT_OUTPUT")" log
export MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-balanced}"
export GRADIENT_GPU_RESERVE_GIB="${GRADIENT_GPU_RESERVE_GIB:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "PLLaMA-7B gradient computation on $GPU_COUNT visible A40 GPUs"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-inherited}"
echo "Device map: $MODEL_DEVICE_MAP"
echo "Model: $MODEL_PATH"
echo "Output: $GRADIENT_OUTPUT"
echo "Calibration: WikiText-2, nsamples=128, seqlen=1024, seed=0"
echo "============================================================"

python lib/gradient_computation.py \
  --model "$MODEL_PATH" \
  --llama_version 2 \
  --nsamples 128 \
  --seqlen 1024 \
  --seed 0 \
  --scale 100 \
  --accumulation_steps 2 \
  --output_path "$GRADIENT_OUTPUT" \
  --enable_checkpointing

python - "$GRADIENT_OUTPUT" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
gradients = torch.load(path, map_location="cpu")
if not isinstance(gradients, dict) or not gradients:
    raise RuntimeError(f"Invalid gradient file: {path}")
print("Gradient verification passed")
print("Path:", path)
print("Tensor count:", len(gradients))
print("First keys:", list(gradients)[:5])
print("Size (GiB):", f"{path.stat().st_size / (1024 ** 3):.3f}")
PY
