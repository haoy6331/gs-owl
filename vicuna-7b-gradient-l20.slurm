#!/bin/bash
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

pick_existing_model() {
  local explicit_path="$1"
  shift
  if [[ -n "$explicit_path" ]]; then
    echo "$explicit_path"
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "$1"
}

REPO_DIR="${REPO_DIR:-/home/yh114/workdir/Prune}"
MODEL_PATH="$(pick_existing_model "${MODEL_PATH:-${VICUNA_7B_MODEL_PATH:-/home/yh114/workdir/DSnoT/models/Vicuna-7B}}" \
  "/home/yh114/workdir/DSnoT/models/Vicuna-7B" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.5" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.3" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.1" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b" \
  "/home/yh114/workdir/models/vicuna-7b-v1.5" \
  "/home/yh114/workdir/models/vicuna-7b")"
SAVE_ROOT="${SAVE_ROOT:-/home/yh114/workdir/Pruner-Zero/gradients}"
GRADIENT_OUTPUT="${GRADIENT_OUTPUT:-${VICUNA_7B_GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/gradients_l2_Vicuna-7B_128_0.pth}}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Vicuna model directory does not exist: $MODEL_PATH" >&2
  echo "Set VICUNA_7B_MODEL_PATH to the merged Hugging Face Vicuna-7B directory." >&2
  exit 1
fi

cd "$REPO_DIR"
mkdir -p log "$SAVE_ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Vicuna-7B offline gradient computation"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-assigned-by-slurm}"
echo "Model: $MODEL_PATH"
echo "Gradient output: $GRADIENT_OUTPUT"

python lib/gradient_computation.py \
  --model "$MODEL_PATH" \
  --llama_version 1 \
  --nsamples 128 \
  --seqlen 1024 \
  --seed 0 \
  --scale 100 \
  --save_root "$SAVE_ROOT" \
  --output_path "$GRADIENT_OUTPUT" \
  --enable_checkpointing
