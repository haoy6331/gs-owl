#!/usr/bin/env bash
set -euo pipefail

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

MODEL_PATH="$(pick_existing_model "${MODEL_PATH:-${VICUNA_7B_MODEL_PATH:-/home/yh114/workdir/DSnoT/models/Vicuna-7B}}" \
  "/home/yh114/workdir/DSnoT/models/Vicuna-7B" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.5" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.3" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.1" \
  "/home/yh114/workdir/DSnoT/models/vicuna-7b" \
  "/home/yh114/workdir/models/vicuna-7b-v1.5" \
  "/home/yh114/workdir/models/vicuna-7b")"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Vicuna model directory does not exist: $MODEL_PATH" >&2
  echo "Set VICUNA_7B_MODEL_PATH to the merged Hugging Face Vicuna-7B directory." >&2
  exit 1
fi

mkdir -p log

export_args="ALL,VICUNA_7B_MODEL_PATH=${MODEL_PATH}"

job_id="$(sbatch --parsable \
  --job-name="vicuna-grad" \
  --output="log/%j.vicuna-grad.out.txt" \
  --error="log/%j.vicuna-grad.err.txt" \
  --export="$export_args" \
  vicuna-7b-gradient-l20.slurm)"

echo "Submitted Vicuna-7B gradient job: $job_id"
echo "Monitor: squeue -j $job_id -r"
echo "Output: tail -f log/${job_id}.vicuna-grad.out.txt"
echo "Expected output: /home/yh114/workdir/Pruner-Zero/gradients/gradients_l2_Vicuna-7B_128_0.pth"
