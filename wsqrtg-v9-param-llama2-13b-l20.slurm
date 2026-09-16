#!/bin/bash
#SBATCH -J l2-13b-wsqrtg
#SBATCH -o log/%A_%a.l2-13b-wsqrtg.out.txt
#SBATCH -e log/%A_%a.l2-13b-wsqrtg.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%3
#SBATCH --time=12:00:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

pick_existing_path() {
  local explicit_path="$1"
  shift
  if [[ -n "$explicit_path" ]]; then
    echo "$explicit_path"
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "$1"
}

MODEL_PATH="$(pick_existing_path "${LLAMA2_13B_MODEL_PATH:-}" \
  "/home/yh114/workdir/DSnoT/models/Llama-2-13b-hf" \
  "/home/yh114/workdir/models/Llama-2-13b-hf")"
GRADIENT_PATH="$(pick_existing_path "${LLAMA2_13B_GRADIENT_PATH:-}" \
  "/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-13b-hf_128_0.pth")"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama2_13b_l20_retry}"
NUM_SHARDS=6

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$GRADIENT_PATH" ]]; then
  echo "Gradient file does not exist: $GRADIENT_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "LLaMA-2-13B OWL-V9 + wsqrtg parameter sweep retry"
echo "Shard: $SLURM_ARRAY_TASK_ID/$((NUM_SHARDS - 1)) (60 of 360 experiments)"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-assigned-by-slurm}"
echo "Model: $MODEL_PATH"
echo "Gradient: $GRADIENT_PATH"
echo "Output: $OUTPUT_ROOT"

python scripts/run_v9_shard.py \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS" \
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
  --extra-args "--gradient_path $GRADIENT_PATH" \
  --resume
