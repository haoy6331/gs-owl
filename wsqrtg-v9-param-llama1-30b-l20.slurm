#!/bin/bash
#SBATCH -J l1-30b-wsqrtg
#SBATCH -o log/%A_%a.l1-30b-wsqrtg.out.txt
#SBATCH -e log/%A_%a.l1-30b-wsqrtg.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --array=0-11%1
#SBATCH --time=12:00:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

MODEL_PATH="${LLAMA1_30B_MODEL_PATH:-/home/yh114/workdir/DSnoT/models/decapoda-research-llama-30B-hf}"
GRADIENT_PATH="${LLAMA1_30B_GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_decapoda-research-llama-30B-hf_128_0.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama1_30b_l20_4gpu}"
NUM_SHARDS=12

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

echo "LLaMA-1-30B OWL-V9 + wsqrtg parameter sweep (4 x L20)"
echo "Shard: $SLURM_ARRAY_TASK_ID/$((NUM_SHARDS - 1)) (30 of 360 experiments)"
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
