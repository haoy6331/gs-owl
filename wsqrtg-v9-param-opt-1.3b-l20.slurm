#!/bin/bash
#SBATCH -J opt-1.3b-wsqrtg
#SBATCH -o log/%A_%a.opt-1.3b-wsqrtg.out.txt
#SBATCH -e log/%A_%a.opt-1.3b-wsqrtg.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%1
#SBATCH --time=12:00:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

MODEL_PATH="${OPT_1_3B_MODEL_PATH:-/home/yh114/workdir/DSnoT/models/opt-1.3b}"
GRADIENT_PATH="${OPT_1_3B_GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/opt/gradients_l2_opt-1.3b_128_0.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_opt_1_3b}"
NUM_SHARDS="${NUM_SHARDS:-6}"

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

echo "OPT-1.3B OWL-V9 + wsqrtg parameter sweep (1 x L20)"
echo "Shard: $SLURM_ARRAY_TASK_ID/$((NUM_SHARDS - 1))"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-assigned-by-slurm}"
echo "Model: $MODEL_PATH"
echo "Gradient: $GRADIENT_PATH"
echo "Output: $OUTPUT_ROOT"

python scripts/run_v9_shard.py \
  --main main_opt.py \
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
