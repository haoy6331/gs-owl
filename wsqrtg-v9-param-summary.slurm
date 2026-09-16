#!/bin/bash
#SBATCH -J wsqrtg-v9-sum
#SBATCH -o log/%j.wsqrtg-v9-summary.out.txt
#SBATCH -e log/%j.wsqrtg-v9-summary.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

python scripts/summarize_v9_param_sweep.py \
  --output-root "$OUTPUT_ROOT"

echo "Best by sparsity:"
cat "$OUTPUT_ROOT/best_by_sparsity.csv"
