#!/bin/bash
#SBATCH -J l2-13b-wsqrtg-sum
#SBATCH -o log/%j.l2-13b-wsqrtg-summary.out.txt
#SBATCH -e log/%j.l2-13b-wsqrtg-summary.err.txt
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

OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama2_13b_l20_retry}"

python scripts/summarize_v9_param_sweep.py \
  --output-root "$OUTPUT_ROOT"

echo "LLaMA-2-13B best:"
cat "$OUTPUT_ROOT/best_by_sparsity.csv"
