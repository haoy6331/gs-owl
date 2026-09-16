#!/bin/bash
#SBATCH -J l1-30b-summary
#SBATCH -o log/%j.l1-30b-summary.out.txt
#SBATCH -e log/%j.l1-30b-summary.err.txt
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

python scripts/summarize_v9_param_sweep.py \
  --output-root "${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama1_30b_l20}"
