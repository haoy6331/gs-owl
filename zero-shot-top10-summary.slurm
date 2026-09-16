#!/bin/bash
#SBATCH -J zs-top10-summary
#SBATCH -o log/%j.zs-top10-summary.out.txt
#SBATCH -e log/%j.zs-top10-summary.err.txt
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

OUTPUT_ROOT="${ZERO_SHOT_OUTPUT_ROOT:-owl/zero_shot_wsqrtg_v9_remaining}"

python scripts/summarize_zero_shot_topk.py \
  --output-root "$OUTPUT_ROOT"
