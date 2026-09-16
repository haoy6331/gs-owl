#!/bin/bash
#SBATCH -J pl-mcq-clean-sum
#SBATCH -o log/%j.pl-mcq-clean-summary.out.txt
#SBATCH -e log/%j.pl-mcq-clean-summary.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" 

MODEL_SIZE="${MODEL_SIZE:?MODEL_SIZE must be exported as 7b or 13b}" \
MODE=summary \
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be exported}" \
bash run-pllama-plant-mcq-fixed-clean.sh
