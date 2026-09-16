#!/bin/bash
#SBATCH -J pl13b-mcq-clean
#SBATCH -o log/%A_%a.pl13b-mcq-clean.out.txt
#SBATCH -e log/%A_%a.pl13b-mcq-clean.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --array=0-3%1
#SBATCH --time=24:00:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" 
mkdir -p log

case "$SLURM_ARRAY_TASK_ID" in
  0) MODE=dense; SPARSITIES="0.5 0.6 0.7" ;;
  1) MODE=fixed; SPARSITIES="0.5" ;;
  2) MODE=fixed; SPARSITIES="0.6" ;;
  3) MODE=fixed; SPARSITIES="0.7" ;;
  *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

MODEL_SIZE=13b \
MODE="$MODE" \
SPARSITIES="$SPARSITIES" \
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/pllama_13b_plant_mcq_l20_fixed_clean_v2}" \
QUESTIONS="${QUESTIONS:-data/plant_mcq/mobiplant_expert_clean_v2.jsonl}" \
REQUIRE_L20=1 \
RESUME="${RESUME:-1}" \
bash run-pllama-plant-mcq-fixed-clean.sh
