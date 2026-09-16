#!/bin/bash
#SBATCH -J zs-top10
#SBATCH -o log/%A_%a.zs-top10.out.txt
#SBATCH -e log/%A_%a.zs-top10.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=0-89%6
#SBATCH --time=12:00:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

LOCAL_DATASETS_PATH="${LOCAL_DATASETS_PATH:-/home/yh114/workdir/Pruner-Zero/data_hf}"
CANDIDATES_CSV="${CANDIDATES_CSV:-owl/zero_shot_wsqrtg_v9_remaining/zero_shot_candidates_top10_ready.csv}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Top10 zero-shot array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-assigned-by-slurm}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "Candidates: $CANDIDATES_CSV"

python scripts/run_zero_shot_topk_array_task.py \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --candidates-csv "$CANDIDATES_CSV" \
  --offline \
  --local-datasets-path "$LOCAL_DATASETS_PATH" \
  --resume
