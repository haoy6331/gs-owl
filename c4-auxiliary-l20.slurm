#!/bin/bash
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

MODEL_KEY="${MODEL_KEY:?MODEL_KEY must be exported by the submit script}"
REPO_DIR="${REPO_DIR:-/home/yh114/workdir/Prune}"

cd "$REPO_DIR"
mkdir -p log

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-balanced}"
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "GS-OWL C4 cross-dataset evaluation"
echo "Model key: $MODEL_KEY"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-assigned-by-slurm}"
echo "Allocated GPUs: ${SLURM_GPUS_ON_NODE:-unknown}"
echo "Device map: $MODEL_DEVICE_MAP"

MODELS="$MODEL_KEY" bash run-c4-auxiliary.sh
