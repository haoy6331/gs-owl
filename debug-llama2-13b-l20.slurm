#!/bin/bash
#SBATCH -J debug-l2-13b
#SBATCH -o log/%j.debug-l2-13b-l20.out.txt
#SBATCH -e log/%j.debug-l2-13b-l20.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3
#SBATCH --time=06:00:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

GPU_COUNT="${GPU_COUNT:-3}"
DEBUG_GPU_ID="${DEBUG_GPU_ID:-0,1,2}"
NSAMPLES="${NSAMPLES:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/debug_llama2_13b_l20_${GPU_COUNT}gpu}"

echo "LLaMA-2-13B L20 debug"
echo "Node: $SLURMD_NODENAME"
echo "SLURM CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-assigned-by-slurm}"
echo "GPU_COUNT: $GPU_COUNT"
echo "DEBUG_GPU_ID: $DEBUG_GPU_ID"
echo "NSAMPLES: $NSAMPLES"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"

GPU_ID="$DEBUG_GPU_ID" \
NSAMPLES="$NSAMPLES" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
bash run-a40-debug-llama2-13b.sh
