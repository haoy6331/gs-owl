#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${1:-}"
case "$MODEL_KEY" in
  llama1_13b)
    GPU_COUNT=2
    JOB_NAME="c4-l1-13b"
    ;;
  llama2_13b)
    GPU_COUNT=3
    JOB_NAME="c4-l2-13b"
    ;;
  llama1_30b)
    GPU_COUNT=4
    JOB_NAME="c4-l1-30b"
    ;;
  *)
    echo "Usage: bash submit-c4-auxiliary-l20.sh {llama1_13b|llama2_13b|llama1_30b}" >&2
    exit 2
    ;;
esac

mkdir -p log

job_id="$(sbatch --parsable \
  --job-name="$JOB_NAME" \
  --gres="gpu:${GPU_COUNT}" \
  --cpus-per-task="$GPU_COUNT" \
  --output="log/%j.${JOB_NAME}.out.txt" \
  --error="log/%j.${JOB_NAME}.err.txt" \
  --export="ALL,MODEL_KEY=${MODEL_KEY}" \
  c4-auxiliary-l20.slurm)"

echo "Submitted $MODEL_KEY C4 evaluation: $job_id"
echo "L20 GPUs on one node: $GPU_COUNT"
echo "Monitor: squeue -j $job_id -r"
echo "Output: tail -f log/${job_id}.${JOB_NAME}.out.txt"
echo "Error:  tail -f log/${job_id}.${JOB_NAME}.err.txt"
