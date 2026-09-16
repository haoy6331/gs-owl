#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${1:-}"
VARIANT="${2:-depth_normalized}"

case "$MODEL_KEY" in
  llama1_13b)
    GPU_COUNT=2
    JOB_NAME="osla-depth-13b"
    ;;
  llama1_30b)
    GPU_COUNT=4
    JOB_NAME="osla-depth-30b"
    ;;
  *)
    echo "Usage: bash submit-osla-depth-l20.sh {llama1_13b|llama1_30b} [depth_absolute|depth_normalized]" >&2
    exit 2
    ;;
esac

case "$VARIANT" in
  depth_absolute|depth_normalized) ;;
  *)
    echo "Unknown depth variant: $VARIANT" >&2
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
  --export="ALL,MODEL_KEY=${MODEL_KEY},VARIANT=${VARIANT}" \
  osla-depth-l20.slurm)"

echo "Submitted $MODEL_KEY $VARIANT: $job_id"
echo "L20 GPUs on one node: $GPU_COUNT"
echo "Monitor: squeue -j $job_id -r"
echo "Output: tail -f log/${job_id}.${JOB_NAME}.out.txt"
echo "Error:  tail -f log/${job_id}.${JOB_NAME}.err.txt"
