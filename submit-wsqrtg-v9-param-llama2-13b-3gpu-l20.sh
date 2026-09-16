#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama2_13b_l20_2gpu}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
NUM_SHARDS="${NUM_SHARDS:-6}"
GPU_COUNT="${GPU_COUNT:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$GPU_COUNT}"
NODELIST="${NODELIST:-}"
EXCLUDE_NODE="${EXCLUDE_NODE:-}"

mkdir -p log "$OUTPUT_ROOT"

sbatch_args=(
  --parsable
  --array="0-$((NUM_SHARDS - 1))%${MAX_PARALLEL}"
  --gres="gpu:${GPU_COUNT}"
  --cpus-per-task="$CPUS_PER_TASK"
  --export="ALL,OUTPUT_ROOT=$OUTPUT_ROOT,NUM_SHARDS=$NUM_SHARDS"
)
if [[ -n "$NODELIST" ]]; then
  sbatch_args+=(--nodelist="$NODELIST")
fi
if [[ -n "$EXCLUDE_NODE" ]]; then
  sbatch_args+=(--exclude="$EXCLUDE_NODE")
fi

array_job_id="$(sbatch "${sbatch_args[@]}" wsqrtg-v9-param-llama2-13b-3gpu-l20.slurm)"
echo "Submitted LLaMA-2-13B three-GPU L20 array: ${array_job_id}"

summary_job_id="$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  --export="ALL,OUTPUT_ROOT=$OUTPUT_ROOT" \
  wsqrtg-v9-param-summary.slurm)"
echo "Submitted summary job: ${summary_job_id}"

echo "Array: 0-$((NUM_SHARDS - 1))%${MAX_PARALLEL}; ${NUM_SHARDS} shards = 360 experiments"
echo "Each array task requests ${GPU_COUNT} L20 GPUs on one node."
echo "CPUs per task: ${CPUS_PER_TASK}"
echo "Output root: $OUTPUT_ROOT"
echo "Progress: find ${OUTPUT_ROOT} -name result.json | wc -l"
echo "Monitor: squeue -j ${array_job_id},${summary_job_id} -r"
