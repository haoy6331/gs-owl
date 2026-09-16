#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_opt_1_3b}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
NUM_SHARDS="${NUM_SHARDS:-6}"
NODELIST="${NODELIST:-}"
EXCLUDE_NODE="${EXCLUDE_NODE:-}"

mkdir -p log "$OUTPUT_ROOT"

sbatch_args=(
  --parsable
  --array="0-$((NUM_SHARDS - 1))%${MAX_PARALLEL}"
  --export="ALL,OUTPUT_ROOT=$OUTPUT_ROOT,NUM_SHARDS=$NUM_SHARDS"
)
if [[ -n "$NODELIST" ]]; then
  sbatch_args+=(--nodelist="$NODELIST")
fi
if [[ -n "$EXCLUDE_NODE" ]]; then
  sbatch_args+=(--exclude="$EXCLUDE_NODE")
fi

array_job_id="$(sbatch "${sbatch_args[@]}" wsqrtg-v9-param-opt-1.3b-l20.slurm)"
echo "Submitted OPT-1.3B wsqrtg array: ${array_job_id}"

summary_job_id="$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  --export="ALL,OUTPUT_ROOT=$OUTPUT_ROOT" \
  wsqrtg-v9-param-summary.slurm)"
echo "Submitted summary job: ${summary_job_id}"

echo "Array: 0-$((NUM_SHARDS - 1))%${MAX_PARALLEL}; ${NUM_SHARDS} shards = 360 experiments"
echo "Each array task requests 1 L20 GPU."
echo "Output root: $OUTPUT_ROOT"
echo "Progress: find ${OUTPUT_ROOT} -name result.json | wc -l"
echo "Monitor: squeue -j ${array_job_id},${summary_job_id} -r"
