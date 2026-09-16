#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama2_13b_l20_retry}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
EXCLUDE_NODE="${EXCLUDE_NODE:-}"

mkdir -p log "$OUTPUT_ROOT"

sbatch_args=(
  --parsable
  --array="0-5%${MAX_PARALLEL}"
  --export="ALL,OUTPUT_ROOT=$OUTPUT_ROOT"
)
if [[ -n "$EXCLUDE_NODE" ]]; then
  sbatch_args+=(--exclude="$EXCLUDE_NODE")
fi

array_job_id="$(sbatch "${sbatch_args[@]}" wsqrtg-v9-param-llama2-13b-l20.slurm)"
echo "Submitted LLaMA-2-13B wsqrtg retry array: ${array_job_id}"

summary_job_id="$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  --export="ALL,OUTPUT_ROOT=$OUTPUT_ROOT" \
  wsqrtg-v9-param-llama2-13b-summary.slurm)"
echo "Submitted summary job: ${summary_job_id}"

echo "Array: 0-5%${MAX_PARALLEL}; 6 shards x 60 runs = 360 experiments"
echo "Progress: find ${OUTPUT_ROOT} -name result.json | wc -l"
echo "Monitor: squeue -j ${array_job_id},${summary_job_id} -r"
