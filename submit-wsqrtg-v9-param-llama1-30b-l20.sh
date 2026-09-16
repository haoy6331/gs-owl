#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama1_30b_l20_4gpu}"
BIG_GPU_COUNT="${BIG_GPU_COUNT:-4}"
BIG_NODE="${BIG_NODE:-}"

mkdir -p log "$OUTPUT_ROOT"

sbatch_args=(
  --parsable
  --gres="gpu:${BIG_GPU_COUNT}"
  --export="ALL,OUTPUT_ROOT=$OUTPUT_ROOT"
)
if [[ -n "$BIG_NODE" ]]; then
  sbatch_args+=(--nodelist="$BIG_NODE")
fi

array_job_id="$(sbatch "${sbatch_args[@]}" wsqrtg-v9-param-llama1-30b-l20.slurm)"
echo "Submitted LLaMA-1-30B wsqrtg sweep array: ${array_job_id}"

summary_job_id="$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  --export="ALL,OUTPUT_ROOT=$OUTPUT_ROOT" \
  wsqrtg-v9-param-llama1-30b-summary.slurm)"
echo "Submitted summary job: ${summary_job_id}"

echo "30B GPU count per task: ${BIG_GPU_COUNT}"
if [[ -n "$BIG_NODE" ]]; then
  echo "Pinned node: ${BIG_NODE}"
else
  echo "No node pinned; Slurm will choose a node with enough GPUs."
fi
echo "Progress: find ${OUTPUT_ROOT} -name result.json | wc -l"
echo "Monitor: squeue -j ${array_job_id},${summary_job_id} -r"
