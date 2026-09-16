#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${MODEL_KEY:-llama2_13b}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
NUM_SHARDS="${NUM_SHARDS:-6}"
EXCLUDE_NODE="${EXCLUDE_NODE:-}"
NODELIST="${NODELIST:-}"

case "$MODEL_KEY" in
  llama2_13b)
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama2_13b_l20_2gpu}"
    ;;
  qwen2_5_7b)
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_qwen2_5_7b_l20_2gpu}"
    ;;
  mistral_7b)
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_mistral_7b_l20_2gpu}"
    ;;
  vicuna_7b)
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_vicuna_7b_l20_2gpu}"
    ;;
  *)
    echo "Unknown MODEL_KEY: $MODEL_KEY" >&2
    echo "Supported: llama2_13b, qwen2_5_7b, mistral_7b, vicuna_7b" >&2
    exit 1
    ;;
esac

mkdir -p log "$OUTPUT_ROOT"

sbatch_args=(
  --parsable
  --job-name="${MODEL_KEY}-2g-sweep"
  --array="0-$((NUM_SHARDS - 1))%${MAX_PARALLEL}"
  --export="ALL,MODEL_KEY=$MODEL_KEY,OUTPUT_ROOT=$OUTPUT_ROOT,NUM_SHARDS=$NUM_SHARDS"
)
if [[ -n "$NODELIST" ]]; then
  sbatch_args+=(--nodelist="$NODELIST")
fi
if [[ -n "$EXCLUDE_NODE" ]]; then
  sbatch_args+=(--exclude="$EXCLUDE_NODE")
fi

array_job_id="$(sbatch "${sbatch_args[@]}" wsqrtg-v9-param-two-gpu-l20.slurm)"
echo "Submitted ${MODEL_KEY} two-GPU L20 array: ${array_job_id}"

summary_job_id="$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  --export="ALL,OUTPUT_ROOT=$OUTPUT_ROOT" \
  wsqrtg-v9-param-summary.slurm)"
echo "Submitted summary job: ${summary_job_id}"

echo "Model key: $MODEL_KEY"
echo "Array: 0-$((NUM_SHARDS - 1))%${MAX_PARALLEL}; ${NUM_SHARDS} shards = 360 experiments"
echo "Each array task requests 2 L20 GPUs on one node."
echo "Output root: $OUTPUT_ROOT"
echo "Progress: find ${OUTPUT_ROOT} -name result.json | wc -l"
echo "Monitor: squeue -j ${array_job_id},${summary_job_id} -r"
