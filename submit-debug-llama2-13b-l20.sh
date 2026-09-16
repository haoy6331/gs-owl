#!/usr/bin/env bash
set -euo pipefail

GPU_COUNT="${GPU_COUNT:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((GPU_COUNT * 2))}"
NSAMPLES="${NSAMPLES:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/debug_llama2_13b_l20_${GPU_COUNT}gpu}"
NODELIST="${NODELIST:-}"
EXCLUDE_NODE="${EXCLUDE_NODE:-}"

gpu_ids() {
  local n="$1"
  local ids=""
  local i
  for ((i = 0; i < n; i++)); do
    if [[ -n "$ids" ]]; then
      ids+=","
    fi
    ids+="$i"
  done
  echo "$ids"
}

DEBUG_GPU_ID="${DEBUG_GPU_ID:-$(gpu_ids "$GPU_COUNT")}"

mkdir -p log "$OUTPUT_ROOT"

sbatch_args=(
  --parsable
  --gres="gpu:${GPU_COUNT}"
  --cpus-per-task="$CPUS_PER_TASK"
  --export="ALL,GPU_COUNT=$GPU_COUNT,DEBUG_GPU_ID=$DEBUG_GPU_ID,NSAMPLES=$NSAMPLES,OUTPUT_ROOT=$OUTPUT_ROOT"
)
if [[ -n "$NODELIST" ]]; then
  sbatch_args+=(--nodelist="$NODELIST")
fi
if [[ -n "$EXCLUDE_NODE" ]]; then
  sbatch_args+=(--exclude="$EXCLUDE_NODE")
fi

job_id="$(sbatch "${sbatch_args[@]}" debug-llama2-13b-l20.slurm)"
echo "Submitted LLaMA-2-13B L20 debug job: ${job_id}"
echo "GPU_COUNT: $GPU_COUNT"
echo "DEBUG_GPU_ID: $DEBUG_GPU_ID"
echo "NSAMPLES: $NSAMPLES"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "Monitor: squeue -j ${job_id} -r"
echo "Log: log/${job_id}.debug-l2-13b-l20.out.txt"
