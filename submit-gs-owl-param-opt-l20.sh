#!/usr/bin/env bash
set -euo pipefail

# OPT-1.3B is already complete and OPT-13B runs on the interactive A100 node.
MODELS="${MODELS:-opt2_7b}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
OPT1_NUM_SHARDS="${OPT1_NUM_SHARDS:-6}"
OPT2_NUM_SHARDS="${OPT2_NUM_SHARDS:-12}"
NODELIST="${NODELIST:-}"
EXCLUDE_NODE="${EXCLUDE_NODE:-}"

mkdir -p log

for model_key in $MODELS; do
  case "$model_key" in
    opt1_3b)
      gpus=1
      num_shards="$OPT1_NUM_SHARDS"
      output_root="owl/gs_owl_param_opt_1_3b_l20"
      walltime="1-00:00:00"
      ;;
    opt2_7b)
      gpus=1
      num_shards="$OPT2_NUM_SHARDS"
      output_root="owl/gs_owl_param_opt_2_7b_l20"
      walltime="1-00:00:00"
      ;;
    *)
      echo "Unknown MODEL_KEY: $model_key" >&2
      exit 1
      ;;
  esac

  sbatch_args=(
    --parsable
    --job-name="${model_key}-gs-owl"
    --gres="gpu:${gpus}"
    --time="$walltime"
    --array="0-$((num_shards - 1))%${MAX_PARALLEL}"
    --export="ALL,MODEL_KEY=$model_key,OUTPUT_ROOT=$output_root,NUM_SHARDS=$num_shards"
  )
  if [[ -n "$NODELIST" ]]; then
    sbatch_args+=(--nodelist="$NODELIST")
  fi
  if [[ -n "$EXCLUDE_NODE" ]]; then
    sbatch_args+=(--exclude="$EXCLUDE_NODE")
  fi

  array_job_id="$(sbatch "${sbatch_args[@]}" gs-owl-param-opt-l20.slurm)"
  echo "Submitted $model_key GS-OWL array: $array_job_id; GPUs/task=$gpus; shards=$num_shards"

  summary_job_id="$(sbatch --parsable \
    --dependency="afterany:${array_job_id}" \
    --export="ALL,OUTPUT_ROOT=$output_root" \
    wsqrtg-v9-param-summary.slurm)"
  echo "Submitted $model_key summary: $summary_job_id"
  echo "Output: $output_root"
done

echo "Each model contains exactly 360 parameter combinations."
echo "Monitor: squeue -u $USER -p gpu-l20 -r"
