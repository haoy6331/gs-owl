#!/usr/bin/env bash
set -euo pipefail

MODELS="${MODELS:-opt1_3b opt2_7b}"
METHODS="${METHODS:-dense magnitude wanda sparsegpt pruner-zero}"
EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-0}"
NODELIST="${NODELIST:-}"
EXCLUDE_NODE="${EXCLUDE_NODE:-}"

mkdir -p log

for model_key in $MODELS; do
  case "$model_key" in
    opt1_3b)
      gpus=1
      output_root="owl/opt_1_3b_baselines_l20"
      walltime="12:00:00"
      ;;
    opt2_7b)
      gpus=1
      output_root="owl/opt_2_7b_baselines_l20"
      walltime="1-00:00:00"
      ;;
    *)
      echo "Unknown MODEL_KEY: $model_key" >&2
      exit 1
      ;;
  esac

  sbatch_args=(
    --parsable
    --job-name="${model_key}-baseline"
    --gres="gpu:${gpus}"
    --time="$walltime"
    --export="ALL,MODEL_KEY=$model_key,OUTPUT_ROOT=$output_root,METHODS=$METHODS,EVAL_ZERO_SHOT=$EVAL_ZERO_SHOT"
  )
  if [[ -n "$NODELIST" ]]; then
    sbatch_args+=(--nodelist="$NODELIST")
  fi
  if [[ -n "$EXCLUDE_NODE" ]]; then
    sbatch_args+=(--exclude="$EXCLUDE_NODE")
  fi

  job_id="$(sbatch "${sbatch_args[@]}" opt-baselines-l20.slurm)"
  echo "Submitted $model_key baseline: job $job_id; GPUs=$gpus; output=$output_root"
done

echo "Monitor: squeue -u $USER -p gpu-l20 -r"
