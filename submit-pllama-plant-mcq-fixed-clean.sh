#!/usr/bin/env bash
set -euo pipefail

QUESTIONS="${QUESTIONS:-data/plant_mcq/mobiplant_expert_clean_v2.jsonl}"
OUTPUT_7B="${OUTPUT_7B:-owl/pllama_7b_plant_mcq_l20_fixed_clean_v2}"
OUTPUT_13B="${OUTPUT_13B:-owl/pllama_13b_plant_mcq_l20_fixed_clean_v2}"
MAX_PARALLEL_7B="${MAX_PARALLEL_7B:-1}"
MAX_PARALLEL_13B="${MAX_PARALLEL_13B:-1}"

[[ -s "$QUESTIONS" ]] || {
  echo "Missing clean question set: $QUESTIONS" >&2
  exit 1
}
mkdir -p log "$OUTPUT_7B" "$OUTPUT_13B"

job_7b="$(sbatch --parsable \
  --array="0-3%${MAX_PARALLEL_7B}" \
  --export="ALL,QUESTIONS=$QUESTIONS,OUTPUT_ROOT=$OUTPUT_7B" \
  pllama-7b-plant-mcq-fixed-clean.slurm)"
echo "Submitted PLLaMA-7B fixed clean evaluation: $job_7b"

job_13b="$(sbatch --parsable \
  --array="0-3%${MAX_PARALLEL_13B}" \
  --export="ALL,QUESTIONS=$QUESTIONS,OUTPUT_ROOT=$OUTPUT_13B" \
  pllama-13b-plant-mcq-fixed-clean.slurm)"
echo "Submitted PLLaMA-13B fixed clean evaluation: $job_13b"

summary_7b="$(sbatch --parsable \
  --dependency="afterok:${job_7b}" \
  --export="ALL,MODEL_SIZE=7b,OUTPUT_ROOT=$OUTPUT_7B" \
  pllama-plant-mcq-fixed-clean-summary.slurm)"
summary_13b="$(sbatch --parsable \
  --dependency="afterok:${job_13b}" \
  --export="ALL,MODEL_SIZE=13b,OUTPUT_ROOT=$OUTPUT_13B" \
  pllama-plant-mcq-fixed-clean-summary.slurm)"

echo "Submitted 7B summary: $summary_7b"
echo "Submitted 13B summary: $summary_13b"
echo "Question set: $QUESTIONS"
echo "Fixed parameters (H_m, lambda, alpha, beta) = (5, 0.08, 0.20, 0.5)"
echo "Monitor: squeue -j ${job_7b},${job_13b},${summary_7b},${summary_13b} -r"
