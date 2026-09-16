#!/usr/bin/env bash
set -euo pipefail

TOP_K="${TOP_K:-10}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
GPU_COUNT="${GPU_COUNT:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
OUTPUT_ROOT="${ZERO_SHOT_OUTPUT_ROOT:-owl/zero_shot_wsqrtg_v9_remaining}"
CANDIDATES_CSV="${CANDIDATES_CSV:-$OUTPUT_ROOT/zero_shot_candidates_top10_qwen_mistral.csv}"
EXCLUDE_NODE="${EXCLUDE_NODE:-}"

QWEN_MODEL="${QWEN_MODEL:-/home/yh114/workdir/DSnoT/models/Qwen2.5-7B}"
QWEN_GRAD="${QWEN_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/qwen2/gradients_l2_Qwen2.5-7B_128_0.pth}"
QWEN_PPL_ROOT="${QWEN_PPL_ROOT:-owl/owl_v9_wsqrtg_param_qwen2_5_7b_l20_2gpu}"
QWEN_SUMMARY="${QWEN_SUMMARY:-$QWEN_PPL_ROOT/summary_all.csv}"

MISTRAL_MODEL="${MISTRAL_MODEL:-/home/yh114/workdir/DSnoT/models/Mistral-7B}"
MISTRAL_GRAD="${MISTRAL_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/mistral/gradients_l2_Mistral-7B_128_0.pth}"
MISTRAL_PPL_ROOT="${MISTRAL_PPL_ROOT:-owl/owl_v9_wsqrtg_param_mistral_7b_l20_2gpu}"
MISTRAL_SUMMARY="${MISTRAL_SUMMARY:-$MISTRAL_PPL_ROOT/summary_all.csv}"

ensure_summary() {
  local output_root="$1"
  local summary_csv="$2"
  if [[ -f "$summary_csv" ]]; then
    return 0
  fi

  local result_count
  result_count="$(find "$output_root" -name result.json 2>/dev/null | wc -l)"
  if [[ "$result_count" -gt 0 ]]; then
    echo "summary_all.csv not found, but found ${result_count} result.json files."
    echo "Generating summary first: $summary_csv"
    python scripts/summarize_v9_param_sweep.py \
      --output-root "$output_root"
    return 0
  fi

  echo "summary_all.csv not found: $summary_csv" >&2
  echo "No PPL result.json files found under: $output_root" >&2
  exit 1
}

mkdir -p log "$OUTPUT_ROOT"

ensure_summary "$QWEN_PPL_ROOT" "$QWEN_SUMMARY"
ensure_summary "$MISTRAL_PPL_ROOT" "$MISTRAL_SUMMARY"

python scripts/make_zero_shot_topk_candidates.py \
  --top-k "$TOP_K" \
  --output-csv "$CANDIDATES_CSV" \
  --job "qwen2_5_7b|$QWEN_MODEL|$QWEN_GRAD|$QWEN_SUMMARY|$OUTPUT_ROOT" \
  --job "mistral_7b|$MISTRAL_MODEL|$MISTRAL_GRAD|$MISTRAL_SUMMARY|$OUTPUT_ROOT"

TASK_COUNT="$(python -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], encoding="utf-8-sig"))))' "$CANDIDATES_CSV")"
if [[ "$TASK_COUNT" -lt 1 ]]; then
  echo "No zero-shot candidates generated: $CANDIDATES_CSV" >&2
  exit 1
fi
LAST_TASK=$((TASK_COUNT - 1))

sbatch_args=(
  --parsable
  --array="0-${LAST_TASK}%${MAX_PARALLEL}"
  --gres="gpu:${GPU_COUNT}"
  --cpus-per-task="$CPUS_PER_TASK"
  --export="ALL,CANDIDATES_CSV=$CANDIDATES_CSV,ZERO_SHOT_OUTPUT_ROOT=$OUTPUT_ROOT"
)
if [[ -n "$EXCLUDE_NODE" ]]; then
  sbatch_args+=(--exclude="$EXCLUDE_NODE")
fi

array_job_id="$(sbatch "${sbatch_args[@]}" zero-shot-top10-array-l20.slurm)"
echo "Submitted Qwen2.5-7B + Mistral-7B Top${TOP_K} zero-shot array: ${array_job_id}"

summary_job_id="$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  --export="ALL,ZERO_SHOT_OUTPUT_ROOT=$OUTPUT_ROOT" \
  zero-shot-top10-summary.slurm)"
echo "Submitted zero-shot summary job: ${summary_job_id}"

echo "Candidates: $CANDIDATES_CSV"
echo "Array: 0-${LAST_TASK}%${MAX_PARALLEL}; each task uses ${GPU_COUNT} L20 GPUs on one node"
echo "CPUs per task: ${CPUS_PER_TASK}"
echo "Progress:"
echo "  find ${OUTPUT_ROOT}/qwen2_5_7b -name zero_shot_result.json | wc -l"
echo "  find ${OUTPUT_ROOT}/mistral_7b -name zero_shot_result.json | wc -l"
echo "Monitor: squeue -j ${array_job_id},${summary_job_id} -r"
