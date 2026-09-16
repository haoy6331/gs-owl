#!/usr/bin/env bash
set -euo pipefail

TOP_K="${TOP_K:-10}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
GPU_COUNT="${GPU_COUNT:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
OUTPUT_ROOT="${ZERO_SHOT_OUTPUT_ROOT:-owl/zero_shot_wsqrtg_v9_remaining}"
CANDIDATES_CSV="${CANDIDATES_CSV:-$OUTPUT_ROOT/zero_shot_candidates_top10_opt_1_3b.csv}"
EXCLUDE_NODE="${EXCLUDE_NODE:-}"

OPT_MODEL="${OPT_MODEL:-/home/yh114/workdir/DSnoT/models/opt-1.3b}"
OPT_GRAD="${OPT_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/opt/gradients_l2_opt-1.3b_128_0.pth}"
OPT_PPL_ROOT="${OPT_PPL_ROOT:-owl/owl_v9_wsqrtg_param_opt_1_3b}"
OPT_SUMMARY="${OPT_SUMMARY:-$OPT_PPL_ROOT/summary_all.csv}"

mkdir -p log "$OUTPUT_ROOT"

if [[ ! -f "$OPT_SUMMARY" ]]; then
  result_count="$(find "$OPT_PPL_ROOT" -name result.json 2>/dev/null | wc -l)"
  if [[ "$result_count" -gt 0 ]]; then
    echo "summary_all.csv not found, but found ${result_count} result.json files."
    echo "Generating summary first: $OPT_SUMMARY"
    python scripts/summarize_v9_param_sweep.py \
      --output-root "$OPT_PPL_ROOT"
  else
    echo "summary_all.csv not found: $OPT_SUMMARY" >&2
    echo "No PPL result.json files found under: $OPT_PPL_ROOT" >&2
    exit 1
  fi
fi

python scripts/make_zero_shot_topk_candidates.py \
  --top-k "$TOP_K" \
  --output-csv "$CANDIDATES_CSV" \
  --job "opt_1_3b|$OPT_MODEL|$OPT_GRAD|$OPT_SUMMARY|$OUTPUT_ROOT|main_opt.py"

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
echo "Submitted OPT-1.3B Top${TOP_K} zero-shot array: ${array_job_id}"

summary_job_id="$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  --export="ALL,ZERO_SHOT_OUTPUT_ROOT=$OUTPUT_ROOT" \
  zero-shot-top10-summary.slurm)"
echo "Submitted zero-shot summary job: ${summary_job_id}"

echo "Candidates: $CANDIDATES_CSV"
echo "Array: 0-${LAST_TASK}%${MAX_PARALLEL}; each task uses ${GPU_COUNT} L20 GPU"
echo "CPUs per task: ${CPUS_PER_TASK}"
echo "Progress: find ${OUTPUT_ROOT}/opt_1_3b -name zero_shot_result.json | wc -l"
echo "Monitor: squeue -j ${array_job_id},${summary_job_id} -r"
