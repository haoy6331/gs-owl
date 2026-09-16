#!/usr/bin/env bash
set -euo pipefail

TOP_K="${TOP_K:-10}"
RANK_START="${RANK_START:-1}"
RANK_END="${RANK_END:-$TOP_K}"
SPARSITIES="${SPARSITIES:-}"
TASK_PROFILE="${TASK_PROFILE:-all}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
BIG_GPU_COUNT="${BIG_GPU_COUNT:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
BIG_NODE="${BIG_NODE:-}"
PPL_OUTPUT_ROOT="${PPL_OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama1_30b_l20_4gpu}"
OUTPUT_ROOT="${ZERO_SHOT_OUTPUT_ROOT:-owl/zero_shot_wsqrtg_v9_remaining}"
if [[ -z "${CANDIDATES_CSV:-}" ]]; then
  if [[ "$RANK_START" == "1" && "$RANK_END" == "$TOP_K" \
      && -z "$SPARSITIES" && "$TASK_PROFILE" == "all" ]]; then
    CANDIDATES_CSV="$OUTPUT_ROOT/zero_shot_candidates_top10_llama1_30b.csv"
  else
    CANDIDATES_CSV="$OUTPUT_ROOT/zero_shot_candidates_r${RANK_START}_${RANK_END}_llama1_30b.csv"
  fi
fi

LLAMA1_30B_MODEL="${LLAMA1_30B_MODEL:-/home/yh114/workdir/DSnoT/models/decapoda-research-llama-30B-hf}"
LLAMA1_30B_GRAD="${LLAMA1_30B_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_decapoda-research-llama-30B-hf_128_0.pth}"
LLAMA1_30B_SUMMARY="${LLAMA1_30B_SUMMARY:-$PPL_OUTPUT_ROOT/summary_all.csv}"

mkdir -p log "$OUTPUT_ROOT"

if [[ ! -f "$LLAMA1_30B_SUMMARY" ]]; then
  result_count="$(find "$PPL_OUTPUT_ROOT" -name result.json 2>/dev/null | wc -l)"
  if [[ "$result_count" -gt 0 ]]; then
    echo "summary_all.csv not found, but found ${result_count} result.json files."
    echo "Generating summary first: $LLAMA1_30B_SUMMARY"
    python scripts/summarize_v9_param_sweep.py \
      --output-root "$PPL_OUTPUT_ROOT"
  else
    echo "summary_all.csv not found: $LLAMA1_30B_SUMMARY" >&2
    echo "No PPL result.json files found under: $PPL_OUTPUT_ROOT" >&2
    exit 1
  fi
fi

CANDIDATE_ARGS=(
  --top-k "$TOP_K"
  --rank-start "$RANK_START"
  --rank-end "$RANK_END"
  --task-profile "$TASK_PROFILE"
)
for sparsity in $SPARSITIES; do
  CANDIDATE_ARGS+=(--sparsity "$sparsity")
done

python scripts/make_zero_shot_topk_candidates.py \
  "${CANDIDATE_ARGS[@]}" \
  --output-csv "$CANDIDATES_CSV" \
  --job "llama1_30b|$LLAMA1_30B_MODEL|$LLAMA1_30B_GRAD|$LLAMA1_30B_SUMMARY|$OUTPUT_ROOT"

TASK_COUNT="$(python -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], encoding="utf-8-sig"))))' "$CANDIDATES_CSV")"
if [[ "$TASK_COUNT" -lt 1 ]]; then
  echo "No zero-shot candidates generated: $CANDIDATES_CSV" >&2
  exit 1
fi
LAST_TASK=$((TASK_COUNT - 1))
ARRAY_SPEC="${ARRAY_SPEC:-0-${LAST_TASK}}"
if [[ "$ARRAY_SPEC" == *"%"* ]]; then
  ARRAY_EXPR="$ARRAY_SPEC"
else
  ARRAY_EXPR="${ARRAY_SPEC}%${MAX_PARALLEL}"
fi

sbatch_args=(
  --parsable
  --array="$ARRAY_EXPR"
  --gres="gpu:${BIG_GPU_COUNT}"
  --cpus-per-task="$CPUS_PER_TASK"
  --export="ALL,CANDIDATES_CSV=$CANDIDATES_CSV,ZERO_SHOT_OUTPUT_ROOT=$OUTPUT_ROOT"
)
if [[ -n "$BIG_NODE" ]]; then
  sbatch_args+=(--nodelist="$BIG_NODE")
fi

array_job_id="$(sbatch "${sbatch_args[@]}" zero-shot-top10-array-l20.slurm)"
echo "Submitted LLaMA-1-30B ranks ${RANK_START}-${RANK_END} zero-shot array: ${array_job_id}"

summary_job_id="$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  --export="ALL,ZERO_SHOT_OUTPUT_ROOT=$OUTPUT_ROOT" \
  zero-shot-top10-summary.slurm)"
echo "Submitted zero-shot summary job: ${summary_job_id}"

echo "Candidates: $CANDIDATES_CSV"
echo "30B GPU count per task: ${BIG_GPU_COUNT}"
echo "CPUs per task: ${CPUS_PER_TASK}"
echo "Sparsities: ${SPARSITIES:-all}; task profile: $TASK_PROFILE"
echo "Array: ${ARRAY_EXPR}"
echo "Progress: find ${OUTPUT_ROOT}/llama1_30b -name zero_shot_result.json | wc -l"
echo "Monitor: squeue -j ${array_job_id},${summary_job_id} -r"
