#!/usr/bin/env bash
set -euo pipefail

TOP_K="${TOP_K:-10}"
RANK_START="${RANK_START:-1}"
RANK_END="${RANK_END:-$TOP_K}"
SPARSITIES="${SPARSITIES:-}"
TASK_PROFILE="${TASK_PROFILE:-all}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
GPU_COUNT="${GPU_COUNT:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-6}"
PPL_OUTPUT_ROOT="${PPL_OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama2_13b_a40_2gpu_final}"
OUTPUT_ROOT="${ZERO_SHOT_OUTPUT_ROOT:-owl/zero_shot_wsqrtg_v9_remaining}"
if [[ -z "${CANDIDATES_CSV:-}" ]]; then
  if [[ "$RANK_START" == "1" && "$RANK_END" == "$TOP_K" \
      && -z "$SPARSITIES" && "$TASK_PROFILE" == "all" ]]; then
    CANDIDATES_CSV="$OUTPUT_ROOT/zero_shot_candidates_top10_llama2_13b.csv"
  else
    CANDIDATES_CSV="$OUTPUT_ROOT/zero_shot_candidates_r${RANK_START}_${RANK_END}_llama2_13b.csv"
  fi
fi
EXCLUDE_NODE="${EXCLUDE_NODE:-}"

pick_existing_path() {
  local explicit_path="$1"
  shift
  if [[ -n "$explicit_path" ]]; then
    echo "$explicit_path"
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "$1"
}

LLAMA2_13B_MODEL="$(pick_existing_path "${LLAMA2_13B_MODEL:-}" \
  "/home/yh114/workdir/DSnoT/models/Llama-2-13b-hf" \
  "/home/yh114/workdir/models/Llama-2-13b-hf")"
LLAMA2_13B_GRAD="$(pick_existing_path "${LLAMA2_13B_GRAD:-}" \
  "/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-13b-hf_128_0.pth")"
LLAMA2_13B_SUMMARY="${LLAMA2_13B_SUMMARY:-$PPL_OUTPUT_ROOT/summary_all.csv}"

mkdir -p log "$OUTPUT_ROOT"

if [[ ! -f "$LLAMA2_13B_SUMMARY" ]]; then
  result_count="$(find "$PPL_OUTPUT_ROOT" -name result.json 2>/dev/null | wc -l)"
  if [[ "$result_count" -gt 0 ]]; then
    echo "summary_all.csv not found, but found ${result_count} result.json files."
    echo "Generating summary first: $LLAMA2_13B_SUMMARY"
    python scripts/summarize_v9_param_sweep.py \
      --output-root "$PPL_OUTPUT_ROOT"
  else
    echo "summary_all.csv not found: $LLAMA2_13B_SUMMARY" >&2
    echo "No PPL result.json files found under: $PPL_OUTPUT_ROOT" >&2
    echo "Please run PPL sweep first:" >&2
    echo "  bash submit-wsqrtg-v9-param-llama2-13b-l20.sh" >&2
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
  --job "llama2_13b|$LLAMA2_13B_MODEL|$LLAMA2_13B_GRAD|$LLAMA2_13B_SUMMARY|$OUTPUT_ROOT"

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
  --gres="gpu:${GPU_COUNT}"
  --cpus-per-task="$CPUS_PER_TASK"
  --export="ALL,CANDIDATES_CSV=$CANDIDATES_CSV,ZERO_SHOT_OUTPUT_ROOT=$OUTPUT_ROOT"
)
if [[ -n "$EXCLUDE_NODE" ]]; then
  sbatch_args+=(--exclude="$EXCLUDE_NODE")
fi

array_job_id="$(sbatch "${sbatch_args[@]}" zero-shot-top10-array-l20.slurm)"
echo "Submitted LLaMA-2-13B ranks ${RANK_START}-${RANK_END} zero-shot array: ${array_job_id}"

summary_job_id="$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  --export="ALL,ZERO_SHOT_OUTPUT_ROOT=$OUTPUT_ROOT" \
  zero-shot-top10-summary.slurm)"
echo "Submitted zero-shot summary job: ${summary_job_id}"

echo "Candidates: $CANDIDATES_CSV"
echo "Sparsities: ${SPARSITIES:-all}; task profile: $TASK_PROFILE"
echo "Array: ${ARRAY_EXPR}; each task uses ${GPU_COUNT} L20 GPUs on one node"
echo "CPUs per task: ${CPUS_PER_TASK}"
echo "Progress: find ${OUTPUT_ROOT}/llama2_13b -name zero_shot_result.json | wc -l"
echo "Monitor: squeue -j ${array_job_id},${summary_job_id} -r"
