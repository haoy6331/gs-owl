#!/usr/bin/env bash
set -euo pipefail

TOP_K="${TOP_K:-10}"
RANK_START="${RANK_START:-1}"
RANK_END="${RANK_END:-$TOP_K}"
SPARSITIES="${SPARSITIES:-}"
TASK_PROFILE="${TASK_PROFILE:-all}"
MODE="${MODE:-ready}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
OUTPUT_ROOT="${ZERO_SHOT_OUTPUT_ROOT:-owl/zero_shot_wsqrtg_v9_remaining}"
if [[ -z "${CANDIDATES_CSV:-}" ]]; then
  if [[ "$RANK_START" == "1" && "$RANK_END" == "$TOP_K" \
      && -z "$SPARSITIES" && "$TASK_PROFILE" == "all" \
      && "$MODE" == "ready" ]]; then
    CANDIDATES_CSV="$OUTPUT_ROOT/zero_shot_candidates_top10_ready.csv"
  else
    CANDIDATES_CSV="$OUTPUT_ROOT/zero_shot_candidates_r${RANK_START}_${RANK_END}_${MODE}.csv"
  fi
fi
EXCLUDE_NODE="${EXCLUDE_NODE:-}"

LLAMA1_7B_MODEL="${LLAMA1_7B_MODEL:-/home/yh114/workdir/models/decapoda-research-llama-7B-hf}"
LLAMA1_7B_GRAD="${LLAMA1_7B_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-7B-hf_128_0_fixed.pth}"
LLAMA1_7B_SUMMARY="${LLAMA1_7B_SUMMARY:-owl/owl_v9_wsqrtg_param_a40/summary_all.csv}"

LLAMA2_7B_MODEL="${LLAMA2_7B_MODEL:-/home/yh114/workdir/DSnoT/models/Llama-2-7b-hf}"
LLAMA2_7B_GRAD="${LLAMA2_7B_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-7b-hf_128_0.pth}"
LLAMA2_7B_SUMMARY="${LLAMA2_7B_SUMMARY:-owl/owl_v9_wsqrtg_param_llama2_7b_l20/summary_all.csv}"

LLAMA1_13B_MODEL="${LLAMA1_13B_MODEL:-/home/yh114/workdir/DSnoT/models/decapoda-research-llama-13B-hf}"
LLAMA1_13B_GRAD="${LLAMA1_13B_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-13B-hf_128_0.pth}"
LLAMA1_13B_SUMMARY="${LLAMA1_13B_SUMMARY:-owl/owl_v9_wsqrtg_param_13b_l20/llama1_13b/summary_all.csv}"

mkdir -p log "$OUTPUT_ROOT"

CANDIDATE_ARGS=(
  --top-k "$TOP_K"
  --rank-start "$RANK_START"
  --rank-end "$RANK_END"
  --task-profile "$TASK_PROFILE"
)
for sparsity in $SPARSITIES; do
  CANDIDATE_ARGS+=(--sparsity "$sparsity")
done

JOBS=(
  --job "llama1_7b|$LLAMA1_7B_MODEL|$LLAMA1_7B_GRAD|$LLAMA1_7B_SUMMARY|$OUTPUT_ROOT"
  --job "llama2_7b|$LLAMA2_7B_MODEL|$LLAMA2_7B_GRAD|$LLAMA2_7B_SUMMARY|$OUTPUT_ROOT"
)
if [[ "$MODE" == "ready" ]]; then
  JOBS+=(--job "llama1_13b|$LLAMA1_13B_MODEL|$LLAMA1_13B_GRAD|$LLAMA1_13B_SUMMARY|$OUTPUT_ROOT")
elif [[ "$MODE" != "target7b" ]]; then
  echo "Unsupported MODE for this submitter: $MODE (use ready or target7b)" >&2
  exit 1
fi

python scripts/make_zero_shot_topk_candidates.py \
  "${CANDIDATE_ARGS[@]}" \
  --output-csv "$CANDIDATES_CSV" \
  "${JOBS[@]}"

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
  --export="ALL,CANDIDATES_CSV=$CANDIDATES_CSV,ZERO_SHOT_OUTPUT_ROOT=$OUTPUT_ROOT"
)
if [[ -n "$EXCLUDE_NODE" ]]; then
  sbatch_args+=(--exclude="$EXCLUDE_NODE")
fi

array_job_id="$(sbatch "${sbatch_args[@]}" zero-shot-top10-array-l20.slurm)"
echo "Submitted ${MODE} ranks ${RANK_START}-${RANK_END} zero-shot array: ${array_job_id}"

summary_job_id="$(sbatch --parsable \
  --dependency="afterany:${array_job_id}" \
  --export="ALL,ZERO_SHOT_OUTPUT_ROOT=$OUTPUT_ROOT" \
  zero-shot-top10-summary.slurm)"
echo "Submitted zero-shot summary job: ${summary_job_id}"

echo "Candidates: $CANDIDATES_CSV"
echo "Sparsities: ${SPARSITIES:-all}; task profile: $TASK_PROFILE"
echo "Array: ${ARRAY_EXPR}; each task uses one L20 GPU"
echo "Progress: find ${OUTPUT_ROOT} -name zero_shot_result.json | wc -l"
echo "Monitor: squeue -j ${array_job_id},${summary_job_id} -r"
