#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-ready}"
TOP_K="${TOP_K:-10}"
RANK_START="${RANK_START:-1}"
RANK_END="${RANK_END:-$TOP_K}"
SPARSITIES="${SPARSITIES:-}"
TASK_PROFILE="${TASK_PROFILE:-all}"
GPU_ID="${GPU_ID:-0}"
LOCAL_DATASETS_PATH="${LOCAL_DATASETS_PATH:-/home/yh114/workdir/Pruner-Zero/data_hf}"
OUTPUT_ROOT="${ZERO_SHOT_OUTPUT_ROOT:-owl/zero_shot_wsqrtg_v9_remaining}"
if [[ -z "${CANDIDATES_CSV:-}" ]]; then
  if [[ "$RANK_START" == "1" && "$RANK_END" == "$TOP_K" \
      && -z "$SPARSITIES" && "$TASK_PROFILE" == "all" ]]; then
    CANDIDATES_CSV="$OUTPUT_ROOT/zero_shot_candidates_top10_${MODE}.csv"
  else
    CANDIDATES_CSV="$OUTPUT_ROOT/zero_shot_candidates_r${RANK_START}_${RANK_END}_${MODE}.csv"
  fi
fi
START_TASK="${START_TASK:-0}"
END_TASK="${END_TASK:-auto}"
BUILD_CANDIDATES="${BUILD_CANDIDATES:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

mkdir -p "$OUTPUT_ROOT" log

CANDIDATE_ARGS=(
  --top-k "$TOP_K"
  --rank-start "$RANK_START"
  --rank-end "$RANK_END"
  --task-profile "$TASK_PROFILE"
)
for sparsity in $SPARSITIES; do
  CANDIDATE_ARGS+=(--sparsity "$sparsity")
done

ensure_summary() {
  local output_root="$1"
  local summary_csv="$2"
  if [[ -f "$summary_csv" ]]; then
    return 0
  fi
  local result_count
  result_count="$(find "$output_root" -name result.json 2>/dev/null | wc -l)"
  if [[ "$result_count" -gt 0 ]]; then
    echo "summary_all.csv not found; generating from $result_count result.json files:"
    echo "  $output_root"
    python scripts/summarize_v9_param_sweep.py --output-root "$output_root"
  fi
  if [[ ! -f "$summary_csv" ]]; then
    echo "summary_all.csv not found: $summary_csv" >&2
    exit 1
  fi
}

set_7b_paths() {
  LLAMA1_7B_MODEL="${LLAMA1_7B_MODEL:-/home/yh114/workdir/models/decapoda-research-llama-7B-hf}"
  LLAMA1_7B_GRAD="${LLAMA1_7B_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-7B-hf_128_0_fixed.pth}"
  LLAMA1_7B_SUMMARY="${LLAMA1_7B_SUMMARY:-owl/owl_v9_wsqrtg_param_a40/summary_all.csv}"

  LLAMA2_7B_MODEL="${LLAMA2_7B_MODEL:-/home/yh114/workdir/DSnoT/models/Llama-2-7b-hf}"
  LLAMA2_7B_GRAD="${LLAMA2_7B_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-7b-hf_128_0.pth}"
  LLAMA2_7B_SUMMARY="${LLAMA2_7B_SUMMARY:-owl/owl_v9_wsqrtg_param_llama2_7b_l20/summary_all.csv}"
}

build_candidates_ready() {
  set_7b_paths
  LLAMA1_13B_MODEL="${LLAMA1_13B_MODEL:-/home/yh114/workdir/DSnoT/models/decapoda-research-llama-13B-hf}"
  LLAMA1_13B_GRAD="${LLAMA1_13B_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-13B-hf_128_0.pth}"
  LLAMA1_13B_SUMMARY="${LLAMA1_13B_SUMMARY:-owl/owl_v9_wsqrtg_param_13b_l20/llama1_13b/summary_all.csv}"

  python scripts/make_zero_shot_topk_candidates.py \
    "${CANDIDATE_ARGS[@]}" \
    --output-csv "$CANDIDATES_CSV" \
    --job "llama1_7b|$LLAMA1_7B_MODEL|$LLAMA1_7B_GRAD|$LLAMA1_7B_SUMMARY|$OUTPUT_ROOT" \
    --job "llama2_7b|$LLAMA2_7B_MODEL|$LLAMA2_7B_GRAD|$LLAMA2_7B_SUMMARY|$OUTPUT_ROOT" \
    --job "llama1_13b|$LLAMA1_13B_MODEL|$LLAMA1_13B_GRAD|$LLAMA1_13B_SUMMARY|$OUTPUT_ROOT"
}

build_candidates_target7b() {
  set_7b_paths
  python scripts/make_zero_shot_topk_candidates.py \
    "${CANDIDATE_ARGS[@]}" \
    --output-csv "$CANDIDATES_CSV" \
    --job "llama1_7b|$LLAMA1_7B_MODEL|$LLAMA1_7B_GRAD|$LLAMA1_7B_SUMMARY|$OUTPUT_ROOT" \
    --job "llama2_7b|$LLAMA2_7B_MODEL|$LLAMA2_7B_GRAD|$LLAMA2_7B_SUMMARY|$OUTPUT_ROOT"
}

build_candidates_single() {
  local label="$1"
  local model_path="$2"
  local gradient_path="$3"
  local ppl_output_root="$4"
  local summary_csv="${SUMMARY_CSV:-$ppl_output_root/summary_all.csv}"
  ensure_summary "$ppl_output_root" "$summary_csv"
  python scripts/make_zero_shot_topk_candidates.py \
    "${CANDIDATE_ARGS[@]}" \
    --output-csv "$CANDIDATES_CSV" \
    --job "$label|$model_path|$gradient_path|$summary_csv|$OUTPUT_ROOT"
}

if [[ "$BUILD_CANDIDATES" == "1" ]]; then
  case "$MODE" in
    ready)
      build_candidates_ready
      ;;
    target7b)
      build_candidates_target7b
      ;;
    llama1_7b)
      set_7b_paths
      build_candidates_single \
        "llama1_7b" \
        "$LLAMA1_7B_MODEL" \
        "$LLAMA1_7B_GRAD" \
        "$(dirname "$LLAMA1_7B_SUMMARY")"
      ;;
    llama2_7b)
      set_7b_paths
      build_candidates_single \
        "llama2_7b" \
        "$LLAMA2_7B_MODEL" \
        "$LLAMA2_7B_GRAD" \
        "$(dirname "$LLAMA2_7B_SUMMARY")"
      ;;
    llama2_13b)
      build_candidates_single \
        "llama2_13b" \
        "${LLAMA2_13B_MODEL:-/home/yh114/workdir/DSnoT/models/Llama-2-13b-hf}" \
        "${LLAMA2_13B_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-13b-hf_128_0.pth}" \
        "${PPL_OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama2_13b_a40_2gpu_final}"
      ;;
    qwen2_5_7b)
      build_candidates_single \
        "qwen2_5_7b" \
        "${QWEN2_5_7B_MODEL:-/home/yh114/workdir/DSnoT/models/Qwen2.5-7B}" \
        "${QWEN2_5_7B_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/qwen2/gradients_l2_Qwen2.5-7B_128_0.pth}" \
        "${PPL_OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_qwen2_5_7b_a100}"
      ;;
    llama1_30b)
      build_candidates_single \
        "llama1_30b" \
        "${LLAMA1_30B_MODEL:-/home/yh114/workdir/DSnoT/models/decapoda-research-llama-30B-hf}" \
        "${LLAMA1_30B_GRAD:-/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_decapoda-research-llama-30B-hf_128_0.pth}" \
        "${PPL_OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama1_30b_l20_4gpu}"
      ;;
    custom)
      if [[ ! -f "$CANDIDATES_CSV" ]]; then
        echo "MODE=custom requires an existing CANDIDATES_CSV: $CANDIDATES_CSV" >&2
        exit 1
      fi
      ;;
    *)
      echo "Unknown MODE: $MODE" >&2
      echo "Supported: ready, target7b, llama1_7b, llama2_7b, llama2_13b, qwen2_5_7b, llama1_30b, custom" >&2
      exit 1
      ;;
  esac
fi

TASK_COUNT="$(python -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], encoding="utf-8-sig"))))' "$CANDIDATES_CSV")"
if [[ "$TASK_COUNT" -lt 1 ]]; then
  echo "No zero-shot candidates found: $CANDIDATES_CSV" >&2
  exit 1
fi

if [[ "$END_TASK" == "auto" ]]; then
  END_TASK=$((TASK_COUNT - 1))
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "Terminal run: OWL-V9 + wsqrtg zero-shot"
echo "MODE: $MODE"
echo "PPL ranks: $RANK_START-$RANK_END"
echo "Sparsities: ${SPARSITIES:-all}"
echo "Task profile: $TASK_PROFILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Candidates: $CANDIDATES_CSV"
echo "Tasks: $START_TASK to $END_TASK of $((TASK_COUNT - 1))"
echo "Offline datasets: $LOCAL_DATASETS_PATH"
echo "Output: $OUTPUT_ROOT"
echo "============================================================"

for task_id in $(seq "$START_TASK" "$END_TASK"); do
  echo "Running zero-shot task $task_id/$((TASK_COUNT - 1))"
  python scripts/run_zero_shot_topk_array_task.py \
    --task-id "$task_id" \
    --candidates-csv "$CANDIDATES_CSV" \
    --offline \
    --local-datasets-path "$LOCAL_DATASETS_PATH" \
    --resume
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  python scripts/summarize_zero_shot_topk.py --output-root "$OUTPUT_ROOT"
fi

echo "Done."
echo "Summary: $OUTPUT_ROOT/zero_shot_topk_summary.csv"
echo "Best: $OUTPUT_ROOT/zero_shot_topk_best_by_sparsity.csv"
echo "Oracle: $OUTPUT_ROOT/zero_shot_oracle_by_sparsity.csv"
