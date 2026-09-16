#!/usr/bin/env bash
set -euo pipefail

TOP_K="${TOP_K:-10}"
GPU_ID="${GPU_ID:-0,1}"
LOCAL_DATASETS_PATH="${LOCAL_DATASETS_PATH:-/home/yh114/workdir/Pruner-Zero/data_hf}"
OUTPUT_ROOT="${ZERO_SHOT_OUTPUT_ROOT:-owl/zero_shot_wsqrtg_v9_qwen_mistral_a40_test}"
CANDIDATES_CSV="${CANDIDATES_CSV:-$OUTPUT_ROOT/zero_shot_candidates_top10_qwen_mistral.csv}"
START_TASK="${START_TASK:-0}"
END_TASK="${END_TASK:-0}"
BUILD_CANDIDATES="${BUILD_CANDIDATES:-1}"
BUILD_ONLY="${BUILD_ONLY:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-0}"

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
    echo "summary_all.csv not found; generating from $result_count result.json files:"
    echo "  $output_root"
    python scripts/summarize_v9_param_sweep.py --output-root "$output_root"
  fi

  if [[ ! -f "$summary_csv" ]]; then
    echo "summary_all.csv not found: $summary_csv" >&2
    exit 1
  fi
}

mkdir -p "$OUTPUT_ROOT" log

if [[ "$BUILD_CANDIDATES" == "1" ]]; then
  ensure_summary "$QWEN_PPL_ROOT" "$QWEN_SUMMARY"
  ensure_summary "$MISTRAL_PPL_ROOT" "$MISTRAL_SUMMARY"

  python scripts/make_zero_shot_topk_candidates.py \
    --top-k "$TOP_K" \
    --output-csv "$CANDIDATES_CSV" \
    --job "qwen2_5_7b|$QWEN_MODEL|$QWEN_GRAD|$QWEN_SUMMARY|$OUTPUT_ROOT" \
    --job "mistral_7b|$MISTRAL_MODEL|$MISTRAL_GRAD|$MISTRAL_SUMMARY|$OUTPUT_ROOT"
fi

TASK_COUNT="$(python -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], encoding="utf-8-sig"))))' "$CANDIDATES_CSV")"
if [[ "$TASK_COUNT" -lt 1 ]]; then
  echo "No zero-shot candidates found: $CANDIDATES_CSV" >&2
  exit 1
fi

if [[ "$BUILD_ONLY" == "1" ]]; then
  echo "Candidate CSV built only: $CANDIDATES_CSV"
  echo "Task count: $TASK_COUNT"
  exit 0
fi

if [[ "$END_TASK" == "auto" ]]; then
  END_TASK=$((TASK_COUNT - 1))
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "A40 terminal run: Qwen2.5-7B + Mistral-7B Top${TOP_K} zero-shot"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
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
echo "Progress:"
echo "  find $OUTPUT_ROOT/qwen2_5_7b -name zero_shot_result.json | wc -l"
echo "  find $OUTPUT_ROOT/mistral_7b -name zero_shot_result.json | wc -l"
