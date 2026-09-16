#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-baseline}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-}"
START_TASK="${START_TASK:-0}"
END_TASK="${END_TASK:-11}"
RESUME="${RESUME:-1}"

MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/PLLaMa-7b-base}"
QUESTIONS="${QUESTIONS:-data/plant_mcq/mobiplant_expert_all.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/pllama_plant_mcq_baselines}"
DENSE_RESULT="${DENSE_RESULT:-$OUTPUT_ROOT/dense_result.json}"
MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-auto}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MODEL_DEVICE_MAP
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# Dense is evaluated separately. The baseline set contains Magnitude, Wanda,
# SparseGPT, and the original OWL at three sparsity levels.
TASKS=(
  "magnitude|0.5"
  "magnitude|0.6"
  "magnitude|0.7"
  "wanda|0.5"
  "wanda|0.6"
  "wanda|0.7"
  "sparsegpt|0.5"
  "sparsegpt|0.6"
  "sparsegpt|0.7"
  "wanda-owl|0.5"
  "wanda-owl|0.6"
  "wanda-owl|0.7"
)

value_tag() {
  echo "$1" | tr '.' 'p' | tr '-' 'm'
}

result_path() {
  local method="$1" sparsity="$2"
  echo "$OUTPUT_ROOT/$method/s_$(value_tag "$sparsity")/plant_mcq_result.json"
}

prepare_check() {
  [[ -s "$QUESTIONS" ]] || {
    echo "Missing question file: $QUESTIONS" >&2
    exit 1
  }
  [[ -d "$MODEL_PATH" ]] || {
    echo "Missing model directory: $MODEL_PATH" >&2
    exit 1
  }
}

run_dense() {
  prepare_check
  mkdir -p "$OUTPUT_ROOT"
  "$PYTHON_BIN" scripts/eval_plant_mcq.py \
    --model "dense=$MODEL_PATH" \
    --questions "$QUESTIONS" \
    --output "$OUTPUT_ROOT/dense_result.json" \
    --device-map "$MODEL_DEVICE_MAP"
}

run_baselines() {
  prepare_check
  local task_count="${#TASKS[@]}"
  (( START_TASK >= 0 && START_TASK < task_count )) || {
    echo "START_TASK must be between 0 and $((task_count - 1))" >&2
    exit 2
  }
  (( END_TASK >= START_TASK && END_TASK < task_count )) || {
    echo "END_TASK must be between START_TASK and $((task_count - 1))" >&2
    exit 2
  }

  for (( index=START_TASK; index<=END_TASK; index++ )); do
    IFS='|' read -r method sparsity <<< "${TASKS[$index]}"
    result="$(result_path "$method" "$sparsity")"
    result_dir="$(dirname "$result")"
    mkdir -p "$result_dir"

    if [[ "$RESUME" == "1" && -s "$result" ]]; then
      echo "[$((index + 1))/${task_count}] skip existing result: $result"
      continue
    fi

    echo "======================================================================"
    echo "PLLaMA plant-MCQ baseline [$((index + 1))/${task_count}]"
    echo "GPU: ${CUDA_VISIBLE_DEVICES:-inherited}"
    echo "method=$method sparsity=$sparsity"
    echo "No pruned checkpoint will be saved."
    echo "======================================================================"

    method_args=()
    if [[ "$method" == "wanda-owl" ]]; then
      case "$sparsity" in
        0.5) method_args+=(--Hyper_m 5.0 --Lamda 0.06) ;;
        0.6) method_args+=(--Hyper_m 5.0 --Lamda 0.12) ;;
        0.7) method_args+=(--Hyper_m 6.0 --Lamda 0.18) ;;
      esac
    fi

    "$PYTHON_BIN" main.py \
      --model "$MODEL_PATH" \
      --cache_dir llm_weights \
      --prune_method "$method" \
      --sparsity_ratio "$sparsity" \
      --sparsity_type unstructured \
      --nsamples 128 \
      --seed 0 \
      --skip_ppl_eval \
      --eval_plant_mcq \
      --plant_mcq_questions "$QUESTIONS" \
      --plant_mcq_label "${method}_s${sparsity}" \
      --plant_mcq_output "$result" \
      --save "$result_dir" \
      "${method_args[@]}"
  done
}

run_summary() {
  "$PYTHON_BIN" scripts/summarize_plant_mcq_methods.py \
    --root "$OUTPUT_ROOT" \
    --dense-result "$DENSE_RESULT"
}

case "$MODE" in
  dense) run_dense ;;
  baseline) run_baselines ;;
  summary) run_summary ;;
  *)
    echo "MODE must be dense, baseline, or summary" >&2
    exit 2
    ;;
esac
