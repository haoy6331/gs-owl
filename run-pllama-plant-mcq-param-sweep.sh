#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-sweep}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-}"
START_TASK="${START_TASK:-0}"
END_TASK="${END_TASK:-59}"
RESUME="${RESUME:-1}"

MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/PLLaMa-7b-base}"
GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/pllama/gradients_l2_PLLaMa-7b-base_128_0.pth}"
QUESTIONS="${QUESTIONS:-data/plant_mcq/mobiplant_expert_all.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/pllama_plant_mcq_top20_sweep}"
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

# Top-20 PPL-ranked LLaMA-2-7B candidates for each sparsity. PLLaMA evaluates
# all 60 candidates directly on the fixed plant-MCQ set without saving models.
TASKS=(
  "0.5|5.0|0.06|0.15"
  "0.5|5.0|0.06|0.10"
  "0.5|5.0|0.06|0.20"
  "0.5|6.0|0.06|0.20"
  "0.5|6.0|0.06|0.25"
  "0.5|5.0|0.06|0.25"
  "0.5|6.0|0.06|0.15"
  "0.5|6.0|0.06|0.10"
  "0.5|6.0|0.08|0.10"
  "0.5|6.0|0.08|0.15"
  "0.5|5.0|0.08|0.10"
  "0.5|5.0|0.08|0.15"
  "0.5|6.0|0.08|0.20"
  "0.5|6.0|0.08|0.25"
  "0.5|5.0|0.08|0.20"
  "0.5|5.0|0.08|0.25"
  "0.5|7.0|0.04|0.15"
  "0.5|7.0|0.04|0.20"
  "0.5|7.0|0.04|0.10"
  "0.5|7.0|0.04|0.25"

  "0.6|5.0|0.12|0.10"
  "0.6|5.0|0.12|0.15"
  "0.6|5.0|0.12|0.20"
  "0.6|5.0|0.12|0.25"
  "0.6|6.0|0.12|0.10"
  "0.6|6.0|0.12|0.15"
  "0.6|5.0|0.10|0.10"
  "0.6|6.0|0.12|0.20"
  "0.6|5.0|0.10|0.20"
  "0.6|5.0|0.10|0.15"
  "0.6|6.0|0.12|0.25"
  "0.6|5.0|0.10|0.25"
  "0.6|6.0|0.10|0.10"
  "0.6|6.0|0.10|0.15"
  "0.6|6.0|0.10|0.20"
  "0.6|6.0|0.10|0.25"
  "0.6|5.0|0.14|0.10"
  "0.6|5.0|0.14|0.15"
  "0.6|6.0|0.14|0.10"
  "0.6|5.0|0.14|0.20"

  "0.7|6.0|0.18|0.10"
  "0.7|6.0|0.16|0.10"
  "0.7|6.0|0.18|0.15"
  "0.7|6.0|0.18|0.20"
  "0.7|6.0|0.18|0.25"
  "0.7|6.0|0.16|0.15"
  "0.7|6.0|0.16|0.20"
  "0.7|6.0|0.16|0.25"
  "0.7|7.0|0.18|0.10"
  "0.7|7.0|0.16|0.10"
  "0.7|7.0|0.16|0.15"
  "0.7|7.0|0.18|0.20"
  "0.7|7.0|0.18|0.15"
  "0.7|7.0|0.16|0.25"
  "0.7|7.0|0.16|0.20"
  "0.7|7.0|0.18|0.25"
  "0.7|6.0|0.14|0.20"
  "0.7|6.0|0.14|0.15"
  "0.7|6.0|0.14|0.10"
  "0.7|6.0|0.20|0.10"
)

value_tag() {
  echo "$1" | tr '.' 'p' | tr '-' 'm'
}

result_path() {
  local sparsity="$1" hyper_m="$2" lamda="$3" alpha="$4"
  echo "$OUTPUT_ROOT/s_$(value_tag "$sparsity")/hm_$(value_tag "$hyper_m")/la_$(value_tag "$lamda")/alpha_$(value_tag "$alpha")/plant_mcq_result.json"
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
  [[ -f "$GRADIENT_PATH" ]] || {
    echo "Missing gradient file: $GRADIENT_PATH" >&2
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

run_sweep() {
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
    IFS='|' read -r sparsity hyper_m lamda alpha <<< "${TASKS[$index]}"
    result="$(result_path "$sparsity" "$hyper_m" "$lamda" "$alpha")"
    result_dir="$(dirname "$result")"
    mkdir -p "$result_dir"

    if [[ "$RESUME" == "1" && -s "$result" ]]; then
      echo "[$((index + 1))/${task_count}] skip existing result: $result"
      continue
    fi

    echo "======================================================================"
    echo "PLLaMA plant-MCQ parameter sweep [$((index + 1))/${task_count}]"
    echo "GPU: ${CUDA_VISIBLE_DEVICES:-inherited}"
    echo "sparsity=$sparsity Hyper_m=$hyper_m Lamda=$lamda Owl_alpha=$alpha"
    echo "No pruned checkpoint will be saved."
    echo "======================================================================"

    "$PYTHON_BIN" main.py \
      --model "$MODEL_PATH" \
      --cache_dir llm_weights \
      --prune_method owl-v9-wsqrtg \
      --sparsity_ratio "$sparsity" \
      --sparsity_type unstructured \
      --nsamples 128 \
      --seed 0 \
      --Hyper_m "$hyper_m" \
      --Lamda "$lamda" \
      --Owl_alpha "$alpha" \
      --gradient_path "$GRADIENT_PATH" \
      --skip_ppl_eval \
      --eval_plant_mcq \
      --plant_mcq_questions "$QUESTIONS" \
      --plant_mcq_label "s${sparsity}_hm${hyper_m}_la${lamda}_a${alpha}" \
      --plant_mcq_output "$result" \
      --save "$result_dir"
  done
}

run_summary() {
  "$PYTHON_BIN" scripts/summarize_plant_mcq_sweep.py \
    --root "$OUTPUT_ROOT" \
    --dense-result "$DENSE_RESULT"
}

case "$MODE" in
  dense) run_dense ;;
  sweep) run_sweep ;;
  summary) run_summary ;;
  *)
    echo "MODE must be dense, sweep, or summary" >&2
    exit 2
    ;;
esac
