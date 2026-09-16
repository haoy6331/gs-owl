#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-sweep}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-}"
START_TASK="${START_TASK:-0}"
END_TASK="${END_TASK:-59}"
RESUME="${RESUME:-1}"
EXPECTED_GPUS="${EXPECTED_GPUS:-2}"

MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/PLLaMa-13b-base}"
GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/pllama/gradients_l2_PLLaMa-13b-base_128_0.pth}"
QUESTIONS="${QUESTIONS:-data/plant_mcq/mobiplant_expert_all.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/pllama_13b_plant_mcq_top20_sweep}"
DENSE_RESULT="${DENSE_RESULT:-$OUTPUT_ROOT/dense_result.json}"
MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-balanced}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MODEL_DEVICE_MAP
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# Each block contains the PPL Top-20 settings from the completed
# LLaMA-2-13B sweep for one target sparsity (50%, 60%, and 70%).
TASKS=(
  "0.5|6.0|0.04|0.20"
  "0.5|6.0|0.04|0.25"
  "0.5|6.0|0.04|0.15"
  "0.5|6.0|0.04|0.10"
  "0.5|5.0|0.04|0.10"
  "0.5|7.0|0.04|0.10"
  "0.5|7.0|0.04|0.15"
  "0.5|7.0|0.04|0.25"
  "0.5|7.0|0.04|0.20"
  "0.5|5.0|0.04|0.15"
  "0.5|5.0|0.04|0.20"
  "0.5|5.0|0.04|0.25"
  "0.5|6.0|0.02|0.25"
  "0.5|6.0|0.02|0.10"
  "0.5|6.0|0.02|0.15"
  "0.5|6.0|0.02|0.20"
  "0.5|7.0|0.06|0.10"
  "0.5|7.0|0.06|0.25"
  "0.5|7.0|0.06|0.20"
  "0.5|7.0|0.06|0.15"

  "0.6|6.0|0.08|0.25"
  "0.6|6.0|0.08|0.20"
  "0.6|6.0|0.08|0.15"
  "0.6|7.0|0.10|0.20"
  "0.6|7.0|0.10|0.15"
  "0.6|6.0|0.08|0.10"
  "0.6|7.0|0.10|0.10"
  "0.6|7.0|0.10|0.25"
  "0.6|7.0|0.08|0.10"
  "0.6|7.0|0.08|0.15"
  "0.6|7.0|0.08|0.25"
  "0.6|7.0|0.08|0.20"
  "0.6|6.0|0.10|0.25"
  "0.6|7.0|0.12|0.25"
  "0.6|6.0|0.10|0.20"
  "0.6|6.0|0.06|0.20"
  "0.6|6.0|0.06|0.10"
  "0.6|6.0|0.06|0.15"
  "0.6|7.0|0.12|0.20"
  "0.6|6.0|0.06|0.25"

  "0.7|6.0|0.12|0.15"
  "0.7|6.0|0.12|0.10"
  "0.7|6.0|0.12|0.20"
  "0.7|6.0|0.12|0.25"
  "0.7|6.0|0.14|0.10"
  "0.7|6.0|0.14|0.15"
  "0.7|6.0|0.14|0.20"
  "0.7|6.0|0.14|0.25"
  "0.7|7.0|0.16|0.10"
  "0.7|5.0|0.12|0.10"
  "0.7|5.0|0.12|0.20"
  "0.7|5.0|0.12|0.15"
  "0.7|5.0|0.12|0.25"
  "0.7|7.0|0.16|0.15"
  "0.7|7.0|0.16|0.20"
  "0.7|7.0|0.16|0.25"
  "0.7|7.0|0.14|0.10"
  "0.7|7.0|0.14|0.20"
  "0.7|7.0|0.14|0.15"
  "0.7|7.0|0.14|0.25"
)

value_tag() {
  echo "$1" | tr '.' 'p' | tr '-' 'm'
}

result_path() {
  local sparsity="$1" hyper_m="$2" lamda="$3" alpha="$4"
  echo "$OUTPUT_ROOT/s_$(value_tag "$sparsity")/hm_$(value_tag "$hyper_m")/la_$(value_tag "$lamda")/alpha_$(value_tag "$alpha")/plant_mcq_result.json"
}

check_gpu_count() {
  local gpu_count
  gpu_count="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
  if (( gpu_count != EXPECTED_GPUS )); then
    echo "Expected exactly $EXPECTED_GPUS visible GPUs, but PyTorch sees $gpu_count." >&2
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
  fi
  echo "Visible GPUs: $gpu_count"
}

check_common_inputs() {
  [[ -s "$QUESTIONS" ]] || {
    echo "Missing question file: $QUESTIONS" >&2
    exit 1
  }
  [[ -d "$MODEL_PATH" ]] || {
    echo "Missing model directory: $MODEL_PATH" >&2
    exit 1
  }
  check_gpu_count
}

check_sweep_inputs() {
  check_common_inputs
  [[ -f "$GRADIENT_PATH" ]] || {
    echo "Missing gradient file: $GRADIENT_PATH" >&2
    echo "Run bash run-pllama-13b-gradient.sh first." >&2
    exit 1
  }
}

run_dense() {
  check_common_inputs
  mkdir -p "$OUTPUT_ROOT"
  if [[ "$RESUME" == "1" && -s "$DENSE_RESULT" ]]; then
    echo "[skip] existing Dense result: $DENSE_RESULT"
    return
  fi
  "$PYTHON_BIN" scripts/eval_plant_mcq.py \
    --model "dense=$MODEL_PATH" \
    --questions "$QUESTIONS" \
    --output "$DENSE_RESULT" \
    --device-map "$MODEL_DEVICE_MAP"
}

run_sweep() {
  check_sweep_inputs
  local task_count="${#TASKS[@]}"
  if (( task_count != 60 )); then
    echo "Internal error: expected 60 candidates, found $task_count." >&2
    exit 2
  fi
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
    echo "PLLaMA-13B plant-MCQ sweep [$((index + 1))/${task_count}]"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-inherited}"
    echo "Device map: $MODEL_DEVICE_MAP"
    echo "sparsity=$sparsity Hyper_m=$hyper_m Lamda=$lamda Owl_alpha=$alpha"
    echo "The pruned model will be evaluated in memory and will not be saved."
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
  all)
    run_dense
    run_sweep
    run_summary
    ;;
  *)
    echo "MODE must be dense, sweep, summary, or all" >&2
    exit 2
    ;;
esac
