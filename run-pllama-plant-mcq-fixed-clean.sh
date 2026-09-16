#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-all}"
MODEL_SIZE="${MODEL_SIZE:-7b}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-}"
RESUME="${RESUME:-1}"

SOURCE_QUESTIONS="${SOURCE_QUESTIONS:-data/plant_mcq/mobiplant_expert_all.jsonl}"
QUESTIONS="${QUESTIONS:-data/plant_mcq/mobiplant_expert_clean_v2.jsonl}"
SPARSITIES="${SPARSITIES:-0.5 0.6 0.7}"
HYPER_M="${HYPER_M:-5}"
LAMDA="${LAMDA:-0.08}"
OWL_ALPHA="${OWL_ALPHA:-0.20}"
GRAD_BETA="${GRAD_BETA:-0.5}"

case "$MODEL_SIZE" in
  7b)
    DEFAULT_MODEL_PATH="/home/yh114/workdir/DSnoT/models/PLLaMa-7b-base"
    DEFAULT_GRADIENT_PATH="/home/yh114/workdir/Pruner-Zero/gradients/pllama/gradients_l2_PLLaMa-7b-base_128_0.pth"
    DEFAULT_OUTPUT_ROOT="owl/pllama_7b_plant_mcq_l20_fixed_clean_v2"
    DEFAULT_DEVICE_MAP="auto"
    MIN_GPUS="${MIN_GPUS:-1}"
    ;;
  13b)
    DEFAULT_MODEL_PATH="/home/yh114/workdir/DSnoT/models/PLLaMa-13b-base"
    DEFAULT_GRADIENT_PATH="/home/yh114/workdir/Pruner-Zero/gradients/pllama/gradients_l2_PLLaMa-13b-base_128_0.pth"
    DEFAULT_OUTPUT_ROOT="owl/pllama_13b_plant_mcq_l20_fixed_clean_v2"
    DEFAULT_DEVICE_MAP="balanced"
    MIN_GPUS="${MIN_GPUS:-2}"
    ;;
  *)
    echo "MODEL_SIZE must be 7b or 13b" >&2
    exit 2
    ;;
esac

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
GRADIENT_PATH="${GRADIENT_PATH:-$DEFAULT_GRADIENT_PATH}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-$DEFAULT_DEVICE_MAP}"
DENSE_RESULT="${DENSE_RESULT:-$OUTPUT_ROOT/dense_result.json}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MODEL_DEVICE_MAP
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

value_tag() {
  echo "$1" | tr '.' 'p' | tr '-' 'm'
}

result_path() {
  local sparsity="$1"
  echo "$OUTPUT_ROOT/s_$(value_tag "$sparsity")/fixed_hm_$(value_tag "$HYPER_M")_la_$(value_tag "$LAMDA")_alpha_$(value_tag "$OWL_ALPHA")_beta_$(value_tag "$GRAD_BETA")/plant_mcq_result.json"
}

prepare_questions() {
  if [[ -s "$QUESTIONS" && "${FORCE_PREPARE:-0}" != "1" ]]; then
    echo "[skip] clean question set already exists: $QUESTIONS"
    return
  fi
  [[ -s "$SOURCE_QUESTIONS" ]] || {
    echo "Missing source question file: $SOURCE_QUESTIONS" >&2
    exit 1
  }
  "$PYTHON_BIN" scripts/prepare_plant_mcq.py \
    --local-json "$SOURCE_QUESTIONS" \
    --output "$QUESTIONS" \
    --preserve-ids \
    --seed 2026
}

check_gpu_count() {
  if [[ "${SKIP_GPU_CHECK:-0}" == "1" ]]; then
    return
  fi
  local gpu_count
  gpu_count="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
  if (( gpu_count < MIN_GPUS )); then
    echo "PLLaMA-$MODEL_SIZE requires at least $MIN_GPUS visible GPU(s), but PyTorch sees $gpu_count." >&2
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
  fi
  echo "Visible GPUs: $gpu_count"
  local gpu_names
  gpu_names="$($PYTHON_BIN -c 'import torch; print("\n".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))')"
  echo "GPU models:"
  echo "$gpu_names"
  if [[ "${REQUIRE_L20:-1}" == "1" ]]; then
    while IFS= read -r gpu_name; do
      [[ "$gpu_name" == *L20* ]] || {
        echo "Expected NVIDIA L20 GPUs, but detected: $gpu_name" >&2
        exit 1
      }
    done <<< "$gpu_names"
  fi
}

check_model_inputs() {
  [[ -s "$QUESTIONS" ]] || {
    echo "Missing clean question file: $QUESTIONS" >&2
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
  check_gpu_count
}

run_dense() {
  [[ -s "$QUESTIONS" ]] || { echo "Missing clean question file: $QUESTIONS" >&2; exit 1; }
  [[ -d "$MODEL_PATH" ]] || { echo "Missing model directory: $MODEL_PATH" >&2; exit 1; }
  check_gpu_count
  mkdir -p "$OUTPUT_ROOT"
  if [[ "$RESUME" == "1" && -s "$DENSE_RESULT" ]]; then
    echo "[skip] existing dense result: $DENSE_RESULT"
    return
  fi
  "$PYTHON_BIN" scripts/eval_plant_mcq.py \
    --model "dense=$MODEL_PATH" \
    --questions "$QUESTIONS" \
    --output "$DENSE_RESULT" \
    --device-map "$MODEL_DEVICE_MAP" \
    --max-length 2048
}

run_fixed() {
  check_model_inputs
  for sparsity in $SPARSITIES; do
    local result result_dir
    result="$(result_path "$sparsity")"
    result_dir="$(dirname "$result")"
    mkdir -p "$result_dir"
    if [[ "$RESUME" == "1" && -s "$result" ]]; then
      echo "[skip] existing fixed result: $result"
      continue
    fi

    echo "================================================================================================"
    echo "PLLaMA-$MODEL_SIZE plant-MCQ fixed protocol"
    echo "sparsity=$sparsity (H_m, lambda, alpha, beta)=($HYPER_M, $LAMDA, $OWL_ALPHA, $GRAD_BETA)"
    echo "questions=$QUESTIONS"
    echo "output=$result"
    echo "================================================================================================"

    "$PYTHON_BIN" main.py \
      --model "$MODEL_PATH" \
      --cache_dir llm_weights \
      --prune_method owl-v9-wsqrtg \
      --sparsity_ratio "$sparsity" \
      --sparsity_type unstructured \
      --nsamples 128 \
      --seed 0 \
      --Hyper_m "$HYPER_M" \
      --Lamda "$LAMDA" \
      --Owl_alpha "$OWL_ALPHA" \
      --Grad_beta "$GRAD_BETA" \
      --gradient_path "$GRADIENT_PATH" \
      --skip_ppl_eval \
      --eval_plant_mcq \
      --plant_mcq_questions "$QUESTIONS" \
      --plant_mcq_max_length 2048 \
      --plant_mcq_label "pllama_${MODEL_SIZE}_s${sparsity}_fixed" \
      --plant_mcq_output "$result" \
      --save "$result_dir"
  done
}

run_summary() {
  "$PYTHON_BIN" scripts/summarize_plant_mcq_fixed.py \
    --root "$OUTPUT_ROOT" \
    --dense-result "$DENSE_RESULT" \
    --hyper-m "$HYPER_M" \
    --lamda "$LAMDA" \
    --alpha "$OWL_ALPHA" \
    --beta "$GRAD_BETA" \
    --require-gpu-substring L20
}

case "$MODE" in
  prepare) prepare_questions ;;
  dense) run_dense ;;
  fixed) run_fixed ;;
  summary) run_summary ;;
  all)
    prepare_questions
    run_dense
    run_fixed
    run_summary
    ;;
  *)
    echo "MODE must be prepare, dense, fixed, summary, or all" >&2
    exit 2
    ;;
esac
