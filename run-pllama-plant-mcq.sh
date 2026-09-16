#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-all}"
SPARSITIES="${SPARSITIES:-0.5 0.6 0.7}"
EVAL_GROUP="${EVAL_GROUP:-all}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/PLLaMa-7b-base}"
GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/pllama/gradients_l2_PLLaMa-7b-base_128_0.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-owl/pllama_plant_mcq}"
QUESTIONS="${QUESTIONS:-data/plant_mcq/mobiplant_expert_all.jsonl}"
PREPARE_LOCAL_JSON="${PREPARE_LOCAL_JSON:-}"
MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-auto}"

export MODEL_DEVICE_MAP
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SKIP_CUDA_EMPTY_CACHE="${SKIP_CUDA_EMPTY_CACHE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

checkpoint_dir() {
  local sparsity="$1"
  local tag="${sparsity/./p}"
  echo "$OUTPUT_ROOT/checkpoints/gs_owl_s_${tag}"
}

checkpoint_complete() {
  local checkpoint="$1"
  [[ -f "$checkpoint/config.json" ]] && {
    compgen -G "$checkpoint/*.safetensors" >/dev/null ||
      compgen -G "$checkpoint/pytorch_model*.bin" >/dev/null
  }
}

prepare_questions() {
  if [[ -s "$QUESTIONS" && "${FORCE_PREPARE:-0}" != "1" ]]; then
    echo "Question set already exists: $QUESTIONS"
    return
  fi

  local args=(
    scripts/prepare_plant_mcq.py
    --output "$QUESTIONS"
    --seed 2026
  )
  if [[ -n "$PREPARE_LOCAL_JSON" ]]; then
    args+=(--local-json "$PREPARE_LOCAL_JSON")
  fi
  "$PYTHON_BIN" "${args[@]}"
}

parameters_for_sparsity() {
  case "$1" in
    0.5) echo "5.0 0.06 0.15" ;;
    0.6) echo "5.0 0.12 0.10" ;;
    0.7) echo "6.0 0.18 0.10" ;;
    *)
      echo "Unsupported sparsity: $1 (expected 0.5, 0.6, or 0.7)" >&2
      exit 2
      ;;
  esac
}

prune_models() {
  [[ -d "$MODEL_PATH" ]] || { echo "Missing model directory: $MODEL_PATH" >&2; exit 1; }
  [[ -f "$GRADIENT_PATH" ]] || { echo "Missing gradient file: $GRADIENT_PATH" >&2; exit 1; }

  for sparsity in $SPARSITIES; do
    read -r hyper_m lamda owl_alpha <<< "$(parameters_for_sparsity "$sparsity")"
    checkpoint="$(checkpoint_dir "$sparsity")"
    result_dir="$OUTPUT_ROOT/pruning/s_${sparsity/./p}"

    if checkpoint_complete "$checkpoint" && [[ "${FORCE_PRUNE:-0}" != "1" ]]; then
      echo "[skip] checkpoint complete: $checkpoint"
      continue
    fi

    echo "======================================================================"
    echo "PLLaMA-7B GS-OWL pruning"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-inherited}"
    echo "sparsity=$sparsity Hyper_m=$hyper_m Lamda=$lamda Owl_alpha=$owl_alpha"
    echo "checkpoint=$checkpoint"
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
      --Owl_alpha "$owl_alpha" \
      --gradient_path "$GRADIENT_PATH" \
      --skip_ppl_eval \
      --save "$result_dir" \
      --save_model "$checkpoint"
  done
}

eval_models() {
  [[ -s "$QUESTIONS" ]] || { echo "Missing question set: $QUESTIONS" >&2; exit 1; }
  local args=(scripts/eval_plant_mcq.py --questions "$QUESTIONS" --device-map "$MODEL_DEVICE_MAP")

  case "$EVAL_GROUP" in
    all)
      args+=(--model "dense=$MODEL_PATH")
      for sparsity in 0.5 0.6 0.7; do
        checkpoint="$(checkpoint_dir "$sparsity")"
        checkpoint_complete "$checkpoint" || { echo "Incomplete checkpoint: $checkpoint" >&2; exit 1; }
        args+=(--model "gs_owl_${sparsity/./p}=$checkpoint")
      done
      ;;
    dense_s05)
      checkpoint="$(checkpoint_dir 0.5)"
      checkpoint_complete "$checkpoint" || { echo "Incomplete checkpoint: $checkpoint" >&2; exit 1; }
      args+=(--model "dense=$MODEL_PATH" --model "gs_owl_0p5=$checkpoint")
      ;;
    s06_s07)
      for sparsity in 0.6 0.7; do
        checkpoint="$(checkpoint_dir "$sparsity")"
        checkpoint_complete "$checkpoint" || { echo "Incomplete checkpoint: $checkpoint" >&2; exit 1; }
        args+=(--model "gs_owl_${sparsity/./p}=$checkpoint")
      done
      ;;
    *)
      echo "Unsupported EVAL_GROUP: $EVAL_GROUP (all, dense_s05, or s06_s07)" >&2
      exit 2
      ;;
  esac

  mkdir -p "$OUTPUT_ROOT"
  args+=(--output "$OUTPUT_ROOT/results_${EVAL_GROUP}.json")
  "$PYTHON_BIN" "${args[@]}"
}

summarize_results() {
  local all_result="$OUTPUT_ROOT/results_all.json"
  local first_shard="$OUTPUT_ROOT/results_dense_s05.json"
  local second_shard="$OUTPUT_ROOT/results_s06_s07.json"
  local args=(scripts/summarize_plant_mcq.py --output "$OUTPUT_ROOT/results_combined.json")

  if [[ -s "$all_result" ]]; then
    args+=(--input "$all_result")
  else
    [[ -s "$first_shard" ]] || { echo "Missing result shard: $first_shard" >&2; exit 1; }
    [[ -s "$second_shard" ]] || { echo "Missing result shard: $second_shard" >&2; exit 1; }
    args+=(--input "$first_shard" --input "$second_shard")
  fi
  "$PYTHON_BIN" "${args[@]}"
}

case "$MODE" in
  prepare) prepare_questions ;;
  prune) prune_models ;;
  eval) eval_models ;;
  summary) summarize_results ;;
  all)
    prepare_questions
    prune_models
    eval_models
    summarize_results
    ;;
  *)
    echo "Unsupported MODE: $MODE (prepare, prune, eval, summary, or all)" >&2
    exit 2
    ;;
esac
