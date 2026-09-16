#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${MODEL_KEY:-llama2_13b}"
GPU_ID="${GPU_ID:-0,1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

case "$MODEL_KEY" in
  llama1_7b)
    MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/models/decapoda-research-llama-7B-hf}"
    GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-7B-hf_128_0_fixed.pth}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama1_7b_a100}"
    ;;
  llama2_7b)
    MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/Llama-2-7b-hf}"
    GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-7b-hf_128_0.pth}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama2_7b_a100}"
    ;;
  llama1_13b)
    MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/decapoda-research-llama-13B-hf}"
    GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-13B-hf_128_0.pth}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama1_13b_a100}"
    ;;
  llama2_13b)
    MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/Llama-2-13b-hf}"
    GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-13b-hf_128_0.pth}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama2_13b_a100}"
    ;;
  qwen2_5_7b)
    MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/Qwen2.5-7B}"
    GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/qwen2/gradients_l2_Qwen2.5-7B_128_0.pth}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_qwen2_5_7b_a100}"
    ;;
  mistral_7b)
    MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/Mistral-7B}"
    GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/mistral/gradients_l2_Mistral-7B_128_0.pth}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_mistral_7b_l20_2gpu}"
    ;;
  llama1_30b)
    MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/models/decapoda-research-llama-30B-hf}"
    GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-30B-hf_128_0.pth}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama1_30b_a100}"
    ;;
  *)
    echo "Unknown MODEL_KEY: $MODEL_KEY" >&2
    echo "Supported: llama1_7b, llama2_7b, llama1_13b, llama2_13b, qwen2_5_7b, mistral_7b, llama1_30b" >&2
    exit 1
    ;;
esac

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$GRADIENT_PATH" ]]; then
  echo "Gradient file does not exist: $GRADIENT_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" log
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "A100 terminal run: OWL-V9 + wsqrtg parameter sweep"
echo "MODEL_KEY: $MODEL_KEY"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Gradient: $GRADIENT_PATH"
echo "Output: $OUTPUT_ROOT"
echo "Total: 3 sparsities x 3 Hyper_m x 10 Lamda x 4 Owl_alpha = 360"
echo "============================================================"

python scripts/run_v9_param_sweep.py \
  --model "$MODEL_PATH" \
  --method owl-v9-wsqrtg \
  --output-root "$OUTPUT_ROOT" \
  --sparsities 0.5 0.6 0.7 \
  --hyper-ms 5 6 7 \
  --lamdas 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 \
  --owl-alphas 0.10 0.15 0.20 0.25 \
  --sparsity-type unstructured \
  --nsamples 128 \
  --seed 0 \
  --extra-args "--gradient_path $GRADIENT_PATH" \
  --resume

if [[ "$RUN_SUMMARY" == "1" ]]; then
  python scripts/summarize_v9_param_sweep.py --output-root "$OUTPUT_ROOT"
fi

echo "Done."
echo "Summary: $OUTPUT_ROOT/summary_all.csv"
echo "Best: $OUTPUT_ROOT/best_by_sparsity.csv"
