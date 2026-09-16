#!/bin/bash
#SBATCH -J wsqrtg-2gpu
#SBATCH -o log/%A_%a.wsqrtg-2gpu.out.txt
#SBATCH -e log/%A_%a.wsqrtg-2gpu.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --array=0-5%1
#SBATCH --time=12:00:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

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

MODEL_KEY="${MODEL_KEY:-llama2_13b}"

case "$MODEL_KEY" in
  llama2_13b)
    MODEL_PATH="$(pick_existing_path "${MODEL_PATH:-${LLAMA2_13B_MODEL_PATH:-}}" \
      "/home/yh114/workdir/DSnoT/models/Llama-2-13b-hf" \
      "/home/yh114/workdir/models/Llama-2-13b-hf")"
    GRADIENT_PATH="$(pick_existing_path "${GRADIENT_PATH:-${LLAMA2_13B_GRADIENT_PATH:-}}" \
      "/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-13b-hf_128_0.pth")"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_llama2_13b_l20_2gpu}"
    ;;
  qwen2_5_7b)
    MODEL_PATH="$(pick_existing_path "${MODEL_PATH:-${QWEN2_5_7B_MODEL_PATH:-}}" \
      "/home/yh114/workdir/DSnoT/models/Qwen2.5-7B")"
    GRADIENT_PATH="$(pick_existing_path "${GRADIENT_PATH:-${QWEN2_5_7B_GRADIENT_PATH:-}}" \
      "/home/yh114/workdir/Pruner-Zero/gradients/qwen2/gradients_l2_Qwen2.5-7B_128_0.pth")"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_qwen2_5_7b_l20_2gpu}"
    ;;
  mistral_7b)
    MODEL_PATH="$(pick_existing_path "${MODEL_PATH:-${MISTRAL_7B_MODEL_PATH:-}}" \
      "/home/yh114/workdir/DSnoT/models/Mistral-7B")"
    GRADIENT_PATH="$(pick_existing_path "${GRADIENT_PATH:-${MISTRAL_7B_GRADIENT_PATH:-}}" \
      "/home/yh114/workdir/Pruner-Zero/gradients/mistral/gradients_l2_Mistral-7B_128_0.pth")"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_mistral_7b_l20_2gpu}"
    ;;
  vicuna_7b)
    MODEL_PATH="$(pick_existing_path "${MODEL_PATH:-${VICUNA_7B_MODEL_PATH:-/home/yh114/workdir/DSnoT/models/Vicuna-7B}}" \
      "/home/yh114/workdir/DSnoT/models/Vicuna-7B" \
      "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.5" \
      "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.3" \
      "/home/yh114/workdir/DSnoT/models/vicuna-7b-v1.1" \
      "/home/yh114/workdir/DSnoT/models/vicuna-7b" \
      "/home/yh114/workdir/models/vicuna-7b-v1.5" \
      "/home/yh114/workdir/models/vicuna-7b")"
    GRADIENT_PATH="$(pick_existing_path "${GRADIENT_PATH:-${VICUNA_7B_GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/gradients_l2_Vicuna-7B_128_0.pth}}" \
      "/home/yh114/workdir/Pruner-Zero/gradients/gradients_l2_Vicuna-7B_128_0.pth" \
      "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_vicuna-7b-v1.5_128_0.pth" \
      "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_vicuna-7b-v1.3_128_0.pth" \
      "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_vicuna-7b-v1.1_128_0.pth" \
      "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_vicuna-7b_128_0.pth" \
      "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_Vicuna-7B_128_0.pth" \
      "/home/yh114/workdir/Pruner-Zero/gradients/vicuna/gradients_l2_vicuna-7b-v1.5_128_0.pth" \
      "/home/yh114/workdir/Pruner-Zero/gradients/vicuna/gradients_l2_vicuna-7b_128_0.pth")"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/owl_v9_wsqrtg_param_vicuna_7b_l20_2gpu}"
    ;;
  *)
    echo "Unknown MODEL_KEY: $MODEL_KEY" >&2
    echo "Supported: llama2_13b, qwen2_5_7b, mistral_7b, vicuna_7b" >&2
    exit 1
    ;;
esac

NUM_SHARDS="${NUM_SHARDS:-6}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$GRADIENT_PATH" ]]; then
  echo "Gradient file does not exist: $GRADIENT_PATH" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "OWL-V9 + wsqrtg two-GPU L20 parameter sweep"
echo "MODEL_KEY: $MODEL_KEY"
echo "Shard: $SLURM_ARRAY_TASK_ID/$((NUM_SHARDS - 1))"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-assigned-by-slurm}"
echo "Model: $MODEL_PATH"
echo "Gradient: $GRADIENT_PATH"
echo "Output: $OUTPUT_ROOT"

python scripts/run_v9_shard.py \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS" \
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
