#!/bin/bash
#SBATCH -J opt-baseline
#SBATCH -o log/%j.opt-baseline-l20.out.txt
#SBATCH -e log/%j.opt-baseline-l20.err.txt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

set -eo pipefail

source ~/.bashrc
conda activate ZW-Prune
set -u

MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required}"
METHODS="${METHODS:-dense magnitude wanda sparsegpt pruner-zero}"
EVAL_ZERO_SHOT="${EVAL_ZERO_SHOT:-0}"
LOCAL_DATASETS_PATH="${LOCAL_DATASETS_PATH:-/home/yh114/workdir/Pruner-Zero/data_hf}"
JSON_TREE="${JSON_TREE:-data/pruner-zero_tree.json}"

case "$MODEL_KEY" in
  opt1_3b)
    MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/opt-1.3b}"
    GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/opt/gradients_l2_opt-1.3b_128_0.pth}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/opt_1_3b_baselines_l20}"
    export MODEL_DEVICE_MAP="auto"
    ;;
  opt2_7b)
    MODEL_PATH="${MODEL_PATH:-/home/yh114/workdir/DSnoT/models/opt-2.7b}"
    GRADIENT_PATH="${GRADIENT_PATH:-/home/yh114/workdir/Pruner-Zero/gradients/opt/gradients_l2_opt-2.7b_128_0.pth}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-owl/opt_2_7b_baselines_l20}"
    export MODEL_DEVICE_MAP="auto"
    ;;
  *)
    echo "Unknown MODEL_KEY: $MODEL_KEY" >&2
    echo "Supported: opt1_3b, opt2_7b" >&2
    exit 1
    ;;
esac

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 1
fi
if [[ "$METHODS" == *"pruner-zero"* && ! -f "$GRADIENT_PATH" ]]; then
  echo "Gradient file does not exist: $GRADIENT_PATH" >&2
  exit 1
fi
if [[ "$METHODS" == *"pruner-zero"* && ! -f "$JSON_TREE" ]]; then
  echo "Pruner-Zero tree does not exist: $JSON_TREE" >&2
  exit 1
fi

read -r -a METHOD_ARGS <<< "$METHODS"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p log "$OUTPUT_ROOT"

echo "============================================================"
echo "OPT baseline experiments on L20"
echo "MODEL_KEY: $MODEL_KEY"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-assigned-by-slurm}"
echo "Visible GPU count: ${SLURM_GPUS_ON_NODE:-unknown}"
echo "Device map: $MODEL_DEVICE_MAP"
echo "Model: $MODEL_PATH"
echo "Gradient: $GRADIENT_PATH"
echo "Methods: $METHODS"
echo "Zero-shot: $EVAL_ZERO_SHOT"
echo "Output: $OUTPUT_ROOT"
echo "============================================================"

command=(
  python scripts/run_opt_baselines.py
  --main main_opt.py
  --model "$MODEL_PATH"
  --gradient-path "$GRADIENT_PATH"
  --json-tree "$JSON_TREE"
  --output-root "$OUTPUT_ROOT"
  --methods "${METHOD_ARGS[@]}"
  --sparsities 0.5 0.6 0.7
  --sparsity-type unstructured
  --nsamples 128
  --seed 0
  --resume
)

if [[ "$EVAL_ZERO_SHOT" == "1" ]]; then
  command+=(--eval-zero-shot --offline --local-datasets-path "$LOCAL_DATASETS_PATH")
fi

"${command[@]}"
python scripts/summarize_opt_baselines.py --output-root "$OUTPUT_ROOT"
