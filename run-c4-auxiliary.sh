#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yh114/miniconda3/envs/ZW-Prune/bin/python}"
MODELS="${MODELS:-llama1_7b llama2_7b}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"

read -r -a MODEL_ARGS <<< "${MODELS}"

CMD=(
  "${PYTHON_BIN}" scripts/run_c4_auxiliary.py
  --models "${MODEL_ARGS[@]}"
  --start-index "${START_INDEX}"
  --resume
)

if [[ -n "${END_INDEX}" ]]; then
  CMD+=(--end-index "${END_INDEX}")
fi

"${CMD[@]}"
