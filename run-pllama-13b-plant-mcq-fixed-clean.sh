#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_SIZE=13b
exec bash "$SCRIPT_DIR/run-pllama-plant-mcq-fixed-clean.sh" "$@"
