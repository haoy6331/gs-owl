#!/usr/bin/env bash
set -euo pipefail

MECHANISM_ROOT="${MECHANISM_ROOT:-owl/osla_mechanism_ablation}"
COMPONENT_ROOT="${COMPONENT_ROOT:-owl/v9_component_ablation}"
SENSITIVITY_ROOT="${SENSITIVITY_ROOT:-owl/layer_sensitivity_wsqrtg}"
OUT_DIR="${OUT_DIR:-owl/analysis/osla_mechanism}"

python scripts/plot_osla_mechanism.py \
  --mechanism-root "$MECHANISM_ROOT" \
  --component-root "$COMPONENT_ROOT" \
  --sensitivity-root "$SENSITIVITY_ROOT" \
  --out-dir "$OUT_DIR" \
  --formats pdf png \
  --dpi 300
