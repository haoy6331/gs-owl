#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-paper/figures}"
FORMATS="${FORMATS:-png pdf svg}"
DPI="${DPI:-400}"

python scripts/draw_main_method_figure.py \
  --out-dir "$OUT_DIR" \
  --formats $FORMATS \
  --dpi "$DPI"
