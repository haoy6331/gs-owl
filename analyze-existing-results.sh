#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-owl}"
OUT_DIR="${OUT_DIR:-owl/analysis}"
METHODS="${METHODS:-owl-v9-wsqrtg wanda-owl-v9 wanda-owl}"
SPARSITIES="${SPARSITIES:-0.5 0.6 0.7}"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Layer distribution visualization"
echo "ROOT: $ROOT"
echo "METHODS: $METHODS"
echo "SPARSITIES: $SPARSITIES"
echo "Output: $OUT_DIR/layer_distribution"
echo "============================================================"

python scripts/plot_layer_distribution.py \
  --root "$ROOT" \
  --out-dir "$OUT_DIR/layer_distribution" \
  --select best \
  --methods $METHODS \
  --sparsities $SPARSITIES

echo "============================================================"
echo "Efficiency analysis"
echo "Output: $OUT_DIR/efficiency"
echo "============================================================"

python scripts/analyze_efficiency.py \
  --root "$ROOT" \
  --out-dir "$OUT_DIR/efficiency" \
  --methods $METHODS \
  --sparsities $SPARSITIES

echo "Done."
echo "Layer CSV: $OUT_DIR/layer_distribution/layer_distribution.csv"
echo "Efficiency CSV: $OUT_DIR/efficiency/efficiency_summary.csv"
