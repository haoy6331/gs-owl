#!/usr/bin/env bash
set -euo pipefail

ABLATION_ROOT="${ABLATION_ROOT:-owl/strict_ablation_wsqrtg_owl_v9}"
OUT_DIR="${OUT_DIR:-owl/analysis/paper_visualizations}"
MODELS="${MODELS:-llama1_7b llama2_7b}"
LAYER_SPARSITY="${LAYER_SPARSITY:-0.6}"
HEATMAP_SPARSITIES="${HEATMAP_SPARSITIES:-0.5 0.6 0.7}"
FORMATS="${FORMATS:-png pdf}"
DPI="${DPI:-400}"
FIGURE="${FIGURE:-all}"

python scripts/plot_strict_ablation_visuals.py \
  --ablation-root "$ABLATION_ROOT" \
  --out-dir "$OUT_DIR" \
  --models $MODELS \
  --layer-sparsity "$LAYER_SPARSITY" \
  --heatmap-sparsities $HEATMAP_SPARSITIES \
  --formats $FORMATS \
  --dpi "$DPI" \
  --figure "$FIGURE"
