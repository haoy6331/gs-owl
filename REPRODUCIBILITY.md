# GS-OWL Reproducibility Notes

This repository contains the cleaned source code and experimental protocol for
GS-OWL. Model weights, gradient checkpoints, generated masks, experiment
results, local build directories, and temporary paper packages are excluded
from version control.

## Main components

- `main.py`, `lib/`, and `scripts/`: pruning, evaluation, and analysis code.
- `data/plant_mcq/`: the cleaned 564-question plant-science benchmark and its audit files.
- `paper/computer_engineering_latex/`: LaTeX source, figure-generation scripts, and canonical figures.
- `lm-evaluation-harness/`: the evaluation harness used by the zero-shot experiments.

## Fixed main protocol

The fixed main protocol uses the tuple `(5, 0.08, 0.20, 0.5)` for
`(H_m, lambda, alpha, beta)`, 128 calibration samples, seed 0, and the same
configuration for all formal tests. The zero-shot protocol uses one sparse
mask per model-sparsity pair across all seven downstream tasks.

## Paper build

The canonical LaTeX source is in `paper/computer_engineering_latex/`. Generated
PDF and PNG files are intentionally excluded; use the drawing scripts and
LaTeX sources in that directory to rebuild them.
