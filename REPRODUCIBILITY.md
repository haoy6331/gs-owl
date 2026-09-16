# GS-OWL Source Repository

This repository is a cleaned source-code copy of the Pruner-Zero/GS-OWL
experiments. Model weights, gradient checkpoints, generated experiment
results, local build directories, and temporary paper packages are excluded
from version control.

## Main components

- `main.py`, `lib/`, and `scripts/`: pruning, evaluation, and experiment code.
- `data/plant_mcq/`: the cleaned 564-question plant-science benchmark and its audit files.
- `paper/computer_engineering_latex/`: LaTeX source, figure-generation scripts, and canonical figures.
- `lm-evaluation-harness/`: the evaluation harness used by the zero-shot experiments.
- `run-*.sh`, `submit-*.sh`, and `*.slurm`: local and L20 cluster entry points.

## Fixed PLLaMA plant-MCQ protocol

The cleaned PLLaMA protocol uses the fixed tuple `(5, 0.08, 0.20, 0.5)` for
`(H_m, lambda, alpha, beta)`, 128 calibration samples, seed 0, and the same
question file for Dense, 50%, 60%, and 70% sparsity.

For an L20 cluster:

```bash
bash submit-pllama-plant-mcq-fixed-clean.sh
```

The submission scripts verify that the allocated GPUs are NVIDIA L20 cards
before running. Results are written under `owl/` locally and are intentionally
ignored by Git.

## Paper build

The canonical LaTeX source is in `paper/computer_engineering_latex/`. See its
README for XeLaTeX/Tectonic compilation and figure-generation commands.
