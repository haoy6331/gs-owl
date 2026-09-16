# GS-OWL Main Framework V2 - Design and Notation Notes

## What was removed from V1

- Removed the symmetric OSLA/GWS flowchart layout and the late-stage merge.
- Removed large enclosing cards, decorative backgrounds, repeated explanatory sentences, and formula-heavy modules.
- Removed random checkerboard masks and unrelated mini charts.
- Replaced generic process boxes with model stacks, activation matrices, saliency matrices, a row-wise threshold, and a mask-consistent sparse matrix.

## How V2 shows dual-scale coordination

- The dense LLM provides a model-level activation view and a magnified view of layer `l`.
- OSLA uses layer activation outlier counts and exceedance severity to generate a nonuniform layer sparsity vector.
- The highlighted budget `s_l` drops directly onto the GWS row-wise selection boundary.
- GWS ranks connections inside the selected layer with the paper's saliency definition.
- The displayed binary mask has a structured row-wise deletion region, and the sparse weight matrix uses exactly the same zero pattern.
- Repeating the constrained selection across layers yields a sparse LLM with nonuniform layer densities under the global target sparsity.

## Text-formula-meaning check

| Figure element | Manuscript notation | Meaning |
|---|---|---|
| Activation matrices | `X_l` | Calibration activations used to derive layer-level outlier evidence. |
| Outlier count | `c_l` | Layer-level outlier proportion/count statistic. |
| Outlier severity | `v_l` | Relative threshold-exceedance severity statistic. |
| OSLA output | `s = OSLA(c, v; s)` | Compact operator shorthand for the manuscript mapping from `c`, `v`, and global target `s` to layer sparsities `s=[s_1,...,s_L]`. |
| Selected layer budget | `s_l` | Sparsity assigned to layer `l`; it constrains within-layer pruning. |
| Weight and gradient | `W_{l,j}`, `G^(l)` | Weight matrix and offline nonnegative aggregated gradient for the selected layer/module. |
| GWS score | `M_{ij}^{(l)}=|W_{ij}^{(l)}|sqrt(G_{ij}^{(l)})` | Connection retention saliency in Eq. (GWS). |
| Row deletion count | `q_{l,j}=floor(s_l d_{l,j}^{in})` | Number of lowest-saliency elements removed from each output row. |
| Binary mask | `B_{l,j}` | Zero denotes a removed connection and one denotes a retained connection. |
| Sparse weight | `W_tilde_{l,j}=W_{l,j} odot B_{l,j}` | Pruned weight with retained values unchanged. |

## Author confirmation

- The figure intentionally uses the compact operator shorthand `s = OSLA(c, v; s)` instead of reproducing the full density mapping, depth constraint, and budget-projection equations. The first `s` is typeset as a vector and the final `s` is the scalar global target.
- The selected-layer gradient matrix is labeled `G^(l)` for visual economy; its element-level definition remains `G_{ij}^{(l)}` in the GWS formula.
- The paper source and Word manuscript have not been modified. The recommended Overleaf asset is `GS_OWL_framework_v2.pdf`.
