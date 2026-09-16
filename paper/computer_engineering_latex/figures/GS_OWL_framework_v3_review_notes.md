# GS-OWL Framework V3 - Symbol Check and Revision Log

## Symbol check

| Figure symbol | Manuscript definition | Role in the figure |
|---|---|---|
| `D_X`, `D_G` | Activation calibration set and gradient calibration set | Calibration inputs used for forward activation statistics and offline aggregated gradients. |
| `X^(1)`, `X^(l)`, `X^(L)` | Schematic activation matrices for representative layers; the manuscript defines channel activation energy as `X_{l,j,c}` | Their highlighted-cell count represents outlier proportion and highlight depth represents exceedance severity. |
| `c_l` | Component of the outlier-count/proportion vector `c=[c_1,...,c_L]` | Count-anchored layer evidence supplied to OSLA. |
| `v_l` | Component of the exceedance-severity vector `v=[v_1,...,v_L]` | Severity correction supplied to OSLA. |
| `s=[s_1,...,s_l,...,s_L]` | Layer sparsity vector produced after bounded mapping, depth constraints, and budget projection | Nonuniform layer budgets under scalar global target sparsity `s`. Boldface distinguishes the vector from the scalar target. |
| `s_l` | Sparsity assigned to layer `l` | Controls the per-row deletion quota in the selector. |
| `W^(l)` | Schematic selected-layer weight matrix; element notation is `W_{ij}^{(l)}` in the GWS formula | Weight magnitude input to GWS. |
| `G^(l)` | Nonnegative offline aggregated gradient matrix; element notation is `G_{ij}^{(l)}` | Gradient sensitivity input to GWS. |
| `M^(l)` | GWS connection-saliency matrix | Supplies the within-row connection order. |
| `M_{ij}^{(l)}=|W_{ij}^{(l)}|sqrt(G_{ij}^{(l)})` | Manuscript GWS definition | Compresses gradient dynamic range while retaining weight magnitude. |
| `q_{l,j}=floor(s_l d_{l,j}^{in})` | Manuscript row-wise deletion quota | Number of lowest-saliency entries removed from each output row of matrix `j` in layer `l`. |
| `B^(l)` | Schematic selected-layer binary mask; the formal manuscript matrix is `B_{l,j}` | Each row has the same deletion quota, but retained positions differ by row. |
| `W_tilde^(l)=W^(l) odot B^(l)` | Sparse-weight definition corresponding to `W_tilde_{l,j}=W_{l,j} odot B_{l,j}` | Retained weights are unchanged and deleted entries use exactly the mask's zero pattern. |

## Revision log

1. Replaced the structured left-column/right-column mask with a row-wise top-k mask mapped back to original connection coordinates.
2. Enforced four retained and three deleted entries in every displayed row; no column is entirely retained or deleted.
3. Made the sparse-weight zero pattern exactly identical to the binary mask.
4. Replaced the orange line through the blue matrix with an arrowed OSLA control path that enters the selector from above.
5. Made the GWS sorted rows enter the same selector horizontally, forming an explicit two-input coordination node.
6. Moved `q_{l,j}` inside the selector and visualized its threshold on descending saliency rows.
7. Increased activation-outlier count differences across `X^(1)`, `X^(l)`, and `X^(L)` and added multiple orange intensity levels.
8. Split the outlier legend into count/proportion and shade/severity encodings.
9. Replaced the ambiguous OSLA self-referential formula with the explicit layer sparsity vector and layer indices `1,...,l,...,L`.
10. Redesigned `W^(l)`, `G^(l)`, and `M^(l)` with visibly different distributions and replaced the copied sorting matrix with three descending representative rows.
11. Replaced word-token cards and the floating `f_W` label with compact `D_X,D_G` calibration input and explicit forward/gradient branches.
12. Rebuilt the sparse LLM with the same layer outlines as the dense LLM; only internal grid occupancy changes across layers.
13. Reduced title size, removed long decorative section rules, and kept only one framed mechanism node: budget-constrained row-wise selection.
14. Replaced unsupported Unicode element-wise-product glyphs with editable vector circle-dot symbols in SVG/PDF/PNG outputs.

## Validation

- Canvas: 170 mm x 74 mm, suitable for a 17 cm two-column figure.
- SVG: editable text and vector primitives only; no embedded raster images.
- PNG: 4016 x 1748 pixels with 600 dpi metadata.
- PDF: one 170 mm x 74 mm page with embedded SimSun and Times New Roman fonts.
- Color and grayscale previews were visually checked at final paper width.
- The manuscript figure has not been replaced automatically.
