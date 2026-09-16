# GS-OWL Framework V4 - Layout Revision Notes

## Canvas and module movement

| Item | V3 | V4 | Change |
|---|---:|---:|---:|
| Canvas | 170 mm x 74 mm; viewBox 1700 x 740 | 170 mm x 64 mm; viewBox 1700 x 640 | Height reduced by 10 mm / 100 px (13.5%). |
| GWS section title | y=340 | y=272 | Moved upward 68 px. |
| Budget selector | x=995, y=365, w=248, h=221 | x=976, y=288, w=288, h=224 | Moved upward 77 px and widened 16.1%. |
| Binary mask | y=420, 16 px cells | y=360, 14 px cells | Moved upward 60 px. |
| Sparse weight | y=420, 16 px cells | y=360, 14 px cells | Moved upward 60 px and aligned with the mask. |
| Sparse LLM | Below the sparse-weight formula, 6 layers | Right of the sparse weight, 5 layers | Converted to a compact horizontal output group. |
| Budget bars | Centered near x=951 | Centered at x=1120 | Aligned directly over the selector center. |
| Dense model | 6 layers, 96 x 18 layer outline | 5 layers, 84 x 16 layer outline | Input area compressed; output stack uses the same outline. |

## Spacing and attachment fixes

1. Split the activation-statistics legend into two independent entries with separate swatches and text.
2. Moved the severity legend left so its text no longer enters the OSLA frame in the PDF rendering.
3. Placed `c_l` and `v_l` above their own arrows with clear separation from the OSLA border.
4. Kept only the gray `layer l local magnification` annotation on the model-to-GWS branch.
5. Added `weight` above `W^(l)` and `aggregated gradient` above `G^(l)`.
6. Increased the vertical spacing below `W^(l)`, `G^(l)`, `M^(l)`, `B^(l)`, and `W_tilde^(l)` so labels no longer touch matrix borders.
7. Replaced the direct multiplication mark between `W^(l)` and `G^(l)` with a two-input GWS operator node.
8. Placed `within-layer order` above the sorted rows and `row-wise descending sort` below them.
9. Replaced unsupported floor-bracket glyphs with editable vector floor brackets around `s_l d_{l,j}^{in}`.
10. Shortened the selector title from `budget-constrained row-wise selection` to `budget-constrained selection` and increased internal padding.
11. Removed the `odot W^(l)` text from the mask-to-weight arrow.
12. Kept the complete sparse-weight formula once below the two aligned matrices.
13. Moved the final sparse LLM to the right of the sparse weight and reduced it from 6 to 5 schematic layers.
14. Shortened the final output label to `LLM satisfying the global target sparsity` and kept it away from the canvas edge.

## Validation

- Color, grayscale, and PDF renderings were checked at 17 cm paper width.
- SVG contains editable text and vector primitives only; no raster image is embedded.
- The 600 dpi PNG is 4016 x 1512 pixels.
- The PDF is one 170 mm x 64 mm page with embedded SimSun and Times New Roman fonts.
- Every displayed mask row retains four entries and deletes three entries; sparse-weight zero positions match the mask exactly.
- The manuscript's current Figure 1 has not been replaced.
