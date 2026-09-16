# GS-OWL framework final revision notes

## Output artifacts

- `GS_OWL_framework_final.svg`: editable vector master, 170 mm x 64 mm.
- `GS_OWL_framework_final.pdf`: one-page vector PDF for Overleaf.
- `GS_OWL_framework_final_600dpi.png`: 4016 x 1512 px, 600 dpi.
- `GS_OWL_framework_final_grayscale.png`: grayscale print preview.
- `GS_OWL_framework_final_preview_17cm.png`: 17 cm, 300 dpi review preview.
- `../draw_gs_owl_framework_final.py`: editable drawing source.

## Final layout changes

- The non-uniform layer-budget chart was moved 80 px left, shortening the OSLA output arrow. The label `全局预算投影` is placed above that arrow.
- The highlighted layer index `l` is directly below its budget bar. The orange budget-control arrow starts below the axis labels and does not cross the bar chart or the character `l`.
- The label `第 l 层预算 s_l` is placed to the right of the vertical control arrow with independent spacing.
- The layer-budget vector is written as bold `s = [s_1, ..., s_l, ..., s_L]`; the scalar global sparsity remains plain `s`.
- The selector formula uses vector-drawn floor brackets and represents the number of lowest-saliency entries deleted from row `j`: `q_{l,j} = floor(s_l d_{l,j}^{in})`.
- The W and G inputs enter GWS through separate upper and lower arrows. Their vertical separation is 32 px, increased from 16 px in the earlier version.
- The activation legends were shortened to `数量：异常比例` and `深浅：异常严重度`. Their swatches remain separate and preserve the count/depth encoding.
- Redundant labels were removed: `前向统计`, `第 l 层局部放大`, `层内次序`, `逐行降序排序`, and `映射回原连接位置`. The sorting block now uses the single label `逐行排序`.
- The selector height was reduced from 224 px to 200 px after removing the redundant mapping sentence, while retaining the title, formula, selected/deleted rows, and sufficient inner margins.

## Enlarged output group

- B and sparse-W matrix cells were enlarged from 14 px to 18 px. Each matrix therefore increased from 98 x 70 px to 126 x 90 px, or about 1.29x in both dimensions.
- The output axis moved from y=400 to y=384; the enlarged matrices begin at y=339, 21 px above the previous placement.
- The output equation was enlarged and retained once below the matrices: `W-tilde^(l) = W^(l) odot B^(l)`.
- The final sparse LLM now uses four equal-size layers with a 168 x 48 px layer body and 10 px inter-layer spacing.
- The complete sparse-LLM stack is about 180 x 222 px. Relative to the immediately preceding final version, this is 1.15x wider and 1.22x taller; relative to the early v4 stack, it is about 1.96x wider and 2.41x taller.
- The final-model title moved upward by 78 px and the stack top moved upward by 72 px, using the previously empty upper-right area.
- All four layer widths remain identical. Only internal retained/deleted textures vary, avoiding the impression of structured layer-width pruning.

## Validation results

- SVG canvas: 170 mm x 64 mm, viewBox `0 0 1700 640`.
- SVG contains 49 editable text nodes and no embedded raster image nodes.
- PDF: one page, 170.000 mm x 64.000 mm.
- SimSun and Times New Roman font subsets used by visible text are embedded in the PDF. The unused ReportLab Helvetica resource does not occur in extracted text spans.
- The binary mask and sparse-weight matrix have the same 15 deleted positions.
- Color, grayscale, 600 dpi, 17 cm preview, and PDF-rendered previews were visually checked for overlap, clipping, font substitution, and texture legibility.
- The manuscript Figure 1 has not been replaced.
