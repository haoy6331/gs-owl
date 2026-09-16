"""Draw the balanced-density GS-OWL framework figure.

The approved three-column layout and right-side output chain are retained.  The
left input column and the OSLA/GWS panels use larger graphical elements and
tighter internal spacing for better visual balance at the journal's 17 cm width.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import draw_gs_owl_framework_submission as base
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
W, H = 1700, 880
WIDTH_MM, HEIGHT_MM = 170, 88

# The reusable canvas classes read geometry from their defining module.
base.W = W
base.H = H
base.WIDTH_MM = WIDTH_MM
base.HEIGHT_MM = HEIGHT_MM


INK = "#30343B"
MODEL_DARK = "#44484E"
TEXT = "#202124"
MID = "#A8ADB2"
LIGHT = "#DDE1E5"
PALE = "#F5F6F7"
OSLA = "#D27A3F"
OSLA_DARK = "#A85A2B"
GWS = "#4C78A8"
GWS_DARK = "#355E88"
OSLA_FILL = "#FCF7F3"
GWS_FILL = "#F5F8FB"
OUT_FILL = "#F7F7F6"
WHITE = "#FFFFFF"
ACCENT = OSLA


def arrow_h(c, x1, y, x2, color=INK, sw=2.2):
    base.arrow_h(c, x1, y, x2, color, sw=sw)


def arrow_v(c, x, y1, y2, color=INK, sw=2.2):
    base.arrow_v(c, x, y1, y2, color, sw=sw)


def local_arrow_h(c, x1, y, x2, color=INK, sw=1.7):
    """Compact arrow for the local elementwise-product relationship."""
    c.line(x1, y, x2 - 8, y, stroke=color, sw=sw)
    c.polygon([(x2 - 8, y - 5), (x2, y), (x2 - 8, y + 5)], fill=color)


def region(c, x, y, w, h, fill, title):
    c.rect(x, y, w, h, fill=fill, stroke="#C7CCD1", sw=1.25)
    if title:
        c.text(x + 18, y + 30, title, size=23, color=TEXT, family="cn", bold=True)


def token_rows(c, x, y, widths, *, row_step=27, height=18, gap=6):
    fills = [base.blend(GWS, 0.67), base.blend(MID, 0.50),
             base.blend(OSLA, 0.69)]
    for rr, row in enumerate(widths):
        xx = x
        for cc, width in enumerate(row):
            fill = fills[(rr + cc) % len(fills)]
            c.rect(xx, y + rr * row_step, width, height,
                   fill=fill, stroke=WHITE, sw=1)
            xx += width + gap


def grid(c, x, y, rows, cols, cw, ch, *, dark_cells=None, border=INK):
    dark_cells = dark_cells or set()
    for rr in range(rows):
        for cc in range(cols):
            fill = MID if (rr, cc) in dark_cells else LIGHT
            c.rect(x + cc * cw, y + rr * ch, cw, ch,
                   fill=fill, stroke=WHITE, sw=0.8)
    c.rect(x, y, cols * cw, rows * ch, fill="none", stroke=border, sw=1.8)


def layer_stack(c, x, y, *, n=6, w=176, h=46, gap=12, highlight=2,
                extra_before_highlight=0):
    for idx in range(n):
        yy = (y + idx * (h + gap)
              + (extra_before_highlight if idx >= highlight else 0))
        stroke = ACCENT if idx == highlight else INK
        sw = 3.0 if idx == highlight else 1.6
        c.rect(x, yy, w, h, fill=WHITE, stroke=stroke, sw=sw)
        grid(c, x + 8, yy + 7, 3, 12, (w - 16) / 12, (h - 14) / 3,
             dark_cells={(idx % 3, (idx * 3 + 2) % 12)}, border=LIGHT)
    return y + n * h + (n - 1) * gap + extra_before_highlight


ACTIVATION_OUTLIERS = [
    {(1, 1): 0.48, (3, 4): 0.76},
    {(0, 1): 0.32, (1, 4): 0.70, (2, 3): 0.88,
     (3, 1): 0.54, (4, 4): 0.96},
    {(0, 3): 0.84, (2, 0): 0.44, (3, 3): 0.65},
]


def activation_matrix(c, x, y, variant, cell=18):
    outliers = ACTIVATION_OUTLIERS[variant]
    for rr in range(5):
        for cc in range(5):
            severity = outliers.get((rr, cc))
            fill = (base.blend(OSLA, 0.78 - 0.60 * severity)
                    if severity is not None else base.blend(MID, 0.70))
            c.rect(x + cc * cell, y + rr * cell, cell, cell,
                   fill=fill, stroke=WHITE, sw=0.8)
    c.rect(x, y, 5 * cell, 5 * cell, fill="none", stroke=MID, sw=1.55)


def value_matrix(c, x, y, values, cell, color, border):
    for rr, row in enumerate(values):
        for cc, value in enumerate(row):
            fill = base.blend(color, 0.88 - 0.70 * value)
            c.rect(x + cc * cell, y + rr * cell, cell, cell,
                   fill=fill, stroke=WHITE, sw=0.8)
    c.rect(x, y, len(values[0]) * cell, len(values) * cell,
           fill="none", stroke=border, sw=1.6)


def bar_distribution(c, x, y, heights, width=16, gap=9, base_y=None):
    base_y = base_y or y + max(heights)
    for idx, height in enumerate(heights):
        fill = (OSLA if idx == len(heights) // 2
                else base.blend(OSLA, 0.48 + 0.06 * (idx % 2)))
        c.rect(x + idx * (width + gap), base_y - height, width, height,
               fill=fill, stroke=OSLA_DARK, sw=0.65)
    c.line(x - 4, base_y, x + len(heights) * (width + gap) - gap + 4,
           base_y, stroke=INK, sw=1.2)


def intensity_strip(c, x, y, n=7, cell=18, height=24):
    for idx in range(n):
        c.rect(x + idx * cell, y, cell, height,
               fill=base.blend(OSLA, 0.88 - idx * 0.11), stroke=WHITE, sw=1)
    c.rect(x, y, n * cell, height, fill="none", stroke=OSLA_DARK, sw=1.3)


def row_bars(c, x, y, rows=4, cols=8, cell_w=17, cell_h=14, sorted_rows=False):
    values = base.SORTED_DESC if sorted_rows else base.M_VALUES
    for rr in range(rows):
        row = values[rr]
        for cc in range(min(cols, len(row))):
            value = row[cc]
            c.rect(x + cc * cell_w, y + rr * (cell_h + 7), cell_w, cell_h,
                   fill=base.blend(GWS, 0.88 - 0.68 * value),
                   stroke=WHITE, sw=0.8)
        c.rect(x, y + rr * (cell_h + 7), len(row) * cell_w, cell_h,
               fill="none", stroke=GWS_DARK, sw=1.15)


def selected_rows(c, x, y, rows=4, cell_w=21, cell_h=14):
    for rr, row in enumerate(base.SORTED_DESC[:rows]):
        for cc, value in enumerate(row):
            xx = x + cc * cell_w
            yy = y + rr * (cell_h + 7)
            if cc >= base.KEEP:
                c.rect(xx, yy, cell_w, cell_h, fill=WHITE, stroke=MID, sw=0.75)
                c.line(xx + 2, yy + cell_h - 2, xx + cell_w - 2, yy + 2,
                       stroke=MID, sw=0.8)
            else:
                c.rect(xx, yy, cell_w, cell_h,
                       fill=base.blend(GWS, 0.88 - 0.68 * value),
                       stroke=WHITE, sw=0.75)
        c.rect(x, y + rr * (cell_h + 7), len(row) * cell_w, cell_h,
               fill="none", stroke=INK, sw=1.15)
    threshold_x = x + base.KEEP * cell_w
    c.line(threshold_x, y - 7, threshold_x,
           y + rows * (cell_h + 7) - 7, stroke=OSLA, sw=2.6)


def mask(c, x, y, rows=5, cols=7, cell=13, density=0.58):
    for rr in range(rows):
        for cc in range(cols):
            score = ((rr * 7 + cc * 5 + rows) % 19) / 18
            kept = score < density
            xx, yy = x + cc * cell, y + rr * cell
            c.rect(xx, yy, cell, cell,
                   fill=INK if kept else WHITE, stroke=WHITE, sw=0.7)
            if not kept:
                c.line(xx + 2, yy + cell - 2, xx + cell - 2, yy + 2,
                       stroke=MID, sw=0.8)
    c.rect(x, y, cols * cell, rows * cell, fill="none", stroke=INK, sw=1.6)


SPARSE_MODEL_PATTERNS = [
    [
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    ],
    [
        [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    ],
    [
        [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    ],
    [
        [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    ],
]


def sparse_model(c, x, y, *, n=4, w=340, h=36, gap=8):
    for idx in range(n):
        yy = y + idx * (h + gap)
        c.rect(x, yy, w, h, fill=WHITE, stroke=INK, sw=1.55)
        rows, cols = 3, 11
        cw, ch = (w - 16) / cols, (h - 10) / rows
        for rr in range(rows):
            for cc in range(cols):
                xx = x + 8 + cc * cw
                cy = yy + 5 + rr * ch
                if SPARSE_MODEL_PATTERNS[idx][rr][cc]:
                    c.rect(xx, cy, cw, ch, fill=MODEL_DARK, stroke=WHITE, sw=0.5)
                else:
                    c.rect(xx, cy, cw, ch, fill=WHITE, stroke=LIGHT, sw=0.5)
                    # Sparse hatching is deliberately intermittent to avoid moire.
                    if (rr * 7 + cc * 5 + idx * 3) % 3 == 0:
                        c.line(xx + 2, cy + ch - 2, xx + cw - 2, cy + 2,
                               stroke=base.blend(MID, 0.12), sw=0.65)


def layer_mask(density, offset=0, rows=5, cols=7):
    keep = max(1, min(cols - 1, round(cols * density)))
    result = []
    for rr in range(rows):
        scores = [((cc * 5 + rr * 7 + offset * 3) % 19, cc) for cc in range(cols)]
        kept = {cc for _, cc in sorted(scores)[:keep]}
        result.append([1 if cc in kept else 0 for cc in range(cols)])
    return result


def layer_matrix(c, x, y, *, cell=10, mode="mask", mask_data=None):
    weights = base.W_VALUES
    mask_data = mask_data or base.MASK_ORIGINAL
    rows, cols = len(weights), len(weights[0])
    for rr in range(rows):
        for cc in range(cols):
            xx, yy = x + cc * cell, y + rr * cell
            kept = bool(mask_data[rr][cc])
            if mode == "mask":
                fill = INK if kept else WHITE
            elif mode == "weight":
                fill = base.blend(INK, 0.20 + 0.66 * (1 - weights[rr][cc]))
            else:
                fill = (base.blend(INK, 0.20 + 0.66 * (1 - weights[rr][cc]))
                        if kept else WHITE)
            c.rect(xx, yy, cell, cell, fill=fill, stroke=WHITE, sw=0.7)
            if mode != "weight" and not kept:
                c.line(xx + 2, yy + cell - 2, xx + cell - 2, yy + 2,
                       stroke=MID, sw=0.8)
    c.rect(x, y, cols * cell, rows * cell, fill="none", stroke=INK, sw=1.6)


def draw_wireframe_legacy(c):
    outer_y, outer_h = 24, 832
    left_x, left_w = 24, 290
    mid_x, mid_w = 330, 820
    right_x, right_w = 1166, 510
    top_y, top_h = 24, 370
    bottom_y, bottom_h = 410, 446

    region(c, left_x, outer_y, left_w, outer_h, PALE, "")
    region(c, mid_x, top_y, mid_w, top_h, OSLA_FILL, "层间预算（OSLA）")
    region(c, mid_x, bottom_y, mid_w, bottom_h, GWS_FILL, "层内评分（GWS）")
    region(c, right_x, outer_y, right_w, outer_h, OUT_FILL, "协同剪枝与输出")

    # Left: calibration tokens and a full-height dense model.
    c.text(52, 55, "校准集", size=21, color=INK, family="cn", bold=True)
    token_rows(c, 52, 82, [[54, 72, 38], [42, 66, 58], [64, 48, 50]])
    arrow_v(c, 145, 186, 218, MID, sw=2.0)
    c.text(52, 240, "稠密LLM", size=21, color=INK, family="cn")
    stack_bottom = layer_stack(c, 74, 266, n=6, w=190, h=48, gap=14, highlight=2)
    c.text(30, 405, "第", size=16, color=ACCENT, family="cn")
    c.formula(47, 405, [base.Seg("l", 17)], color=ACCENT)
    c.text(57, 405, "层", size=16, color=ACCENT, family="cn")

    # Forward-activation and layer-local extraction connectors.
    c.polyline([(264, 318), (300, 318), (300, 174), (330, 174)],
               stroke=ACCENT, sw=2.4)
    arrow_h(c, 314, 174, 346, ACCENT, sw=2.4)
    c.polyline([(264, 410), (300, 410), (300, 574), (330, 574)],
               stroke=MID, sw=2.4)
    arrow_h(c, 314, 574, 346, MID, sw=2.4)
    c.line(264, 410, 292, 410, stroke=ACCENT, sw=3.0)

    # OSLA row: activation matrices, two graphical statistics, and OSLA node.
    for idx, x in enumerate([360, 468, 576]):
        activation_matrix(c, x, 102, idx)
        c.formula(x + 45, 218, [base.Seg("X", 23, bold=True),
                                base.Seg(["(1)", "(l)", "(L)"][idx], 15, 7)],
                  anchor="middle", color=INK)
    arrow_h(c, 674, 148, 704, MID, sw=2.0)
    bar_distribution(c, 710, 108, [20, 36, 52, 30, 64], width=14, gap=7, base_y=176)
    intensity_strip(c, 706, 218, n=7, cell=15)
    c.formula(820, 145, [base.Seg("c", 22), base.Seg("l", 14, -5)], color=ACCENT)
    c.formula(820, 232, [base.Seg("v", 22), base.Seg("l", 14, -5)], color=ACCENT)
    arrow_h(c, 842, 148, 884, ACCENT, sw=2.2)
    arrow_h(c, 842, 230, 884, ACCENT, sw=2.2)
    c.rect(884, 102, 220, 164, fill=WHITE, stroke=ACCENT, sw=2.0)
    c.text(994, 142, "OSLA", size=26, color=INK, anchor="middle", family="times")
    for yy in [172, 202, 232]:
        c.rect(920, yy, 148, 17, fill=LIGHT, stroke=MID, sw=0.8)
    arrow_h(c, 1104, 184, right_x, ACCENT, sw=2.4)

    # GWS row: large W/G, scoring node, M, and before/after row ordering.
    grid(c, 354, 494, 6, 7, 20, 20,
         dark_cells={(0, 1), (1, 5), (2, 3), (4, 0), (5, 6)}, border=INK)
    grid(c, 354, 650, 6, 7, 20, 20,
         dark_cells={(0, 5), (1, 1), (2, 6), (3, 2), (5, 4)}, border=MID)
    c.formula(424, 632, [base.Seg("W", 25, bold=True), base.Seg("(l)", 16, 8)],
              anchor="middle", color=INK)
    c.formula(424, 788, [base.Seg("G", 25, bold=True), base.Seg("(l)", 16, 8)],
              anchor="middle", color=INK)
    c.polyline([(494, 554), (520, 554), (520, 584), (546, 584)], stroke=MID, sw=2.0)
    c.polyline([(494, 710), (520, 710), (520, 616), (546, 616)], stroke=MID, sw=2.0)
    arrow_h(c, 530, 584, 558, MID, sw=2.0)
    arrow_h(c, 530, 616, 558, MID, sw=2.0)
    c.rect(558, 556, 86, 88, fill=WHITE, stroke=MID, sw=1.8)
    c.text(601, 608, "GWS", size=20, color=INK, anchor="middle", family="times")
    arrow_h(c, 644, 600, 674, MID, sw=2.2)
    grid(c, 674, 510, 7, 7, 22, 22,
         dark_cells={(0, 2), (0, 6), (1, 0), (2, 4), (3, 1), (4, 5), (5, 3), (6, 6)},
         border=MID)
    c.formula(751, 688, [base.Seg("M", 25, bold=True), base.Seg("(l)", 16, 8)],
              anchor="middle", color=INK)
    arrow_h(c, 828, 600, 852, MID, sw=2.2)
    row_bars(c, 858, 488, rows=4, cols=8, cell_w=15, cell_h=14, sorted_rows=False)
    arrow_v(c, 918, 584, 610, MID, sw=1.8)
    row_bars(c, 858, 620, rows=4, cols=8, cell_w=15, cell_h=14, sorted_rows=True)
    c.text(918, 476, "排序前", size=18, color=INK, anchor="middle", family="cn")
    c.text(918, 744, "逐行排序", size=18, color=INK, anchor="middle", family="cn")
    c.rect(1000, 512, 126, 124, fill=WHITE, stroke=LIGHT, sw=1.2)
    for rr in range(5):
        for cc in range(5):
            c.circle(1016 + cc * 23, 528 + rr * 22, 5,
                     fill=MID if (rr * 3 + cc) % 4 else WHITE, stroke=MID, sw=1)
    arrow_h(c, 984, 600, 1000, MID, sw=2.0)
    arrow_h(c, 1126, 600, right_x, MID, sw=2.4)

    # Right: a continuous vertical coordination and output chain.
    c.formula(1418, 64, [base.Seg("s", 24, bold=True)],
              anchor="middle", color=INK)
    bar_distribution(c, 1292, 84, [48, 70, 42, 82, 58, 96, 52, 76, 64],
                     width=18, gap=10, base_y=190)
    c.text(1298, 214, "1", size=16, color=INK, family="times", italic=True)
    c.text(1374, 214, "...", size=16, color=INK, family="times")
    c.text(1418, 214, "l", size=16, color=INK, family="times", italic=True)
    c.text(1470, 214, "...", size=16, color=INK, family="times")
    c.text(1546, 214, "L", size=16, color=INK, family="times", italic=True)
    arrow_v(c, 1418, 224, 252, ACCENT, sw=2.2)
    c.formula(1440, 246, [base.Seg("s", 20), base.Seg("l", 13, -5)], color=ACCENT)

    c.rect(1264, 254, 308, 136, fill=WHITE, stroke=INK, sw=1.8)
    c.text(1418, 282, "预算掩码", size=21, color=INK, anchor="middle", family="cn")
    row_bars(c, 1332, 300, rows=4, cols=8, cell_w=21, cell_h=14, sorted_rows=True)
    c.line(1437, 292, 1437, 378, stroke=ACCENT, sw=2.6)
    arrow_v(c, 1418, 390, 420, INK, sw=2.0)

    mask(c, 1288, 424, rows=5, cols=7, cell=14, density=0.58)
    arrow_h(c, 1390, 459, 1436, INK, sw=2.0)
    mask(c, 1448, 424, rows=5, cols=7, cell=14, density=0.68)
    c.formula(1337, 512, [base.Seg("B", 22, bold=True), base.Seg("(l)", 14, 7)],
              anchor="middle", color=INK)
    c.formula(1497, 512, [base.Seg("W̃", 22, bold=True), base.Seg("(l)", 14, 7)],
              anchor="middle", color=INK)
    arrow_v(c, 1418, 514, 536, INK, sw=2.0)

    for idx, (x, density, suffix) in enumerate([
        (1248, 0.72, "(1)"), (1372, 0.55, "(l)"), (1496, 0.42, "(L)"),
    ]):
        mask(c, x, 540, rows=4, cols=6, cell=11, density=density)
        c.formula(x + 33, 602, [base.Seg("B", 18, bold=True), base.Seg(suffix, 12, 6)],
                  anchor="middle", color=INK)
    c.polyline([(1281, 608), (1281, 620), (1418, 620)], stroke=MID, sw=1.7)
    c.polyline([(1405, 608), (1405, 620), (1418, 620)], stroke=MID, sw=1.7)
    c.polyline([(1529, 608), (1529, 620), (1418, 620)], stroke=MID, sw=1.7)
    arrow_v(c, 1418, 620, 642, INK, sw=2.0)

    c.text(1418, 664, "稀疏LLM（s）", size=21, color=INK,
           anchor="middle", family="cn")
    sparse_model(c, 1238, 678, n=4, w=360, h=23, gap=5)


def draw_refined(c):
    outer_y, outer_h = 24, 832
    left_x, left_w = 24, 290
    mid_x, mid_w = 330, 820
    right_x, right_w = 1166, 510
    top_y, top_h = 24, 370
    bottom_y, bottom_h = 410, 446

    region(c, left_x, outer_y, left_w, outer_h, PALE, "")
    region(c, mid_x, top_y, mid_w, top_h, OSLA_FILL, "层间预算（OSLA）")
    region(c, mid_x, bottom_y, mid_w, bottom_h, GWS_FILL, "层内评分（GWS）")
    region(c, right_x, outer_y, right_w, outer_h, OUT_FILL, "协同剪枝与输出")

    # Left column: larger calibration tokens and a denser full-height model stack.
    c.text(48, 58, "校准集", size=21, color=INK, family="cn")
    token_rows(c, 48, 82,
               [[62, 82, 44], [48, 76, 66], [72, 56, 58]],
               row_step=27, height=18, gap=6)
    arrow_v(c, 158, 165, 202, MID, sw=2.0)
    c.text(48, 228, "稠密LLM", size=21, color=INK, family="cn")
    layer_stack(c, 52, 250, n=6, w=234, h=63, gap=19, highlight=2,
                extra_before_highlight=12)
    c.text(62, 417, "第", size=15, color=ACCENT, family="cn")
    c.formula(77, 417, [base.Seg("l", 16)], color=ACCENT)
    c.text(86, 417, "层", size=15, color=ACCENT, family="cn")

    branch_x, branch_y = 286, 457
    c.circle(branch_x, branch_y, 5.5, fill=WHITE, stroke=ACCENT, sw=2.0)
    c.circle(branch_x, branch_y, 2.1, fill=ACCENT)

    # OSLA branch leaves the shared node, clears the region divider, and enters X^(l).
    c.polyline([(branch_x, branch_y), (314, branch_y), (314, 382),
                (454, 382), (454, 190)], stroke=ACCENT, sw=2.5)
    arrow_h(c, 454, 190, 467, ACCENT, sw=2.5)

    # GWS branch: the same node feeds both W^(l) and G^(l).
    c.polyline([(branch_x, branch_y), (320, branch_y), (320, 734)],
               stroke=GWS, sw=2.4)
    arrow_h(c, 320, 558, 346, GWS, sw=2.4)
    arrow_h(c, 320, 734, 346, GWS, sw=2.4)

    # OSLA group: enlarged matrices/statistics and tighter, balanced spacing.
    for idx, x in enumerate([340, 467, 594]):
        activation_matrix(c, x, 138, idx, cell=21)
        c.formula(x + 52.5, 274, [base.Seg("X", 23, bold=True),
                                base.Seg(["(1)", "(l)", "(L)"][idx], 15, 7)],
                  anchor="middle", color=INK)
    c.text(456, 124, "...", size=17, color=MID, anchor="middle", family="times")
    c.text(583, 124, "...", size=17, color=MID, anchor="middle", family="times")
    arrow_h(c, 699, 190, 712, OSLA, sw=2.0)
    bar_distribution(c, 716, 150, [22, 42, 60, 34, 72],
                     width=15, gap=7, base_y=232)
    intensity_strip(c, 710, 267, n=7, cell=16, height=28)
    c.formula(840, 194, [base.Seg("p", 22), base.Seg("l", 14, -5)],
              anchor="middle", color=ACCENT)
    c.formula(840, 282, [base.Seg("v", 22), base.Seg("l", 14, -5)],
              anchor="middle", color=ACCENT)
    arrow_h(c, 860, 194, 878, ACCENT, sw=2.2)
    arrow_h(c, 860, 282, 878, ACCENT, sw=2.2)
    c.rect(878, 126, 236, 190, fill=WHITE, stroke=OSLA, sw=2.0)
    c.text(996, 162, "OSLA", size=25, color=OSLA_DARK,
           anchor="middle", family="times", bold=True)

    # Compact, paper-faithful mechanism: count/severity fusion produces
    # \widetilde d_l, then the depth constraint precedes budget projection.
    c.rect(892, 180, 92, 30, fill=base.blend(OSLA, 0.88),
           stroke=base.blend(OSLA, 0.42), sw=0.85)
    c.rect(892, 254, 92, 30, fill=base.blend(OSLA, 0.84),
           stroke=base.blend(OSLA, 0.42), sw=0.85)
    c.text(938, 201, "比例锚定", size=15, color=TEXT,
           anchor="middle", family="cn")
    c.text(938, 275, "严重度修正", size=15, color=TEXT,
           anchor="middle", family="cn")
    c.polyline([(878, 194), (886, 194), (886, 195), (892, 195)],
               stroke=OSLA, sw=1.45)
    c.polyline([(878, 282), (886, 282), (886, 269), (892, 269)],
               stroke=OSLA, sw=1.45)
    c.polyline([(984, 195), (990, 195), (990, 228)], stroke=OSLA, sw=1.55)
    c.polyline([(984, 269), (990, 269), (990, 242)], stroke=OSLA, sw=1.55)
    c.circle(999, 235, 9, fill=WHITE, stroke=OSLA_DARK, sw=1.45)
    c.circle(999, 235, 2.3, fill=OSLA_DARK)
    arrow_h(c, 1008, 235, 1025, OSLA, sw=1.6)
    c.formula(1044, 204, [base.Seg("d̃", 20), base.Seg("l", 12, -5)],
              anchor="middle", color=TEXT)
    arrow_h(c, 1061, 235, 1073, OSLA, sw=1.6)
    c.line(1075, 211, 1075, 259, stroke=OSLA_DARK, sw=3.0)
    c.text(1075, 290, "深度约束", size=15, color=TEXT,
           anchor="middle", family="cn")
    arrow_h(c, 1078, 235, 1114, OSLA, sw=1.6)

    # OSLA output reaches the layer-budget distribution horizontally.
    arrow_h(c, 1114, 220, 1284, ACCENT, sw=2.4)
    c.text(1204, 201, "预算投影", size=14, color=OSLA_DARK,
           anchor="middle", family="cn")

    # GWS area: larger matrices and denser graphical progression.
    value_matrix(c, 346, 492, base.W_VALUES, 22, INK, INK)
    value_matrix(c, 346, 668, base.G_VALUES, 22, GWS, GWS_DARK)
    c.formula(423, 650, [base.Seg("W", 25, bold=True), base.Seg("(l)", 16, 8)],
              anchor="middle", color=INK)
    c.formula(423, 826, [base.Seg("G", 25, bold=True), base.Seg("(l)", 16, 8)],
              anchor="middle", color=GWS_DARK)
    c.polyline([(500, 558), (522, 558), (522, 620), (538, 620)], stroke=GWS, sw=2.0)
    c.polyline([(500, 734), (522, 734), (522, 662), (538, 662)], stroke=GWS, sw=2.0)
    arrow_h(c, 530, 620, 550, GWS, sw=2.0)
    arrow_h(c, 530, 662, 550, GWS, sw=2.0)
    c.rect(550, 594, 96, 94, fill=WHITE, stroke=GWS_DARK, sw=1.8)
    c.text(598, 648, "GWS", size=20, color=GWS_DARK,
           anchor="middle", family="times", bold=True)
    arrow_h(c, 646, 641, 670, GWS, sw=2.2)
    value_matrix(c, 670, 542, base.M_VALUES, 23, GWS, GWS_DARK)
    c.formula(750.5, 730, [base.Seg("M", 25, bold=True), base.Seg("(l)", 16, 8)],
              anchor="middle", color=GWS_DARK)
    c.formula(750.5, 784, [
        base.Seg("M", 22), base.Seg("ij", 14, -5), base.Seg("(l)", 14, 7),
        base.Seg(" = |W", 22), base.Seg("ij", 14, -5), base.Seg("(l)", 14, 7),
        base.Seg("|√G", 22), base.Seg("ij", 14, -5), base.Seg("(l)", 14, 7),
    ], anchor="middle", color=TEXT)
    arrow_h(c, 831, 623, 852, GWS, sw=2.2)
    row_bars(c, 856, 496, rows=4, cols=8, cell_w=17, cell_h=16, sorted_rows=False)
    arrow_v(c, 924, 584, 622, GWS, sw=1.8)
    row_bars(c, 856, 632, rows=4, cols=8, cell_w=17, cell_h=16, sorted_rows=True)
    c.text(924, 482, "排序前", size=17, color=TEXT, anchor="middle", family="cn")
    c.text(924, 750, "逐行排序", size=17, color=TEXT, anchor="middle", family="cn")
    c.text(1073, 482, "位置恢复", size=17, color=TEXT,
           anchor="middle", family="cn")
    c.rect(1012, 496, 122, 122, fill=WHITE, stroke=LIGHT, sw=1.2)
    for rr in range(5):
        for cc in range(5):
            value = base.M_VALUES[rr][(cc + rr) % 7]
            kept = value >= 0.46
            c.circle(1027 + cc * 23, 511 + rr * 23, 5.4,
                     fill=(base.blend(GWS, 0.34 + 0.40 * (1 - value))
                           if kept else WHITE),
                     stroke=GWS_DARK if kept else MID, sw=0.9)
    c.polyline([(992, 675), (1002, 675), (1002, 557), (1012, 557)],
               stroke=GWS, sw=1.8)
    arrow_h(c, 1002, 557, 1012, GWS, sw=1.8)

    # GWS sorting result enters the budget gate from the left without crossing modules.
    c.polyline([(1134, 557), (1148, 557), (1148, 542),
                (1204, 542), (1204, 398), (1248, 398)],
               stroke=GWS, sw=2.15)
    arrow_h(c, 1248, 398, 1264, GWS, sw=2.15)

    # Right column: aligned budget distribution and vertical coordination chain.
    c.formula(1418, 92, [
        base.Seg("s", 22, bold=True), base.Seg(" = [s", 20),
        base.Seg("1", 13, -5), base.Seg(", ..., s", 20), base.Seg("l", 13, -5),
        base.Seg(", ..., s", 20), base.Seg("L", 13, -5), base.Seg("]", 20),
    ], anchor="middle", color=INK)
    bar_x, bar_base, selected_bar = 1292, 230, 4
    heights = [48, 70, 42, 82, 58, 96, 52, 76, 64]
    for idx, height in enumerate(heights):
        c.rect(bar_x + idx * 28, bar_base - height, 18, height,
               fill=ACCENT if idx == selected_bar else base.blend(OSLA, 0.50),
               stroke=OSLA_DARK, sw=0.7)
    c.line(bar_x - 4, bar_base, bar_x + 9 * 28 - 10,
           bar_base, stroke=INK, sw=1.2)
    for idx, label in {0: "1", 2: "...", 4: "l", 6: "...", 8: "L"}.items():
        c.text(bar_x + idx * 28 + 9, 254, label, size=16, color=INK,
               anchor="middle", family="times", italic=label != "...")
    selected_x = bar_x + selected_bar * 28 + 9
    arrow_v(c, selected_x, 270, 304, ACCENT, sw=2.2)
    c.formula(selected_x + 22, 288, [base.Seg("s", 20), base.Seg("l", 13, -5)],
              color=ACCENT)

    gate_x, gate_y, gate_w, gate_h = 1264, 304, 308, 122
    c.rect(gate_x, gate_y, gate_w, gate_h, fill=WHITE, stroke=INK, sw=1.55)
    c.text(1418, 332, "预算掩码", size=21, color=TEXT,
           anchor="middle", family="cn", bold=True)
    selected_rows(c, 1332, 348, rows=4, cell_w=21, cell_h=14)

    # Gate outputs only B^(l); W^(l) is an independent, vertically aligned operand.
    w_x, b_x, out_x, mcell = 1402, 1402, 1580, 7
    w_y, b_y, out_y = 432, 500, 467
    matrix_w, matrix_h = 7 * mcell, 5 * mcell
    w_center = w_x + matrix_w / 2
    b_center = b_x + matrix_w / 2
    out_center = out_x + matrix_w / 2
    product_x, product_y = 1532, 484

    # The mask output uses a separate port and enters B^(l) from its left edge.
    # This prevents the black output path from visually extending the orange threshold.
    mask_port_x = 1388
    c.line(mask_port_x, gate_y + gate_h, mask_port_x, b_y + matrix_h / 2,
           stroke=INK, sw=1.85)
    local_arrow_h(c, mask_port_x, b_y + matrix_h / 2, b_x, INK, sw=1.85)

    layer_matrix(c, w_x, w_y, cell=mcell, mode="weight",
                 mask_data=base.MASK_ORIGINAL)
    layer_matrix(c, b_x, b_y, cell=mcell, mode="mask",
                 mask_data=base.MASK_ORIGINAL)
    layer_matrix(c, out_x, out_y, cell=mcell, mode="sparse",
                 mask_data=base.MASK_ORIGINAL)
    c.formula(w_center, 494,
              [base.Seg("W", 20, bold=True), base.Seg("(l)", 13, 6)],
              anchor="middle", color=INK)
    c.formula(b_center, 562,
              [base.Seg("B", 20, bold=True), base.Seg("(l)", 13, 6)],
              anchor="middle", color=INK)
    c.formula(out_center, 529,
              [base.Seg("W̃", 20, bold=True), base.Seg("(l)", 13, 6)],
              anchor="middle", color=INK)

    # Two independent short diagonal arrows converge on the mathematical odot.
    c.line(w_x + matrix_w, w_y + matrix_h / 2, product_x - 13, product_y - 7,
           stroke=INK, sw=1.65)
    c.polygon([(product_x - 20, product_y - 13),
               (product_x - 13, product_y - 7),
               (product_x - 22, product_y - 4)], fill=INK)
    c.line(b_x + matrix_w, b_y + matrix_h / 2, product_x - 13, product_y + 7,
           stroke=INK, sw=1.65)
    c.polygon([(product_x - 22, product_y + 4),
               (product_x - 13, product_y + 7),
               (product_x - 20, product_y + 13)], fill=INK)
    c.text(product_x, product_y + 9, "⊙", size=26, color=INK,
           anchor="middle", family="math")
    local_arrow_h(c, product_x + 16, product_y, out_x, INK, sw=1.75)

    c.formula(1518, 590, [
        base.Seg("W̃", 21, bold=True), base.Seg("(l)", 13, 7),
        base.Seg(" = W", 21, bold=True), base.Seg("(l)", 13, 7),
        base.Seg("⊙", 21, italic=False, family="math"),
        base.Seg("B", 21, bold=True), base.Seg("(l)", 13, 7),
    ], anchor="middle", color=INK)

    # Different layers are shown as sparse weights rather than binary masks.
    thumb_specs = [
        (1248, 0.72, 1, "(1)"),
        (1372, 0.56, 3, "(l)"),
        (1496, 0.42, 5, "(L)"),
    ]
    for x, density, offset, suffix in thumb_specs:
        thumb_mask = layer_mask(density, offset=offset, rows=5, cols=7)
        layer_matrix(c, x, 602, cell=8, mode="sparse", mask_data=thumb_mask)
        c.formula(x + 28, 664, [base.Seg("W̃", 17, bold=True), base.Seg(suffix, 11, 5)],
                  anchor="middle", color=INK)
    # Side exits keep all three aggregation lines away from the labels.
    c.polyline([(1248, 622), (1232, 622), (1232, 688)], stroke=MID, sw=1.55)
    c.polyline([(1372, 622), (1356, 622), (1356, 688)], stroke=MID, sw=1.55)
    c.polyline([(1552, 622), (1568, 622), (1568, 688)], stroke=MID, sw=1.55)
    c.line(1232, 688, 1568, 688, stroke=MID, sw=1.55)
    c.line(1232, 688, 1232, 750, stroke=INK, sw=1.8)
    local_arrow_h(c, 1232, 750, 1248, INK, sw=1.8)

    c.text(1418, 718, "稀疏LLM（s）", size=21, color=TEXT,
           anchor="middle", family="cn", bold=True)
    sparse_model(c, 1248, 738, n=4, w=340, h=24, gap=4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / "GS_OWL_framework_balanced.svg"
    pdf = OUT / "GS_OWL_framework_balanced.pdf"
    png = OUT / "GS_OWL_framework_balanced.png"
    png_600 = OUT / "GS_OWL_framework_balanced_600dpi.png"
    gray = OUT / "GS_OWL_framework_balanced_grayscale.png"
    preview = OUT / "GS_OWL_framework_balanced_preview_17cm.png"
    detail = OUT / "GS_OWL_framework_balanced_right_detail.png"
    base.render_svg(svg, draw_refined)
    pdf_canvas = base.PDFCanvas(pdf)
    draw_refined(pdf_canvas)
    pdf_canvas.save()
    base.render_png(png, draw_refined,
                    round(WIDTH_MM / 25.4 * 600), dpi=600)
    shutil.copy2(png, png_600)
    base.render_png(preview, draw_refined,
                    round(WIDTH_MM / 25.4 * 300), dpi=300)
    with Image.open(preview) as image:
        image.convert("L").save(gray, dpi=(300, 300), optimize=True)
        scale = image.width / W
        right_crop = (
            round(1166 * scale), round(24 * scale),
            round(1676 * scale), round(856 * scale),
        )
        image.crop(right_crop).resize(
            (round((right_crop[2] - right_crop[0]) * 1.8),
             round((right_crop[3] - right_crop[1]) * 1.8)),
            Image.Resampling.LANCZOS,
        ).save(detail, dpi=(300, 300), optimize=True)
    for path in [svg, pdf, png, png_600, gray, preview, detail]:
        print(path)


if __name__ == "__main__":
    main()
