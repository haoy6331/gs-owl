"""Low-fidelity three-column/two-row layout study for the GS-OWL framework.

This file intentionally uses neutral fills, placeholders, and orthogonal arrows.
It is a layout review artifact, not the final publication artwork.
"""

from __future__ import annotations

from pathlib import Path

import draw_gs_owl_framework_submission as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
W, H = 1700, 820
WIDTH_MM, HEIGHT_MM = 170, 82

# The reusable canvas classes read geometry from their defining module.
base.W = W
base.H = H
base.WIDTH_MM = WIDTH_MM
base.HEIGHT_MM = HEIGHT_MM


INK = "#3F444A"
MID = "#8C9299"
LIGHT = "#D7DADF"
PALE = "#F4F5F6"
OSLA_FILL = "#FAF5F1"
GWS_FILL = "#F2F6F9"
OUT_FILL = "#F5F5F4"
WHITE = "#FFFFFF"
ACCENT = "#9A6A4C"


def arrow_h(c, x1, y, x2, color=INK, sw=2.2):
    base.arrow_h(c, x1, y, x2, color, sw=sw)


def arrow_v(c, x, y1, y2, color=INK, sw=2.2):
    base.arrow_v(c, x, y1, y2, color, sw=sw)


def region(c, x, y, w, h, fill, title):
    c.rect(x, y, w, h, fill=fill, stroke=LIGHT, sw=1.4)
    if title:
        c.text(x + 18, y + 30, title, size=24, color=INK, family="cn", bold=True)


def token_rows(c, x, y, widths):
    for rr, row in enumerate(widths):
        xx = x
        for cc, width in enumerate(row):
            fill = LIGHT if (rr + cc) % 3 else MID
            c.rect(xx, y + rr * 24, width, 15, fill=fill, stroke=WHITE, sw=1)
            xx += width + 5


def grid(c, x, y, rows, cols, cw, ch, *, dark_cells=None, border=INK):
    dark_cells = dark_cells or set()
    for rr in range(rows):
        for cc in range(cols):
            fill = MID if (rr, cc) in dark_cells else LIGHT
            c.rect(x + cc * cw, y + rr * ch, cw, ch,
                   fill=fill, stroke=WHITE, sw=0.8)
    c.rect(x, y, cols * cw, rows * ch, fill="none", stroke=border, sw=1.8)


def layer_stack(c, x, y, *, n=6, w=176, h=46, gap=12, highlight=2):
    for idx in range(n):
        yy = y + idx * (h + gap)
        stroke = ACCENT if idx == highlight else INK
        sw = 3.0 if idx == highlight else 1.6
        c.rect(x, yy, w, h, fill=WHITE, stroke=stroke, sw=sw)
        grid(c, x + 8, yy + 7, 3, 12, (w - 16) / 12, (h - 14) / 3,
             dark_cells={(idx % 3, (idx * 3 + 2) % 12)}, border=LIGHT)
    return y + n * h + (n - 1) * gap


def activation_matrix(c, x, y, variant):
    highlights = [
        {(1, 1)},
        {(0, 1), (2, 4), (3, 2)},
        {(0, 4), (1, 2), (2, 0), (3, 3), (4, 1)},
    ][variant]
    grid(c, x, y, 5, 5, 18, 18, dark_cells=highlights, border=MID)


def bar_distribution(c, x, y, heights, width=16, gap=9, base_y=None):
    base_y = base_y or y + max(heights)
    for idx, height in enumerate(heights):
        fill = MID if idx != len(heights) // 2 else ACCENT
        c.rect(x + idx * (width + gap), base_y - height, width, height,
               fill=fill, stroke=INK, sw=0.7)
    c.line(x - 4, base_y, x + len(heights) * (width + gap) - gap + 4,
           base_y, stroke=INK, sw=1.2)


def intensity_strip(c, x, y, n=7, cell=18):
    shades = ["#ECEDEE", "#DADDE0", "#C8CCD0", "#B3B8BD",
              "#9EA4AA", "#858C93", "#676E75"]
    for idx in range(n):
        c.rect(x + idx * cell, y, cell, 24, fill=shades[idx], stroke=WHITE, sw=1)
    c.rect(x, y, n * cell, 24, fill="none", stroke=INK, sw=1.3)


def row_bars(c, x, y, rows=4, cols=8, cell_w=17, cell_h=14, sorted_rows=False):
    shades = ["#6D737A", "#888E95", "#A3A8AE", "#BEC2C7",
              "#D4D7DA", "#E4E6E8", "#F0F1F2", WHITE]
    for rr in range(rows):
        order = list(range(cols))
        if not sorted_rows:
            shift = (rr * 3 + 1) % cols
            order = order[shift:] + order[:shift]
        for cc, shade_idx in enumerate(order):
            c.rect(x + cc * cell_w, y + rr * (cell_h + 7), cell_w, cell_h,
                   fill=shades[shade_idx], stroke=WHITE, sw=0.8)
        c.rect(x, y + rr * (cell_h + 7), cols * cell_w, cell_h,
               fill="none", stroke=INK, sw=1.2)


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


def sparse_model(c, x, y, *, n=4, w=310, h=48, gap=10):
    densities = [0.68, 0.48, 0.61, 0.42]
    for idx in range(n):
        yy = y + idx * (h + gap)
        c.rect(x, yy, w, h, fill=WHITE, stroke=INK, sw=1.8)
        rows, cols = 3, 18
        cw, ch = (w - 16) / cols, (h - 12) / rows
        for rr in range(rows):
            for cc in range(cols):
                score = ((rr * 11 + cc * 7 + idx * 5) % 31) / 30
                xx = x + 8 + cc * cw
                cy = yy + 6 + rr * ch
                if score < densities[idx]:
                    c.rect(xx, cy, cw, ch, fill=INK, stroke=WHITE, sw=0.5)
                else:
                    c.rect(xx, cy, cw, ch, fill=WHITE, stroke=LIGHT, sw=0.5)
                    c.line(xx + 1, cy + ch - 1, xx + cw - 1, cy + 1,
                           stroke=MID, sw=0.7)


def draw_wireframe(c):
    outer_y, outer_h = 24, 772
    left_x, left_w = 24, 290
    mid_x, mid_w = 330, 820
    right_x, right_w = 1166, 510
    top_y, top_h = 24, 370
    bottom_y, bottom_h = 410, 386

    region(c, left_x, outer_y, left_w, outer_h, PALE, "")
    region(c, mid_x, top_y, mid_w, top_h, OSLA_FILL, "层间预算（OSLA）")
    region(c, mid_x, bottom_y, mid_w, bottom_h, GWS_FILL, "层内评分（GWS）")
    region(c, right_x, outer_y, right_w, outer_h, OUT_FILL, "协同剪枝与输出")

    # Left: calibration tokens and a full-height dense model.
    c.text(52, 88, "校准集", size=21, color=INK, family="cn")
    token_rows(c, 52, 108, [[54, 72, 38], [42, 66, 58], [64, 48, 50]])
    arrow_v(c, 145, 186, 218, MID, sw=2.0)
    c.text(52, 240, "稠密LLM", size=21, color=INK, family="cn")
    stack_bottom = layer_stack(c, 74, 266, n=6, w=190, h=48, gap=14, highlight=2)
    c.text(30, 422, "第", size=16, color=ACCENT, family="cn")
    c.formula(47, 422, [base.Seg("l", 17)], color=ACCENT)
    c.text(57, 422, "层", size=16, color=ACCENT, family="cn")

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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / "GS_OWL_framework_dense_layout_wireframe.svg"
    preview = OUT / "GS_OWL_framework_dense_layout_wireframe_preview_17cm.png"
    base.render_svg(svg, draw_wireframe)
    base.render_png(preview, draw_wireframe,
                    round(WIDTH_MM / 25.4 * 300), dpi=300)
    print(svg)
    print(preview)


if __name__ == "__main__":
    main()
