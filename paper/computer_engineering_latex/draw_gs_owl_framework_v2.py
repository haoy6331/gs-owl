"""Draw the second-generation GS-OWL paper framework figure.

This version uses a global-model view, an OSLA budget controller, a magnified
layer-local GWS view, budget-constrained row selection, and a sparse-LLM output.
The geometry is authored at 170 mm x 74 mm for a two-column journal figure.
"""

from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
W, H = 1700, 740
WIDTH_MM, HEIGHT_MM = 170, 74

SIMSUN = Path(r"C:\Windows\Fonts\simsun.ttc")
TIMES = Path(r"C:\Windows\Fonts\times.ttf")
TIMES_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")
TIMES_ITALIC = Path(r"C:\Windows\Fonts\timesi.ttf")
TIMES_BOLD_ITALIC = Path(r"C:\Windows\Fonts\timesbi.ttf")


@dataclass(frozen=True)
class Theme:
    osla: str
    osla_light: str
    gws: str
    gws_light: str
    dark: str
    text: str
    neutral: str
    pale: str
    white: str = "#FFFFFF"


COLOR = Theme(
    osla="#D27A3F",
    osla_light="#F4E2D5",
    gws="#4C78A8",
    gws_light="#DCE7F2",
    dark="#30343B",
    text="#202124",
    neutral="#A7ADB4",
    pale="#F4F5F6",
)

GRAY = Theme(
    osla="#575757",
    osla_light="#D8D8D8",
    gws="#858585",
    gws_light="#E8E8E8",
    dark="#202020",
    text="#111111",
    neutral="#969696",
    pale="#F3F3F3",
)

WIREFRAME = Theme(
    osla="#666666",
    osla_light="#FFFFFF",
    gws="#777777",
    gws_light="#FFFFFF",
    dark="#333333",
    text="#222222",
    neutral="#A0A0A0",
    pale="#FFFFFF",
)


@dataclass(frozen=True)
class Seg:
    text: str
    size: float = 31
    rise: float = 0
    bold: bool = False
    italic: bool = True
    family: str = "times"


def rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def blend(color: str, white_amount: float) -> str:
    base = rgb(color)
    out = tuple(round(c + (255 - c) * white_amount) for c in base)
    return "#%02X%02X%02X" % out


class SVGCanvas:
    def __init__(self):
        self.parts: list[str] = []

    def rect(self, x, y, w, h, *, fill, stroke="none", sw=0, dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{da}/>'
        )

    def line(self, x1, y1, x2, y2, *, stroke, sw=3, dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linecap="round"{da}/>'
        )

    def polyline(self, points, *, stroke, sw=3, fill="none", dash=None):
        pts = " ".join(f"{x},{y}" for x, y in points)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"{da}/>'
        )

    def polygon(self, points, *, fill, stroke="none", sw=0):
        pts = " ".join(f"{x},{y}" for x, y in points)
        self.parts.append(
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def circle(self, x, y, r, *, fill, stroke="none", sw=0):
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def text(self, x, y, value, *, size=30, color="#202124", anchor="start", bold=False,
             italic=False, family="cn"):
        fam = "SimSun, 'Source Han Serif SC', serif" if family == "cn" else "'Times New Roman', Times, serif"
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{color}" text-anchor="{anchor}" '
            f'font-family="{fam}" font-size="{size}" font-weight="{700 if bold else 400}" '
            f'font-style="{"italic" if italic else "normal"}">{html.escape(value)}</text>'
        )

    def formula(self, x, y, segments: Sequence[Seg], *, anchor="start", color="#202124"):
        estimate = sum(len(seg.text) * seg.size * 0.51 for seg in segments)
        if anchor == "middle":
            x -= estimate / 2
        elif anchor == "end":
            x -= estimate
        spans = []
        for seg in segments:
            fam = "'Times New Roman', Times, serif" if seg.family == "times" else "SimSun, serif"
            shift = "baseline" if seg.rise == 0 else ("super" if seg.rise > 0 else "sub")
            spans.append(
                f'<tspan font-family="{fam}" font-size="{seg.size}" '
                f'font-weight="{700 if seg.bold else 400}" font-style="{"italic" if seg.italic else "normal"}" '
                f'baseline-shift="{shift}">{html.escape(seg.text)}</tspan>'
            )
        self.parts.append(f'<text x="{x:.1f}" y="{y}" fill="{color}">{"".join(spans)}</text>')

    def save(self, path: Path):
        doc = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm" '
            f'viewBox="0 0 {W} {H}">\n<rect width="{W}" height="{H}" fill="#FFFFFF"/>\n'
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )
        path.write_text(doc, encoding="utf-8")


class RasterCanvas:
    def __init__(self, width_px: int):
        self.s = width_px / W
        self.image = Image.new("RGB", (width_px, round(H * self.s)), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.font_cache = {}

    def _font(self, size, family="cn", bold=False, italic=False):
        px = max(8, round(size * self.s))
        key = (px, family, bold, italic)
        if key not in self.font_cache:
            if family == "cn":
                path = SIMSUN
            elif bold and italic:
                path = TIMES_BOLD_ITALIC
            elif bold:
                path = TIMES_BOLD
            elif italic:
                path = TIMES_ITALIC
            else:
                path = TIMES
            self.font_cache[key] = ImageFont.truetype(str(path), px)
        return self.font_cache[key]

    def xy(self, x, y):
        return round(x * self.s), round(y * self.s)

    def rect(self, x, y, w, h, *, fill, stroke="none", sw=0, dash=None):
        box = [*self.xy(x, y), *self.xy(x + w, y + h)]
        if fill != "none":
            self.draw.rectangle(box, fill=fill)
        if stroke != "none" and sw:
            width = max(1, round(sw * self.s))
            if dash:
                self._dashed_rect(x, y, w, h, stroke, width)
            else:
                self.draw.rectangle(box, outline=stroke, width=width)

    def _dashed_rect(self, x, y, w, h, color, width):
        step, dash = 18, 11
        for xx in range(round(x), round(x + w), step):
            self.draw.line([self.xy(xx, y), self.xy(min(xx + dash, x + w), y)], fill=color, width=width)
            self.draw.line([self.xy(xx, y + h), self.xy(min(xx + dash, x + w), y + h)], fill=color, width=width)
        for yy in range(round(y), round(y + h), step):
            self.draw.line([self.xy(x, yy), self.xy(x, min(yy + dash, y + h))], fill=color, width=width)
            self.draw.line([self.xy(x + w, yy), self.xy(x + w, min(yy + dash, y + h))], fill=color, width=width)

    def line(self, x1, y1, x2, y2, *, stroke, sw=3, dash=None):
        width = max(1, round(sw * self.s))
        if not dash:
            self.draw.line([self.xy(x1, y1), self.xy(x2, y2)], fill=stroke, width=width)
            return
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        pos = 0
        while pos < length:
            end = min(pos + 11, length)
            self.draw.line(
                [self.xy(x1 + ux * pos, y1 + uy * pos), self.xy(x1 + ux * end, y1 + uy * end)],
                fill=stroke,
                width=width,
            )
            pos += 18

    def polyline(self, points, *, stroke, sw=3, fill="none", dash=None):
        for a, b in zip(points, points[1:]):
            self.line(a[0], a[1], b[0], b[1], stroke=stroke, sw=sw, dash=dash)

    def polygon(self, points, *, fill, stroke="none", sw=0):
        pts = [self.xy(x, y) for x, y in points]
        self.draw.polygon(pts, fill=fill)
        if stroke != "none" and sw:
            self.draw.line(pts + [pts[0]], fill=stroke, width=max(1, round(sw * self.s)))

    def circle(self, x, y, r, *, fill, stroke="none", sw=0):
        box = [*self.xy(x - r, y - r), *self.xy(x + r, y + r)]
        self.draw.ellipse(box, fill=fill, outline=None if stroke == "none" else stroke,
                          width=max(1, round(sw * self.s)) if sw else 1)

    def text(self, x, y, value, *, size=30, color="#202124", anchor="start", bold=False,
             italic=False, family="cn"):
        font = self._font(size, family, bold, italic)
        box = self.draw.textbbox((0, 0), value, font=font)
        width = box[2] - box[0]
        px, py = self.xy(x, y)
        if anchor == "middle":
            px -= width // 2
        elif anchor == "end":
            px -= width
        self.draw.text((px, py - round(size * self.s)), value, font=font, fill=color)

    def formula(self, x, y, segments: Sequence[Seg], *, anchor="start", color="#202124"):
        widths = []
        for seg in segments:
            font = self._font(seg.size, seg.family, seg.bold, seg.italic)
            box = self.draw.textbbox((0, 0), seg.text, font=font)
            widths.append(box[2] - box[0])
        total = sum(widths)
        px, py = self.xy(x, y)
        if anchor == "middle":
            px -= total // 2
        elif anchor == "end":
            px -= total
        for seg, width in zip(segments, widths):
            font = self._font(seg.size, seg.family, seg.bold, seg.italic)
            self.draw.text(
                (px, py - round(seg.size * self.s) - round(seg.rise * self.s)),
                seg.text,
                font=font,
                fill=color,
            )
            px += width


class PDFCanvas:
    def __init__(self, path: Path):
        self.width = WIDTH_MM / 25.4 * 72
        self.height = HEIGHT_MM / 25.4 * 72
        self.s = self.width / W
        pdfmetrics.registerFont(TTFont("GS-SimSun", str(SIMSUN), subfontIndex=0))
        pdfmetrics.registerFont(TTFont("GS-Times", str(TIMES)))
        pdfmetrics.registerFont(TTFont("GS-Times-Bold", str(TIMES_BOLD)))
        pdfmetrics.registerFont(TTFont("GS-Times-Italic", str(TIMES_ITALIC)))
        pdfmetrics.registerFont(TTFont("GS-Times-BoldItalic", str(TIMES_BOLD_ITALIC)))
        self.c = canvas.Canvas(str(path), pagesize=(self.width, self.height), pageCompression=1)
        self.c.setTitle("GS-OWL framework v2")

    def X(self, x): return x * self.s
    def Y(self, y): return self.height - y * self.s
    def C(self, value):
        a = rgb(value)
        return tuple(v / 255 for v in a)

    def rect(self, x, y, w, h, *, fill, stroke="none", sw=0, dash=None):
        if fill != "none":
            self.c.setFillColorRGB(*self.C(fill))
        if stroke != "none":
            self.c.setStrokeColorRGB(*self.C(stroke))
        self.c.setLineWidth(sw * self.s)
        self.c.setDash([11 * self.s, 7 * self.s] if dash else [])
        self.c.rect(self.X(x), self.Y(y + h), self.X(w), self.X(h),
                    fill=fill != "none", stroke=stroke != "none" and sw > 0)
        self.c.setDash([])

    def line(self, x1, y1, x2, y2, *, stroke, sw=3, dash=None):
        self.c.setStrokeColorRGB(*self.C(stroke))
        self.c.setLineWidth(sw * self.s)
        self.c.setLineCap(1)
        self.c.setDash([11 * self.s, 7 * self.s] if dash else [])
        self.c.line(self.X(x1), self.Y(y1), self.X(x2), self.Y(y2))
        self.c.setDash([])

    def polyline(self, points, *, stroke, sw=3, fill="none", dash=None):
        p = self.c.beginPath()
        p.moveTo(self.X(points[0][0]), self.Y(points[0][1]))
        for x, y in points[1:]:
            p.lineTo(self.X(x), self.Y(y))
        self.c.setStrokeColorRGB(*self.C(stroke))
        self.c.setLineWidth(sw * self.s)
        self.c.setDash([11 * self.s, 7 * self.s] if dash else [])
        self.c.drawPath(p, fill=0, stroke=1)
        self.c.setDash([])

    def polygon(self, points, *, fill, stroke="none", sw=0):
        p = self.c.beginPath()
        p.moveTo(self.X(points[0][0]), self.Y(points[0][1]))
        for x, y in points[1:]:
            p.lineTo(self.X(x), self.Y(y))
        p.close()
        self.c.setFillColorRGB(*self.C(fill))
        if stroke != "none":
            self.c.setStrokeColorRGB(*self.C(stroke))
            self.c.setLineWidth(sw * self.s)
        self.c.drawPath(p, fill=1, stroke=stroke != "none" and sw > 0)

    def circle(self, x, y, r, *, fill, stroke="none", sw=0):
        self.c.setFillColorRGB(*self.C(fill))
        if stroke != "none":
            self.c.setStrokeColorRGB(*self.C(stroke))
            self.c.setLineWidth(sw * self.s)
        self.c.circle(self.X(x), self.Y(y), self.X(r), fill=1, stroke=stroke != "none" and sw > 0)

    def _font_name(self, family, bold, italic):
        if family == "cn": return "GS-SimSun"
        if bold and italic: return "GS-Times-BoldItalic"
        if bold: return "GS-Times-Bold"
        if italic: return "GS-Times-Italic"
        return "GS-Times"

    def text(self, x, y, value, *, size=30, color="#202124", anchor="start", bold=False,
             italic=False, family="cn"):
        font = self._font_name(family, bold, italic)
        fs = size * self.s
        self.c.setFont(font, fs)
        self.c.setFillColorRGB(*self.C(color))
        width = pdfmetrics.stringWidth(value, font, fs)
        px = self.X(x)
        if anchor == "middle": px -= width / 2
        elif anchor == "end": px -= width
        self.c.drawString(px, self.Y(y), value)

    def formula(self, x, y, segments: Sequence[Seg], *, anchor="start", color="#202124"):
        data, total = [], 0
        for seg in segments:
            font = self._font_name(seg.family, seg.bold, seg.italic)
            fs = seg.size * self.s
            width = pdfmetrics.stringWidth(seg.text, font, fs)
            data.append((seg, font, fs, width))
            total += width
        px = self.X(x)
        if anchor == "middle": px -= total / 2
        elif anchor == "end": px -= total
        for seg, font, fs, width in data:
            self.c.setFont(font, fs)
            self.c.setFillColorRGB(*self.C(color))
            self.c.drawString(px, self.Y(y) + seg.rise * self.s, seg.text)
            px += width

    def save(self):
        self.c.showPage()
        self.c.save()


def arrow_h(c, x1, y, x2, color, sw=4, dash=None):
    c.line(x1, y, x2 - 13, y, stroke=color, sw=sw, dash=dash)
    c.polygon([(x2 - 13, y - 8), (x2, y), (x2 - 13, y + 8)], fill=color)


def arrow_v(c, x, y1, y2, color, sw=4):
    c.line(x, y1, x, y2 - 13, stroke=color, sw=sw)
    c.polygon([(x - 8, y2 - 13), (x, y2), (x + 8, y2 - 13)], fill=color)


def group_title(c, x, y, title, color, width):
    c.text(x, y, title, size=35, color=color, bold=True, family="cn")
    c.line(x, y + 12, x + width, y + 12, stroke=blend(color, 0.52), sw=2)


def matrix(c, x, y, values, cell, base, border, *, highlight=None, hatch=None, no_fill=False):
    rows, cols = len(values), len(values[0])
    for rr in range(rows):
        for cc in range(cols):
            v = values[rr][cc]
            fill = "#FFFFFF" if no_fill else blend(base, 0.88 - 0.62 * v)
            if highlight and (rr, cc) in highlight:
                fill = highlight[(rr, cc)]
            c.rect(x + cc * cell, y + rr * cell, cell, cell, fill=fill, stroke="#FFFFFF", sw=1)
            if hatch and (rr, cc) in hatch:
                c.line(x + cc * cell + 2, y + (rr + 1) * cell - 2,
                       x + (cc + 1) * cell - 2, y + rr * cell + 2,
                       stroke=border, sw=1.2)
    c.rect(x, y, cols * cell, rows * cell, fill="none", stroke=border, sw=2)


def transformer_stack(c, x, y, theme: Theme, *, sparse=False, selected_index=None,
                      n=6, lw=102, lh=31, gap=7, x_offset=4):
    for idx in range(n - 1, -1, -1):
        yy = y + idx * (lh + gap)
        selected = idx == selected_index
        stroke = theme.osla if selected else theme.dark
        sw = 4 if selected else 2
        c.rect(x + idx * x_offset, yy, lw, lh, fill=theme.white, stroke=stroke, sw=sw)
        if sparse:
            keep_pattern = [0.70, 0.56, 0.78, 0.48, 0.64, 0.42]
            keep = keep_pattern[idx % len(keep_pattern)]
            cells = 10
            for j in range(cells):
                if j / cells < keep:
                    c.rect(x + idx * x_offset + 7 + j * ((lw - 18) / cells), yy + max(4, lh * 0.28),
                           max(4, (lw - 28) / cells), max(7, lh * 0.42), fill=theme.dark)
                else:
                    xx = x + idx * x_offset + 7 + j * ((lw - 18) / cells)
                    c.line(xx, yy + lh - 5, xx + max(4, (lw - 28) / cells), yy + 5,
                           stroke=theme.neutral, sw=1)
    return x + selected_index * x_offset + lw if selected_index is not None else x + (n - 1) * x_offset + lw


W_VALUES = [
    [0.20, 0.44, 0.72, 0.31, 0.80, 0.55, 0.66],
    [0.25, 0.51, 0.76, 0.38, 0.86, 0.58, 0.69],
    [0.18, 0.47, 0.70, 0.34, 0.82, 0.61, 0.74],
    [0.22, 0.49, 0.79, 0.36, 0.88, 0.57, 0.71],
    [0.16, 0.42, 0.68, 0.29, 0.77, 0.53, 0.64],
]

G_VALUES = [
    [0.12, 0.30, 0.56, 0.24, 0.91, 0.48, 0.65],
    [0.15, 0.35, 0.61, 0.28, 0.95, 0.52, 0.70],
    [0.10, 0.32, 0.53, 0.25, 0.88, 0.57, 0.73],
    [0.13, 0.34, 0.64, 0.27, 0.97, 0.50, 0.68],
    [0.09, 0.29, 0.49, 0.21, 0.85, 0.46, 0.62],
]

M_VALUES = [
    [abs(w) * math.sqrt(g) for w, g in zip(wrow, grow)]
    for wrow, grow in zip(W_VALUES, G_VALUES)
]
max_m = max(max(row) for row in M_VALUES)
M_VALUES = [[v / max_m for v in row] for row in M_VALUES]
SORTED_M = [sorted(row) for row in M_VALUES]
Q = 3
MASK_SORTED = [[0 if cc < Q else 1 for cc in range(7)] for _ in range(5)]
SPARSE_W = [[w if MASK_SORTED[r][cc] else 0 for cc, w in enumerate(row)] for r, row in enumerate(W_VALUES)]


def draw_wireframe(c):
    t = WIREFRAME
    group_title(c, 300, 55, "模型级：层间预算分配（OSLA）", t.dark, 760)
    group_title(c, 300, 385, "连接级：层内连接排序（GWS）", t.dark, 650)
    group_title(c, 1010, 335, "预算约束掩码生成", t.dark, 300)
    group_title(c, 1390, 285, "稀疏模型", t.dark, 230)

    c.text(32, 72, "校准序列", size=30, color=t.dark, family="cn")
    for i, width in enumerate([135, 165, 115]):
        c.line(32, 105 + i * 25, 32 + width, 105 + i * 25, stroke=t.neutral, sw=4)
    arrow_h(c, 72, 190, 145, t.dark)
    transformer_stack(c, 145, 92, t, selected_index=3)
    c.text(175, 340, "稠密LLM", size=30, color=t.dark, family="cn")

    for i, x in enumerate([320, 415, 510]):
        c.rect(x, 110, 72, 58, fill=t.white, stroke=t.neutral, sw=2)
    arrow_h(c, 590, 140, 680, t.dark)
    c.rect(685, 102, 115, 76, fill=t.white, stroke=t.dark, sw=2)
    c.text(742, 147, "OSLA", size=30, color=t.dark, anchor="middle", family="times")
    arrow_h(c, 815, 140, 875, t.dark)
    for i, h in enumerate([40, 58, 32, 66, 48, 72, 35, 55]):
        c.rect(885 + i * 27, 210 - h, 16, h, fill=t.white, stroke=t.neutral, sw=2)
    arrow_v(c, 1046, 220, 374, t.dark, sw=5)

    c.polyline([(250, 226), (250, 470), (285, 470)], stroke=t.neutral, sw=3)
    c.rect(300, 435, 112, 80, fill=t.white, stroke=t.neutral, sw=2)
    c.rect(445, 435, 112, 80, fill=t.white, stroke=t.neutral, sw=2)
    c.text(428, 482, "+", size=34, color=t.dark, anchor="middle", family="times")
    arrow_h(c, 575, 475, 635, t.dark)
    c.rect(650, 435, 112, 80, fill=t.white, stroke=t.neutral, sw=2)
    arrow_h(c, 780, 475, 840, t.dark)
    c.rect(855, 435, 112, 80, fill=t.white, stroke=t.neutral, sw=2)
    arrow_h(c, 975, 475, 1015, t.dark)

    c.rect(1030, 420, 112, 80, fill=t.white, stroke=t.neutral, sw=2)
    arrow_h(c, 1150, 460, 1192, t.dark, sw=5)
    c.rect(1200, 420, 112, 80, fill=t.white, stroke=t.dark, sw=3)
    arrow_h(c, 1325, 460, 1380, t.dark, sw=5)
    c.rect(1390, 405, 72, 58, fill=t.white, stroke=t.neutral, sw=2)
    c.circle(1480, 436, 10, fill=t.white, stroke=t.dark, sw=2)
    c.circle(1480, 436, 3, fill=t.dark)
    c.rect(1510, 405, 72, 58, fill=t.white, stroke=t.neutral, sw=2)
    arrow_v(c, 1518, 492, 545, t.dark)
    transformer_stack(c, 1450, 545, t, sparse=True, n=5, lw=120, lh=20, gap=4, x_offset=5)


def draw_final(c, t: Theme):
    # Global input and dense model.
    c.text(30, 60, "校准序列", size=31, color=t.text, family="cn")
    token_rows = [
        [("the", 29), ("leaf", 36), ("...", 29)],
        [("root", 35), ("cells", 36), ("...", 29)],
        [("x1", 29), ("...", 31), ("xN", 31)],
    ]
    for rr, row in enumerate(token_rows):
        xx, yy = 24, 82 + rr * 38
        for text_value, tw in row:
            c.rect(xx, yy, tw, 27, fill=t.pale, stroke=t.neutral, sw=1.2)
            c.text(xx + tw / 2, yy + 19, text_value, size=17, color=t.text,
                   anchor="middle", family="times")
            xx += tw + 4
    arrow_h(c, 78, 203, 166, t.dark, sw=3.7)
    selected_right = transformer_stack(c, 166, 83, t, selected_index=3)
    c.text(217, 342, "稠密LLM", size=31, color=t.text, anchor="middle", family="cn")
    c.formula(217, 371, [Seg("f", 29), Seg("W", 20, -7, bold=True)], anchor="middle", color=t.text)

    # Model-level OSLA.
    group_title(c, 300, 55, "模型级：层间预算分配（OSLA）", t.osla, 815)
    arrow_h(c, selected_right + 10, 150, 302, t.osla, sw=4)
    outliers = [
        {(0, 3): blend(t.osla, 0.25), (2, 1): blend(t.osla, 0.48)},
        {(0, 1): blend(t.osla, 0.38), (1, 4): blend(t.osla, 0.08), (3, 2): blend(t.osla, 0.55)},
        {(1, 2): blend(t.osla, 0.32), (2, 4): blend(t.osla, 0.13)},
    ]
    for idx, x in enumerate([310, 420, 530]):
        values = [[0.12 + ((r * 3 + cc * 2 + idx) % 6) * 0.07 for cc in range(5)] for r in range(4)]
        matrix(c, x, 104, values, 15, t.neutral, t.neutral, highlight=outliers[idx])
        label = ["X₁", "Xₗ", "Xᴸ"][idx]
        c.text(x + 37, 184, label, size=27, color=t.text, anchor="middle", family="times", italic=True)
    c.text(326, 226, "异常数量", size=28, color=t.text, family="cn")
    c.formula(438, 226, [Seg("c", 30), Seg("l", 19, -7)], color=t.osla)
    c.text(477, 226, "异常强度", size=28, color=t.text, family="cn")
    c.formula(588, 226, [Seg("v", 30), Seg("l", 19, -7)], color=t.osla)

    arrow_h(c, 622, 145, 672, t.osla, sw=4)
    c.rect(682, 102, 120, 82, fill=t.white, stroke=t.osla, sw=3)
    c.text(742, 139, "OSLA", size=32, color=t.osla, anchor="middle", family="times")
    c.text(742, 170, "计数锚定", size=26, color=t.text, anchor="middle", family="cn")
    c.text(742, 217, "深度约束", size=27, color=t.text, anchor="middle", family="cn")
    c.text(742, 249, "全局预算投影", size=27, color=t.text, anchor="middle", family="cn")
    arrow_h(c, 812, 145, 862, t.osla, sw=4)

    budget = [0.48, 0.55, 0.43, 0.62, 0.51, 0.66, 0.46, 0.58, 0.52]
    bar_x, base_y = 880, 237
    selected_bar = 7
    for i, val in enumerate(budget):
        bh = 58 + (val - 0.43) * 210
        fill = t.osla if i == selected_bar else blend(t.osla, 0.58)
        sw = 2.5 if i == selected_bar else 1.4
        c.rect(bar_x + i * 26, base_y - bh, 16, bh, fill=fill, stroke=t.osla, sw=sw)
    c.line(bar_x - 5, base_y, bar_x + 9 * 26 - 5, base_y, stroke=t.neutral, sw=1.5)
    c.formula(989, 274, [
        Seg("s", 31, bold=True), Seg(" = OSLA(c, v; s)", 31),
    ], anchor="middle", color=t.osla)

    # Layer-local magnification and GWS matrices.
    group_title(c, 300, 384, "连接级：层内连接排序（GWS）", t.gws, 665)
    c.polyline([(selected_right - 48, 226), (selected_right - 48, 452), (292, 452)],
               stroke=t.neutral, sw=2.5)
    arrow_h(c, 292, 452, 312, t.neutral, sw=2.5)
    c.text(245, 430, "第 l 层放大", size=27, color=t.text, anchor="middle", family="cn")

    matrix(c, 318, 444, W_VALUES, 16, t.dark, t.dark)
    c.formula(374, 542, [Seg("W", 31, bold=True), Seg("l,j", 20, -7)], anchor="middle", color=t.text)
    c.text(443, 488, "×", size=37, color=t.dark, anchor="middle", family="times")
    matrix(c, 476, 444, G_VALUES, 16, t.gws, t.gws)
    c.formula(532, 542, [Seg("G", 31, bold=True), Seg("(l)", 20, 9)], anchor="middle", color=t.gws)

    arrow_h(c, 606, 484, 652, t.gws, sw=4)
    matrix(c, 664, 444, M_VALUES, 16, t.gws, t.gws)
    c.formula(720, 542, [Seg("M", 31, bold=True), Seg("(l)", 20, 9)], anchor="middle", color=t.gws)
    c.formula(720, 588, [
        Seg("M", 31), Seg("ij", 20, -7), Seg("(l)", 20, 9), Seg(" = |W", 31),
        Seg("ij", 20, -7), Seg("(l)", 20, 9), Seg("|√G", 31), Seg("ij", 20, -7), Seg("(l)", 20, 9),
    ], anchor="middle", color=t.text)
    c.text(720, 621, "动态范围压缩", size=27, color=t.gws, anchor="middle", family="cn")

    arrow_h(c, 792, 484, 838, t.gws, sw=4)
    matrix(c, 850, 444, SORTED_M, 16, t.gws, t.gws)
    c.text(906, 542, "逐行排序", size=29, color=t.text, anchor="middle", family="cn")

    # OSLA budget directly constrains row-wise GWS selection.
    group_title(c, 1095, 352, "预算约束掩码生成", t.dark, 225)
    selected_bar_x = bar_x + selected_bar * 26 + 8
    arrow_v(c, selected_bar_x, base_y + 8, 430, t.osla, sw=4)
    c.formula(selected_bar_x - 13, 371, [Seg("s", 31, bold=True), Seg("l", 20, -7)],
              anchor="end", color=t.osla)
    arrow_h(c, 970, 484, 1012, t.gws, sw=4)

    # Sorted M and its deterministic row-quota mask.
    matrix(c, 1022, 438, SORTED_M, 16, t.gws, t.gws)
    threshold_x = 1022 + Q * 16
    c.line(threshold_x, 430, threshold_x, 526, stroke=t.osla, sw=4)
    c.formula(threshold_x, 420, [Seg("q", 29), Seg("l,j", 19, -7)], anchor="middle", color=t.osla)
    arrow_h(c, 1143, 478, 1177, t.dark, sw=5)
    deleted = {(r, cc) for r in range(5) for cc in range(Q)}
    mask_highlight = {
        (r, cc): t.dark if MASK_SORTED[r][cc] else t.white
        for r in range(5) for cc in range(7)
    }
    matrix(c, 1186, 438, MASK_SORTED, 16, t.dark, t.dark,
           highlight=mask_highlight, hatch=deleted, no_fill=True)
    c.formula(1242, 542, [Seg("B", 31, bold=True), Seg("l,j", 20, -7)], anchor="middle", color=t.dark)
    c.text(1132, 590, "逐行保留高显著性连接", size=27, color=t.text, anchor="middle", family="cn")

    # Sparse weight and sparse model output. The zero pattern matches B exactly.
    group_title(c, 1430, 352, "稀疏模型", t.dark, 225)
    matrix(c, 1368, 425, W_VALUES[:4], 12, t.dark, t.dark)
    c.formula(1410, 500, [Seg("W", 28, bold=True), Seg("l,j", 18, -6)], anchor="middle", color=t.text)
    c.circle(1461, 458, 9, fill=t.white, stroke=t.dark, sw=1.8)
    c.circle(1461, 458, 2.7, fill=t.dark)
    mask_small = [row[:] for row in MASK_SORTED[:4]]
    highlight_small = {
        (r, cc): t.dark if mask_small[r][cc] else t.white
        for r in range(4) for cc in range(7)
    }
    matrix(c, 1482, 425, mask_small, 12, t.dark, t.dark,
           highlight=highlight_small, hatch={(r, cc) for r in range(4) for cc in range(Q)}, no_fill=True)
    c.formula(1524, 500, [Seg("B", 28, bold=True), Seg("l,j", 18, -6)], anchor="middle", color=t.text)
    arrow_h(c, 1574, 467, 1610, t.dark, sw=4)
    matrix(c, 1615, 425, SPARSE_W[:4], 10, t.dark, t.dark,
           highlight={(r, cc): (t.white if cc < Q else blend(t.dark, 0.25 + 0.55 * (1 - W_VALUES[r][cc])))
                      for r in range(4) for cc in range(7)},
           hatch={(r, cc) for r in range(4) for cc in range(Q)}, no_fill=True)

    c.formula(1490, 545, [
        Seg("W̃", 31, bold=True), Seg("l,j", 20, -7), Seg(" = W", 31, bold=True), Seg("l,j", 20, -7),
    ], anchor="end", color=t.dark)
    c.circle(1505, 536, 10, fill=t.white, stroke=t.dark, sw=2)
    c.circle(1505, 536, 3, fill=t.dark)
    c.formula(1520, 545, [Seg("B", 31, bold=True), Seg("l,j", 20, -7)], color=t.dark)
    arrow_v(c, 1530, 565, 592, t.dark, sw=4)
    transformer_stack(c, 1450, 580, t, sparse=True, n=5, lw=120, lh=16, gap=3, x_offset=5)
    c.text(1550, 704, "满足全局目标稀疏率的", size=28, color=t.text,
           anchor="middle", family="cn")
    c.text(1550, 733, "稀疏LLM", size=28, color=t.text,
           anchor="middle", family="cn")


def render_svg(path: Path, drawer, theme=None):
    c = SVGCanvas()
    drawer(c) if theme is None else drawer(c, theme)
    c.save(path)


def render_png(path: Path, drawer, width_px: int, theme=None, dpi=300):
    c = RasterCanvas(width_px)
    drawer(c) if theme is None else drawer(c, theme)
    c.image.save(path, dpi=(dpi, dpi), optimize=True)


def render_pdf(path: Path, drawer, theme):
    c = PDFCanvas(path)
    drawer(c, theme)
    c.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wireframe-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    wire_svg = OUT / "GS_OWL_framework_v2_wireframe.svg"
    wire_png = OUT / "GS_OWL_framework_v2_wireframe.png"
    render_svg(wire_svg, draw_wireframe)
    render_png(wire_png, draw_wireframe, 2200, dpi=200)
    print(wire_svg)
    print(wire_png)
    if args.wireframe_only:
        return

    svg = OUT / "GS_OWL_framework_v2.svg"
    pdf = OUT / "GS_OWL_framework_v2.pdf"
    png = OUT / "GS_OWL_framework_v2_600dpi.png"
    gray = OUT / "GS_OWL_framework_v2_grayscale.png"
    preview = OUT / "GS_OWL_framework_v2_preview_17cm.png"
    render_svg(svg, draw_final, COLOR)
    render_pdf(pdf, draw_final, COLOR)
    render_png(png, draw_final, round(WIDTH_MM / 25.4 * 600), COLOR, dpi=600)
    render_png(gray, draw_final, round(WIDTH_MM / 25.4 * 300), GRAY, dpi=300)
    render_png(preview, draw_final, round(WIDTH_MM / 25.4 * 300), COLOR, dpi=300)
    for path in [svg, pdf, png, gray, preview]:
        print(path)


if __name__ == "__main__":
    main()
