"""Draw the submission-ready GS-OWL paper framework figure.

This version uses a global-model view, an OSLA budget controller, a magnified
layer-local GWS view, budget-constrained row selection, and a sparse-LLM output.
The geometry is authored at 170 mm x 64 mm for a two-column journal figure.
This variant minimizes explanatory text and uses a unified output-chain texture.
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
W, H = 1700, 640
WIDTH_MM, HEIGHT_MM = 170, 64

SIMSUN = Path(r"C:\Windows\Fonts\simsun.ttc")
TIMES = Path(r"C:\Windows\Fonts\times.ttf")
TIMES_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")
TIMES_ITALIC = Path(r"C:\Windows\Fonts\timesi.ttf")
TIMES_BOLD_ITALIC = Path(r"C:\Windows\Fonts\timesbi.ttf")
CAMBRIA = Path(r"C:\Windows\Fonts\cambria.ttc")


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
        if family == "cn":
            fam = "SimSun, 'Source Han Serif SC', serif"
        elif family == "math":
            fam = "'Cambria Math', Cambria, serif"
        else:
            fam = "'Times New Roman', Times, serif"
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
            if seg.family == "times":
                fam = "'Times New Roman', Times, serif"
            elif seg.family == "math":
                fam = "'Cambria Math', Cambria, serif"
            else:
                fam = "SimSun, serif"
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
                self.font_cache[key] = ImageFont.truetype(str(path), px)
                return self.font_cache[key]
            elif family == "math":
                self.font_cache[key] = ImageFont.truetype(str(CAMBRIA), px, index=1)
                return self.font_cache[key]
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
        pdfmetrics.registerFont(TTFont("GS-Cambria-Math", str(CAMBRIA), subfontIndex=1))
        self.c = canvas.Canvas(str(path), pagesize=(self.width, self.height), pageCompression=1)
        self.c.setTitle("GS-OWL framework final")

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
        if family == "math": return "GS-Cambria-Math"
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
                      n=6, lw=96, lh=18, gap=3, x_offset=3):
    for idx in range(n - 1, -1, -1):
        yy = y + idx * (lh + gap)
        selected = idx == selected_index
        stroke = theme.osla if selected else theme.dark
        sw = 4 if selected else 2
        c.rect(x + idx * x_offset, yy, lw, lh, fill=theme.white, stroke=stroke, sw=sw)
        if sparse:
            densities = [0.68, 0.52, 0.74, 0.46, 0.61, 0.43]
            density = densities[idx % len(densities)]
            rows, cols = 2, 12
            cell_w = (lw - 14) / cols
            cell_h = (lh - 7) / rows
            for rr in range(rows):
                for cc in range(cols):
                    score = ((cc * 5 + rr * 7 + idx * 3) % 23) / 22
                    xx = x + idx * x_offset + 7 + cc * cell_w
                    cy = yy + 3.5 + rr * cell_h
                    if score < density:
                        c.rect(xx, cy, max(2.5, cell_w - 1.2), max(2.5, cell_h - 1.2),
                               fill=theme.dark)
                    else:
                        c.line(xx + 0.5, cy + cell_h - 1.5,
                               xx + cell_w - 1.5, cy + 0.5,
                               stroke=theme.neutral, sw=0.9)
    return x + selected_index * x_offset + lw if selected_index is not None else x + (n - 1) * x_offset + lw


def sparse_submission_stack(c, x, y, theme: Theme, *, n=4, lw=144, lh=36,
                            gap=8, x_offset=4):
    """Draw a compact sparse model using the same cell/hatch language as B and W-tilde."""
    densities = [0.62, 0.48, 0.68, 0.54]
    rows, cols = 3, 12
    for idx in range(n - 1, -1, -1):
        xx = x + idx * x_offset
        yy = y + idx * (lh + gap)
        c.rect(xx, yy, lw, lh, fill=theme.white, stroke=theme.dark, sw=2.4)
        cell_w = (lw - 12) / cols
        cell_h = (lh - 8) / rows
        for rr in range(rows):
            for cc in range(cols):
                score = ((cc * 7 + rr * 5 + idx * 4) % 29) / 28
                cx = xx + 6 + cc * cell_w
                cy = yy + 4 + rr * cell_h
                if score < densities[idx]:
                    c.rect(cx, cy, cell_w, cell_h, fill=theme.dark,
                           stroke=theme.white, sw=0.65)
                else:
                    c.rect(cx, cy, cell_w, cell_h, fill=theme.white,
                           stroke=blend(theme.dark, 0.72), sw=0.55)
                    c.line(cx + 1.2, cy + cell_h - 1.2,
                           cx + cell_w - 1.2, cy + 1.2,
                           stroke=theme.neutral, sw=0.85)


W_VALUES = [
    [0.82, 0.20, 0.55, 0.34, 0.90, 0.48, 0.70],
    [0.30, 0.88, 0.45, 0.76, 0.25, 0.65, 0.53],
    [0.58, 0.41, 0.93, 0.22, 0.67, 0.81, 0.36],
    [0.46, 0.74, 0.28, 0.89, 0.61, 0.33, 0.79],
    [0.91, 0.37, 0.69, 0.52, 0.31, 0.85, 0.44],
]

G_VALUES = [
    [0.16, 0.92, 0.35, 0.74, 0.22, 0.68, 0.48],
    [0.81, 0.27, 0.95, 0.31, 0.72, 0.44, 0.60],
    [0.24, 0.83, 0.18, 0.91, 0.56, 0.29, 0.76],
    [0.66, 0.21, 0.84, 0.26, 0.93, 0.49, 0.34],
    [0.20, 0.78, 0.42, 0.69, 0.88, 0.25, 0.97],
]

M_VALUES = [
    [abs(w) * math.sqrt(g) for w, g in zip(wrow, grow)]
    for wrow, grow in zip(W_VALUES, G_VALUES)
]
max_m = max(max(row) for row in M_VALUES)
M_VALUES = [[v / max_m for v in row] for row in M_VALUES]
SORTED_DESC = [sorted(row, reverse=True) for row in M_VALUES]
Q = 3
KEEP = len(W_VALUES[0]) - Q


def topk_mask(values, keep):
    out = []
    for row in values:
        retained = set(sorted(range(len(row)), key=lambda idx: (-row[idx], idx))[:keep])
        out.append([1 if idx in retained else 0 for idx in range(len(row))])
    return out


MASK_ORIGINAL = topk_mask(M_VALUES, KEEP)
SPARSE_W = [
    [w if MASK_ORIGINAL[rr][cc] else 0 for cc, w in enumerate(row)]
    for rr, row in enumerate(W_VALUES)
]


def matrix_label(c, x, y, symbol, color, *, prefix="", suffix="(l)"):
    segments = []
    if prefix:
        segments.append(Seg(prefix, 29, bold=False))
    segments.extend([Seg(symbol, 31, bold=True), Seg(suffix, 19, 9)])
    c.formula(x, y, segments, anchor="middle", color=color)


def draw_sorted_rows(c, x, y, rows, cell, t: Theme, *, selected=False):
    for rr, row in enumerate(rows):
        for cc, value in enumerate(row):
            if selected and cc >= KEEP:
                c.rect(x + cc * cell, y + rr * (cell + 5), cell, cell,
                       fill=t.white, stroke=t.neutral, sw=1)
                c.line(x + cc * cell + 2, y + rr * (cell + 5) + cell - 2,
                       x + (cc + 1) * cell - 2, y + rr * (cell + 5) + 2,
                       stroke=t.neutral, sw=1)
            else:
                c.rect(x + cc * cell, y + rr * (cell + 5), cell, cell,
                       fill=blend(t.gws, 0.88 - 0.62 * value), stroke=t.white, sw=1)
        c.rect(x, y + rr * (cell + 5), len(row) * cell, cell,
               fill="none", stroke=t.gws if not selected else t.dark, sw=1.6)
    if selected:
        threshold_x = x + KEEP * cell
        c.line(threshold_x, y - 5, threshold_x,
               y + len(rows) * (cell + 5) - 5, stroke=t.osla, sw=3)


def draw_final_v3_reference(c, t: Theme):
    # Compact calibration input and dense model.
    c.text(26, 55, "校准集", size=29, color=t.text, family="cn", bold=True)
    c.formula(73, 104, [
        Seg("D", 30), Seg("X", 18, -7), Seg(",  D", 30), Seg("G", 18, -7),
    ], anchor="middle", color=t.text)
    arrow_h(c, 42, 137, 112, t.dark, sw=3.5)
    selected_right = transformer_stack(c, 112, 115, t, selected_index=3)
    c.text(166, 266, "稠密LLM", size=29, color=t.text, anchor="middle", family="cn")

    # Model-level activation statistics and OSLA budget allocation.
    c.text(270, 48, "模型级：OSLA层间预算", size=28, color=t.osla, family="cn", bold=True)
    c.polyline([(selected_right + 3, 184), (245, 184), (245, 132), (266, 132)],
               stroke=t.osla, sw=2.8)
    arrow_h(c, 266, 132, 282, t.osla, sw=2.8)
    c.text(246, 112, "前向统计", size=22, color=t.osla, anchor="middle", family="cn")

    outliers = [
        {(2, 1): blend(t.osla, 0.55)},
        {(0, 0): blend(t.osla, 0.52), (0, 4): blend(t.osla, 0.08),
         (2, 2): blend(t.osla, 0.30), (3, 4): blend(t.osla, 0.16)},
        {(0, 3): blend(t.osla, 0.18), (3, 0): blend(t.osla, 0.48)},
    ]
    activation_x = [292, 395, 498]
    supers = ["(1)", "(l)", "(L)"]
    for idx, x in enumerate(activation_x):
        values = [[0.13 + ((r * 4 + cc * 3 + idx * 2) % 7) * 0.045 for cc in range(5)] for r in range(4)]
        matrix(c, x, 91, values, 14, t.neutral, t.neutral, highlight=outliers[idx])
        c.formula(x + 35, 170, [Seg("X", 27, bold=True), Seg(supers[idx], 17, 8)],
                  anchor="middle", color=t.text)

    c.rect(302, 193, 12, 12, fill=blend(t.osla, 0.18), stroke=t.osla, sw=1)
    c.text(320, 204, "数量：异常比例", size=21, color=t.text, family="cn")
    for ii, shade in enumerate([0.65, 0.38, 0.10]):
        c.rect(455 + ii * 14, 193, 12, 12, fill=blend(t.osla, shade), stroke=t.osla, sw=0.7)
    c.text(501, 204, "深浅：异常严重度", size=21, color=t.text, family="cn")

    c.formula(590, 126, [Seg("c", 27, bold=True), Seg("l", 17, -6)], anchor="middle", color=t.osla)
    c.formula(590, 169, [Seg("v", 27, bold=True), Seg("l", 17, -6)], anchor="middle", color=t.osla)
    arrow_h(c, 605, 119, 647, t.osla, sw=3)
    arrow_h(c, 605, 162, 647, t.osla, sw=3)
    c.rect(655, 91, 142, 102, fill=t.white, stroke=t.osla, sw=2.5)
    c.text(726, 124, "OSLA", size=29, color=t.osla, anchor="middle", family="times")
    c.text(726, 153, "比例锚定", size=23, color=t.text, anchor="middle", family="cn")
    c.text(726, 181, "严重度修正", size=23, color=t.text, anchor="middle", family="cn")
    c.text(726, 241, "深度约束 · 预算投影", size=21, color=t.text, anchor="middle", family="cn")
    arrow_h(c, 805, 143, 837, t.osla, sw=3.5)

    c.text(850, 51, "非均匀层预算", size=24, color=t.osla, family="cn", bold=True)
    c.formula(951, 84, [
        Seg("s", 29, bold=True), Seg(" = [s", 27), Seg("1", 17, -6), Seg(", ..., s", 27),
        Seg("l", 17, -6), Seg(", ..., s", 27), Seg("L", 17, -6), Seg("]", 27),
    ], anchor="middle", color=t.osla)
    budget = [0.49, 0.57, 0.45, 0.61, 0.53, 0.66, 0.47, 0.59, 0.51]
    bar_x, base_y = 852, 225
    selected_bar = 4
    for i, val in enumerate(budget):
        bh = 48 + (val - 0.45) * 250
        fill = t.osla if i == selected_bar else blend(t.osla, 0.63)
        c.rect(bar_x + i * 23, base_y - bh, 14, bh, fill=fill, stroke=t.osla,
               sw=2.3 if i == selected_bar else 1.2)
    c.line(bar_x - 3, base_y, bar_x + 9 * 23 - 5, base_y, stroke=t.neutral, sw=1.3)
    for idx, label in {0: "1", 2: "...", 4: "l", 6: "...", 8: "L"}.items():
        c.text(bar_x + idx * 23 + 7, 247, label, size=18, color=t.text,
               anchor="middle", family="times", italic=True)

    # Layer-local GWS path.
    c.text(270, 340, "连接级：GWS层内排序", size=28, color=t.gws, family="cn", bold=True)
    c.polyline([(selected_right + 3, 184), (233, 184), (233, 414), (270, 414)],
               stroke=t.neutral, sw=2.5)
    arrow_h(c, 270, 414, 292, t.neutral, sw=2.5)
    c.text(162, 399, "第", size=22, color=t.text, family="cn")
    c.formula(194, 399, [Seg("l", 23)], anchor="middle", color=t.text)
    c.text(208, 399, "层局部放大", size=22, color=t.text, family="cn")
    c.text(230, 377, "权重 / 反向梯度", size=19, color=t.text,
           anchor="middle", family="cn")

    matrix(c, 300, 390, W_VALUES, 14, t.dark, t.dark)
    matrix_label(c, 349, 484, "W", t.text)
    c.text(419, 431, "×", size=34, color=t.dark, anchor="middle", family="times")
    matrix(c, 450, 390, G_VALUES, 14, t.gws, t.gws)
    matrix_label(c, 499, 484, "G", t.gws)

    arrow_h(c, 554, 425, 594, t.gws, sw=3.5)
    matrix(c, 604, 390, M_VALUES, 14, t.gws, t.gws)
    matrix_label(c, 653, 484, "M", t.gws)
    c.formula(653, 535, [
        Seg("M", 28), Seg("ij", 18, -6), Seg("(l)", 18, 8), Seg(" = |W", 28),
        Seg("ij", 18, -6), Seg("(l)", 18, 8), Seg("|√G", 28), Seg("ij", 18, -6), Seg("(l)", 18, 8),
    ], anchor="middle", color=t.text)
    c.text(653, 564, "动态范围压缩", size=21, color=t.gws, anchor="middle", family="cn")

    arrow_h(c, 708, 425, 755, t.gws, sw=3.5)
    draw_sorted_rows(c, 770, 397, SORTED_DESC[:3], 15, t)
    c.text(822, 484, "逐行降序排序", size=22, color=t.text, anchor="middle", family="cn")

    # Explicit two-input selector: OSLA controls how many, GWS controls which.
    selector_x, selector_y, selector_w, selector_h = 995, 365, 248, 221
    selected_budget_x = bar_x + selected_bar * 23 + 7
    c.polyline([(selected_budget_x, base_y + 6), (selected_budget_x, 326),
                (1119, 326), (1119, selector_y - 14)], stroke=t.osla, sw=3.5)
    arrow_v(c, 1119, selector_y - 14, selector_y + 2, t.osla, sw=3.5)
    c.formula(1005, 313, [Seg("s", 27, bold=True), Seg("l", 17, -6)], color=t.osla)
    c.text(1052, 312, "层预算", size=20, color=t.osla, family="cn")

    arrow_h(c, 886, 425, selector_x, t.gws, sw=3.8)
    c.text(939, 411, "层内次序", size=20, color=t.gws, anchor="middle", family="cn")
    c.rect(selector_x, selector_y, selector_w, selector_h,
           fill=t.white, stroke=t.dark, sw=2.4)
    c.text(selector_x + selector_w / 2, selector_y + 34, "预算约束逐行选择",
           size=27, color=t.dark, anchor="middle", family="cn", bold=True)
    c.formula(selector_x + selector_w / 2, selector_y + 69, [
        Seg("q", 26), Seg("l,j", 17, -6), Seg(" = floor(s", 24), Seg("l", 16, -6),
        Seg(" d", 24), Seg("l,j", 16, -6), Seg("in", 16, 7), Seg(")", 24),
    ], anchor="middle", color=t.text)
    draw_sorted_rows(c, selector_x + 55, selector_y + 90, SORTED_DESC[:3], 16, t, selected=True)
    c.text(selector_x + 84, selector_y + 184, "保留", size=19, color=t.dark,
           anchor="middle", family="cn")
    c.text(selector_x + 178, selector_y + 184, "删除", size=19, color=t.neutral,
           anchor="middle", family="cn")
    c.text(selector_x + selector_w / 2, selector_y + 210, "映射回原连接位置",
           size=20, color=t.text, anchor="middle", family="cn")

    # Non-structured row-wise mask: identical row quotas, different positions.
    arrow_h(c, selector_x + selector_w + 8, 465, 1278, t.dark, sw=4)
    mask_highlight = {
        (rr, cc): t.dark if MASK_ORIGINAL[rr][cc] else t.white
        for rr in range(5) for cc in range(7)
    }
    deleted = {(rr, cc) for rr in range(5) for cc in range(7) if not MASK_ORIGINAL[rr][cc]}
    matrix(c, 1288, 420, MASK_ORIGINAL, 16, t.dark, t.dark,
           highlight=mask_highlight, hatch=deleted, no_fill=True)
    matrix_label(c, 1344, 521, "B", t.dark)

    arrow_h(c, 1410, 460, 1492, t.dark, sw=4)
    c.circle(1440, 437, 7, fill=t.white, stroke=t.dark, sw=1.6)
    c.circle(1440, 437, 2.2, fill=t.dark)
    c.formula(1468, 443, [Seg("W", 23, bold=True), Seg("(l)", 15, 7)],
              anchor="middle", color=t.dark)
    sparse_highlight = {
        (rr, cc): (blend(t.dark, 0.22 + 0.62 * (1 - W_VALUES[rr][cc]))
                   if MASK_ORIGINAL[rr][cc] else t.white)
        for rr in range(5) for cc in range(7)
    }
    matrix(c, 1502, 420, SPARSE_W, 16, t.dark, t.dark,
           highlight=sparse_highlight, hatch=deleted, no_fill=True)
    matrix_label(c, 1558, 521, "W̃", t.dark)
    c.formula(1540, 555, [
        Seg("W̃", 25, bold=True), Seg("(l)", 16, 8), Seg(" = W", 25, bold=True),
        Seg("(l)", 16, 8),
    ], anchor="end", color=t.dark)
    c.circle(1552, 547, 7.5, fill=t.white, stroke=t.dark, sw=1.6)
    c.circle(1552, 547, 2.3, fill=t.dark)
    c.formula(1565, 555, [Seg("B", 25, bold=True), Seg("(l)", 16, 8)],
              color=t.dark)

    c.line(1558, 560, 1558, 566, stroke=t.dark, sw=3.2)
    c.polygon([(1551, 566), (1558, 574), (1565, 566)], fill=t.dark)
    transformer_stack(c, 1502, 574, t, sparse=True, n=6, lw=96, lh=18, gap=3, x_offset=3)
    c.text(1555, 730, "满足全局目标的稀疏LLM", size=24, color=t.text,
           anchor="middle", family="cn")


def draw_final_v4(c, t: Theme):
    cell = 14

    # Compact calibration input and dense model.
    c.text(24, 40, "校准集", size=25, color=t.text, family="cn", bold=True)
    c.formula(64, 80, [
        Seg("D", 28), Seg("X", 17, -6), Seg(",  D", 28), Seg("G", 17, -6),
    ], anchor="middle", color=t.text)
    arrow_h(c, 40, 112, 96, t.dark, sw=3.2)
    selected_right = transformer_stack(
        c, 96, 88, t, selected_index=2, n=5, lw=84, lh=16, gap=3, x_offset=2
    )
    c.text(142, 216, "稠密LLM", size=25, color=t.text, anchor="middle", family="cn")

    # Model-level OSLA path.
    c.text(248, 40, "模型级：OSLA 层间预算", size=24, color=t.osla,
           family="cn", bold=True)
    c.polyline([(selected_right + 4, 134), (224, 134), (224, 120), (248, 120)],
               stroke=t.osla, sw=2.8)
    arrow_h(c, 248, 120, 272, t.osla, sw=2.8)

    outliers = [
        {(2, 1): blend(t.osla, 0.55)},
        {(0, 0): blend(t.osla, 0.52), (0, 4): blend(t.osla, 0.08),
         (2, 2): blend(t.osla, 0.30), (3, 4): blend(t.osla, 0.16)},
        {(0, 3): blend(t.osla, 0.18), (3, 0): blend(t.osla, 0.48)},
    ]
    for idx, x in enumerate([272, 376, 480]):
        values = [[0.13 + ((r * 4 + cc * 3 + idx * 2) % 7) * 0.045
                   for cc in range(5)] for r in range(4)]
        matrix(c, x, 88, values, cell, t.neutral, t.neutral, highlight=outliers[idx])
        c.formula(x + 35, 178, [
            Seg("X", 27, bold=True), Seg(["(1)", "(l)", "(L)"][idx], 17, 8),
        ], anchor="middle", color=t.text)

    # Independent outlier-statistic legends.
    c.rect(248, 192, 14, 14, fill=blend(t.osla, 0.18), stroke=t.osla, sw=1)
    c.text(272, 204, "数量：异常比例", size=20, color=t.text, family="cn")
    for ii, shade in enumerate([0.68, 0.38, 0.08]):
        c.rect(248 + ii * 18, 220, 14, 14, fill=blend(t.osla, shade),
               stroke=t.osla, sw=0.8)
    c.text(312, 232, "深浅：异常严重度", size=20, color=t.text, family="cn")

    # Count and severity enter OSLA through separate arrows.
    c.formula(624, 108, [Seg("c", 26, bold=True), Seg("l", 16, -6)],
              anchor="middle", color=t.osla)
    c.formula(624, 160, [Seg("v", 26, bold=True), Seg("l", 16, -6)],
              anchor="middle", color=t.osla)
    arrow_h(c, 640, 120, 672, t.osla, sw=3)
    arrow_h(c, 640, 172, 672, t.osla, sw=3)
    c.rect(672, 88, 152, 112, fill=t.white, stroke=t.osla, sw=2.5)
    c.text(748, 124, "OSLA", size=28, color=t.osla, anchor="middle", family="times")
    c.text(748, 154, "比例锚定", size=23, color=t.text, anchor="middle", family="cn")
    c.text(748, 184, "严重度修正", size=23, color=t.text, anchor="middle", family="cn")
    c.text(748, 248, "深度约束", size=21, color=t.text,
           anchor="middle", family="cn")
    c.text(876, 128, "全局预算投影", size=20, color=t.osla,
           anchor="middle", family="cn")
    arrow_h(c, 832, 144, 920, t.osla, sw=3.4)

    # Layer budget bars are centered directly over the selector.
    c.text(1040, 40, "非均匀层预算", size=22, color=t.osla,
           anchor="middle", family="cn", bold=True)
    c.formula(1040, 72, [
        Seg("s", 28, bold=True), Seg(" = [s", 26), Seg("1", 16, -6), Seg(", ..., s", 26),
        Seg("l", 16, -6), Seg(", ..., s", 26), Seg("L", 16, -6), Seg("]", 26),
    ], anchor="middle", color=t.osla)
    budget = [0.49, 0.57, 0.45, 0.61, 0.53, 0.66, 0.47, 0.59, 0.51]
    bar_x, base_y, selected_bar = 936, 216, 4
    for i, value in enumerate(budget):
        bh = 48 + (value - 0.45) * 240
        c.rect(bar_x + i * 24, base_y - bh, 16, bh,
               fill=t.osla if i == selected_bar else blend(t.osla, 0.63),
               stroke=t.osla, sw=2.2 if i == selected_bar else 1.2)
    c.line(bar_x - 4, base_y, bar_x + 9 * 24 - 4, base_y,
           stroke=t.neutral, sw=1.3)
    for idx, label in {0: "1", 2: "...", 4: "l", 6: "...", 8: "L"}.items():
        c.text(bar_x + idx * 24 + 8, 238, label, size=18, color=t.text,
               anchor="middle", family="times", italic=True)

    selector_x, selector_y, selector_w, selector_h = 896, 288, 288, 200
    arrow_v(c, 1040, 252, selector_y, t.osla, sw=3.5)
    c.text(1064, 274, "第", size=20, color=t.osla, family="cn")
    c.formula(1090, 274, [Seg("l", 22)], anchor="middle", color=t.osla)
    c.text(1102, 274, "层预算", size=20, color=t.osla, family="cn")
    c.formula(1166, 274, [Seg("s", 25, bold=True), Seg("l", 16, -6)],
              color=t.osla)

    # Connection-level GWS path, moved upward and arranged on an 8 px grid.
    c.text(248, 272, "连接级：GWS 层内排序", size=24, color=t.gws,
           family="cn", bold=True)
    c.polyline([(selected_right + 4, 134), (208, 134), (208, 400), (272, 400)],
               stroke=t.neutral, sw=2.5)
    arrow_h(c, 272, 400, 288, t.neutral, sw=2.5)

    # Weight and gradient are distinct inputs to a small GWS fusion operator.
    c.text(337, 322, "权重", size=21, color=t.text, anchor="middle", family="cn")
    matrix(c, 288, 336, W_VALUES, cell, t.dark, t.dark)
    matrix_label(c, 337, 438, "W", t.text)

    c.text(337, 458, "聚合梯度", size=21, color=t.gws, anchor="middle", family="cn")
    matrix(c, 288, 472, G_VALUES, cell, t.gws, t.gws)
    matrix_label(c, 337, 574, "G", t.gws)

    c.polyline([(386, 371), (400, 371), (400, 384)], stroke=t.gws, sw=2.6)
    c.polyline([(386, 507), (408, 507), (408, 416)], stroke=t.gws, sw=2.6)
    arrow_h(c, 400, 384, 432, t.gws, sw=2.6)
    arrow_h(c, 408, 416, 432, t.gws, sw=2.6)
    c.rect(432, 368, 72, 64, fill=t.white, stroke=t.gws, sw=2.2)
    c.text(468, 407, "GWS", size=20, color=t.gws, anchor="middle", family="times")
    arrow_h(c, 504, 400, 520, t.gws, sw=3.2)

    matrix(c, 520, 360, M_VALUES, cell, t.gws, t.gws)
    matrix_label(c, 569, 472, "M", t.gws)
    c.formula(569, 516, [
        Seg("M", 27), Seg("ij", 17, -6), Seg("(l)", 17, 8), Seg(" = |W", 27),
        Seg("ij", 17, -6), Seg("(l)", 17, 8), Seg("|√G", 27),
        Seg("ij", 17, -6), Seg("(l)", 17, 8),
    ], anchor="middle", color=t.text)
    c.text(569, 548, "动态范围压缩", size=20, color=t.gws,
           anchor="middle", family="cn")

    arrow_h(c, 624, 400, 664, t.gws, sw=3.2)
    c.text(721, 352, "逐行排序", size=20, color=t.gws,
           anchor="middle", family="cn")
    draw_sorted_rows(c, 672, 368, SORTED_DESC[:3], cell, t)
    arrow_h(c, 784, 400, selector_x, t.gws, sw=3.5)

    # Central budget-constrained selector.
    c.rect(selector_x, selector_y, selector_w, selector_h,
           fill=t.white, stroke=t.dark, sw=3.4)
    c.text(1040, 324, "预算约束选择", size=27, color=t.dark,
           anchor="middle", family="cn", bold=True)
    c.formula(938, 360, [
        Seg("q", 26), Seg("l,j", 17, -6), Seg(" =", 25),
    ], color=t.text)
    c.formula(1048, 360, [
        Seg("s", 25), Seg("l", 16, -6), Seg(" d", 25), Seg("l,j", 16, -6),
        Seg("in", 16, 7),
    ], color=t.text)
    c.line(1036, 334, 1036, 364, stroke=t.text, sw=1.8)
    c.line(1036, 364, 1044, 364, stroke=t.text, sw=1.8)
    c.line(1130, 334, 1130, 364, stroke=t.text, sw=1.8)
    c.line(1122, 364, 1130, 364, stroke=t.text, sw=1.8)
    draw_sorted_rows(c, 991, 384, SORTED_DESC[:3], cell, t, selected=True)
    c.text(1020, 466, "保留", size=20, color=t.dark,
           anchor="middle", family="cn")
    c.text(1090, 466, "删除", size=20, color=t.neutral,
           anchor="middle", family="cn")

    # Enlarged, upward-shifted output group uses the available upper-right space.
    output_cell = 18
    output_y = 339
    output_axis_y = 384
    mask_x = 1202
    sparse_x = 1348

    arrow_h(c, 1184, output_axis_y, mask_x, t.dark, sw=3.8)
    mask_highlight = {
        (rr, cc): t.dark if MASK_ORIGINAL[rr][cc] else t.white
        for rr in range(5) for cc in range(7)
    }
    deleted = {(rr, cc) for rr in range(5) for cc in range(7)
               if not MASK_ORIGINAL[rr][cc]}
    matrix(c, mask_x, output_y, MASK_ORIGINAL, output_cell, t.dark, t.dark,
           highlight=mask_highlight, hatch=deleted, no_fill=True)
    matrix_label(c, mask_x + 63, 470, "B", t.dark)

    arrow_h(c, mask_x + 132, output_axis_y, sparse_x, t.dark, sw=3.8)
    sparse_highlight = {
        (rr, cc): (blend(t.dark, 0.22 + 0.62 * (1 - W_VALUES[rr][cc]))
                   if MASK_ORIGINAL[rr][cc] else t.white)
        for rr in range(5) for cc in range(7)
    }
    matrix(c, sparse_x, output_y, SPARSE_W, output_cell, t.dark, t.dark,
           highlight=sparse_highlight, hatch=deleted, no_fill=True)
    matrix_label(c, sparse_x + 63, 470, "W̃", t.dark)

    c.formula(1332, 520, [
        Seg("W̃", 27, bold=True), Seg("(l)", 17, 8), Seg(" = W", 27, bold=True),
        Seg("(l)", 17, 8),
    ], anchor="end", color=t.dark)
    c.circle(1345, 512, 8, fill=t.white, stroke=t.dark, sw=1.7)
    c.circle(1345, 512, 2.5, fill=t.dark)
    c.formula(1360, 520, [Seg("B", 27, bold=True), Seg("(l)", 17, 8)],
              color=t.dark)

    final_model_x = 1490
    arrow_h(c, sparse_x + 132, output_axis_y, final_model_x, t.dark, sw=3.5)
    c.text(1580, 210, "稀疏LLM", size=25, color=t.text,
           anchor="middle", family="cn")
    transformer_stack(
        c, final_model_x, 232, t, sparse=True, n=4,
        lw=168, lh=48, gap=10, x_offset=4
    )
    c.text(1570, 500, "全局稀疏率", size=21, color=t.text,
           anchor="end", family="cn")
    c.formula(1586, 500, [Seg("s", 24)], color=t.text)


def draw_submission(c, t: Theme):
    cell = 14

    # Compact calibration input and dense-model anchor.
    c.formula(64, 78, [
        Seg("D", 27), Seg("X", 16, -6), Seg(",  D", 27), Seg("G", 16, -6),
    ], anchor="middle", color=t.text)
    arrow_h(c, 40, 112, 96, t.dark, sw=3.0)
    selected_right = transformer_stack(
        c, 96, 88, t, selected_index=2, n=5, lw=84, lh=16, gap=3, x_offset=2
    )
    c.text(142, 216, "稠密LLM", size=23, color=t.text,
           anchor="middle", family="cn")

    # OSLA: activation statistics to a projected layer budget.
    c.text(248, 40, "层间预算", size=24, color=t.osla,
           family="cn", bold=True)
    c.formula(354, 40, [Seg("(OSLA)", 22, bold=True)], color=t.osla)
    c.polyline([(selected_right + 4, 134), (224, 134), (224, 120), (248, 120)],
               stroke=t.osla, sw=2.8)
    arrow_h(c, 248, 120, 272, t.osla, sw=2.8)

    outliers = [
        {(2, 1): blend(t.osla, 0.55)},
        {(0, 0): blend(t.osla, 0.52), (0, 4): blend(t.osla, 0.08),
         (2, 2): blend(t.osla, 0.30), (3, 4): blend(t.osla, 0.16)},
        {(0, 3): blend(t.osla, 0.18), (3, 0): blend(t.osla, 0.48)},
    ]
    for idx, x in enumerate([272, 376, 480]):
        values = [[0.13 + ((r * 4 + cc * 3 + idx * 2) % 7) * 0.045
                   for cc in range(5)] for r in range(4)]
        matrix(c, x, 88, values, cell, t.neutral, t.neutral,
               highlight=outliers[idx])
        c.formula(x + 35, 178, [
            Seg("X", 27, bold=True), Seg(["(1)", "(l)", "(L)"][idx], 17, 8),
        ], anchor="middle", color=t.text)

    # Minimal count/severity visual keys; c_l and v_l remain on the OSLA inputs.
    c.rect(248, 194, 14, 14, fill=blend(t.osla, 0.18), stroke=t.osla, sw=1)
    for ii, shade in enumerate([0.68, 0.38, 0.08]):
        c.rect(292 + ii * 18, 194, 14, 14, fill=blend(t.osla, shade),
               stroke=t.osla, sw=0.8)

    c.formula(624, 108, [Seg("c", 26, bold=True), Seg("l", 16, -6)],
              anchor="middle", color=t.osla)
    c.formula(624, 176, [Seg("v", 26, bold=True), Seg("l", 16, -6)],
              anchor="middle", color=t.osla)
    arrow_h(c, 640, 120, 672, t.osla, sw=3.0)
    arrow_h(c, 640, 188, 672, t.osla, sw=3.0)
    c.rect(672, 80, 152, 140, fill=t.white, stroke=t.osla, sw=2.5)
    c.text(748, 112, "OSLA", size=27, color=t.osla,
           anchor="middle", family="times")
    c.text(748, 144, "比例锚定", size=21, color=t.text,
           anchor="middle", family="cn")
    c.text(748, 174, "严重度修正", size=21, color=t.text,
           anchor="middle", family="cn")
    c.text(748, 204, "深度约束", size=21, color=t.text,
           anchor="middle", family="cn")
    c.text(870, 128, "预算投影", size=19, color=t.osla,
           anchor="middle", family="cn")
    arrow_h(c, 832, 144, 916, t.osla, sw=3.2)

    # The bar chart itself represents the non-uniform vector; only its symbol remains.
    c.formula(1036, 100, [Seg("s", 27, bold=True)],
              anchor="middle", color=t.osla)
    budget = [0.49, 0.57, 0.45, 0.61, 0.53, 0.66, 0.47, 0.59, 0.51]
    bar_x, base_y, selected_bar = 932, 216, 4
    for i, value in enumerate(budget):
        bh = 48 + (value - 0.45) * 240
        c.rect(bar_x + i * 24, base_y - bh, 16, bh,
               fill=t.osla if i == selected_bar else blend(t.osla, 0.63),
               stroke=t.osla, sw=2.2 if i == selected_bar else 1.2)
    c.line(bar_x - 4, base_y, bar_x + 9 * 24 - 4, base_y,
           stroke=t.neutral, sw=1.3)
    for idx, label in {0: "1", 2: "...", 4: "l", 6: "...", 8: "L"}.items():
        c.text(bar_x + idx * 24 + 8, 238, label, size=18, color=t.text,
               anchor="middle", family="times", italic=True)

    selector_x, selector_y, selector_w, selector_h = 900, 300, 272, 200
    selected_budget_x = bar_x + selected_bar * 24 + 8
    arrow_v(c, selected_budget_x, 252, selector_y, t.osla, sw=3.3)
    c.formula(selected_budget_x + 24, 282,
              [Seg("s", 24, bold=True), Seg("l", 16, -6)], color=t.osla)

    # GWS: W and G enter one compact scoring node, followed by row-wise ordering.
    c.text(248, 272, "层内评分", size=24, color=t.gws,
           family="cn", bold=True)
    c.formula(354, 272, [Seg("(GWS)", 22, bold=True)], color=t.gws)
    c.polyline([(selected_right + 4, 134), (208, 134), (208, 400), (272, 400)],
               stroke=t.neutral, sw=2.5)
    arrow_h(c, 272, 400, 288, t.neutral, sw=2.5)

    matrix(c, 288, 336, W_VALUES, cell, t.dark, t.dark)
    matrix_label(c, 337, 438, "W", t.text)
    matrix(c, 288, 472, G_VALUES, cell, t.gws, t.gws)
    matrix_label(c, 337, 574, "G", t.gws)

    c.polyline([(386, 371), (402, 371), (402, 386)], stroke=t.gws, sw=2.5)
    c.polyline([(386, 507), (410, 507), (410, 414)], stroke=t.gws, sw=2.5)
    arrow_h(c, 402, 386, 434, t.gws, sw=2.5)
    arrow_h(c, 410, 414, 434, t.gws, sw=2.5)
    c.rect(434, 372, 64, 56, fill=t.white, stroke=t.gws, sw=2.1)
    c.text(466, 406, "GWS", size=19, color=t.gws,
           anchor="middle", family="times")
    arrow_h(c, 498, 400, 516, t.gws, sw=3.0)

    matrix(c, 516, 360, M_VALUES, cell, t.gws, t.gws)
    matrix_label(c, 565, 472, "M", t.gws)
    c.formula(565, 516, [
        Seg("M", 27), Seg("ij", 17, -6), Seg("(l)", 17, 8), Seg(" = |W", 27),
        Seg("ij", 17, -6), Seg("(l)", 17, 8), Seg("|√G", 27),
        Seg("ij", 17, -6), Seg("(l)", 17, 8),
    ], anchor="middle", color=t.text)

    arrow_h(c, 622, 400, 638, t.gws, sw=3.0)
    c.text(687, 352, "逐行排序", size=19, color=t.gws,
           anchor="middle", family="cn")
    draw_sorted_rows(c, 638, 368, SORTED_DESC[:3], cell, t)
    arrow_h(c, 744, 400, selector_x, t.gws, sw=3.2)

    # Budget mask: one definition and three representative rows.
    c.rect(selector_x, selector_y, selector_w, selector_h,
           fill=t.white, stroke=t.dark, sw=2.8)
    c.text(selector_x + selector_w / 2, 334, "预算掩码", size=25,
           color=t.dark, anchor="middle", family="cn", bold=True)
    c.formula(944, 372, [
        Seg("q", 26), Seg("l,j", 17, -6), Seg(" =", 25),
    ], color=t.text)
    c.formula(1032, 372, [
        Seg("s", 25), Seg("l", 16, -6), Seg(" d", 25), Seg("l,j", 16, -6),
        Seg("in", 16, 7),
    ], color=t.text)
    c.line(1020, 346, 1020, 376, stroke=t.text, sw=1.8)
    c.line(1020, 376, 1028, 376, stroke=t.text, sw=1.8)
    c.line(1126, 346, 1126, 376, stroke=t.text, sw=1.8)
    c.line(1118, 376, 1126, 376, stroke=t.text, sw=1.8)
    draw_sorted_rows(c, 987, 398, SORTED_DESC[:3], cell, t, selected=True)
    c.text(1016, 478, "保留", size=19, color=t.dark,
           anchor="middle", family="cn")
    c.text(1080, 478, "删除", size=19, color=t.neutral,
           anchor="middle", family="cn")

    # Unified horizontal output chain.
    output_y, output_axis_y = 365, 400
    mask_x, sparse_x, model_x = 1200, 1326, 1452
    arrow_h(c, selector_x + selector_w + 4, output_axis_y, mask_x, t.dark, sw=3.0)

    mask_highlight = {
        (rr, cc): t.dark if MASK_ORIGINAL[rr][cc] else t.white
        for rr in range(5) for cc in range(7)
    }
    deleted = {(rr, cc) for rr in range(5) for cc in range(7)
               if not MASK_ORIGINAL[rr][cc]}
    matrix(c, mask_x, output_y, MASK_ORIGINAL, cell, t.dark, t.dark,
           highlight=mask_highlight, hatch=deleted, no_fill=True)
    matrix_label(c, mask_x + 49, 470, "B", t.dark)

    arrow_h(c, mask_x + 102, output_axis_y, sparse_x, t.dark, sw=3.0)
    sparse_highlight = {
        (rr, cc): (blend(t.dark, 0.22 + 0.62 * (1 - W_VALUES[rr][cc]))
                   if MASK_ORIGINAL[rr][cc] else t.white)
        for rr in range(5) for cc in range(7)
    }
    matrix(c, sparse_x, output_y, SPARSE_W, cell, t.dark, t.dark,
           highlight=sparse_highlight, hatch=deleted, no_fill=True)
    matrix_label(c, sparse_x + 49, 470, "W̃", t.dark)

    arrow_h(c, sparse_x + 102, output_axis_y, model_x, t.dark, sw=3.0)
    c.text(model_x + 24, 286, "稀疏LLM", size=22, color=t.text,
           family="cn")
    c.formula(model_x + 104, 286, [Seg("(s)", 20)], color=t.text)
    sparse_submission_stack(c, model_x, 316, t, n=4, lw=144, lh=36,
                            gap=8, x_offset=4)

    c.formula(1300, 530, [
        Seg("W̃", 26, bold=True), Seg("(l)", 17, 8), Seg(" = W", 26, bold=True),
        Seg("(l)", 17, 8),
    ], anchor="end", color=t.dark)
    c.circle(1313, 522, 7.5, fill=t.white, stroke=t.dark, sw=1.6)
    c.circle(1313, 522, 2.3, fill=t.dark)
    c.formula(1328, 530, [Seg("B", 26, bold=True), Seg("(l)", 17, 8)],
              color=t.dark)


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
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    svg = OUT / "GS_OWL_framework_submission.svg"
    pdf = OUT / "GS_OWL_framework_submission.pdf"
    png = OUT / "GS_OWL_framework_submission_600dpi.png"
    gray = OUT / "GS_OWL_framework_submission_grayscale.png"
    preview = OUT / "GS_OWL_framework_submission_preview_17cm.png"
    render_svg(svg, draw_submission, COLOR)
    render_pdf(pdf, draw_submission, COLOR)
    render_png(png, draw_submission, round(WIDTH_MM / 25.4 * 600), COLOR, dpi=600)
    render_png(gray, draw_submission, round(WIDTH_MM / 25.4 * 300), GRAY, dpi=300)
    render_png(preview, draw_submission, round(WIDTH_MM / 25.4 * 300), COLOR, dpi=300)
    for path in [svg, pdf, png, gray, preview]:
        print(path)


if __name__ == "__main__":
    main()
