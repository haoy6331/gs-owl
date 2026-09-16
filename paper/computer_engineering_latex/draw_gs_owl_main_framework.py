"""Draw the GS-OWL main framework as editable SVG, PDF, and PNG assets.

The geometry is authored at the journal insertion size (170 mm x 74 mm).
All labels and formulas are kept as text in the SVG and PDF outputs.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"

VIEW_W = 1600
VIEW_H = 640
WIDTH_MM = 170
HEIGHT_MM = 68

SIMSUN = Path(r"C:\Windows\Fonts\simsun.ttc")
TIMES = Path(r"C:\Windows\Fonts\times.ttf")
TIMES_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")
TIMES_ITALIC = Path(r"C:\Windows\Fonts\timesi.ttf")
TIMES_BOLD_ITALIC = Path(r"C:\Windows\Fonts\timesbi.ttf")


@dataclass(frozen=True)
class Theme:
    osla: str
    osla_fill: str
    gws: str
    gws_fill: str
    merge: str
    merge_fill: str
    text: str
    neutral: str
    light: str
    white: str = "#FFFFFF"


COLOR = Theme(
    osla="#405B6D",
    osla_fill="#F0F3F5",
    gws="#875F5F",
    gws_fill="#F7F2F2",
    merge="#30343B",
    merge_fill="#F6F7F8",
    text="#202124",
    neutral="#A8ADB2",
    light="#E6E8EA",
)

BW = Theme(
    osla="#303030",
    osla_fill="#F2F2F2",
    gws="#666666",
    gws_fill="#FAFAFA",
    merge="#171717",
    merge_fill="#F5F5F5",
    text="#111111",
    neutral="#8A8A8A",
    light="#DEDEDE",
)


@dataclass(frozen=True)
class Seg:
    text: str
    size: float = 23
    rise: float = 0
    family: str = "times"
    bold: bool = False
    italic: bool = True


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def lighten(value: str, amount: float) -> str:
    rgb = hex_rgb(value)
    out = tuple(round(c + (255 - c) * amount) for c in rgb)
    return "#%02X%02X%02X" % out


class SVG:
    def __init__(self, theme: Theme):
        self.t = theme
        self.parts: list[str] = []

    def rect(self, x, y, w, h, *, fill, stroke, sw=3, dash=None, rx=0):
        attrs = [
            f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"',
            f'fill="{fill}"', f'stroke="{stroke}"', f'stroke-width="{sw}"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if rx:
            attrs.extend([f'rx="{rx}"', f'ry="{rx}"'])
        self.parts.append(f"<rect {' '.join(attrs)}/>")

    def line(self, x1, y1, x2, y2, *, stroke, sw=3, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{dash_attr}/>'
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

    def text(self, x, y, value, *, size=24, fill=None, anchor="start", bold=False,
             family="cn", italic=False, letter_spacing=0):
        fam = "SimSun, 'Songti SC', serif" if family == "cn" else "'Times New Roman', Times, serif"
        weight = "700" if bold else "400"
        style = "italic" if italic else "normal"
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{fill or self.t.text}" text-anchor="{anchor}" '
            f'font-family="{fam}" font-size="{size}" font-weight="{weight}" '
            f'font-style="{style}" letter-spacing="{letter_spacing}">{html.escape(value)}</text>'
        )

    def formula(self, x, y, segments: Sequence[Seg], *, anchor="start", fill=None):
        total = sum(seg.size * 0.54 * len(seg.text) for seg in segments)
        if anchor == "middle":
            x -= total / 2
        elif anchor == "end":
            x -= total
        tspans = []
        for seg in segments:
            fam = "'Times New Roman', Times, serif" if seg.family == "times" else "SimSun, serif"
            weight = "700" if seg.bold else "400"
            style = "italic" if seg.italic else "normal"
            shift = "baseline" if seg.rise == 0 else ("super" if seg.rise > 0 else "sub")
            tspans.append(
                f'<tspan font-family="{fam}" font-size="{seg.size}" font-weight="{weight}" '
                f'font-style="{style}" baseline-shift="{shift}">{html.escape(seg.text)}</tspan>'
            )
        self.parts.append(
            f'<text x="{x:.1f}" y="{y}" fill="{fill or self.t.text}">{"".join(tspans)}</text>'
        )

    def render(self) -> str:
        defs = """
<defs>
  <marker id="arrow-osla" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="#405B6D"/>
  </marker>
  <marker id="arrow-gws" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="#875F5F"/>
  </marker>
  <marker id="arrow-bw1" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="#303030"/>
  </marker>
  <marker id="arrow-bw2" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="#666666"/>
  </marker>
  <marker id="arrow-merge" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="#30343B"/>
  </marker>
  <marker id="arrow-merge-bw" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="#171717"/>
  </marker>
</defs>"""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm" '
            f'viewBox="0 0 {VIEW_W} {VIEW_H}">\n{defs}\n'
            f'<rect width="{VIEW_W}" height="{VIEW_H}" fill="#FFFFFF"/>\n'
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )


class Raster:
    def __init__(self, theme: Theme, width_px: int):
        self.t = theme
        self.s = width_px / VIEW_W
        self.im = Image.new("RGB", (width_px, round(VIEW_H * self.s)), "white")
        self.d = ImageDraw.Draw(self.im)
        self.fonts = {}

    def _font(self, size, family="cn", bold=False, italic=False):
        key = (round(size * self.s), family, bold, italic)
        if key not in self.fonts:
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
            self.fonts[key] = ImageFont.truetype(str(path), max(8, key[0]))
        return self.fonts[key]

    def _xy(self, x, y):
        return round(x * self.s), round(y * self.s)

    def rect(self, x, y, w, h, *, fill, stroke, sw=3, dash=None, rx=0):
        box = [*self._xy(x, y), *self._xy(x + w, y + h)]
        width = max(1, round(sw * self.s))
        if fill != "none":
            self.d.rectangle(box, fill=fill)
        if dash:
            self._dashed_rect(x, y, w, h, stroke, width)
        else:
            self.d.rectangle(box, outline=stroke, width=width)

    def _dashed_rect(self, x, y, w, h, color, width):
        dash, gap = 12, 7
        for a, b, horizontal in ((x, x + w, True), (y, y + h, False)):
            pos = a
            while pos < b:
                end = min(pos + dash, b)
                if horizontal:
                    self.d.line([self._xy(pos, y), self._xy(end, y)], fill=color, width=width)
                    self.d.line([self._xy(pos, y + h), self._xy(end, y + h)], fill=color, width=width)
                else:
                    self.d.line([self._xy(x, pos), self._xy(x, end)], fill=color, width=width)
                    self.d.line([self._xy(x + w, pos), self._xy(x + w, end)], fill=color, width=width)
                pos += dash + gap

    def line(self, x1, y1, x2, y2, *, stroke, sw=3, dash=None):
        width = max(1, round(sw * self.s))
        if not dash:
            self.d.line([self._xy(x1, y1), self._xy(x2, y2)], fill=stroke, width=width)
            return
        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        pos = 0
        while pos < length:
            end = min(pos + 12, length)
            self.d.line(
                [self._xy(x1 + ux * pos, y1 + uy * pos), self._xy(x1 + ux * end, y1 + uy * end)],
                fill=stroke,
                width=width,
            )
            pos += 19

    def polygon(self, points, *, fill, stroke="none", sw=0):
        pts = [self._xy(x, y) for x, y in points]
        self.d.polygon(pts, fill=fill)
        if stroke != "none" and sw:
            self.d.line(pts + [pts[0]], fill=stroke, width=max(1, round(sw * self.s)))

    def circle(self, x, y, r, *, fill, stroke="none", sw=0):
        box = [*self._xy(x - r, y - r), *self._xy(x + r, y + r)]
        self.d.ellipse(box, fill=fill, outline=None if stroke == "none" else stroke,
                       width=max(1, round(sw * self.s)) if sw else 1)

    def text(self, x, y, value, *, size=24, fill=None, anchor="start", bold=False,
             family="cn", italic=False, letter_spacing=0):
        font = self._font(size, family, bold, italic)
        bbox = self.d.textbbox((0, 0), value, font=font)
        width = bbox[2] - bbox[0]
        px, py = self._xy(x, y)
        if anchor == "middle":
            px -= width // 2
        elif anchor == "end":
            px -= width
        self.d.text((px, py - round(size * self.s)), value, font=font, fill=fill or self.t.text)

    def formula(self, x, y, segments: Sequence[Seg], *, anchor="start", fill=None):
        widths = []
        for seg in segments:
            font = self._font(seg.size, seg.family, seg.bold, seg.italic)
            bbox = self.d.textbbox((0, 0), seg.text, font=font)
            widths.append(bbox[2] - bbox[0])
        total = sum(widths)
        px, py = self._xy(x, y)
        if anchor == "middle":
            px -= total // 2
        elif anchor == "end":
            px -= total
        for seg, width in zip(segments, widths):
            font = self._font(seg.size, seg.family, seg.bold, seg.italic)
            rise = seg.rise * self.s
            self.d.text((px, py - round(seg.size * self.s) - rise), seg.text, font=font,
                        fill=fill or self.t.text)
            px += width


class PDF:
    def __init__(self, path: Path, theme: Theme):
        self.t = theme
        self.width = WIDTH_MM / 25.4 * 72
        self.height = HEIGHT_MM / 25.4 * 72
        self.s = self.width / VIEW_W
        pdfmetrics.registerFont(TTFont("SimSunGS", str(SIMSUN), subfontIndex=0))
        pdfmetrics.registerFont(TTFont("TimesGS", str(TIMES)))
        pdfmetrics.registerFont(TTFont("TimesGS-Bold", str(TIMES_BOLD)))
        pdfmetrics.registerFont(TTFont("TimesGS-Italic", str(TIMES_ITALIC)))
        pdfmetrics.registerFont(TTFont("TimesGS-BoldItalic", str(TIMES_BOLD_ITALIC)))
        self.c = canvas.Canvas(str(path), pagesize=(self.width, self.height), pageCompression=1)
        self.c.setTitle("GS-OWL dual-scale sensitivity coordination framework")

    def _x(self, x): return x * self.s
    def _y(self, y): return self.height - y * self.s
    def _color(self, value):
        r, g, b = hex_rgb(value)
        return r / 255, g / 255, b / 255

    def rect(self, x, y, w, h, *, fill, stroke, sw=3, dash=None, rx=0):
        if fill != "none":
            self.c.setFillColorRGB(*self._color(fill))
        self.c.setStrokeColorRGB(*self._color(stroke))
        self.c.setLineWidth(sw * self.s)
        self.c.setDash([12 * self.s, 7 * self.s] if dash else [])
        self.c.rect(
            self._x(x), self._y(y + h), self._x(w), self._x(h),
            fill=fill != "none", stroke=1,
        )
        self.c.setDash([])

    def line(self, x1, y1, x2, y2, *, stroke, sw=3, dash=None):
        self.c.setStrokeColorRGB(*self._color(stroke))
        self.c.setLineWidth(sw * self.s)
        self.c.setLineCap(1)
        self.c.setDash([12 * self.s, 7 * self.s] if dash else [])
        self.c.line(self._x(x1), self._y(y1), self._x(x2), self._y(y2))
        self.c.setDash([])

    def polygon(self, points, *, fill, stroke="none", sw=0):
        p = self.c.beginPath()
        p.moveTo(self._x(points[0][0]), self._y(points[0][1]))
        for x, y in points[1:]:
            p.lineTo(self._x(x), self._y(y))
        p.close()
        self.c.setFillColorRGB(*self._color(fill))
        if stroke != "none":
            self.c.setStrokeColorRGB(*self._color(stroke))
            self.c.setLineWidth(sw * self.s)
        self.c.drawPath(p, fill=1, stroke=stroke != "none")

    def circle(self, x, y, r, *, fill, stroke="none", sw=0):
        self.c.setFillColorRGB(*self._color(fill))
        if stroke != "none":
            self.c.setStrokeColorRGB(*self._color(stroke))
            self.c.setLineWidth(sw * self.s)
        self.c.circle(self._x(x), self._y(y), self._x(r), fill=1, stroke=stroke != "none")

    def text(self, x, y, value, *, size=24, fill=None, anchor="start", bold=False,
             family="cn", italic=False, letter_spacing=0):
        if family == "cn":
            font = "SimSunGS"
        elif bold and italic:
            font = "TimesGS-BoldItalic"
        elif bold:
            font = "TimesGS-Bold"
        elif italic:
            font = "TimesGS-Italic"
        else:
            font = "TimesGS"
        fs = size * self.s
        self.c.setFont(font, fs)
        self.c.setFillColorRGB(*self._color(fill or self.t.text))
        width = pdfmetrics.stringWidth(value, font, fs)
        px = self._x(x)
        if anchor == "middle": px -= width / 2
        elif anchor == "end": px -= width
        self.c.drawString(px, self._y(y), value)

    def formula(self, x, y, segments: Sequence[Seg], *, anchor="start", fill=None):
        run_data = []
        total = 0
        for seg in segments:
            if seg.bold and seg.italic: font = "TimesGS-BoldItalic"
            elif seg.bold: font = "TimesGS-Bold"
            elif seg.italic: font = "TimesGS-Italic"
            else: font = "TimesGS"
            fs = seg.size * self.s
            width = pdfmetrics.stringWidth(seg.text, font, fs)
            run_data.append((seg, font, fs, width))
            total += width
        px = self._x(x)
        if anchor == "middle": px -= total / 2
        elif anchor == "end": px -= total
        for seg, font, fs, width in run_data:
            self.c.setFont(font, fs)
            self.c.setFillColorRGB(*self._color(fill or self.t.text))
            self.c.drawString(px, self._y(y) + seg.rise * self.s, seg.text)
            px += width

    def save(self):
        self.c.showPage()
        self.c.save()


def arrow(r, x1, y, x2, *, color, dashed=False):
    r.line(x1, y, x2 - 10, y, stroke=color, sw=3, dash="12 7" if dashed else None)
    r.polygon([(x2 - 11, y - 7), (x2, y), (x2 - 11, y + 7)], fill=color)


def lane_label(r, x, y, name, desc, color, *, square=False):
    if square:
        r.rect(x, y - 13, 24, 24, fill=color, stroke=color, sw=1)
    else:
        r.circle(x + 12, y - 1, 12, fill=color)
    r.text(x + 38, y + 7, name, size=29, fill=color, bold=True, family="times")
    r.text(x + 132, y + 7, desc, size=23, fill=r.t.text, family="cn")


def module(r, x, y, w, h, title, subtitle, color, fill, *, dashed=False):
    r.rect(x, y, w, h, fill=fill, stroke=color, sw=3, dash="12 7" if dashed else None)
    r.text(x + 18, y + 36, title, size=28, fill=color, bold=True, family="cn")
    if subtitle:
        r.text(x + 18, y + 64, subtitle, size=20, fill=r.t.text, family="cn")
    r.line(x + 18, y + 76, x + w - 18, y + 76, stroke=lighten(color, 0.58), sw=2,
           dash="10 6" if dashed else None)


def matrix(r, x, y, rows, cols, cell, values, color, *, binary=False, dashed=False):
    for rr in range(rows):
        for cc in range(cols):
            v = values[rr][cc]
            if binary:
                fill = color if v else r.t.white
            else:
                fill = lighten(color, 0.82 - 0.55 * v)
            r.rect(x + cc * cell, y + rr * cell, cell, cell, fill=fill,
                   stroke=r.t.white, sw=1.2)
    r.rect(x, y, cols * cell, rows * cell, fill="none", stroke=color, sw=2,
           dash="9 5" if dashed else None)


def draw_diagram(r, theme: Theme):
    top_y, bot_y, h = 105, 365, 160
    xs, ws = [180, 445, 720], [225, 235, 225]

    def compact_box(x, y, w, title, color, fill, dashed=False):
        r.rect(x, y, w, h, fill=fill, stroke=color, sw=3, dash="11 7" if dashed else None)
        r.rect(x, y, 8, h, fill=color, stroke=color, sw=0)
        r.text(x + 24, y + 37, title, size=25, fill=color, bold=True, family="cn")
        r.line(x + 24, y + 52, x + w - 18, y + 52, stroke=lighten(color, 0.64), sw=1.6,
               dash="9 6" if dashed else None)

    def path_tag(y, name, question, color, square=False):
        if square:
            r.rect(42, y + 38, 24, 24, fill=color, stroke=color, sw=1)
        else:
            r.circle(54, y + 50, 12, fill=color)
        r.text(82, y + 50, name, size=27, fill=color, bold=True, family="times")
        r.text(42, y + 88, question, size=19, fill=theme.text, family="cn")

    path_tag(top_y, "OSLA", "每层剪多少", theme.osla)
    path_tag(bot_y, "GWS", "层内剪哪些", theme.gws, square=True)

    compact_box(xs[0], top_y, ws[0], "异常统计", theme.osla, theme.osla_fill)
    compact_box(xs[1], top_y, ws[1], "OSLA层敏感性", theme.osla, theme.osla_fill)
    compact_box(xs[2], top_y, ws[2], "层预算", theme.osla, theme.osla_fill)
    compact_box(xs[0], bot_y, ws[0], "权重与梯度", theme.gws, theme.gws_fill, dashed=True)
    compact_box(xs[1], bot_y, ws[1], "GWS显著性", theme.gws, theme.gws_fill, dashed=True)
    compact_box(xs[2], bot_y, ws[2], "行内稳定排序", theme.gws, theme.gws_fill, dashed=True)

    # OSLA: element outliers -> anchored sensitivity -> layer budget.
    r.line(202, 192, 382, 192, stroke=theme.neutral, sw=1.7, dash="8 6")
    for px, py, outlier in [(210, 203, False), (232, 184, False), (254, 201, False),
                            (277, 172, True), (300, 187, False), (323, 154, True),
                            (346, 181, False), (371, 164, True)]:
        r.circle(px, py, 5 if not outlier else 7, fill=theme.osla if outlier else theme.white,
                 stroke=theme.osla, sw=1.7)
    r.formula(292, 232, [
        Seg("O", 18), Seg("l,j,r,c", 12, -5), Seg(" = |W", 18), Seg("l,j,r,c", 12, -5),
        Seg("|√X", 18), Seg("l,j,c", 12, -5),
    ], anchor="middle")
    r.formula(292, 254, [
        Seg("c", 20, bold=True), Seg("l", 13, -5), Seg(" ,  v", 20, bold=True), Seg("l", 13, -5),
    ], anchor="middle", fill=theme.osla)

    r.rect(472, 176, 72, 16, fill=lighten(theme.osla, 0.42), stroke=theme.osla, sw=1.3)
    r.rect(578, 176, 78, 16, fill=lighten(theme.osla, 0.64), stroke=theme.osla, sw=1.3)
    r.formula(562, 228, [
        Seg("d̃", 19, bold=True), Seg("l", 12, -5), Seg(" = (1−α)d", 19),
        Seg("c", 12, 8), Seg("l", 12, -5), Seg(" + αd", 19), Seg("v", 12, 8), Seg("l", 12, -5),
    ], anchor="middle")

    for i, bh in enumerate([34, 49, 27, 58, 42, 30]):
        r.rect(746 + i * 28, 226 - bh, 16, bh, fill=lighten(theme.osla, 0.36 + (i % 2) * 0.17),
               stroke=theme.osla, sw=1)
    r.line(740, 226, 918, 226, stroke=theme.neutral, sw=1.4)
    r.formula(833, 249, [
        Seg("C", 18), Seg("dep", 11, -5), Seg("  →  Π", 18), Seg("s", 11, -5),
        Seg("  →  {s", 18, bold=True), Seg("l", 11, -5), Seg("}", 18, bold=True),
    ], anchor="middle", fill=theme.osla)

    # GWS: weights and offline gradients -> saliency -> stable row order.
    vals_w = [[0.2, 0.75, 0.45], [0.88, 0.34, 0.62]]
    vals_g = [[0.72, 0.26, 0.9], [0.4, 0.82, 0.46]]
    matrix(r, 216, 444, 2, 3, 24, vals_w, theme.gws, dashed=True)
    matrix(r, 316, 444, 2, 3, 24, vals_g, theme.gws, dashed=True)
    r.formula(252, 432, [Seg("W", 20, bold=True), Seg("l,j", 13, -5)], anchor="middle", fill=theme.gws)
    r.formula(352, 432, [Seg("G", 20, bold=True), Seg("(l)", 13, 8)], anchor="middle", fill=theme.gws)
    r.text(292, 514, "离线聚合梯度", size=17, anchor="middle", family="cn")

    vals_m = [[0.16, 0.48, 0.78, 0.31], [0.88, 0.42, 0.64, 0.2]]
    matrix(r, 514, 438, 2, 4, 25, vals_m, theme.gws, dashed=True)
    r.formula(562, 513, [
        Seg("M", 18), Seg("ij", 12, -5), Seg("(l)", 12, 8), Seg(" = |W", 18), Seg("ij", 12, -5),
        Seg("(l)", 12, 8), Seg("|√G", 18), Seg("ij", 12, -5), Seg("(l)", 12, 8),
    ], anchor="middle")

    order = [0.10, 0.23, 0.38, 0.55, 0.72, 0.9]
    for i, v in enumerate(order):
        r.rect(748 + i * 29, 437, 27, 40, fill=lighten(theme.gws, 0.84 - 0.58 * v),
               stroke=theme.white, sw=1)
    r.rect(748, 437, 172, 40, fill="none", stroke=theme.gws, sw=1.8, dash="8 5")
    r.formula(834, 514, [
        Seg("rank", 19), Seg("↑", 12, 8), Seg("(M", 19), Seg("l,j,r,:", 12, -5), Seg(")", 19),
    ], anchor="middle", fill=theme.gws)

    # Horizontal branch arrows.
    arrow(r, 412, 185, 437, color=theme.osla)
    arrow(r, 687, 185, 712, color=theme.osla)
    arrow(r, 952, 185, 992, color=theme.osla)
    arrow(r, 412, 445, 437, color=theme.gws, dashed=True)
    arrow(r, 687, 445, 712, color=theme.gws, dashed=True)
    arrow(r, 952, 445, 992, color=theme.gws, dashed=True)

    # Compact merge node with one visual operation and two equations.
    mx, my, mw, mh = 1002, 105, 318, 420
    r.rect(mx, my, mw, mh, fill=theme.merge_fill, stroke=theme.merge, sw=4)
    r.text(mx + mw / 2, 145, "预算约束逐行剪枝", size=27, fill=theme.merge,
           anchor="middle", bold=True, family="cn")
    r.line(mx + 22, 160, mx + mw - 22, 160, stroke=theme.neutral, sw=1.7)

    r.circle(mx + 35, 190, 8, fill=theme.osla)
    r.text(mx + 52, 198, "层预算", size=18, family="cn")
    r.formula(mx + 148, 198, [Seg("s", 20, bold=True), Seg("l", 13, -5)], fill=theme.osla)
    r.rect(mx + 27, 414, 16, 16, fill=theme.gws, stroke=theme.gws, sw=1)
    r.text(mx + 52, 429, "连接排序", size=18, family="cn")
    r.formula(mx + 154, 429, [Seg("M", 20, bold=True), Seg("(l)", 13, 8)], fill=theme.gws)

    sal = [0.12, 0.76, 0.34, 0.9, 0.55]
    for i, v in enumerate(sal):
        r.rect(mx + 31 + i * 26, 242, 24, 38, fill=lighten(theme.gws, 0.84 - 0.58 * v),
               stroke=theme.white, sw=1)
    r.rect(mx + 31, 242, 128, 38, fill="none", stroke=theme.gws, sw=1.8, dash="8 5")
    arrow(r, mx + 168, 261, mx + 190, color=theme.merge)
    binary_mid = [0, 1, 0, 1, 1]
    for i, v in enumerate(binary_mid):
        r.rect(mx + 197 + i * 23, 242, 21, 38, fill=theme.merge if v else theme.white,
               stroke=theme.white, sw=1)
    r.rect(mx + 197, 242, 113, 38, fill="none", stroke=theme.merge, sw=1.8)
    r.formula(mx + 95, 302, [Seg("M", 18, bold=True), Seg("(l)", 12, 8)],
              anchor="middle", fill=theme.gws)
    r.formula(mx + 254, 302, [Seg("B", 18, bold=True), Seg("l,j,r,:", 12, -5)],
              anchor="middle", fill=theme.merge)

    # q_lj = floor(s_l d_in_lj), drawn with font-independent floor brackets.
    r.formula(mx + 92, 345, [Seg("q", 19), Seg("l,j", 12, -5), Seg(" =", 19)], anchor="middle")
    r.formula(mx + 192, 345, [
        Seg("s", 19), Seg("l", 12, -5), Seg(" d", 19), Seg("in", 12, 8), Seg("l,j", 12, -5),
    ], anchor="middle")
    r.line(mx + 134, 327, mx + 134, 351, stroke=theme.merge, sw=1.8)
    r.line(mx + 134, 351, mx + 142, 351, stroke=theme.merge, sw=1.8)
    r.line(mx + 253, 327, mx + 253, 351, stroke=theme.merge, sw=1.8)
    r.line(mx + 245, 351, mx + 253, 351, stroke=theme.merge, sw=1.8)
    r.formula(mx + mw / 2, 390, [
        Seg("B", 17), Seg("l,j,r,c", 11, -5), Seg(" = I[rank", 17), Seg("↑", 11, 8),
        Seg("(M", 17), Seg("l,j,r,c", 11, -5), Seg(") > q", 17), Seg("l,j", 11, -5), Seg("]", 17),
    ], anchor="middle")

    # Final output.
    ox, oy, ow, oh = 1380, 205, 185, 220
    arrow(r, mx + mw + 9, 315, ox - 10, color=theme.merge)
    r.rect(ox, oy, ow, oh, fill=theme.white, stroke=theme.merge, sw=4)
    r.text(ox + ow / 2, 241, "掩码与稀疏权重", size=21, fill=theme.merge,
           anchor="middle", bold=True, family="cn")
    r.line(ox + 16, 255, ox + ow - 16, 255, stroke=theme.neutral, sw=1.6)
    binary = [[1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]
    matrix(r, ox + 29, 276, 3, 4, 25, binary, theme.merge, binary=True)
    r.formula(ox + ow / 2, 375, [
        Seg("W̃", 18, bold=True), Seg("l,j", 12, -5),
    ], anchor="middle")
    r.formula(ox + ow / 2 - 11, 405, [
        Seg("= W", 18, bold=True), Seg("l,j", 12, -5),
    ], anchor="end")
    r.circle(ox + ow / 2, 399, 7, fill=theme.white, stroke=theme.merge, sw=1.6)
    r.circle(ox + ow / 2, 399, 2.2, fill=theme.merge)
    r.formula(ox + ow / 2 + 12, 405, [
        Seg("B", 18, bold=True), Seg("l,j", 12, -5),
    ], anchor="start")


def write_svg(path: Path, theme: Theme):
    renderer = SVG(theme)
    draw_diagram(renderer, theme)
    path.write_text(renderer.render(), encoding="utf-8")


def write_png(path: Path, theme: Theme, width_px: int, *, grayscale=False):
    renderer = Raster(theme, width_px)
    draw_diagram(renderer, theme)
    image = renderer.im
    if grayscale:
        image = image.convert("L").convert("RGB")
    image.save(path, dpi=(600, 600) if "600dpi" in path.name else (300, 300), optimize=True)


def write_pdf(path: Path, theme: Theme):
    renderer = PDF(path, theme)
    draw_diagram(renderer, theme)
    renderer.save()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / "GS_OWL_main_framework.svg"
    pdf_path = OUT_DIR / "GS_OWL_main_framework.pdf"
    png_600 = OUT_DIR / "GS_OWL_main_framework_600dpi.png"
    preview_color = OUT_DIR / "GS_OWL_main_framework_preview_color.png"
    preview_bw = OUT_DIR / "GS_OWL_main_framework_preview_bw.png"
    preview_17cm = OUT_DIR / "GS_OWL_main_framework_preview_17cm.png"

    write_svg(svg_path, COLOR)
    write_pdf(pdf_path, COLOR)
    write_png(png_600, COLOR, round(WIDTH_MM / 25.4 * 600))
    write_png(preview_color, COLOR, 2400)
    write_png(preview_bw, BW, 2400, grayscale=True)
    write_png(preview_17cm, COLOR, round(WIDTH_MM / 25.4 * 300))

    print(svg_path)
    print(pdf_path)
    print(png_600)
    print(preview_color)
    print(preview_bw)
    print(preview_17cm)


if __name__ == "__main__":
    main()
