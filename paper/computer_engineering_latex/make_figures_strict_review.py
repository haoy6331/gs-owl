import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"


COLOR = {
    "navy": "#1F3B5B",
    "blue": "#4D73A8",
    "teal": "#2A9D8F",
    "orange": "#D9824B",
    "red": "#B85C5C",
    "gray": "#7B8794",
    "dark": "#263238",
    "grid": "#D6DDE3",
    "blue_light": "#EAF1FA",
    "teal_light": "#E6F4F1",
    "orange_light": "#FCEDE3",
    "gray_light": "#F2F4F5",
}

# Palette shared with the approved GS-OWL framework figure.  It is deliberately
# restrained so the three manuscript figures read as one submission set.
SUBMISSION_COLOR = {
    "navy": "#405B6D",
    "blue": "#4C78A8",
    "teal": "#6B7B83",
    "orange": "#D27A3F",
    "red": "#875F5F",
    "gray": "#8A929A",
    "dark": "#30343B",
    "grid": "#D9DEE2",
    "blue_light": "#EDF3F7",
    "teal_light": "#EEF2F3",
    "orange_light": "#FCF1E9",
    "gray_light": "#F5F6F7",
}

GRAY = {
    "navy": "#202020",
    "blue": "#4A4A4A",
    "teal": "#696969",
    "orange": "#8A8A8A",
    "red": "#555555",
    "gray": "#A0A0A0",
    "dark": "#202020",
    "grid": "#D0D0D0",
    "blue_light": "#ECECEC",
    "teal_light": "#E4E4E4",
    "orange_light": "#F3F3F3",
    "gray_light": "#F7F7F7",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "STZhongsong",
                "Microsoft YaHei",
                "SimSun",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_17cm_preview(source_name: str, preview_name: str) -> None:
    source = FIG_DIR / f"{source_name}.png"
    target = FIG_DIR / f"{preview_name}.png"
    target_width = round(17 / 2.54 * 300)
    with Image.open(source) as image:
        target_height = round(image.height * target_width / image.width)
        image.resize((target_width, target_height), Image.Resampling.LANCZOS).save(
            target, dpi=(300, 300), optimize=True
        )


def panel_tag(ax: plt.Axes, label: str, color: str) -> None:
    ax.text(
        0.01,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        weight="bold",
        color=color,
    )


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
    title: str,
    lines: list[str],
    title_color: str,
    title_size: float = 8.3,
    body_size: float = 7.0,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + height - 0.055,
        title,
        ha="left",
        va="center",
        fontsize=title_size,
        color=title_color,
        weight="bold",
    )
    start_y = y + height - 0.135
    for index, line in enumerate(lines):
        ax.text(
            x + 0.018,
            start_y - index * 0.063,
            line,
            ha="left",
            va="center",
            fontsize=body_size,
            color=COLOR["dark"] if edgecolor != "black" else GRAY["dark"],
        )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def draw_score_matrix(
    ax: plt.Axes,
    origin: tuple[float, float],
    color: str,
    pruned_color: str,
) -> None:
    x0, y0 = origin
    values = np.array(
        [
            [0.95, 0.18, 0.61, 0.13, 0.76, 0.24],
            [0.32, 0.84, 0.17, 0.73, 0.22, 0.90],
            [0.69, 0.21, 0.88, 0.29, 0.56, 0.16],
        ]
    )
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            keep = values[row, col] >= 0.35
            ax.add_patch(
                Rectangle(
                    (x0 + col * 0.024, y0 + row * 0.035),
                    0.018,
                    0.025,
                    facecolor=color if keep else pruned_color,
                    edgecolor="white",
                    linewidth=0.4,
                )
            )


def make_overview_academic_previous(theme: dict[str, str], suffix: str) -> None:
    """Draw a restrained three-panel mechanism figure for the paper."""
    fig, ax = plt.subplots(figsize=(7.45, 2.92))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    dark = theme["dark"]
    muted = theme["gray"]
    line = theme["grid"]

    def thin_arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        color: str,
        connectionstyle: str = "arc3",
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8.5,
                linewidth=0.9,
                color=color,
                connectionstyle=connectionstyle,
                zorder=8,
            )
        )

    def panel_heading(x: float, width: float, tag: str, title: str, color: str) -> None:
        ax.text(
            x,
            0.948,
            f"({tag})",
            ha="left",
            va="center",
            fontsize=7.4,
            weight="bold",
            color=dark,
        )
        ax.text(
            x + 0.030,
            0.948,
            title,
            ha="left",
            va="center",
            fontsize=7.4,
            weight="bold",
            color=dark,
        )
        ax.plot([x, x + width], [0.906, 0.906], color=line, linewidth=0.65)
        ax.plot([x, x + 0.052], [0.906, 0.906], color=color, linewidth=1.8)

    def matrix(
        x: float,
        y: float,
        values: np.ndarray,
        color: str,
        width: float,
        height: float,
        binary: bool = False,
    ) -> None:
        rows, cols = values.shape
        cell_w = width / cols
        cell_h = height / rows
        vmax = max(float(np.abs(values).max()), 1e-8)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = float(values[row, col])
                if binary:
                    face = color if value > 0 else "white"
                    edge = "white" if value > 0 else line
                elif value == 0:
                    face = "white"
                    edge = line
                else:
                    face = mcolors.to_rgba(color, 0.18 + 0.72 * abs(value) / vmax)
                    edge = "white"
                ax.add_patch(
                    Rectangle(
                        (x + col * cell_w, y + (rows - 1 - row) * cell_h),
                        cell_w * 0.86,
                        cell_h * 0.82,
                        facecolor=face,
                        edgecolor=edge,
                        linewidth=0.35,
                        zorder=4,
                    )
                )

    # Panel structure
    panel_heading(0.020, 0.145, "a", "校准统计", theme["navy"])
    panel_heading(0.210, 0.415, "b", "层级预算：OSLA", theme["teal"])
    panel_heading(0.682, 0.298, "c", "连接剪枝：GWS", theme["orange"])
    ax.plot([0.185, 0.185], [0.08, 0.90], color=line, linewidth=0.65)
    ax.plot([0.655, 0.655], [0.08, 0.90], color=line, linewidth=0.65)

    # (a) Model and calibration statistics
    layer_y = np.linspace(0.620, 0.785, 6)
    widths = [0.088, 0.095, 0.102, 0.109, 0.116, 0.123]
    for idx, (yy, width) in enumerate(zip(layer_y, widths)):
        ax.add_patch(
            Rectangle(
                (0.091 - width / 2, yy),
                width,
                0.020,
                facecolor=mcolors.to_rgba(theme["blue"], 0.12 + idx * 0.07),
                edgecolor=theme["blue"],
                linewidth=0.55,
            )
        )
    ax.text(0.091, 0.572, r"$L$ 个 Transformer 块", ha="center", va="center",
            fontsize=6.3, color=dark)
    ax.text(0.091, 0.505, r"权重 $\mathbf{W}_l$", ha="center", va="center",
            fontsize=6.6, weight="bold", color=dark)
    ax.plot([0.041, 0.141], [0.465, 0.465], color=line, linewidth=0.6)
    ax.text(0.030, 0.400, r"$\mathcal{D}_X$", fontsize=6.7, weight="bold",
            color=theme["teal"], va="center")
    ax.text(0.061, 0.400, r"$X_j=\|\mathbf{x}_j\|_2^2$", fontsize=6.25,
            color=dark, va="center")
    ax.text(0.030, 0.288, r"$\mathcal{D}_G$", fontsize=6.7, weight="bold",
            color=theme["orange"], va="center")
    ax.text(0.061, 0.288, r"$G_{ij}=\|\{\nabla_{W_{ij}}\mathcal{L}_b\}_b\|_2$",
            fontsize=5.85, color=dark, va="center")
    ax.text(0.091, 0.164, "统计量按层读取并复用", fontsize=6.1,
            color=muted, ha="center")
    thin_arrow((0.159, 0.400), (0.215, 0.666), theme["teal"],
               connectionstyle="arc3,rad=-0.14")
    thin_arrow((0.159, 0.288), (0.692, 0.250), theme["orange"],
               connectionstyle="arc3,rad=0.06")

    # (b) OSLA layer-budget estimation
    step_y = 0.822
    ax.text(0.221, step_y, "异常尾部", fontsize=6.45, weight="bold", color=dark)
    ax.text(0.374, step_y, "计数锚定融合", fontsize=6.45, weight="bold", color=dark)
    ax.text(0.535, step_y, "预算约束", fontsize=6.45, weight="bold", color=dark)

    xs = np.linspace(0.0, 1.0, 48)
    ys = 0.125 * np.exp(-3.5 * xs) + 0.010
    base_x, base_y = 0.223, 0.568
    ax.plot(base_x + xs * 0.115, base_y + ys, color=theme["teal"], linewidth=1.15)
    ax.fill_between(
        base_x + xs * 0.115,
        base_y,
        base_y + ys,
        color=mcolors.to_rgba(theme["teal"], 0.10),
    )
    ax.plot([base_x, base_x + 0.115], [base_y, base_y], color=line, linewidth=0.55)
    tau_x = base_x + 0.081
    ax.plot([tau_x, tau_x], [base_y, base_y + 0.130], color=theme["red"],
            linewidth=0.85, linestyle=(0, (3, 2)))
    ax.text(tau_x + 0.003, base_y + 0.120, r"$\tau_l$", fontsize=5.9,
            color=theme["red"], va="bottom")
    ax.scatter(
        [base_x + 0.091, base_x + 0.101, base_x + 0.110],
        [base_y + 0.035, base_y + 0.058, base_y + 0.091],
        s=[7, 10, 13],
        color=theme["teal"],
        edgecolor="white",
        linewidth=0.35,
        zorder=6,
    )
    ax.text(0.223, 0.500, r"$O_l=|\mathbf{W}_l|\sqrt{\mathbf{X}_l}$",
            fontsize=6.2, color=dark)
    ax.text(0.223, 0.451, r"$p_l$：比例       $v_l$：超阈幅度",
            fontsize=5.95, color=muted)

    layer_x = np.linspace(0, 1, 10)
    count_profile = np.array([0.70, 0.63, 0.58, 0.48, 0.44, 0.46, 0.51, 0.57, 0.66, 0.73])
    severity_profile = count_profile + np.array([0.03, -0.02, 0.02, 0.07, -0.01, 0.02, 0.08, -0.02, 0.04, 0.06])
    px0, py0, pw, ph = 0.375, 0.568, 0.126, 0.153
    ax.plot([px0, px0 + pw], [py0, py0], color=line, linewidth=0.55)
    ax.plot(px0 + layer_x * pw, py0 + count_profile * ph, color=theme["navy"],
            linewidth=1.0, marker="o", markersize=1.8, zorder=5)
    ax.plot(px0 + layer_x * pw, py0 + severity_profile * ph, color=theme["teal"],
            linewidth=1.0, linestyle=(0, (3, 2)), marker="o", markersize=1.6, zorder=5)
    ax.text(px0, 0.540, r"$d_l^p$", fontsize=6.0, color=theme["navy"], weight="bold")
    ax.text(px0 + 0.037, 0.540, r"$d_l^v$", fontsize=6.0, color=theme["teal"], weight="bold")
    ax.text(0.375, 0.474, r"$\widetilde d_l=(1-\alpha)d_l^p+\alpha d_l^v$",
            fontsize=6.0, color=dark)
    ax.text(0.375, 0.430, "保留计数排序，补充尾部幅度", fontsize=5.9, color=muted)

    density = np.array([0.47, 0.51, 0.55, 0.53, 0.50, 0.46, 0.48, 0.52])
    bx0, by0, bw, gap = 0.536, 0.567, 0.0095, 0.0045
    for idx, value in enumerate(density):
        height = 0.155 * value / 0.60
        color = theme["teal"] if idx in (2, 3) else theme["blue"]
        ax.add_patch(Rectangle((bx0 + idx * (bw + gap), by0), bw, height,
                               facecolor=mcolors.to_rgba(color, 0.74),
                               edgecolor="white", linewidth=0.35, zorder=4))
    target_y = by0 + 0.155 * 0.50 / 0.60
    ax.plot([bx0 - 0.004, bx0 + 8 * (bw + gap) - gap],
            [target_y, target_y], color=theme["red"], linewidth=0.9,
            linestyle=(0, (3, 2)), zorder=6)
    ax.plot([bx0 - 0.004, bx0 + 8 * (bw + gap) - gap], [by0, by0],
            color=line, linewidth=0.55)
    ax.text(0.536, 0.500, r"$d_l\in[1-s-\lambda,\,1-s+\lambda]$",
            fontsize=5.85, color=dark)
    ax.text(0.536, 0.455, r"$L^{-1}\sum_l(1-d_l)=s$",
            fontsize=5.95, weight="bold", color=theme["navy"])
    ax.text(0.536, 0.411, r"$s_l=1-d_l$", fontsize=6.2,
            weight="bold", color=theme["teal"])
    thin_arrow((0.343, 0.641), (0.366, 0.641), muted)
    thin_arrow((0.507, 0.641), (0.529, 0.641), muted)

    # (c) Connection score, row-wise mask, and sparse output
    W = np.array(
        [[0.82, 0.23, 0.61, 0.37, 0.72],
         [0.31, 0.91, 0.18, 0.69, 0.28],
         [0.54, 0.41, 0.83, 0.21, 0.63],
         [0.24, 0.67, 0.36, 0.76, 0.17]]
    )
    G = np.array(
        [[0.22, 0.91, 0.48, 0.30, 0.61],
         [0.81, 0.20, 0.73, 0.42, 0.28],
         [0.33, 0.62, 0.24, 0.88, 0.54],
         [0.69, 0.31, 0.58, 0.26, 0.77]]
    )
    M = W * np.sqrt(G)
    threshold = np.sort(M, axis=1)[:, 2:3]
    B = (M >= threshold).astype(float)
    sparse_W = W * B

    ax.text(0.694, 0.825, r"$\mathbf{M}_l=|\mathbf{W}_l|\odot\sqrt{|\mathbf{G}_l|}$",
            fontsize=6.45, weight="bold", color=dark)
    ax.text(0.694, 0.773, "平方根压缩梯度动态范围", fontsize=5.9, color=muted)
    matrix(0.694, 0.550, M, theme["orange"], 0.076, 0.135)
    matrix(0.808, 0.550, B, theme["navy"], 0.076, 0.135, binary=True)
    matrix(0.916, 0.550, sparse_W, theme["blue"], 0.064, 0.135)
    ax.text(0.732, 0.507, r"显著性 $\mathbf{M}_l$", fontsize=5.9, color=dark, ha="center")
    ax.text(0.846, 0.507, r"掩码 $\mathbf{B}_l$", fontsize=5.9, color=dark, ha="center")
    ax.text(0.948, 0.507, r"$\widetilde{\mathbf{W}}_l$", fontsize=6.0, color=dark, ha="center")
    thin_arrow((0.775, 0.619), (0.801, 0.619), muted)
    thin_arrow((0.889, 0.619), (0.910, 0.619), muted)

    ax.text(0.694, 0.365, "逐输出行稳定排序", fontsize=6.15, weight="bold", color=dark)
    ax.text(0.694, 0.310, r"$q_l=\lfloor s_l d_{\mathrm{in}}\rfloor$",
            fontsize=6.45, color=theme["navy"])
    ax.text(0.694, 0.253, "删除每行最低分连接", fontsize=5.95, color=muted)
    ax.plot([0.694, 0.980], [0.195, 0.195], color=line, linewidth=0.6)
    ax.text(0.837, 0.127, "层间非均匀预算  ·  层内逐行定额",
            fontsize=6.15, color=dark, ha="center")

    # Route the layer budget along the panel boundary to avoid crossing formulas.
    ax.plot(
        [0.619, 0.670, 0.670, 0.800],
        [0.412, 0.412, 0.718, 0.718],
        color=theme["teal"],
        linewidth=0.9,
        zorder=7,
    )
    thin_arrow((0.800, 0.718), (0.836, 0.718), theme["teal"])
    ax.text(0.681, 0.736, r"层预算 $s_l$", fontsize=5.85, weight="bold",
            color=theme["teal"], ha="left")

    fig.subplots_adjust(left=0.008, right=0.992, bottom=0.025, top=0.99)
    save(fig, f"method_overview_academic_{suffix}")


def make_overview(theme: dict[str, str], suffix: str) -> None:
    """Draw GS-OWL with the sparse visual language used in recent pruning papers."""
    if suffix == "color":
        # OptiPrune-like publication palette: ink carries structure; pale blue
        # and peach are reserved for the two proposed sensitivity signals.
        theme = {
            **theme,
            "navy": "#4F7398",
            "blue": "#7294B6",
            "teal": "#4F7398",
            "orange": "#D48552",
            "red": "#C56F46",
            "gray": "#73777B",
            "dark": "#202124",
            "grid": "#B8BDC2",
            "blue_light": "#DFE8F2",
            "teal_light": "#DFE8F2",
            "orange_light": "#F4DDCD",
            "gray_light": "#F2F2F2",
        }
    fig, ax = plt.subplots(figsize=(7.35, 3.12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    dark = theme["dark"]
    muted = theme["gray"]
    line = theme["grid"]

    def arrow(start, end, color=muted, width=0.85, scale=7.5) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=scale,
                linewidth=width,
                color=color,
                zorder=10,
            )
        )

    def routed_arrow(points, color, width=0.95) -> None:
        ax.plot(
            [point[0] for point in points[:-1]],
            [point[1] for point in points[:-1]],
            color=color,
            linewidth=width,
            solid_capstyle="round",
            zorder=9,
        )
        arrow(points[-2], points[-1], color=color, width=width)

    def blend(color: str, weight: float) -> tuple[float, float, float]:
        rgb = np.array(mcolors.to_rgb(color))
        return tuple((1.0 - weight) * np.ones(3) + weight * rgb)

    def matrix(
        x: float,
        y: float,
        values: np.ndarray,
        color: str,
        width: float,
        height: float,
        binary: bool = False,
    ) -> None:
        rows, cols = values.shape
        cell_w = width / cols
        cell_h = height / rows
        vmax = max(float(np.abs(values).max()), 1e-8)
        for row in range(rows):
            for col in range(cols):
                value = float(values[row, col])
                if binary:
                    face = dark if value > 0 else "white"
                    edge = "white" if value > 0 else line
                elif value == 0:
                    face = "white"
                    edge = line
                else:
                    ratio = abs(value) / vmax
                    if ratio < 0.35:
                        tone = 0.24
                    elif ratio < 0.60:
                        tone = 0.43
                    elif ratio < 0.82:
                        tone = 0.63
                    else:
                        tone = 0.82
                    face = blend(color, tone)
                    edge = "white"
                ax.add_patch(
                    Rectangle(
                        (x + col * cell_w, y + (rows - 1 - row) * cell_h),
                        cell_w,
                        cell_h,
                        facecolor=face,
                        edgecolor=edge,
                        linewidth=0.40,
                        zorder=4,
                    )
                )
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor="none",
                edgecolor="#55585B",
                linewidth=0.55,
                zorder=5,
            )
        )

    def section_title(x, y, tag, title) -> None:
        ax.text(x, y, tag, color=dark, fontsize=7.7, weight="bold", va="center")
        ax.text(x + 0.037, y, title, color=dark, fontsize=7.7,
                weight="bold", va="center")

    def step_label(x, y, number, label, color) -> None:
        ax.text(x, y, number, fontsize=6.2, weight="bold",
                color=color, ha="left", va="center")
        ax.text(x + 0.015, y, label, fontsize=6.1, weight="bold",
                color=dark, ha="left", va="center")

    # Only hairline rules separate the three logical regions. The figure is
    # intentionally not enclosed in presentation-style cards.
    ax.plot([0.020, 0.690], [0.510, 0.510], color=line, linewidth=0.65)
    ax.plot([0.706, 0.706], [0.055, 0.945], color=line, linewidth=0.65)
    section_title(0.025, 0.935, "(a)", "异常严重度感知层分配（OSLA）")
    section_title(0.025, 0.465, "(b)", "梯度加权连接显著性（GWS）")
    section_title(0.730, 0.935, "(c)", "双尺度协同掩码")

    # Panel (a): outlier statistics -> anchored fusion -> constrained budget.
    step_label(0.040, 0.850, "1", "异常尾部统计", theme["teal"])
    rank = np.array(
        [0.98, 0.79, 0.66, 0.55, 0.46, 0.39, 0.33, 0.28,
         0.24, 0.21, 0.18, 0.16, 0.14, 0.125, 0.112, 0.102]
    )
    rx, ry, rw, rh = 0.043, 0.660, 0.145, 0.135
    xs = np.linspace(0, 1, rank.size)
    ax.plot([rx, rx + rw], [ry, ry], color=line, linewidth=0.55)
    ax.plot(rx + xs * rw, ry + rank * rh, color=theme["teal"],
            linewidth=1.05, marker="o", markersize=1.45)
    tau_y = ry + 0.40 * rh
    ax.plot([rx, rx + rw], [tau_y, tau_y], color=theme["red"],
            linewidth=0.75, linestyle=(0, (3, 2)))
    ax.fill_between(
        rx + xs[:6] * rw,
        tau_y,
        ry + rank[:6] * rh,
        color=theme["blue_light"],
    )
    ax.text(rx + rw - 0.004, tau_y + 0.008, r"$\tau_l$",
            fontsize=5.5, color=theme["red"], ha="right")
    ax.text(0.040, 0.595, r"$O_{l,u}=|W_{l,u}|\sqrt{X_{l,u}}$",
            fontsize=5.8, color=dark)

    step_label(0.238, 0.850, "2", "计数锚定融合", theme["teal"])
    layer_x = np.linspace(0, 1, 10)
    count_profile = np.array([0.64, 0.57, 0.50, 0.44, 0.46, 0.51, 0.48, 0.55, 0.61, 0.67])
    severity_profile = count_profile + np.array([0.01, 0.04, 0.02, 0.07, 0.03, -0.01, 0.06, 0.03, 0.05, 0.02])
    px, py, pw, ph = 0.242, 0.655, 0.154, 0.145
    ax.plot([px, px + pw], [py, py], color=line, linewidth=0.55)
    ax.plot(px + layer_x * pw, py + count_profile * ph,
            color=dark, linewidth=0.95, marker="o", markersize=1.5)
    ax.plot(px + layer_x * pw, py + severity_profile * ph,
            color=theme["teal"], linewidth=0.95, linestyle=(0, (3, 2)),
            marker="o", markersize=1.5)
    ax.text(px, 0.616, r"$p_l$  异常比例", fontsize=5.35,
            color=dark, ha="left")
    ax.text(px + pw, 0.616, r"$v_l$  超阈严重度", fontsize=5.35,
            color=dark, ha="right")
    ax.text(0.319, 0.570,
            r"$\widetilde d_l=(1-\alpha)d_l^c+\alpha d_l^v$",
            fontsize=5.45, color=dark, ha="center")

    step_label(0.447, 0.850, "3", "约束与预算回投", theme["teal"])
    density = np.array([0.48, 0.50, 0.54, 0.56, 0.53, 0.49, 0.46, 0.45, 0.47, 0.50, 0.52])
    dx, dy, dw, dh = 0.451, 0.650, 0.190, 0.150
    core_l, core_r = dx + dw * 0.25, dx + dw * 0.45
    tail_l, tail_r = dx + dw * 0.58, dx + dw * 0.90
    ax.add_patch(Rectangle(
        (core_l, dy), core_r - core_l, dh,
        facecolor=theme["blue_light"], edgecolor="none"))
    ax.add_patch(Rectangle(
        (tail_l, dy), tail_r - tail_l, dh,
        facecolor=theme["orange_light"], edgecolor="none"))
    ax.plot([dx, dx + dw], [dy, dy], color=line, linewidth=0.55)
    ax.plot(
        dx + np.linspace(0, 1, density.size) * dw,
        dy + (density - 0.40) / 0.20 * dh,
        color=theme["teal"], linewidth=1.05, marker="o", markersize=1.55,
    )
    target = dy + (0.50 - 0.40) / 0.20 * dh
    ax.plot([dx, dx + dw], [target, target], color=theme["gray"],
            linewidth=0.65, linestyle=(0, (3, 2)))
    ax.text((core_l + core_r) / 2, dy + dh + 0.008, r"$\mathcal{C}$",
            fontsize=5.4, color=theme["teal"], ha="center")
    ax.text((tail_l + tail_r) / 2, dy + dh + 0.008, r"$\mathcal{T}$",
            fontsize=5.4, color=theme["orange"], ha="center")
    ax.text(0.546, 0.600,
            r"$\mathbf{d}=\Pi_s(\mathcal{C}_{\rm dep}[\widetilde{\mathbf{d}}])$",
            fontsize=5.55, color=dark, ha="center")
    ax.text(0.546, 0.560, r"$\mathbf{s}=\mathbf{1}-\mathbf{d}$",
            fontsize=6.25, color=dark, weight="bold", ha="center")
    arrow((0.198, 0.710), (0.226, 0.710))
    arrow((0.408, 0.710), (0.438, 0.710))

    # Panel (b): weight and offline gradient form a robust connection score.
    W = np.array(
        [[0.82, 0.23, 0.61, 0.37],
         [0.31, 0.91, 0.18, 0.69],
         [0.54, 0.41, 0.83, 0.21]]
    )
    G = np.array(
        [[0.22, 0.91, 0.48, 0.30],
         [0.81, 0.20, 0.73, 0.42],
         [0.33, 0.62, 0.24, 0.88]]
    )
    M = W * np.sqrt(G)
    matrix(0.052, 0.205, W, theme["blue"], 0.115, 0.125)
    matrix(0.233, 0.205, np.sqrt(G), theme["orange"], 0.115, 0.125)
    matrix(0.452, 0.188, M, theme["orange"], 0.145, 0.158)
    ax.text(0.109, 0.360, r"权重幅值 $|\mathbf{W}_l|$", fontsize=5.8,
            weight="bold", color=dark, ha="center")
    ax.text(0.290, 0.360, r"聚合梯度 $\sqrt{|\mathbf{G}_l|}$", fontsize=5.8,
            weight="bold", color=dark, ha="center")
    ax.text(0.199, 0.266, r"$\odot$", fontsize=8.0, color=dark, ha="center")
    ax.text(0.525, 0.362, r"连接显著性 $\mathbf{M}_l$", fontsize=5.9,
            weight="bold", color=dark, ha="center")
    arrow((0.365, 0.266), (0.435, 0.266))
    ax.text(0.333, 0.142,
            r"$M_{l,ij}=|W_{l,ij}|\sqrt{|G_{l,ij}|}$",
            fontsize=6.05, color=dark, ha="center")
    ax.text(0.333, 0.095, "梯度长尾压缩，保留层内排序",
            fontsize=5.45, color=muted, ha="center")

    # Panel (c): the only coupling point is row-wise Bottom-K.
    op_x, op_y, op_w, op_h = 0.758, 0.650, 0.190, 0.125
    ax.add_patch(
        Rectangle(
            (op_x, op_y),
            op_w,
            op_h,
            facecolor="white",
            edgecolor=dark,
            linewidth=0.80,
        )
    )
    ax.text(op_x + op_w / 2, op_y + 0.084,
            r"$q_l=\lfloor s_l d_{\rm in}\rfloor$",
            fontsize=6.0, color=dark, ha="center", va="center")
    ax.text(op_x + op_w / 2, op_y + 0.039,
            r"$\operatorname{BottomK}_{\rm row}(\mathbf{M}_l,q_l)$",
            fontsize=5.75, color=dark, weight="bold", ha="center", va="center")
    ax.text(0.853, 0.610, "每个输出行使用相同层预算",
            fontsize=5.45, color=muted, ha="center")

    # Show the saliency matrix and its row-wise binary decision.
    matrix(0.758, 0.392, M, theme["orange"], 0.080, 0.130)
    threshold = np.sort(M, axis=1)[:, 1:2]
    B = (M >= threshold).astype(float)
    sparse_W = W * B
    matrix(0.875, 0.392, B, theme["navy"], 0.080, 0.130, binary=True)
    ax.text(0.798, 0.355, r"$\mathbf{M}_l$", fontsize=5.8,
            color=dark, weight="bold", ha="center")
    ax.text(0.915, 0.355, r"$\mathbf{B}_l$", fontsize=5.8,
            color=dark, weight="bold", ha="center")
    arrow((0.843, 0.457), (0.865, 0.457))

    matrix(0.821, 0.166, sparse_W, theme["blue"], 0.108, 0.145)
    ax.text(0.875, 0.122, r"$\widetilde{\mathbf{W}}_l"
            r"=\mathbf{W}_l\odot\mathbf{B}_l$",
            fontsize=6.25, color=dark, weight="bold", ha="center")
    arrow((0.915, 0.345), (0.875, 0.315))

    # Two orthogonal routes emphasize that the signals meet only at masking.
    routed_arrow(
        [(0.642, 0.575), (0.695, 0.575), (0.695, 0.735), (0.752, 0.735)],
        theme["teal"],
        width=0.95,
    )
    ax.text(0.666, 0.595, r"层预算 $\mathbf{s}$", fontsize=5.65,
            color=dark, weight="bold", ha="center")
    routed_arrow(
        [(0.603, 0.266), (0.682, 0.266), (0.682, 0.685), (0.752, 0.685)],
        theme["orange"],
        width=0.95,
    )
    ax.text(0.642, 0.286, r"显著性 $\mathbf{M}_l$", fontsize=5.65,
            color=dark, weight="bold", ha="center")

    fig.subplots_adjust(left=0.006, right=0.994, bottom=0.015, top=0.985)
    save(fig, f"method_overview_final_{suffix}")


def method_styles(theme: dict[str, str]) -> dict[str, tuple[str, str, float]]:
    return {
        "Wanda": (theme["gray"], "o", 1.25),
        "SparseGPT": (theme["blue"], "s", 1.35),
        "Pruner-Zero": (theme["orange"], "^", 1.35),
        "GS-OWL": (theme["teal"], "D", 2.15),
    }


def make_ppl_trends(theme: dict[str, str], suffix: str) -> None:
    data = pd.read_csv(DATA_DIR / "ppl_main_results.csv")
    models = [
        "LLaMA-1-7B",
        "LLaMA-1-13B",
        "LLaMA-1-30B",
        "LLaMA-2-7B",
        "LLaMA-2-13B",
    ]
    methods = ["Wanda", "SparseGPT", "Pruner-Zero", "GS-OWL"]
    sparsities = np.array([50, 60, 70])
    styles = method_styles(theme)

    fig = plt.figure(figsize=(7.15, 4.55))
    grid = fig.add_gridspec(2, 6)
    positions = [
        grid[0, 0:2],
        grid[0, 2:4],
        grid[0, 4:6],
        grid[1, 1:3],
        grid[1, 3:5],
    ]
    axes = []
    for position in positions:
        axes.append(fig.add_subplot(position, sharex=axes[0] if axes else None))
    for ax, model, tag in zip(axes, models, ["(a)", "(b)", "(c)", "(d)", "(e)"]):
        for method in methods:
            row = data[(data["model"] == model) & (data["method"] == method)].iloc[0]
            values = row[["sparsity_0.5", "sparsity_0.6", "sparsity_0.7"]].to_numpy(dtype=float)
            color, marker, width = styles[method]
            ax.plot(
                sparsities,
                values,
                label=method,
                color=color,
                marker=marker,
                markersize=4.2 if method != "GS-OWL" else 5.0,
                linewidth=width,
                markerfacecolor="white" if method != "GS-OWL" else color,
                markeredgewidth=0.9,
                zorder=4 if method == "GS-OWL" else 2,
            )
        ax.set_yscale("log")
        ax.set_ylim(4.5, 120)
        ax.set_yticks([5, 10, 20, 50, 100])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_xticks(sparsities)
        ax.grid(axis="y", which="both", color=theme["grid"], linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{tag} {model}", loc="left", color=theme["navy"], weight="bold", pad=7)
        ax.annotate(
            f"{data[(data['model'] == model) & (data['method'] == 'GS-OWL')]['sparsity_0.7'].iloc[0]:.2f}",
            (70, data[(data["model"] == model) & (data["method"] == "GS-OWL")]["sparsity_0.7"].iloc[0]),
            xytext=(-3, 7),
            textcoords="offset points",
            ha="right",
            fontsize=6.6,
            color=theme["teal"],
            weight="bold",
        )
    for ax in axes[3:]:
        ax.set_xlabel("目标稀疏率 / %")
    axes[0].set_ylabel("WikiText-2 PPL / ↓（对数坐标）")
    axes[3].set_ylabel("WikiText-2 PPL / ↓（对数坐标）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.15, w_pad=1.0)
    save(fig, f"main_ppl_trends_{suffix}")


def make_zero_shot_trends(theme: dict[str, str], suffix: str) -> None:
    data = pd.read_csv(DATA_DIR / "zero_shot_main_results.csv")
    models = [
        "LLaMA-1-7B",
        "LLaMA-1-13B",
        "LLaMA-1-30B",
        "LLaMA-2-7B",
        "LLaMA-2-13B",
    ]
    methods = ["Wanda", "SparseGPT", "GS-OWL"]
    sparsities = np.array([50, 60, 70])
    styles = method_styles(theme)

    fig = plt.figure(figsize=(7.15, 4.25))
    grid = fig.add_gridspec(2, 6)
    positions = [
        grid[0, 0:2],
        grid[0, 2:4],
        grid[0, 4:6],
        grid[1, 1:3],
        grid[1, 3:5],
    ]
    axes = []
    for position in positions:
        axes.append(
            fig.add_subplot(
                position,
                sharex=axes[0] if axes else None,
                sharey=axes[0] if axes else None,
            )
        )
    for ax, model, tag in zip(axes, models, ["(a)", "(b)", "(c)", "(d)", "(e)"]):
        for method in methods:
            row = data[(data["model"] == model) & (data["method"] == method)].iloc[0]
            values = row[["sparsity_0.5", "sparsity_0.6", "sparsity_0.7"]].to_numpy(dtype=float)
            color, marker, width = styles[method]
            ax.plot(
                sparsities,
                values,
                color=color,
                marker=marker,
                markersize=4.0 if method != "GS-OWL" else 4.8,
                linewidth=width,
                markerfacecolor="white" if method != "GS-OWL" else color,
                markeredgewidth=0.9,
                label=method,
            )
        ax.set_title(f"{tag} {model.replace('LLaMA-', 'L')}", loc="left", fontsize=8.6, color=theme["navy"], weight="bold", pad=7)
        ax.set_xticks(sparsities)
        ax.set_xlabel("稀疏率 / %")
        ax.set_ylim(30, 66)
        ax.grid(axis="y", color=theme["grid"], linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        gs = data[(data["model"] == model) & (data["method"] == "GS-OWL")]["sparsity_0.7"].iloc[0]
        ax.annotate(
            f"{gs:.1f}",
            (70, gs),
            xytext=(-2, 6),
            textcoords="offset points",
            ha="right",
            fontsize=6.5,
            color=theme["teal"],
            weight="bold",
        )
    axes[0].set_ylabel("七任务平均准确率 / % ↑")
    axes[3].set_ylabel("七任务平均准确率 / % ↑")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.0, w_pad=0.9)
    save(fig, f"zero_shot_trends_{suffix}")


def make_layer_allocation(theme: dict[str, str], suffix: str) -> None:
    data = pd.read_csv(DATA_DIR / "layer_allocation_data.csv")
    columns = [
        ("uniform_sparsity", "均匀分配", theme["gray"], "--", None),
        ("original_owl_sparsity", "原始 OWL", theme["navy"], (0, (4, 2)), "o"),
        ("owl_v9_sparsity", "OSLA", theme["orange"], "-", "s"),
    ]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 4.05),
        sharex="col",
        gridspec_kw={"height_ratios": [2.55, 1.45], "hspace": 0.08, "wspace": 0.14},
    )
    for column_index, (model_key, title, tag) in enumerate(zip(
        ["llama1_7b", "llama2_7b"],
        ["LLaMA-1-7B", "LLaMA-2-7B"],
        ["(a)", "(b)"],
    )):
        subset = data[data["model_key"] == model_key]
        allocation_ax = axes[0, column_index]
        delta_ax = axes[1, column_index]
        for ax in (allocation_ax, delta_ax):
            ax.axvspan(3.5, 10.5, color=theme["blue_light"], zorder=0)
            ax.axvspan(16.5, 30.5, color=theme["orange_light"], zorder=0)
        for column, label, color, line_style, marker in columns:
            allocation_ax.plot(
                subset["layer"],
                100 * subset[column],
                color=color,
                linewidth=1.85 if column == "owl_v9_sparsity" else 1.25,
                linestyle=line_style,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=0.8,
                markevery=4,
                markersize=3.2,
                label=label,
                zorder=4 if column == "owl_v9_sparsity" else 3,
            )
        allocation_ax.set_title(
            f"{tag} {title}", loc="left", weight="bold", color=theme["navy"], pad=7
        )
        allocation_ax.set_xlim(-0.5, 31.5)
        allocation_ax.set_ylim(47, 70)
        allocation_ax.set_yticks([50, 55, 60, 65, 70])
        allocation_ax.grid(axis="y", color=theme["grid"], linewidth=0.5)
        allocation_ax.spines[["top", "right"]].set_visible(False)
        allocation_ax.tick_params(axis="x", labelbottom=False)
        allocation_ax.text(
            7,
            0.95,
            "核心保护区",
            transform=allocation_ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.4,
            color=theme["blue"],
        )
        allocation_ax.text(
            23,
            0.95,
            "中后段上限区",
            transform=allocation_ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.4,
            color=theme["orange"],
        )

        delta = 100 * (
            subset["owl_v9_sparsity"].to_numpy()
            - subset["original_owl_sparsity"].to_numpy()
        )
        bar_colors = np.where(delta <= 0, theme["blue"], theme["orange"])
        delta_ax.bar(
            subset["layer"],
            delta,
            width=0.72,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        delta_ax.axhline(0, color=theme["dark"], linewidth=0.65, zorder=4)
        delta_ax.set_ylim(-1.05, 0.35)
        delta_ax.set_yticks([-1.0, -0.5, 0.0])
        delta_ax.set_xticks(np.arange(0, 32, 5))
        delta_ax.set_xlabel("Transformer 层索引（0起始）")
        delta_ax.grid(axis="y", color=theme["grid"], linewidth=0.45)
        delta_ax.spines[["top", "right"]].set_visible(False)
        delta_ax.text(
            0.985,
            0.10,
            rf"最大 $|\Delta|$={np.max(np.abs(delta)):.1f} 个百分点",
            transform=delta_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.3,
            color=theme["dark"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
        )

    axes[0, 0].set_ylabel("层稀疏率 / %")
    axes[1, 0].set_ylabel("OSLA$-$OWL\n/ 百分点")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.text(
        0.5,
        0.005,
        "差值小于0表示OSLA降低该层稀疏率（保护），大于0表示增加剪枝以回投全局预算。",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=theme["dark"],
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.15, top=0.89)
    save(fig, f"layer_allocation_{suffix}")


def make_parameter_sensitivity(theme: dict[str, str], suffix: str) -> None:
    data = pd.read_csv(DATA_DIR / "parameter_sensitivity_results.csv")
    for column in ["target_sparsity", "ppl_test", "Hyper_m", "Lamda", "Owl_alpha"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()

    sparsities = [0.5, 0.6, 0.7]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), sharey=True)
    cmap = (
        mcolors.LinearSegmentedColormap.from_list(
            "paper_heatmap",
            [theme["blue_light"], theme["teal"], theme["navy"]],
        )
        if suffix == "color"
        else plt.get_cmap("Greys")
    )
    image = None
    for ax, sparsity, tag in zip(axes, sparsities, ["(a)", "(b)", "(c)"]):
        subset = data[np.isclose(data["target_sparsity"], sparsity)]
        grouped = (
            subset.groupby(["Hyper_m", "Lamda"], as_index=False)["ppl_test"]
            .min()
            .sort_values(["Hyper_m", "Lamda"])
        )
        pivot = grouped.pivot(index="Hyper_m", columns="Lamda", values="ppl_test")
        best = np.nanmin(pivot.to_numpy())
        relative = 100 * (pivot.to_numpy() / best - 1)
        clipped = np.clip(relative, 0, 25)
        image = ax.imshow(clipped, aspect="auto", cmap=cmap, vmin=0, vmax=25, origin="lower")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{x:g}" for x in pivot.index])
        ax.set_title(f"{tag} 稀疏率 {int(100*sparsity)}%", loc="left", color=theme["navy"], weight="bold")
        ax.set_xlabel(r"$\lambda$")
        best_position = np.unravel_index(np.nanargmin(relative), relative.shape)
        ax.scatter(
            best_position[1],
            best_position[0],
            marker="*",
            s=52,
            color=theme["orange"] if suffix == "color" else "black",
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
        )
        ax.text(
            0.98,
            0.04,
            f"最优 PPL={best:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.7,
            color=theme["dark"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2),
        )
    axes[0].set_ylabel(r"阈值倍数 $H_m$")
    cbar = fig.colorbar(image, ax=axes, location="right", fraction=0.025, pad=0.025)
    cbar.set_label("相对最优 PPL 退化 / %（上限25%）", fontsize=7.4)
    cbar.ax.tick_params(labelsize=7.0)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.22, top=0.88, wspace=0.16)
    save(fig, f"parameter_sensitivity_{suffix}")


def make_pllama_domain_mcq(theme: dict[str, str], suffix: str) -> None:
    data = pd.read_csv(DATA_DIR / "pllama_domain_mcq_results.csv")
    data["accuracy"] = pd.to_numeric(data["accuracy"], errors="raise") * 100
    data["sparsity"] = pd.to_numeric(data["sparsity"], errors="coerce")

    models = ["PLLaMA-7B", "PLLaMA-13B"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), sharey=True)
    is_color = suffix.endswith("color")
    bar_color = theme["blue"] if is_color else theme["teal"]

    for ax, model, tag in zip(axes, models, ["(a)", "(b)"]):
        subset = data[data["model"] == model]
        dense = subset[subset["method"] == "Dense"].iloc[0]
        pruned = subset[subset["method"] == "GS-OWL"].sort_values("sparsity")
        x = np.arange(len(pruned))
        accuracy = pruned["accuracy"].to_numpy()
        dense_accuracy = float(dense["accuracy"])

        bars = ax.bar(
            x,
            accuracy,
            width=0.58,
            color=bar_color,
            edgecolor=theme["dark"],
            linewidth=0.55,
            zorder=3,
        )
        ax.axhline(
            dense_accuracy,
            color=theme["dark"],
            linewidth=1.1,
            linestyle=(0, (4, 2.5)),
            zorder=4,
        )
        ax.text(
            0.98,
            0.95,
            f"Dense {dense_accuracy:.2f}%  ({int(dense['correct'])}/565)",
            ha="right",
            va="top",
            fontsize=8.0,
            color=theme["dark"],
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.0),
        )

        for bar, (_, row) in zip(bars, pruned.iterrows()):
            value = float(row["accuracy"])
            delta = value - dense_accuracy
            near_reference_above = 0 < delta < 5.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.55 if near_reference_above else value - 1.1,
                f"{value:.2f}%\n{delta:+.2f} pp",
                ha="center",
                va="bottom" if near_reference_above else "top",
                fontsize=8.1,
                color=(
                    theme["dark"]
                    if near_reference_above or not is_color
                    else "white"
                ),
                weight="bold",
                linespacing=1.08,
                bbox=dict(
                    facecolor="white" if near_reference_above else bar_color,
                    edgecolor="none",
                    alpha=0.96,
                    pad=0.45,
                ),
                zorder=5,
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                1.0,
                f"{int(row['correct'])}/565",
                ha="center",
                va="bottom",
                fontsize=7.6,
                color="white" if is_color else theme["dark"],
                weight="bold",
            )

        ax.set_title(f"{tag} {model}", loc="left", color=theme["navy"], weight="bold", pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(100 * value)}%" for value in pruned["sparsity"]])
        ax.set_xlabel("目标稀疏率")
        ax.set_ylim(0, 48)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.grid(axis="y", color=theme["grid"], linewidth=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("植物科学选择题准确率 / %")
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=bar_color, edgecolor=theme["dark"], linewidth=0.55),
        Line2D([0], [0], color=theme["dark"], linewidth=1.1, linestyle=(0, (4, 2.5))),
    ]
    fig.legend(
        legend_handles,
        ["GS-OWL 候选 oracle 上界", "Dense"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.81, wspace=0.17)
    save(fig, f"pllama_domain_mcq_{suffix}")


def make_pllama_domain_mcq_fixed_pending(theme: dict[str, str], suffix: str) -> None:
    """Render the fixed-mask PLLaMA result slots without reusing oracle values."""
    data = pd.read_csv(DATA_DIR / "pllama_domain_mcq_results.csv")
    data["accuracy"] = pd.to_numeric(data["accuracy"], errors="raise") * 100

    models = ["PLLaMA-7B", "PLLaMA-13B"]
    # Keep the pending-result panel shallow enough to remain attached to its
    # bilingual caption in the journal's full-width Word section.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.30))
    is_color = suffix.endswith("color")
    pending_face = theme["orange_light"] if is_color else theme["gray_light"]
    pending_edge = theme["orange"] if is_color else theme["gray"]
    dense_face = theme["blue_light"] if is_color else "#E8E8E8"

    for ax, model, tag in zip(axes, models, ["(a)", "(b)"]):
        dense = data[(data["model"] == model) & (data["method"] == "Dense")].iloc[0]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(
            0.02,
            0.96,
            f"{tag} {model}",
            ha="left",
            va="top",
            fontsize=9.2,
            weight="bold",
            color=theme["navy"],
        )

        ax.add_patch(
            Rectangle(
                (0.04, 0.69),
                0.92,
                0.16,
                facecolor=dense_face,
                edgecolor=theme["dark"],
                linewidth=0.65,
            )
        )
        ax.text(0.08, 0.77, "Dense", ha="left", va="center", fontsize=8.2, weight="bold")
        ax.text(
            0.92,
            0.77,
            f"{float(dense['accuracy']):.2f}%  ({int(dense['correct'])}/565)",
            ha="right",
            va="center",
            fontsize=8.2,
            color=theme["dark"],
        )

        for index, sparsity in enumerate((50, 60, 70)):
            x = 0.04 + index * 0.31
            ax.add_patch(
                Rectangle(
                    (x, 0.17),
                    0.28,
                    0.39,
                    facecolor=pending_face,
                    edgecolor=pending_edge,
                    linewidth=0.75,
                    hatch="///" if not is_color else None,
                )
            )
            ax.text(
                x + 0.14,
                0.49,
                f"稀疏率 {sparsity}%",
                ha="center",
                va="center",
                fontsize=7.6,
                weight="bold",
                color=theme["dark"],
            )
            ax.plot(
                [x + 0.04, x + 0.24],
                [0.43, 0.43],
                color=pending_edge,
                linewidth=0.55,
            )
            ax.text(
                x + 0.14,
                0.35,
                "准确率待补",
                ha="center",
                va="center",
                fontsize=7.5,
                color=theme["dark"],
            )
            ax.text(
                x + 0.14,
                0.24,
                "Mask ID待登记",
                ha="center",
                va="center",
                fontsize=7.0,
                color=theme["gray"],
            )

        ax.text(
            0.50,
            0.075,
            "固定参数 · 随机种子0 · 每档单一掩码",
            ha="center",
            va="center",
            fontsize=7.3,
            color=theme["dark"],
        )

    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.055, top=0.96, wspace=0.12)
    save(fig, f"pllama_domain_mcq_fixed_pending_{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["all", "overview", "submission-color", "strict-review", "fixed-protocol-pending"],
        default="all",
        help="Generate all paper figures or only the method overview.",
    )
    args = parser.parse_args()
    configure()
    if args.only == "submission-color":
        make_layer_allocation(SUBMISSION_COLOR, "submission_color")
        make_pllama_domain_mcq(SUBMISSION_COLOR, "submission_color")
        print(FIG_DIR)
        return
    if args.only == "strict-review":
        make_layer_allocation(SUBMISSION_COLOR, "strict_review_color")
        make_layer_allocation(GRAY, "strict_review_bw")
        make_pllama_domain_mcq(SUBMISSION_COLOR, "strict_review_color")
        make_pllama_domain_mcq(GRAY, "strict_review_bw")
        save_17cm_preview(
            "layer_allocation_strict_review_color",
            "layer_allocation_strict_review_preview_17cm",
        )
        save_17cm_preview(
            "pllama_domain_mcq_strict_review_color",
            "pllama_domain_mcq_strict_review_preview_17cm",
        )
        print(FIG_DIR)
        return
    if args.only == "fixed-protocol-pending":
        make_pllama_domain_mcq_fixed_pending(SUBMISSION_COLOR, "color")
        make_pllama_domain_mcq_fixed_pending(GRAY, "bw")
        save_17cm_preview(
            "pllama_domain_mcq_fixed_pending_color",
            "pllama_domain_mcq_fixed_pending_preview_17cm",
        )
        print(FIG_DIR)
        return
    for theme, suffix in [(COLOR, "color"), (GRAY, "bw")]:
        make_overview(theme, suffix)
        if args.only == "all":
            make_ppl_trends(theme, suffix)
            make_zero_shot_trends(theme, suffix)
            make_layer_allocation(theme, suffix)
            make_parameter_sensitivity(theme, suffix)
            make_pllama_domain_mcq(theme, suffix)
    print(FIG_DIR)


if __name__ == "__main__":
    main()
