import argparse
from pathlib import Path


FONT_LARGE = 13.6
FONT_SMALL = 9.2
FONT_REGULAR = None
FONT_BOLD = None


COLORS = {
    "ink": "#202124",
    "muted": "#626970",
    "line": "#C9CED3",
    "grid": "#E4E8EB",
    "blue": "#2F6FA5",
    "blue_dark": "#214F77",
    "blue_fill": "#EAF2F8",
    "orange": "#C9622B",
    "orange_dark": "#8E411F",
    "orange_fill": "#FAEEE7",
    "teal": "#2A7F73",
    "teal_dark": "#1D5C54",
    "teal_fill": "#E9F4F2",
    "neutral_fill": "#F7F8F9",
    "white": "#FFFFFF",
}


def load_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib and numpy are required. Install them with: pip install matplotlib numpy"
        ) from exc
    return plt, np, FontProperties, FancyArrowPatch, Polygon, Rectangle


def configure_fonts(FontProperties):
    global FONT_REGULAR, FONT_BOLD
    regular_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    regular_path = next((path for path in regular_candidates if path.exists()), None)
    bold_path = next((path for path in bold_candidates if path.exists()), regular_path)
    if regular_path is None:
        raise RuntimeError("No Chinese font found. Install Noto Sans CJK SC or Microsoft YaHei.")
    FONT_REGULAR = FontProperties(fname=str(regular_path))
    FONT_BOLD = FontProperties(fname=str(bold_path), weight="bold")


def add_text(ax, x, y, value, *, large=False, bold=False, color="ink", ha="center",
             va="center", zorder=10, rotation=0):
    return ax.text(
        x,
        y,
        value,
        fontsize=FONT_LARGE if large else FONT_SMALL,
        fontproperties=FONT_BOLD if bold else FONT_REGULAR,
        color=COLORS.get(color, color),
        ha=ha,
        va=va,
        rotation=rotation,
        linespacing=1.18,
        zorder=zorder,
    )


def add_arrow(ax, FancyArrowPatch, start, end, *, color="line", width=1.35,
              mutation=12, connectionstyle="arc3,rad=0"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=width,
        color=COLORS.get(color, color),
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=8,
    )
    ax.add_patch(patch)
    return patch


def add_panel_heading(ax, Rectangle, x0, x1, y, marker, title, color):
    add_text(ax, x0, y, marker, large=True, bold=True, ha="left")
    add_text(ax, x0 + 3.1, y, title, large=True, bold=True, ha="left")
    ax.add_patch(Rectangle(
        (x0, y - 2.0), x1 - x0, 0.25,
        facecolor=COLORS[color], edgecolor="none", zorder=2,
    ))


def hex_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))


def mix_with_white(np, color, amount):
    rgb = np.array(hex_rgb(color))
    return tuple(1.0 - amount * (1.0 - rgb))


def draw_matrix(ax, Rectangle, np, x, y, width, height, values, accent,
                zero_mask=None, highlight_mask=None):
    values = np.asarray(values, dtype=float)
    rows, columns = values.shape
    gap = min(width / columns, height / rows) * 0.075
    cell_width = (width - gap * (columns - 1)) / columns
    cell_height = (height - gap * (rows - 1)) / rows
    low = float(values.min())
    span = max(float(values.max()) - low, 1e-9)
    for row in range(rows):
        for column in range(columns):
            xx = x + column * (cell_width + gap)
            yy = y + (rows - 1 - row) * (cell_height + gap)
            is_zero = zero_mask is not None and bool(zero_mask[row, column])
            highlighted = highlight_mask is not None and bool(highlight_mask[row, column])
            if is_zero:
                face = COLORS["white"]
                edge = COLORS["line"]
                linewidth = 0.72
            else:
                normalized = (float(values[row, column]) - low) / span
                face = mix_with_white(np, COLORS[accent], 0.18 + normalized * 0.72)
                edge = COLORS[accent] if highlighted else COLORS["line"]
                linewidth = 1.2 if highlighted else 0.62
            ax.add_patch(Rectangle(
                (xx, yy), cell_width, cell_height,
                facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=5,
            ))


def draw_dense_model(ax, Rectangle, np, x, y, width, height):
    layers = 7
    row_height = 1.35
    gap = (height - layers * row_height) / (layers - 1)
    for layer in range(layers):
        yy = y + layer * (row_height + gap)
        ax.add_patch(Rectangle(
            (x + 0.10 * (layer % 2), yy), width, row_height,
            facecolor=mix_with_white(np, COLORS["blue"], 0.32 + 0.05 * layer),
            edgecolor=COLORS["blue_dark"], linewidth=0.72, zorder=4,
        ))
        for column in range(7):
            xx = x + 0.52 + column * (width - 1.05) / 7
            ax.add_patch(Rectangle(
                (xx, yy + 0.34), (width - 1.35) / 8.2, 0.67,
                facecolor=COLORS["white"], edgecolor="none", alpha=0.72, zorder=5,
            ))


def draw_activation(ax, Rectangle, np, x, y, width, height):
    values = (0.34, 0.55, 0.80, 0.46, 0.68, 0.29, 0.62, 0.40)
    for index, value in enumerate(values):
        cell_width = width / 9.4
        xx = x + index * width / 8
        ax.add_patch(Rectangle(
            (xx, y), cell_width, height * value,
            facecolor=mix_with_white(np, COLORS["blue"], 0.34 + 0.55 * value),
            edgecolor=COLORS["blue"], linewidth=0.5, zorder=5,
        ))


def draw_layer_profile(ax, Rectangle, x, y, width, height):
    values = [0.53, 0.51, 0.48, 0.45, 0.43, 0.44, 0.47, 0.51, 0.56, 0.59, 0.60, 0.58]
    target = 0.52
    minimum, maximum = 0.40, 0.62
    core_x0 = x + width * 3 / 11
    core_x1 = x + width * 6 / 11
    tail_x0 = x + width * 8 / 11

    ax.add_patch(Rectangle(
        (core_x0, y), core_x1 - core_x0, height,
        facecolor=COLORS["blue_fill"], edgecolor="none", zorder=1,
    ))
    ax.add_patch(Rectangle(
        (tail_x0, y), x + width - tail_x0, height,
        facecolor=COLORS["neutral_fill"], edgecolor="none", zorder=1,
    ))
    ax.plot([x, x], [y, y + height], color=COLORS["line"], linewidth=0.85, zorder=3)
    ax.plot([x, x + width], [y, y], color=COLORS["line"], linewidth=0.85, zorder=3)
    target_y = y + height * (target - minimum) / (maximum - minimum)
    ax.plot([x, x + width], [target_y, target_y], color=COLORS["muted"],
            linewidth=0.9, linestyle=(0, (3, 2)), zorder=3)
    points_x = [x + width * index / (len(values) - 1) for index in range(len(values))]
    points_y = [y + height * (value - minimum) / (maximum - minimum) for value in values]
    ax.plot(points_x, points_y, color=COLORS["blue"], linewidth=2.0, zorder=6)
    ax.scatter(points_x, points_y, s=13, color=COLORS["blue"],
               edgecolor=COLORS["white"], linewidth=0.45, zorder=7)
    add_text(ax, x - 0.9, y + height / 2, r"$s_l$", color="blue_dark")
    add_text(ax, (core_x0 + core_x1) / 2, y + height + 0.7, "核心层保护", color="blue_dark")
    add_text(ax, (tail_x0 + x + width) / 2, y + height + 0.7, "尾部上限", color="muted")


def draw_sparse_model(ax, Rectangle, x, y, width, height):
    patterns = [
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    ]
    sparsities = [".49", ".45", ".47", ".54", ".60", ".58"]
    row_height = height / len(patterns)
    for row, pattern in enumerate(patterns):
        yy = y + (len(patterns) - 1 - row) * row_height
        add_text(ax, x - 0.9, yy + row_height / 2, rf"$L_{{{row + 1}}}$", color="muted", ha="right")
        cell_width = width / len(pattern)
        for column, keep in enumerate(pattern):
            ax.add_patch(Rectangle(
                (x + column * cell_width, yy + 0.31),
                cell_width - 0.10,
                row_height - 0.62,
                facecolor=COLORS["teal"] if keep else COLORS["white"],
                edgecolor=COLORS["teal_dark"] if keep else COLORS["line"],
                linewidth=0.62,
                zorder=5,
            ))
        add_text(ax, x + width + 0.8, yy + row_height / 2,
                 rf"$s_l={sparsities[row]}$", color="muted", ha="left")


def build_figure():
    plt, np, FontProperties, FancyArrowPatch, Polygon, Rectangle = load_plotting()
    configure_fonts(FontProperties)
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "savefig.facecolor": "white",
    })

    figure, ax = plt.subplots(figsize=(15.8, 6.25))
    figure.patch.set_facecolor(COLORS["white"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    # Input and calibration context
    add_text(ax, 7.8, 37.0, "输入与校准", large=True, bold=True)
    ax.add_patch(Rectangle((1.7, 35.0), 12.3, 0.25,
                           facecolor=COLORS["ink"], edgecolor="none"))
    add_text(ax, 7.8, 32.7, "预训练大语言模型", bold=True)
    draw_dense_model(ax, Rectangle, np, 4.2, 18.4, 7.2, 12.0)
    add_text(ax, 7.8, 16.8, r"模型权重 $W_l$", color="muted")
    add_text(ax, 3.0, 13.3, "校准激活", color="muted", ha="left")
    draw_activation(ax, Rectangle, np, 3.1, 10.2, 9.4, 2.1)
    add_text(ax, 13.2, 11.2, r"$X_l$", color="blue_dark", ha="left")
    gradient_preview = np.array([[0.18, 0.74, 0.31, 0.59], [0.67, 0.22, 0.83, 0.41]])
    add_text(ax, 3.0, 7.8, "聚合梯度", color="muted", ha="left")
    draw_matrix(ax, Rectangle, np, 3.1, 3.4, 9.4, 3.4, gradient_preview, "orange")
    add_text(ax, 13.2, 5.1, r"$G_l$", color="orange_dark", ha="left")

    # Parallel innovation mechanisms
    ax.add_patch(Rectangle((17.0, 21.4), 50.8, 13.6,
                           facecolor="#FBFCFD", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((17.0, 2.0), 50.8, 16.2,
                           facecolor="#FFFCFA", edgecolor="none", zorder=0))
    add_panel_heading(ax, Rectangle, 18.2, 66.4, 37.0,
                      "(a)", "OWL-V9：层间稀疏率分配", "blue")
    add_panel_heading(ax, Rectangle, 18.2, 66.4, 17.2,
                      "(b)", "wsqrtg：层内权重显著性", "orange")
    add_arrow(ax, FancyArrowPatch, (14.5, 25.7), (18.5, 29.0),
              color="blue", width=1.45, connectionstyle="arc3,rad=-0.10")
    add_arrow(ax, FancyArrowPatch, (14.5, 7.0), (18.5, 9.5),
              color="orange", width=1.45, connectionstyle="arc3,rad=-0.08")

    # OWL-V9 mechanism
    outlier_values = np.array([
        [0.18, 0.31, 0.88, 0.25, 0.42, 0.76],
        [0.22, 0.79, 0.35, 0.29, 0.91, 0.38],
        [0.61, 0.27, 0.33, 0.82, 0.24, 0.49],
        [0.19, 0.44, 0.73, 0.28, 0.36, 0.86],
    ])
    add_text(ax, 25.1, 32.6, "异常值证据", bold=True)
    add_text(ax, 25.1, 30.3, r"$o_l=|W_l|\sqrt{X_l}$", color="blue_dark")
    draw_matrix(ax, Rectangle, np, 20.7, 23.2, 8.8, 5.2, outlier_values, "blue",
                highlight_mask=outlier_values > 0.70)
    add_text(ax, 25.1, 22.3, r"$\tau_l=H_m\,\mathrm{mean}(o_l)$", color="muted")
    add_arrow(ax, FancyArrowPatch, (30.3, 30.2), (32.5, 30.2), color="blue")

    add_text(ax, 40.2, 32.6, "双证据建模", bold=True)
    add_text(ax, 40.2, 29.2, r"计数：$c_l=\#\{o_l>\tau_l\}/N_l$", color="blue_dark")
    add_text(ax, 40.2, 26.9,
             r"超额：$e_l=\mathbb{E}[(o_l-\tau_l)_+]/(\tau_l+\epsilon)$",
             color="blue_dark")
    add_text(ax, 40.2, 24.6, r"严重度：$v_l=c_l[1+\log(1+e_l)]$", color="blue_dark")
    add_text(ax, 40.2, 22.5, r"$\tilde d_l=(1-\alpha)D(c_l)+\alpha D(v_l)$")
    add_arrow(ax, FancyArrowPatch, (47.7, 30.2), (49.4, 30.2), color="blue")

    add_text(ax, 57.6, 32.6, "约束密度映射", bold=True)
    draw_layer_profile(ax, Rectangle, 51.2, 23.3, 12.9, 6.1)
    add_text(ax, 57.6, 22.1, r"全局重归一化：$\frac{1}{L}\sum_l s_l=s$", color="muted")

    # wsqrtg mechanism
    weights = np.array([
        [0.22, 0.84, 0.47, 0.33, 0.71],
        [0.73, 0.29, 0.91, 0.18, 0.52],
        [0.38, 0.65, 0.24, 0.79, 0.44],
        [0.88, 0.35, 0.58, 0.27, 0.69],
    ])
    gradients = np.array([
        [0.72, 0.14, 0.66, 0.39, 0.81],
        [0.26, 0.87, 0.43, 0.74, 0.31],
        [0.69, 0.34, 0.92, 0.21, 0.57],
        [0.41, 0.78, 0.19, 0.85, 0.48],
    ])
    saliency = np.abs(weights) * np.sqrt(np.abs(gradients))
    add_text(ax, 24.1, 13.8, r"权重幅值 $|W_l|$", bold=True)
    draw_matrix(ax, Rectangle, np, 19.8, 5.9, 8.8, 6.5, weights, "blue")
    add_text(ax, 31.0, 9.1, r"$\odot$")
    add_text(ax, 37.9, 13.8, r"梯度敏感度 $\sqrt{|G_l|}$", bold=True)
    draw_matrix(ax, Rectangle, np, 33.5, 5.9, 8.8, 6.5, np.sqrt(gradients), "orange")
    add_arrow(ax, FancyArrowPatch, (43.1, 9.1), (46.9, 9.1), color="orange")
    add_text(ax, 52.0, 13.8, "权重显著性", bold=True)
    draw_matrix(ax, Rectangle, np, 47.6, 5.9, 8.8, 6.5, saliency, "orange")
    add_text(ax, 52.0, 4.6, r"$M_l=|W_l|\odot\sqrt{|G_l|}$", color="orange_dark")
    add_text(ax, 61.1, 10.6, "一阶敏感性")
    add_text(ax, 61.1, 7.3, "无需 Hessian\n无需权重更新", color="muted")

    # Two mechanisms converge at a decision node
    add_panel_heading(ax, Rectangle, 70.3, 98.4, 37.0,
                      "(c)", "双机制协同剪枝", "teal")
    add_arrow(ax, FancyArrowPatch, (64.9, 27.0), (74.0, 23.5),
              color="blue", width=1.55, mutation=13, connectionstyle="arc3,rad=0.10")
    add_text(ax, 69.0, 27.9, r"层预算 $s_l$", color="blue_dark")
    add_arrow(ax, FancyArrowPatch, (65.0, 9.1), (74.0, 20.0),
              color="orange", width=1.55, mutation=13, connectionstyle="arc3,rad=-0.13")
    add_text(ax, 69.0, 10.3, r"显著性 $M_l$", color="orange_dark")

    diamond = Polygon(
        [[78.0, 27.1], [82.2, 21.7], [78.0, 16.3], [73.8, 21.7]],
        closed=True,
        facecolor=COLORS["teal_fill"],
        edgecolor=COLORS["teal"],
        linewidth=1.15,
        zorder=4,
    )
    ax.add_patch(diamond)
    add_text(ax, 78.0, 22.6, "层内排序", bold=True)
    add_text(ax, 78.0, 20.5, "阈值选择")
    add_text(ax, 78.0, 14.7, r"$K_l=\mathbf{1}[M_l\geq q_l(s_l)]$", color="teal_dark")

    add_arrow(ax, FancyArrowPatch, (82.0, 23.3), (86.2, 30.0),
              color="teal", width=1.55, mutation=13,
              connectionstyle="arc3,rad=-0.10")
    add_text(ax, 91.3, 31.7, "非均匀稀疏模型", bold=True)
    draw_sparse_model(ax, Rectangle, 87.0, 11.0, 9.2, 18.5)

    # Bottom synthesis strip
    ax.add_patch(Rectangle((71.8, 2.4), 26.0, 6.4,
                           facecolor=COLORS["neutral_fill"], edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((73.0, 6.55), 1.1, 1.1,
                           facecolor=COLORS["blue"], edgecolor="none", zorder=3))
    add_text(ax, 74.8, 7.1, "层间：分配剪枝预算", ha="left")
    ax.add_patch(Rectangle((73.0, 4.65), 1.1, 1.1,
                           facecolor=COLORS["orange"], edgecolor="none", zorder=3))
    add_text(ax, 74.8, 5.2, "层内：选择保留权重", ha="left")
    ax.add_patch(Rectangle((73.0, 2.75), 1.1, 1.1,
                           facecolor=COLORS["teal"], edgecolor="none", zorder=3))
    add_text(ax, 74.8, 3.3, "输出：满足目标全局稀疏率", color="teal_dark", ha="left")

    return figure, plt


def main():
    parser = argparse.ArgumentParser(description="Draw the Chinese draft main figure.")
    parser.add_argument("--out-dir", default="paper/figures")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png", "pdf", "svg"],
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, plt = build_figure()
    outputs = []
    for extension in args.formats:
        output = output_dir / f"main_method_overview_cn_draft.{extension}"
        options = {"bbox_inches": "tight", "pad_inches": 0.04, "facecolor": "white"}
        if extension == "png":
            options["dpi"] = args.dpi
        figure.savefig(output, **options)
        outputs.append(output)
    plt.close(figure)

    print("Chinese draft main figure generated:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
