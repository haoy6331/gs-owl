import argparse
from pathlib import Path


def load_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import FancyArrowPatch, Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib and numpy are required. Install them with: pip install matplotlib numpy"
        ) from exc
    return plt, np, FancyArrowPatch, Rectangle


COLORS = {
    "ink": "#202124",
    "muted": "#5F6368",
    "line": "#C9CDD2",
    "grid": "#E4E7EB",
    "panel": "#F7F8FA",
    "blue": "#356FA3",
    "blue_dark": "#234F78",
    "blue_fill": "#EAF2F8",
    "orange": "#C8642D",
    "orange_dark": "#91421F",
    "orange_fill": "#FAEEE6",
    "gray_fill": "#EEF0F2",
    "white": "#FFFFFF",
}


def text(ax, x, y, value, size=9.5, color="ink", weight="normal", ha="center",
         va="center", style="normal", zorder=10, linespacing=1.15):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=COLORS.get(color, color),
        fontweight=weight,
        fontstyle=style,
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=zorder,
    )


def arrow(ax, FancyArrowPatch, start, end, color="line", width=1.35,
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


def panel_heading(ax, Rectangle, x0, x1, y, marker, title, color):
    text(ax, x0, y, marker, size=12.8, color="ink", weight="bold", ha="left")
    text(ax, x0 + 3.0, y, title, size=12.8, color="ink", weight="bold", ha="left")
    ax.add_patch(Rectangle(
        (x0, y - 2.05),
        x1 - x0,
        0.26,
        facecolor=COLORS[color],
        edgecolor="none",
        zorder=2,
    ))


def hex_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))


def mix_with_white(np, color, amount):
    rgb = np.array(hex_rgb(color))
    return tuple(1.0 - amount * (1.0 - rgb))


def draw_matrix(ax, Rectangle, np, x, y, width, height, values, accent,
                zero_mask=None, highlight_mask=None, show_values=False):
    values = np.asarray(values, dtype=float)
    rows, columns = values.shape
    gap = min(width / columns, height / rows) * 0.075
    cell_width = (width - gap * (columns - 1)) / columns
    cell_height = (height - gap * (rows - 1)) / rows
    low = float(values.min())
    span = max(float(values.max()) - low, 1e-9)
    for row in range(rows):
        for column in range(columns):
            cell_x = x + column * (cell_width + gap)
            cell_y = y + (rows - 1 - row) * (cell_height + gap)
            is_zero = zero_mask is not None and bool(zero_mask[row, column])
            is_highlighted = highlight_mask is not None and bool(highlight_mask[row, column])
            if is_zero:
                face = COLORS["white"]
                edge = COLORS["line"]
                linewidth = 0.72
            else:
                normalized = (float(values[row, column]) - low) / span
                face = mix_with_white(np, COLORS[accent], 0.18 + 0.72 * normalized)
                edge = COLORS[accent] if is_highlighted else COLORS["line"]
                linewidth = 1.25 if is_highlighted else 0.62
            ax.add_patch(Rectangle(
                (cell_x, cell_y),
                cell_width,
                cell_height,
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                zorder=5,
            ))
            if show_values and not is_zero:
                text(
                    ax,
                    cell_x + cell_width / 2,
                    cell_y + cell_height / 2,
                    f"{values[row, column]:.1f}",
                    size=6.4,
                    color="ink",
                    zorder=7,
                )


def draw_dense_stack(ax, Rectangle, x, y, width, height):
    layers = 7
    layer_height = 1.35
    gap = (height - layers * layer_height) / (layers - 1)
    for layer in range(layers):
        yy = y + layer * (layer_height + gap)
        ax.add_patch(Rectangle(
            (x + 0.12 * (layer % 2), yy),
            width,
            layer_height,
            facecolor=mix_with_white(__import__("numpy"), COLORS["blue"], 0.32 + 0.05 * layer),
            edgecolor=COLORS["blue_dark"],
            linewidth=0.75,
            zorder=4,
        ))
        for cell in range(7):
            cx = x + 0.55 + cell * (width - 1.1) / 7
            ax.add_patch(Rectangle(
                (cx, yy + 0.34),
                (width - 1.4) / 8.2,
                0.67,
                facecolor=COLORS["white"],
                edgecolor="none",
                alpha=0.72,
                zorder=5,
            ))


def draw_input_tokens(ax, Rectangle, x, y, width, height, accent):
    for index, scale in enumerate((0.34, 0.55, 0.80, 0.46, 0.68, 0.29, 0.62, 0.40)):
        cell_width = width / 9.4
        xx = x + index * width / 8
        ax.add_patch(Rectangle(
            (xx, y),
            cell_width,
            height * scale,
            facecolor=mix_with_white(__import__("numpy"), COLORS[accent], 0.35 + 0.55 * scale),
            edgecolor=COLORS[accent],
            linewidth=0.5,
            zorder=5,
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
        facecolor=COLORS["gray_fill"], edgecolor="none", zorder=1,
    ))

    ax.plot([x, x], [y, y + height], color=COLORS["line"], linewidth=0.85, zorder=3)
    ax.plot([x, x + width], [y, y], color=COLORS["line"], linewidth=0.85, zorder=3)
    target_y = y + height * (target - minimum) / (maximum - minimum)
    ax.plot(
        [x, x + width],
        [target_y, target_y],
        color=COLORS["muted"],
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=3,
    )

    points_x = [x + width * index / (len(values) - 1) for index in range(len(values))]
    points_y = [y + height * (value - minimum) / (maximum - minimum) for value in values]
    ax.plot(points_x, points_y, color=COLORS["blue"], linewidth=2.0, zorder=6)
    ax.scatter(points_x, points_y, s=12, color=COLORS["blue"], edgecolor=COLORS["white"],
               linewidth=0.45, zorder=7)
    text(ax, x - 0.9, y + height / 2, r"$s_l$", size=9.2, color="blue_dark")
    text(ax, x + width - 0.1, target_y + 0.72, r"target $s$", size=7.2,
         color="muted", ha="right")
    text(ax, (core_x0 + core_x1) / 2, y + height + 0.65, "protected core", size=7.1,
         color="blue_dark")
    text(ax, (tail_x0 + x + width) / 2, y + height + 0.65, "tail cap", size=7.1,
         color="muted")


def draw_sparse_stack(ax, Rectangle, x, y, width, height):
    patterns = [
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    ]
    labels = [r"$L_1$", r"$L_5$", r"$L_{10}$", r"$L_{18}$", r"$L_{26}$", r"$L_L$"]
    sparsities = [".49", ".45", ".47", ".54", ".60", ".58"]
    row_height = height / len(patterns)
    for row, pattern in enumerate(patterns):
        yy = y + (len(patterns) - 1 - row) * row_height
        text(ax, x - 1.0, yy + row_height * 0.5, labels[row], size=7.5,
             color="muted", ha="right")
        cell_width = width / len(pattern)
        for column, keep in enumerate(pattern):
            face = COLORS["blue"] if keep else COLORS["white"]
            edge = COLORS["blue_dark"] if keep else COLORS["line"]
            ax.add_patch(Rectangle(
                (x + column * cell_width, yy + 0.32),
                cell_width - 0.10,
                row_height - 0.64,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.62,
                zorder=5,
            ))
        text(ax, x + width + 0.8, yy + row_height * 0.5, rf"$s_l={sparsities[row]}$",
             size=7.2, color="muted", ha="left")


def build_figure():
    plt, np, FancyArrowPatch, Rectangle = load_plotting()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })

    figure, ax = plt.subplots(figsize=(15.8, 6.25))
    figure.patch.set_facecolor(COLORS["white"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    # Common inputs
    text(ax, 7.8, 37.1, "Model inputs", size=12.8, weight="bold")
    ax.add_patch(Rectangle((1.7, 35.1), 12.2, 0.26, facecolor=COLORS["ink"], edgecolor="none"))
    text(ax, 7.8, 32.9, "Pretrained LLM", size=9.4, weight="bold")
    draw_dense_stack(ax, Rectangle, 4.2, 18.2, 7.2, 12.5)
    text(ax, 7.8, 16.6, r"weights $W_l$", size=8.4, color="muted")

    text(ax, 3.0, 13.4, "Calibration", size=8.2, color="muted", ha="left")
    draw_input_tokens(ax, Rectangle, 3.1, 10.3, 9.4, 2.1, "blue")
    text(ax, 13.2, 11.25, r"$X_l$", size=10.8, color="blue_dark", ha="left")

    gradient_preview = np.array([
        [0.18, 0.74, 0.31, 0.59],
        [0.67, 0.22, 0.83, 0.41],
    ])
    text(ax, 3.0, 7.9, "Aggregated gradient", size=8.2, color="muted", ha="left")
    draw_matrix(ax, Rectangle, np, 3.1, 3.4, 9.4, 3.5, gradient_preview, "orange")
    text(ax, 13.2, 5.15, r"$G_l$", size=10.8, color="orange_dark", ha="left")

    # Two method lanes
    ax.add_patch(Rectangle((17.0, 21.4), 50.7, 13.7, facecolor="#FBFCFD", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((17.0, 2.0), 50.7, 16.2, facecolor="#FFFCFA", edgecolor="none", zorder=0))
    panel_heading(ax, Rectangle, 18.2, 66.4, 37.1, "(a)", "OWL-V9: layer sparsity allocation", "blue")
    panel_heading(ax, Rectangle, 18.2, 66.4, 17.3, "(b)", "wsqrtg: gradient-sensitive weight saliency", "orange")

    arrow(ax, FancyArrowPatch, (14.5, 25.7), (18.5, 29.0), "blue", 1.4, 12,
          "arc3,rad=-0.10")
    arrow(ax, FancyArrowPatch, (14.5, 7.0), (18.5, 9.5), "orange", 1.4, 12,
          "arc3,rad=-0.08")

    # OWL-V9 lane
    outlier_values = np.array([
        [0.18, 0.31, 0.88, 0.25, 0.42, 0.76],
        [0.22, 0.79, 0.35, 0.29, 0.91, 0.38],
        [0.61, 0.27, 0.33, 0.82, 0.24, 0.49],
        [0.19, 0.44, 0.73, 0.28, 0.36, 0.86],
    ])
    threshold = 0.70
    text(ax, 25.2, 32.7, "Outlier evidence", size=9.3, weight="bold")
    text(ax, 25.2, 30.5, r"$o_l=|W_l|\sqrt{X_l}$", size=12.5, color="blue_dark")
    draw_matrix(
        ax, Rectangle, np, 20.7, 23.2, 9.0, 5.3, outlier_values, "blue",
        highlight_mask=outlier_values > threshold,
    )
    text(ax, 25.2, 22.3, r"$\tau_l=H_m\,\mathrm{mean}(o_l)$", size=8.5, color="muted")

    arrow(ax, FancyArrowPatch, (30.5, 30.35), (32.6, 30.35), "blue", 1.35, 11)
    text(ax, 40.4, 32.7, "Dual layer evidence", size=9.3, weight="bold")
    text(ax, 40.4, 29.3, r"Count: $c_l=\#\{o_l>\tau_l\}/N_l$", size=9.2,
         color="blue_dark")
    text(ax, 40.4, 27.1,
         r"Excess: $e_l=\mathbb{E}[(o_l-\tau_l)_+]/(\tau_l+\epsilon)$", size=8.4,
         color="blue_dark")
    text(ax, 40.4, 24.9, r"Severity: $v_l=c_l[1+\log(1+e_l)]$", size=9.0,
         color="blue_dark")
    text(ax, 40.4, 22.7, r"$\tilde d_l=(1-\alpha)D(c_l)+\alpha D(v_l)$", size=9.6,
         color="ink")

    arrow(ax, FancyArrowPatch, (47.7, 30.35), (49.4, 30.35), "blue", 1.35, 11)
    text(ax, 57.6, 32.7, "Constrained allocation", size=9.3, weight="bold")
    draw_layer_profile(ax, Rectangle, 51.3, 23.3, 12.8, 6.2)
    text(ax, 57.7, 22.1, r"exact global sparsity: $\frac{1}{L}\sum_l s_l=s$", size=8.0,
         color="muted")

    # wsqrtg lane
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
    text(ax, 24.2, 14.0, r"$|W_l|$", size=11.0, weight="bold")
    draw_matrix(ax, Rectangle, np, 19.8, 5.9, 8.8, 6.6, weights, "blue")
    text(ax, 31.1, 9.2, r"$\odot$", size=18.0, color="muted")
    text(ax, 38.0, 14.0, r"$\sqrt{|G_l|}$", size=11.0, weight="bold")
    draw_matrix(ax, Rectangle, np, 33.6, 5.9, 8.8, 6.6, np.sqrt(gradients), "orange")
    arrow(ax, FancyArrowPatch, (43.2, 9.2), (47.0, 9.2), "orange", 1.35, 11)
    text(ax, 52.1, 14.0, "Weight saliency", size=9.3, weight="bold")
    draw_matrix(ax, Rectangle, np, 47.7, 5.9, 8.8, 6.6, saliency, "orange")
    text(ax, 52.1, 4.6, r"$M_l=|W_l|\odot\sqrt{|G_l|}$", size=11.3,
         color="orange_dark", weight="bold")
    text(ax, 61.2, 10.8, "First-order\nsensitivity", size=8.4, color="muted",
         linespacing=1.2)
    text(ax, 61.2, 7.1, "No Hessian or\nweight update", size=8.4, color="muted",
         linespacing=1.2)

    # Fusion and sparse output
    panel_heading(ax, Rectangle, 70.3, 98.4, 37.1, "(c)", "Layer-wise pruning", "ink")
    arrow(ax, FancyArrowPatch, (64.9, 27.1), (73.1, 24.6), "blue", 1.55, 13,
          "arc3,rad=0.10")
    text(ax, 69.2, 28.0, r"budget $s_l$", size=8.2, color="blue_dark")
    arrow(ax, FancyArrowPatch, (65.0, 9.2), (73.1, 19.0), "orange", 1.55, 13,
          "arc3,rad=-0.13")
    text(ax, 69.0, 10.4, r"saliency $M_l$", size=8.2, color="orange_dark")

    ax.add_patch(Rectangle(
        (73.2, 17.1), 9.8, 9.3,
        facecolor=COLORS["white"], edgecolor=COLORS["line"], linewidth=1.0, zorder=3,
    ))
    text(ax, 78.1, 24.4, "Per-layer ranking", size=9.2, weight="bold")
    text(ax, 78.1, 21.5, r"$K_l=\mathbf{1}[M_l\geq q_l(s_l)]$", size=11.0, color="ink")
    text(ax, 78.1, 18.7, "keep highest scores", size=7.9, color="muted")

    arrow(ax, FancyArrowPatch, (83.1, 24.0), (86.1, 30.0), "ink", 1.4, 12,
          "arc3,rad=-0.10")
    text(ax, 91.3, 31.8, "Sparse LLM", size=10.2, weight="bold")
    draw_sparse_stack(ax, Rectangle, 87.0, 11.0, 9.2, 18.6)

    # Compact legend and outcome
    ax.add_patch(Rectangle((75.3, 8.1), 1.25, 1.25, facecolor=COLORS["blue"],
                           edgecolor=COLORS["blue_dark"], linewidth=0.6))
    text(ax, 77.3, 8.73, "kept", size=7.7, color="muted", ha="left")
    ax.add_patch(Rectangle((80.9, 8.1), 1.25, 1.25, facecolor=COLORS["white"],
                           edgecolor=COLORS["line"], linewidth=0.7))
    text(ax, 82.9, 8.73, "pruned", size=7.7, color="muted", ha="left")
    text(ax, 86.4, 4.9, "one-shot  |  no retraining", size=8.6, color="ink", weight="bold")
    text(ax, 86.4, 2.9, "non-uniform layer sparsity, gradient-aware weight selection",
         size=7.8, color="muted")

    return figure, plt


def main():
    parser = argparse.ArgumentParser(description="Draw the paper main figure for OWL-V9 + wsqrtg.")
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
        output = output_dir / f"main_method_overview.{extension}"
        save_options = {
            "bbox_inches": "tight",
            "pad_inches": 0.04,
            "facecolor": "white",
        }
        if extension == "png":
            save_options["dpi"] = args.dpi
        figure.savefig(output, **save_options)
        outputs.append(output)
    plt.close(figure)

    print("Paper main figure generated:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
