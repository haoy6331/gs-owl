import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


MODEL_LABELS = {
    "llama1_7b": "LLaMA-1-7B",
    "llama2_7b": "LLaMA-2-7B",
    "llama1_13b": "LLaMA-1-13B",
    "llama1_30b": "LLaMA-1-30B",
}

PROGRESSIVE_ORDER = ["count", "anchored", "core", "full"]
PROGRESSIVE_LABELS = {
    "count": "Count",
    "anchored": "+ Severity fusion",
    "core": "+ Core protection",
    "full": "+ Tail restriction (OSLA)",
}
PROGRESSIVE_COLORS = {
    "count": "#777777",
    "anchored": "#4C78A8",
    "core": "#4E8F78",
    "full": "#B5524C",
}

VARIANT_LABELS = {
    "projection_none": "No projection",
    "projection_uniform": "Uniform",
    "projection_protected": "Protected",
    "boundary_none": "No constraints",
    "boundary_default": "Default",
    "boundary_left2": "Shift left",
    "boundary_right2": "Shift right",
    "depth_absolute": "Absolute",
    "depth_normalized": "Normalized",
}


def as_float(value, default=None):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_list(value):
    if isinstance(value, list):
        return [float(item) for item in value]
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [float(item) for item in parsed]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return []


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_path(value, result_path):
    if not value:
        return None
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, result_path.parent / path.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def enrich_result(payload, result_path):
    row = dict(payload)
    row["result_path"] = str(result_path)
    row["target_sparsity"] = as_float(
        row.get("target_sparsity") or row.get("sparsity_ratio")
    )
    row["ppl_test"] = as_float(row.get("ppl_test"))
    row["source_final_ppl"] = as_float(row.get("source_final_ppl"))
    row["layer_density_ratios"] = parse_list(row.get("layer_density_ratios"))
    diagnostics_path = resolve_path(row.get("diagnostics_json"), result_path)
    if diagnostics_path is None:
        candidate = result_path.parent / "v9_layer_diagnostics.json"
        diagnostics_path = candidate if candidate.exists() else None
    diagnostics = load_json(diagnostics_path) if diagnostics_path else {}
    row["diagnostics"] = diagnostics
    if diagnostics.get("layers"):
        row["layer_density_ratios"] = [
            float(layer["target_density_after_projection"])
            for layer in diagnostics["layers"]
        ]
    return row


def collect_results(root):
    rows = []
    if not root.exists():
        return rows
    for result_path in root.rglob("result.json"):
        payload = load_json(result_path)
        if payload.get("status") != "success":
            continue
        rows.append(enrich_result(payload, result_path))
    return rows


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required: pip install matplotlib") from exc

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.2,
        "axes.titleweight": "semibold",
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.solid_capstyle": "round",
    })
    return plt


def save_figure(fig, output_base, formats, dpi):
    paths = []
    for extension in formats:
        output_path = output_base.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(output_path, **kwargs)
        paths.append(output_path)
    return paths


def find_component(rows, model_key, sparsity, component):
    candidates = [
        row for row in rows
        if row.get("model_key") == model_key
        and row.get("v9_component") == component
        and row.get("target_sparsity") is not None
        and abs(row["target_sparsity"] - sparsity) < 1e-7
    ]
    return min(candidates, key=lambda row: row.get("ppl_test", math.inf)) if candidates else None


def find_variant(rows, suite, model_key, sparsity, variant):
    candidates = [
        row for row in rows
        if row.get("suite") == suite
        and row.get("model_key") == model_key
        and row.get("variant") == variant
        and row.get("target_sparsity") is not None
        and abs(row["target_sparsity"] - sparsity) < 1e-7
    ]
    return min(candidates, key=lambda row: row.get("ppl_test", math.inf)) if candidates else None


def reference_row(mechanism_rows, component_rows, suite, model_key, sparsity, variant):
    row = find_variant(mechanism_rows, suite, model_key, sparsity, variant)
    if row is not None:
        return row
    component = None
    if variant in {"projection_protected", "boundary_default"}:
        component = "full"
    elif variant == "boundary_none":
        component = "anchored"
    if component:
        return find_component(component_rows, model_key, sparsity, component)
    return None


def plot_progressive(plt, component_rows, out_dir, formats, dpi):
    models = ["llama1_7b", "llama2_7b"]
    sparsities = [0.6, 0.7]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.3), sharex=True)
    plotted = 0
    for row_index, model_key in enumerate(models):
        for column_index, sparsity in enumerate(sparsities):
            ax = axes[row_index][column_index]
            ax.axvspan(4, 10, color="#DCE8F2", alpha=0.55, linewidth=0)
            ax.axvspan(16, 30, color="#F2E8DD", alpha=0.52, linewidth=0)
            ax.axhline(1.0 - sparsity, color="#222222", linewidth=0.8, linestyle="--")
            for component in PROGRESSIVE_ORDER:
                row = find_component(component_rows, model_key, sparsity, component)
                if row is None or not row["layer_density_ratios"]:
                    continue
                values = row["layer_density_ratios"]
                ax.plot(
                    range(len(values)),
                    values,
                    color=PROGRESSIVE_COLORS[component],
                    linewidth=1.65 if component != "full" else 2.15,
                    marker="o" if component == "full" else None,
                    markersize=2.4,
                    label=PROGRESSIVE_LABELS[component],
                    zorder=3 if component == "full" else 2,
                )
                plotted += 1
            ax.set_title(f"{MODEL_LABELS[model_key]}, {int(sparsity * 100)}% sparsity")
            ax.set_xlim(0, 31)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 1:
                ax.set_xlabel("Transformer layer")
            if column_index == 0:
                ax.set_ylabel("Retained density")
    if plotted == 0:
        plt.close(fig)
        return []
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.275, 0.035, "Core interval", color="#4C78A8", ha="center", fontsize=8.5)
    fig.text(0.72, 0.035, "Tail-restricted interval", color="#9A6B43", ha="center", fontsize=8.5)
    fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.31, wspace=0.18)
    return save_figure(fig, out_dir / "osla_progressive_allocation", formats, dpi)


def post_budget_error(row, sparsity):
    if row is None:
        return None
    budget = row.get("diagnostics", {}).get("budget", {})
    value = as_float(budget.get("post_projection_error"))
    if value is not None:
        return abs(value)
    densities = row.get("layer_density_ratios", [])
    if densities:
        return abs(sum(densities) / len(densities) - (1.0 - sparsity))
    return None


def plot_projection_boundary(plt, mechanism_rows, component_rows, out_dir, formats, dpi):
    models = ["llama1_7b", "llama2_7b"]
    model_colors = ["#4C78A8", "#B5524C"]
    projection_variants = [
        "projection_none", "projection_uniform", "projection_protected"
    ]
    boundary_variants = [
        "boundary_none", "boundary_left2", "boundary_default", "boundary_right2"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    width = 0.34
    any_data = False

    for model_index, model_key in enumerate(models):
        offset = (model_index - 0.5) * width
        ppl_values = []
        error_values = []
        for variant in projection_variants:
            row = reference_row(
                mechanism_rows, component_rows, "projection", model_key, 0.7, variant
            )
            ppl_values.append(row.get("ppl_test") if row else None)
            error_values.append(post_budget_error(row, 0.7))
        valid_ppl = [value for value in ppl_values if value is not None]
        if valid_ppl:
            any_data = True
            baseline = min(valid_ppl)
            axes[0].bar(
                [index + offset for index in range(len(projection_variants))],
                [(value - baseline) if value is not None else 0 for value in ppl_values],
                width=width,
                color=model_colors[model_index],
                alpha=0.88,
                label=MODEL_LABELS[model_key],
            )
        if any(value is not None for value in error_values):
            axes[1].bar(
                [index + offset for index in range(len(projection_variants))],
                [max(value, 1e-9) if value is not None else 1e-9 for value in error_values],
                width=width,
                color=model_colors[model_index],
                alpha=0.88,
            )

        boundary_ppl = []
        for variant in boundary_variants:
            row = reference_row(
                mechanism_rows, component_rows, "boundary", model_key, 0.7, variant
            )
            boundary_ppl.append(row.get("ppl_test") if row else None)
        if any(value is not None for value in boundary_ppl):
            axes[2].plot(
                range(len(boundary_variants)),
                [value if value is not None else math.nan for value in boundary_ppl],
                color=model_colors[model_index],
                marker="o",
                markersize=5,
                linewidth=1.8,
                label=MODEL_LABELS[model_key],
            )

    projection_labels = [VARIANT_LABELS[name] for name in projection_variants]
    boundary_labels = [VARIANT_LABELS[name] for name in boundary_variants]
    axes[0].set_title("(a) Projection quality")
    axes[0].set_ylabel("PPL increase from best")
    axes[0].set_xticks(range(len(projection_labels)), projection_labels, rotation=20, ha="right")
    axes[1].set_title("(b) Global budget error")
    axes[1].set_ylabel("Absolute density error")
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(projection_labels)), projection_labels, rotation=20, ha="right")
    axes[2].set_title("(c) Boundary robustness")
    axes[2].set_ylabel("WikiText2 PPL")
    axes[2].set_xticks(range(len(boundary_labels)), boundary_labels, rotation=20, ha="right")
    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[2].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(top=0.82, bottom=0.27, wspace=0.34)
    if not any_data and not handles:
        plt.close(fig)
        return []
    return save_figure(fig, out_dir / "osla_projection_boundary", formats, dpi)


def plot_depth(plt, mechanism_rows, out_dir, formats, dpi):
    models = ["llama1_13b", "llama1_30b"]
    variants = ["depth_absolute", "depth_normalized"]
    colors = ["#777777", "#B5524C"]
    values = defaultdict(dict)
    for model_key in models:
        normalized = find_variant(
            mechanism_rows, "depth", model_key, 0.7, "depth_normalized"
        )
        absolute = find_variant(
            mechanism_rows, "depth", model_key, 0.7, "depth_absolute"
        )
        if absolute is None and normalized is not None and normalized.get("source_final_ppl") is not None:
            values[model_key]["depth_absolute"] = normalized["source_final_ppl"]
        elif absolute is not None:
            values[model_key]["depth_absolute"] = absolute.get("ppl_test")
        if normalized is not None:
            values[model_key]["depth_normalized"] = normalized.get("ppl_test")
    if not any(values.values()):
        return []

    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    width = 0.34
    for variant_index, variant in enumerate(variants):
        positions = [index + (variant_index - 0.5) * width for index in range(len(models))]
        heights = [values[model].get(variant, math.nan) for model in models]
        bars = ax.bar(
            positions,
            heights,
            width=width,
            color=colors[variant_index],
            alpha=0.9,
            label=VARIANT_LABELS[variant],
        )
        for bar, value in zip(bars, heights):
            if not math.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(range(len(models)), [MODEL_LABELS[model] for model in models])
    ax.set_ylabel("WikiText2 PPL at 70% sparsity")
    ax.set_title("Cross-depth interval transfer")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    return save_figure(fig, out_dir / "osla_cross_depth", formats, dpi)


def rank_values(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0 + 1.0
        for offset in range(position, end):
            ranks[order[offset]] = rank
        position = end
    return ranks


def spearman(left, right):
    if len(left) < 2 or len(left) != len(right):
        return None
    left_rank = rank_values(left)
    right_rank = rank_values(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_rank, right_rank)
    )
    left_scale = sum((x - left_mean) ** 2 for x in left_rank) ** 0.5
    right_scale = sum((y - right_mean) ** 2 for y in right_rank) ** 0.5
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def plot_sensitivity(plt, sensitivity_root, out_dir, formats, dpi):
    datasets = []
    for model_key in ["llama1_7b", "llama2_7b"]:
        path = sensitivity_root / f"{model_key}_layer_sensitivity.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        parsed = []
        for row in rows:
            layer = as_float(row.get("target_layer"))
            sensitivity = as_float(row.get("delta_ppl"))
            if sensitivity is None:
                sensitivity = as_float(row.get("ppl_test"))
            density = as_float(row.get("v9_density"))
            if layer is None or sensitivity is None or density is None:
                continue
            parsed.append((int(layer), sensitivity, density))
        if parsed:
            datasets.append((model_key, sorted(parsed)))
    if not datasets:
        return []

    fig, axes = plt.subplots(len(datasets), 1, figsize=(9.4, 2.75 * len(datasets)), squeeze=False)
    for index, (model_key, rows) in enumerate(datasets):
        ax = axes[index][0]
        layers = [row[0] for row in rows]
        sensitivity = [row[1] for row in rows]
        density = [row[2] for row in rows]
        correlation = spearman(sensitivity, density)
        ax.bar(layers, sensitivity, color="#A8B6C4", width=0.82, label="Single-layer sensitivity")
        twin = ax.twinx()
        twin.plot(layers, density, color="#B5524C", linewidth=2.0, marker="o", markersize=2.8, label="OSLA density")
        ax.axvspan(4, 10, color="#DCE8F2", alpha=0.45, linewidth=0)
        ax.axvspan(16, 30, color="#F2E8DD", alpha=0.42, linewidth=0)
        ax.set_ylabel("Delta PPL")
        twin.set_ylabel("Retained density", color="#8F3F3A")
        ax.set_title(
            f"{MODEL_LABELS[model_key]}: sensitivity-allocation alignment"
            + (f"  (Spearman rho={correlation:.3f})" if correlation is not None else "")
        )
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.7)
        ax.spines[["top"]].set_visible(False)
        twin.spines[["top"]].set_visible(False)
        if index == len(datasets) - 1:
            ax.set_xlabel("Transformer layer")
        left_handles, left_labels = ax.get_legend_handles_labels()
        right_handles, right_labels = twin.get_legend_handles_labels()
        ax.legend(left_handles + right_handles, left_labels + right_labels, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    return save_figure(fig, out_dir / "osla_sensitivity_alignment", formats, dpi)


def write_summary_csv(path, mechanism_rows, component_rows):
    fields = [
        "source", "suite", "variant", "model_key", "target_sparsity", "ppl_test",
        "actual_sparsity", "component", "projection_mode", "interval_mode",
        "post_projection_budget_error", "active_core_layers", "active_tail_layers",
        "result_path",
    ]
    rows = []
    for source, source_rows in [("mechanism", mechanism_rows), ("component", component_rows)]:
        for row in source_rows:
            diagnostics = row.get("diagnostics", {})
            budget = diagnostics.get("budget", {})
            constraints = diagnostics.get("constraints", {})
            rows.append({
                "source": source,
                "suite": row.get("suite", "progressive" if source == "component" else ""),
                "variant": row.get("variant", row.get("v9_component", "")),
                "model_key": row.get("model_key", ""),
                "target_sparsity": row.get("target_sparsity", ""),
                "ppl_test": row.get("ppl_test", ""),
                "actual_sparsity": row.get("actual_sparsity", ""),
                "component": row.get("v9_component", ""),
                "projection_mode": row.get("projection_mode", ""),
                "interval_mode": row.get("interval_mode", ""),
                "post_projection_budget_error": budget.get("post_projection_error", ""),
                "active_core_layers": constraints.get("active_core_layers", ""),
                "active_tail_layers": constraints.get("active_tail_layers", ""),
                "result_path": row.get("result_path", ""),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Create publication-ready OSLA mechanism figures.")
    parser.add_argument("--mechanism-root", default="owl/osla_mechanism_ablation")
    parser.add_argument("--component-root", default="owl/v9_component_ablation")
    parser.add_argument("--sensitivity-root", default="owl/layer_sensitivity_wsqrtg")
    parser.add_argument("--out-dir", default="owl/analysis/osla_mechanism")
    parser.add_argument("--formats", nargs="+", choices=["pdf", "png", "svg"], default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    plt = require_matplotlib()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mechanism_rows = collect_results(Path(args.mechanism_root))
    component_rows = collect_results(Path(args.component_root))

    written = []
    written.extend(plot_progressive(plt, component_rows, out_dir, args.formats, args.dpi))
    written.extend(plot_projection_boundary(
        plt, mechanism_rows, component_rows, out_dir, args.formats, args.dpi
    ))
    written.extend(plot_depth(plt, mechanism_rows, out_dir, args.formats, args.dpi))
    written.extend(plot_sensitivity(
        plt, Path(args.sensitivity_root), out_dir, args.formats, args.dpi
    ))
    write_summary_csv(out_dir / "osla_mechanism_summary.csv", mechanism_rows, component_rows)

    print(f"Mechanism runs: {len(mechanism_rows)}")
    print(f"Component runs: {len(component_rows)}")
    print(f"Summary CSV: {out_dir / 'osla_mechanism_summary.csv'}")
    if written:
        for path in written:
            print(f"Figure: {path}")
    else:
        print("No complete figure could be generated from the available results.")


if __name__ == "__main__":
    main()
