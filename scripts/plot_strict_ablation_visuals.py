import argparse
import csv
import json
import math
import re
from pathlib import Path


METHOD_INFO = {
    "wanda": ("wanda", "uniform"),
    "wanda-owl": ("wanda", "original_owl"),
    "wanda-owl-v9": ("wanda", "owl_v9"),
    "wsqrtg": ("wsqrtg", "uniform"),
    "owl-wsqrtg": ("wsqrtg", "original_owl"),
    "owl-v9-wsqrtg": ("wsqrtg", "owl_v9"),
}

MODEL_LABELS = {
    "llama1_7b": "LLaMA-1-7B",
    "llama2_7b": "LLaMA-2-7B",
    "llama1_13b": "LLaMA-1-13B",
    "llama2_13b": "LLaMA-2-13B",
}

LIST_PATTERNS = {
    "original_outlier": r"Raw true OWL LOD ratios:\s*\[(.*?)\]",
    "v9_count": r"Raw V9 true OWL count ratios:\s*\[(.*?)\]",
    "v9_severity": r"Raw V9 severity-weighted ratios:\s*\[(.*?)\]",
    "adjusted_density": r"Adjusted density ratios:\s*\[(.*?)\]",
    "adjusted_sparsity": r"Adjusted sparsity ratios:\s*\[(.*?)\]",
}

POLICY_METHODS = {
    "uniform": ("wsqrtg", "wanda"),
    "original_owl": ("owl-wsqrtg", "wanda-owl"),
    "owl_v9": ("owl-v9-wsqrtg", "wanda-owl-v9"),
}

COLORS = {
    "uniform": "#6B7280",
    "original_owl": "#0072B2",
    "owl_v9": "#D55E00",
    "count": "#009E73",
    "severity": "#CC79A7",
    "grid": "#D1D5DB",
    "text": "#1F2937",
}


def as_float(value, default=None):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_number_list(text):
    if isinstance(text, list):
        return [float(value) for value in text]
    if text in (None, ""):
        return []
    if isinstance(text, str):
        stripped = text.strip()
        if stripped.startswith("["):
            try:
                loaded = json.loads(stripped)
                if isinstance(loaded, list):
                    return [float(value) for value in loaded]
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return [
            float(value)
            for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        ]
    return []


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


def resolve_stdout_path(result_path, payload):
    candidates = []
    if payload.get("stdout_log"):
        configured = Path(payload["stdout_log"])
        candidates.append(configured if configured.is_absolute() else Path.cwd() / configured)
    candidates.extend([
        result_path.parent / "stdout.log",
        result_path.parent / "zero_shot_stdout.log",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_stdout_arrays(path):
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = {}
    for key, pattern in LIST_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            parsed[key] = parse_number_list(matches[-1])
    return parsed


def infer_model_key(result_path, root, payload):
    if payload.get("model_key"):
        return payload["model_key"]
    try:
        parts = result_path.relative_to(root).parts
    except ValueError:
        parts = result_path.parts
    return parts[0] if parts else "model"


def collect_results(root):
    rows = []
    for result_path in root.rglob("result.json"):
        payload = load_json(result_path)
        if not payload or payload.get("status") != "success":
            continue
        method = payload.get("method", "")
        metric, allocation = METHOD_INFO.get(
            method,
            (payload.get("pruning_metric", ""), payload.get("allocation_method", "")),
        )
        sparsity = as_float(payload.get("target_sparsity") or payload.get("sparsity_ratio"))
        ppl = as_float(payload.get("ppl_test"))
        if sparsity is None or ppl is None:
            continue
        stdout_path = resolve_stdout_path(result_path, payload)
        arrays = parse_stdout_arrays(stdout_path)
        payload_sparsity = parse_number_list(payload.get("layer_sparsity_ratios"))
        payload_density = parse_number_list(payload.get("layer_density_ratios"))
        if payload_sparsity and not arrays.get("adjusted_sparsity"):
            arrays["adjusted_sparsity"] = payload_sparsity
        if payload_density and not arrays.get("adjusted_density"):
            arrays["adjusted_density"] = payload_density
        rows.append({
            "model_key": infer_model_key(result_path, root, payload),
            "method": method,
            "metric": metric,
            "allocation": allocation,
            "target_sparsity": sparsity,
            "ppl": ppl,
            "actual_sparsity": as_float(payload.get("actual_sparsity")),
            "arrays": arrays,
            "stdout_path": str(stdout_path) if stdout_path else "",
            "result_path": str(result_path),
        })
    return rows


def select_row(rows, model_key, sparsity, methods):
    candidates = [
        row for row in rows
        if row["model_key"] == model_key
        and abs(row["target_sparsity"] - sparsity) < 1e-7
        and row["method"] in methods
    ]
    if not candidates:
        return None
    method_order = {method: index for index, method in enumerate(methods)}
    return min(candidates, key=lambda row: method_order.get(row["method"], len(methods)))


def require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install it in ZW-Prune with: pip install matplotlib"
        ) from exc

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    return plt, LinearSegmentedColormap, TwoSlopeNorm, Rectangle


def save_figure(fig, output_base, formats, dpi):
    paths = []
    for extension in formats:
        path = output_base.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def write_layer_csv(path, layer_data):
    fields = [
        "model_key", "layer", "target_sparsity", "uniform_sparsity",
        "original_owl_sparsity", "owl_v9_sparsity", "owl_count_ratio",
        "v9_severity_ratio",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_key, data in layer_data.items():
            for layer in range(data["n_layers"]):
                writer.writerow({
                    "model_key": model_key,
                    "layer": layer,
                    "target_sparsity": data["target_sparsity"],
                    "uniform_sparsity": data["uniform"][layer],
                    "original_owl_sparsity": data["original_owl"][layer],
                    "owl_v9_sparsity": data["owl_v9"][layer],
                    "owl_count_ratio": data["count_ratio"][layer],
                    "v9_severity_ratio": data["severity_ratio"][layer],
                })


def prepare_layer_data(rows, models, target_sparsity):
    layer_data = {}
    missing = []
    for model_key in models:
        selected = {
            policy: select_row(rows, model_key, target_sparsity, methods)
            for policy, methods in POLICY_METHODS.items()
        }
        for policy in ("original_owl", "owl_v9"):
            if selected[policy] is None:
                missing.append(f"{model_key}: {policy} at sparsity {target_sparsity}")
        if selected["original_owl"] is None or selected["owl_v9"] is None:
            continue

        original_arrays = selected["original_owl"]["arrays"]
        v9_arrays = selected["owl_v9"]["arrays"]
        original_sparsity = original_arrays.get("adjusted_sparsity", [])
        v9_sparsity = v9_arrays.get("adjusted_sparsity", [])
        count_ratio = original_arrays.get("original_outlier") or v9_arrays.get("v9_count", [])
        severity_ratio = v9_arrays.get("v9_severity", [])
        lengths = [len(original_sparsity), len(v9_sparsity), len(count_ratio), len(severity_ratio)]
        if not all(lengths) or len(set(lengths)) != 1:
            missing.append(
                f"{model_key}: incomplete layer arrays "
                f"(original={lengths[0]}, v9={lengths[1]}, count={lengths[2]}, severity={lengths[3]})"
            )
            continue
        n_layers = lengths[0]
        layer_data[model_key] = {
            "target_sparsity": target_sparsity,
            "n_layers": n_layers,
            "uniform": [target_sparsity] * n_layers,
            "original_owl": original_sparsity,
            "owl_v9": v9_sparsity,
            "count_ratio": count_ratio,
            "severity_ratio": severity_ratio,
        }
    if missing:
        raise RuntimeError("Cannot build layer-allocation figure:\n  " + "\n  ".join(missing))
    return layer_data


def plot_layer_allocation(rows, models, target_sparsity, out_dir, formats, dpi):
    plt, _, _, _ = require_matplotlib()
    layer_data = prepare_layer_data(rows, models, target_sparsity)
    figure, axes = plt.subplots(
        len(models), 2,
        figsize=(12.8, 3.6 * len(models)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, model_key in enumerate(models):
        data = layer_data[model_key]
        layers = list(range(data["n_layers"]))
        label = MODEL_LABELS.get(model_key, model_key)
        allocation_ax = axes[row_index][0]
        signal_ax = axes[row_index][1]

        allocation_ax.plot(
            layers, [value * 100 for value in data["uniform"]],
            color=COLORS["uniform"], linestyle="--", linewidth=1.8,
            label="Uniform",
        )
        allocation_ax.plot(
            layers, [value * 100 for value in data["original_owl"]],
            color=COLORS["original_owl"], marker="o", markevery=3,
            markersize=3.8, linewidth=2.0, label="Original OWL",
        )
        allocation_ax.plot(
            layers, [value * 100 for value in data["owl_v9"]],
            color=COLORS["owl_v9"], marker="s", markevery=3,
            markersize=3.5, linewidth=2.1, label="OWL-V9",
        )
        allocation_ax.set_title(f"{label}: layer allocation")
        allocation_ax.set_xlabel("Transformer layer")
        allocation_ax.set_ylabel("Layer sparsity (%)")
        allocation_ax.set_xlim(0, data["n_layers"] - 1)
        allocation_ax.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.65)
        allocation_ax.spines[["top", "right"]].set_visible(False)
        allocation_ax.legend(frameon=False, ncol=3, loc="best")

        signal_ax.plot(
            layers, data["count_ratio"],
            color=COLORS["count"], marker="o", markevery=3,
            markersize=3.8, linewidth=2.0, label="OWL outlier count",
        )
        signal_ax.plot(
            layers, data["severity_ratio"],
            color=COLORS["severity"], marker="D", markevery=3,
            markersize=3.4, linewidth=2.0, label="V9 severity-weighted",
        )
        signal_ax.set_title(f"{label}: allocation signals")
        signal_ax.set_xlabel("Transformer layer")
        signal_ax.set_ylabel("Outlier signal (%)")
        signal_ax.set_xlim(0, data["n_layers"] - 1)
        signal_ax.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.65)
        signal_ax.spines[["top", "right"]].set_visible(False)
        signal_ax.legend(frameon=False, loc="best")

    figure.suptitle(
        f"Layer-wise allocation at global sparsity {target_sparsity:.1f}",
        fontsize=13,
        fontweight="semibold",
    )
    paths = save_figure(figure, out_dir / "figure_layer_allocation", formats, dpi)
    plt.close(figure)
    write_layer_csv(out_dir / "figure_layer_allocation_data.csv", layer_data)
    return paths


def build_heatmap_matrices(rows, models, sparsities):
    matrices = {}
    missing = []
    for model_key in models:
        for sparsity in sparsities:
            matrix = [[math.nan] * 3 for _ in range(2)]
            for row_index, metric in enumerate(("wanda", "wsqrtg")):
                for column_index, allocation in enumerate(("uniform", "original_owl", "owl_v9")):
                    candidates = [
                        row for row in rows
                        if row["model_key"] == model_key
                        and abs(row["target_sparsity"] - sparsity) < 1e-7
                        and row["metric"] == metric
                        and row["allocation"] == allocation
                    ]
                    if candidates:
                        matrix[row_index][column_index] = min(row["ppl"] for row in candidates)
                    else:
                        missing.append(f"{model_key}, s={sparsity}, {metric}, {allocation}")
            matrices[(model_key, sparsity)] = matrix
    if missing:
        raise RuntimeError("Cannot build strict-ablation heatmap; missing successful results:\n  " + "\n  ".join(missing))
    return matrices


def write_heatmap_csv(path, matrices):
    fields = [
        "model_key", "target_sparsity", "metric", "allocation", "ppl",
        "wanda_uniform_ppl", "delta_ppl_vs_wanda_uniform",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (model_key, sparsity), matrix in sorted(matrices.items()):
            baseline = matrix[0][0]
            for row_index, metric in enumerate(("wanda", "wsqrtg")):
                for column_index, allocation in enumerate(("uniform", "original_owl", "owl_v9")):
                    ppl = matrix[row_index][column_index]
                    writer.writerow({
                        "model_key": model_key,
                        "target_sparsity": sparsity,
                        "metric": metric,
                        "allocation": allocation,
                        "ppl": ppl,
                        "wanda_uniform_ppl": baseline,
                        "delta_ppl_vs_wanda_uniform": ppl - baseline,
                    })


def plot_ablation_heatmap(rows, models, sparsities, out_dir, formats, dpi):
    plt, LinearSegmentedColormap, TwoSlopeNorm, Rectangle = require_matplotlib()
    matrices = build_heatmap_matrices(rows, models, sparsities)
    deltas = []
    for matrix in matrices.values():
        baseline = matrix[0][0]
        deltas.extend(value - baseline for row in matrix for value in row)
    max_abs = max((abs(value) for value in deltas), default=1.0)
    max_abs = max(max_abs, 1e-6)
    color_map = LinearSegmentedColormap.from_list(
        "ppl_gain",
        [COLORS["original_owl"], "#F7F7F2", COLORS["owl_v9"]],
    )
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    figure, axes = plt.subplots(
        len(models), len(sparsities),
        figsize=(4.0 * len(sparsities), 3.25 * len(models)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for model_index, model_key in enumerate(models):
        for sparsity_index, sparsity in enumerate(sparsities):
            axis = axes[model_index][sparsity_index]
            matrix = matrices[(model_key, sparsity)]
            baseline = matrix[0][0]
            delta_matrix = [[value - baseline for value in row] for row in matrix]
            image = axis.imshow(delta_matrix, cmap=color_map, norm=norm, aspect="auto")

            best_value = min(value for row in matrix for value in row)
            for row_index in range(2):
                for column_index in range(3):
                    ppl = matrix[row_index][column_index]
                    delta = ppl - baseline
                    color = "white" if abs(delta) > max_abs * 0.58 else COLORS["text"]
                    weight = "bold" if abs(ppl - best_value) < 1e-10 else "normal"
                    axis.text(
                        column_index, row_index,
                        f"{ppl:.3f}\n({delta:+.3f})",
                        ha="center", va="center", color=color,
                        fontsize=9, fontweight=weight,
                    )
                    if abs(ppl - best_value) < 1e-10:
                        axis.add_patch(Rectangle(
                            (column_index - 0.48, row_index - 0.48), 0.96, 0.96,
                            fill=False, edgecolor=COLORS["text"], linewidth=2.0,
                        ))

            axis.set_xticks(range(3), ["Uniform", "Original\nOWL", "OWL-V9"])
            axis.set_yticks(range(2), ["Wanda", "wsqrtg"])
            axis.set_title(f"s = {sparsity:.1f}")
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
            if sparsity_index == 0:
                axis.set_ylabel(MODEL_LABELS.get(model_key, model_key), fontweight="semibold")

    figure.suptitle(
        "Strict ablation: pruning metric x layer allocation",
        fontsize=13,
        fontweight="semibold",
    )
    if image is not None:
        colorbar = figure.colorbar(
            image, ax=axes.ravel().tolist(), orientation="horizontal",
            fraction=0.045, pad=0.08, aspect=45,
        )
        colorbar.set_label("Delta PPL vs. Wanda + Uniform (lower is better)")
    paths = save_figure(figure, out_dir / "figure_strict_ablation_heatmap", formats, dpi)
    plt.close(figure)
    write_heatmap_csv(out_dir / "figure_strict_ablation_heatmap_data.csv", matrices)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Create publication-ready layer-allocation and strict-ablation figures."
    )
    parser.add_argument("--ablation-root", default="owl/strict_ablation_wsqrtg_owl_v9")
    parser.add_argument("--out-dir", default="owl/analysis/paper_visualizations")
    parser.add_argument("--models", nargs="+", default=["llama1_7b", "llama2_7b"])
    parser.add_argument("--layer-sparsity", type=float, default=0.6)
    parser.add_argument("--heatmap-sparsities", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--figure", choices=["all", "layer", "heatmap"], default="all")
    args = parser.parse_args()

    root = Path(args.ablation_root)
    if not root.exists():
        raise FileNotFoundError(f"Ablation root not found: {root}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_results(root)
    if not rows:
        raise RuntimeError(f"No successful strict-ablation result.json files found under: {root}")
    print(f"Loaded {len(rows)} successful strict-ablation results from {root}")

    if args.figure in {"all", "layer"}:
        layer_paths = plot_layer_allocation(
            rows, args.models, args.layer_sparsity, out_dir, args.formats, args.dpi
        )
        print("Layer-allocation figure:")
        for path in layer_paths:
            print(f"  {path}")
    if args.figure in {"all", "heatmap"}:
        heatmap_paths = plot_ablation_heatmap(
            rows, args.models, args.heatmap_sparsities, out_dir, args.formats, args.dpi
        )
        print("Strict-ablation heatmap:")
        for path in heatmap_paths:
            print(f"  {path}")
    print(f"Plot data CSV files: {out_dir}")


if __name__ == "__main__":
    main()
