import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


LIST_PATTERNS = {
    "raw_count_ratio": r"Raw V9 true OWL count ratios:\s*\[(.*?)\]",
    "raw_severity_ratio": r"Raw V9 severity-weighted ratios:\s*\[(.*?)\]",
    "count_density": r"Count-OWL density ratios:\s*\[(.*?)\]",
    "severity_density": r"Severity-OWL density ratios:\s*\[(.*?)\]",
    "anchored_density": r"Anchored density ratios before constraints:\s*\[(.*?)\]",
    "adjusted_sparsity": r"Adjusted sparsity ratios:\s*\[(.*?)\]",
    "adjusted_density": r"Adjusted density ratios:\s*\[(.*?)\]",
}

RAW_V9_PATTERN = re.compile(
    r"layer\s+(\d+)\s+V9 raw ratios:\s+count=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)%,\s+"
    r"severity_ratio=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)%.*?"
    r"severity=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s+"
    r"tail_mass=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)%",
    re.DOTALL,
)


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def as_float(value, default=None):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_number_list(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def parse_stdout(stdout_path):
    if not stdout_path or not stdout_path.exists():
        return {}

    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    parsed = {}
    for key, pattern in LIST_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            parsed[key] = parse_number_list(matches[-1])

    raw_rows = {}
    for match in RAW_V9_PATTERN.finditer(text):
        layer = int(match.group(1))
        raw_rows[layer] = {
            "raw_count_ratio": float(match.group(2)),
            "raw_severity_ratio": float(match.group(3)),
            "raw_severity": float(match.group(4)),
            "tail_mass_ratio": float(match.group(5)),
        }
    if raw_rows:
        parsed["raw_rows"] = raw_rows
    return parsed


def rel_parts(path, root):
    try:
        return path.relative_to(root).parts
    except ValueError:
        return path.parts


def infer_identity(result_path, root, payload):
    parts = rel_parts(result_path, root)
    output_root = parts[0] if len(parts) > 0 else ""
    model = parts[1] if len(parts) > 1 else ""
    sparsity_type = payload.get("sparsity_type", parts[2] if len(parts) > 2 else "")
    method = payload.get("method", parts[3] if len(parts) > 3 else "")
    return output_root, model, sparsity_type, method


def resolve_stdout_path(result_path, payload):
    candidates = []
    if payload.get("stdout_log"):
        p = Path(payload["stdout_log"])
        candidates.append(p if p.is_absolute() else Path.cwd() / p)
    candidates.append(result_path.parent / "stdout.log")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def collect_success_results(root):
    rows = []
    for result_path in root.rglob("result.json"):
        payload = load_json(result_path)
        if not payload or payload.get("status") != "success":
            continue
        target_sparsity = as_float(payload.get("target_sparsity"))
        ppl = as_float(payload.get("ppl_test"))
        if target_sparsity is None:
            continue
        output_root, model, sparsity_type, method = infer_identity(result_path, root, payload)
        rows.append({
            "output_root": output_root,
            "model": model,
            "sparsity_type": sparsity_type,
            "method": method,
            "target_sparsity": target_sparsity,
            "actual_sparsity": as_float(payload.get("actual_sparsity")),
            "ppl_test": ppl,
            "Hyper_m": as_float(payload.get("Hyper_m")),
            "Lamda": as_float(payload.get("Lamda")),
            "Owl_alpha": as_float(payload.get("Owl_alpha")),
            "pruning_time_sec": as_float(payload.get("pruning_time_sec")),
            "run_dir": payload.get("run_dir", str(result_path.parent)),
            "stdout_path": resolve_stdout_path(result_path, payload),
            "result_path": result_path,
            "payload": payload,
        })
    return rows


def select_rows(rows, mode, sparsities, methods, output_roots, models):
    def keep(row):
        if sparsities and row["target_sparsity"] not in sparsities:
            return False
        if methods and row["method"] not in methods:
            return False
        if output_roots and row["output_root"] not in output_roots:
            return False
        if models and row["model"] not in models:
            return False
        return True

    rows = [row for row in rows if keep(row)]
    if mode == "all":
        return rows

    grouped = defaultdict(list)
    for row in rows:
        key = (row["output_root"], row["model"], row["sparsity_type"], row["method"], row["target_sparsity"])
        grouped[key].append(row)

    selected = []
    for group in grouped.values():
        candidates = [row for row in group if row["ppl_test"] is not None]
        if candidates:
            selected.append(min(candidates, key=lambda row: row["ppl_test"]))
    return sorted(selected, key=lambda row: (
        row["output_root"], row["model"], row["method"], row["target_sparsity"]
    ))


def value_at(values, index):
    if values and index < len(values):
        return values[index]
    return ""


def build_layer_rows(selected_rows):
    layer_rows = []
    selected_meta = []

    for row in selected_rows:
        parsed = parse_stdout(row["stdout_path"])
        raw_rows = parsed.get("raw_rows", {})

        max_len = 0
        for values in parsed.values():
            if isinstance(values, list):
                max_len = max(max_len, len(values))
        if raw_rows:
            max_len = max(max_len, max(raw_rows) + 1)
        if max_len == 0:
            continue

        selected_meta.append(row)
        for layer in range(max_len):
            raw = raw_rows.get(layer, {})
            layer_rows.append({
                "output_root": row["output_root"],
                "model": row["model"],
                "method": row["method"],
                "target_sparsity": row["target_sparsity"],
                "actual_sparsity": row["actual_sparsity"],
                "ppl_test": row["ppl_test"],
                "Hyper_m": row["Hyper_m"],
                "Lamda": row["Lamda"],
                "Owl_alpha": row["Owl_alpha"],
                "pruning_time_sec": row["pruning_time_sec"],
                "layer": layer,
                "raw_count_ratio": raw.get("raw_count_ratio", value_at(parsed.get("raw_count_ratio"), layer)),
                "raw_severity_ratio": raw.get("raw_severity_ratio", value_at(parsed.get("raw_severity_ratio"), layer)),
                "raw_severity": raw.get("raw_severity", ""),
                "tail_mass_ratio": raw.get("tail_mass_ratio", ""),
                "count_density": value_at(parsed.get("count_density"), layer),
                "severity_density": value_at(parsed.get("severity_density"), layer),
                "anchored_density": value_at(parsed.get("anchored_density"), layer),
                "adjusted_density": value_at(parsed.get("adjusted_density"), layer),
                "adjusted_sparsity": value_at(parsed.get("adjusted_sparsity"), layer),
                "run_dir": row["run_dir"],
                "stdout_path": str(row["stdout_path"]),
            })
    return layer_rows, selected_meta


def write_csv(path, rows):
    fields = [
        "output_root", "model", "method", "target_sparsity", "actual_sparsity",
        "ppl_test", "Hyper_m", "Lamda", "Owl_alpha", "pruning_time_sec",
        "layer", "raw_count_ratio", "raw_severity_ratio", "raw_severity",
        "tail_mass_ratio", "count_density", "severity_density", "anchored_density",
        "adjusted_density", "adjusted_sparsity", "run_dir", "stdout_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"matplotlib unavailable, skip plotting: {exc}")
        return None


def numeric_series(rows, key):
    xs, ys = [], []
    for row in rows:
        value = as_float(row.get(key))
        if value is not None and not math.isnan(value):
            xs.append(int(row["layer"]))
            ys.append(value)
    return xs, ys


def plot_individual(layer_rows, out_dir):
    plt = import_matplotlib()
    if plt is None:
        return

    grouped = defaultdict(list)
    for row in layer_rows:
        key = (row["output_root"], row["model"], row["method"], row["target_sparsity"])
        grouped[key].append(row)

    for key, rows in grouped.items():
        output_root, model, method, sparsity = key
        rows = sorted(rows, key=lambda r: int(r["layer"]))
        prefix = safe_name(f"{output_root}_{model}_{method}_s{sparsity}")

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for metric, label in [
            ("adjusted_density", "Adjusted Density"),
            ("count_density", "Count-OWL Density"),
            ("severity_density", "Severity-OWL Density"),
            ("anchored_density", "Anchored Density"),
        ]:
            xs, ys = numeric_series(rows, metric)
            if xs:
                axes[0].plot(xs, ys, marker="o", linewidth=1.4, markersize=3, label=label)
        axes[0].set_ylabel("Density")
        axes[0].set_title(f"{model} {method} s={sparsity} Layer Density")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)

        for metric, label in [
            ("adjusted_sparsity", "Adjusted Sparsity"),
        ]:
            xs, ys = numeric_series(rows, metric)
            if xs:
                axes[1].plot(xs, ys, marker="o", linewidth=1.4, markersize=3, label=label)
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Sparsity")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_density_sparsity.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        for metric, label in [
            ("raw_count_ratio", "Raw Count Ratio"),
            ("raw_severity_ratio", "Severity-Weighted Ratio"),
            ("tail_mass_ratio", "Tail Mass Ratio"),
        ]:
            xs, ys = numeric_series(rows, metric)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.4, markersize=3, label=label)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Ratio (%)")
        ax.set_title(f"{model} {method} s={sparsity} Outlier Ratios")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_outlier_ratios.png", dpi=200)
        plt.close(fig)


def plot_comparison(layer_rows, out_dir):
    plt = import_matplotlib()
    if plt is None:
        return

    grouped = defaultdict(list)
    for row in layer_rows:
        key = (row["model"], row["target_sparsity"])
        grouped[key].append(row)

    for key, rows in grouped.items():
        model, sparsity = key
        by_run = defaultdict(list)
        for row in rows:
            run_key = (row["output_root"], row["method"])
            by_run[run_key].append(row)

        if len(by_run) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for run_key, run_rows in sorted(by_run.items()):
            output_root, method = run_key
            xs, ys = numeric_series(sorted(run_rows, key=lambda r: int(r["layer"])), "adjusted_density")
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3, label=f"{output_root}:{method}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Adjusted Density")
        ax.set_title(f"{model} s={sparsity} Layer Density Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_{safe_name(model)}_s{sparsity}_density.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Extract and plot layer-wise OWL density/sparsity distributions.")
    parser.add_argument("--root", default="owl", help="Directory to scan.")
    parser.add_argument("--out-dir", default="owl/analysis/layer_distribution")
    parser.add_argument("--select", choices=["best", "all"], default="best")
    parser.add_argument("--sparsities", nargs="*", type=float, default=[0.5, 0.6, 0.7])
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--output-roots", nargs="*", default=[])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_success_results(root)
    selected = select_rows(
        rows,
        mode=args.select,
        sparsities=set(args.sparsities) if args.sparsities else set(),
        methods=set(args.methods),
        output_roots=set(args.output_roots),
        models=set(args.models),
    )
    layer_rows, selected_meta = build_layer_rows(selected)
    write_csv(out_dir / "layer_distribution.csv", layer_rows)

    print(f"Selected runs: {len(selected_meta)}")
    print(f"Layer rows: {len(layer_rows)}")
    print(f"CSV: {out_dir / 'layer_distribution.csv'}")

    if not args.no_plots:
        plot_individual(layer_rows, out_dir)
        plot_comparison(layer_rows, out_dir)
        print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
