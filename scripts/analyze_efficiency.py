import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


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
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_stdout_metric(stdout_path, pattern):
    if not stdout_path or not stdout_path.exists():
        return None
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, text)
    if not match:
        return None
    return as_float(match.group(1))


def collect_results(root):
    rows = []
    for path in root.rglob("result.json"):
        payload = load_json(path)
        if not payload:
            continue
        output_root, model, sparsity_type, method = infer_identity(path, root, payload)
        stdout_path = resolve_stdout_path(path, payload)

        pruning_time = as_float(payload.get("pruning_time_sec"))
        if pruning_time is None:
            pruning_time = parse_stdout_metric(stdout_path, r"pruning time:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

        actual_sparsity = as_float(payload.get("actual_sparsity"))
        if actual_sparsity is None:
            actual_sparsity = parse_stdout_metric(stdout_path, r"sparsity sanity check\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

        ppl = as_float(payload.get("ppl_test"))
        if ppl is None:
            ppl = parse_stdout_metric(stdout_path, r"wikitext perplexity\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

        rows.append({
            "output_root": output_root,
            "model": model,
            "sparsity_type": sparsity_type,
            "method": method,
            "status": payload.get("status", "missing_status"),
            "target_sparsity": as_float(payload.get("target_sparsity")),
            "actual_sparsity": actual_sparsity,
            "sparsity_error": None,
            "ppl_test": ppl,
            "Hyper_m": as_float(payload.get("Hyper_m")),
            "Lamda": as_float(payload.get("Lamda")),
            "Owl_alpha": as_float(payload.get("Owl_alpha")),
            "pruning_time_sec": pruning_time,
            "elapsed_sec": as_float(payload.get("elapsed_sec")),
            "returncode": payload.get("returncode", ""),
            "run_dir": payload.get("run_dir", str(path.parent)),
            "result_json": str(path),
            "stdout_log": str(stdout_path) if stdout_path else "",
        })

    for row in rows:
        if row["target_sparsity"] is not None and row["actual_sparsity"] is not None:
            row["sparsity_error"] = row["actual_sparsity"] - row["target_sparsity"]
    return rows


def mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return ""
    return sum(values) / len(values)


def std(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if len(values) < 2:
        return ""
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["target_sparsity"] is None:
            continue
        key = (
            row["output_root"],
            row["model"],
            row["sparsity_type"],
            row["method"],
            row["target_sparsity"],
        )
        grouped[key].append(row)

    summary = []
    best_rows = []
    for key, group in sorted(grouped.items()):
        output_root, model, sparsity_type, method, target_sparsity = key
        success = [row for row in group if row["status"] == "success"]
        failed = [row for row in group if row["status"] != "success"]
        with_ppl = [row for row in success if row["ppl_test"] is not None]
        best = min(with_ppl, key=lambda row: row["ppl_test"]) if with_ppl else None
        if best:
            best_rows.append(best)

        times = [row["pruning_time_sec"] for row in success]
        actuals = [row["actual_sparsity"] for row in success]
        errors = [row["sparsity_error"] for row in success]
        ppls = [row["ppl_test"] for row in success]

        summary.append({
            "output_root": output_root,
            "model": model,
            "sparsity_type": sparsity_type,
            "method": method,
            "target_sparsity": target_sparsity,
            "success": len(success),
            "failed": len(failed),
            "seen": len(group),
            "mean_pruning_time_sec": mean(times),
            "std_pruning_time_sec": std(times),
            "min_pruning_time_sec": min([t for t in times if t is not None], default=""),
            "max_pruning_time_sec": max([t for t in times if t is not None], default=""),
            "mean_actual_sparsity": mean(actuals),
            "mean_sparsity_error": mean(errors),
            "mean_ppl": mean(ppls),
            "best_ppl": best["ppl_test"] if best else "",
            "best_actual_sparsity": best["actual_sparsity"] if best else "",
            "best_pruning_time_sec": best["pruning_time_sec"] if best else "",
            "best_Hyper_m": best["Hyper_m"] if best else "",
            "best_Lamda": best["Lamda"] if best else "",
            "best_Owl_alpha": best["Owl_alpha"] if best else "",
            "best_run_dir": best["run_dir"] if best else "",
        })
    return summary, best_rows


def write_csv(path, rows, fields):
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


def plot_efficiency(summary_rows, out_dir):
    plt = import_matplotlib()
    if plt is None:
        return

    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[(row["model"], row["method"])].append(row)

    for (model, method), rows in grouped.items():
        rows = sorted(rows, key=lambda row: (row["output_root"], row["target_sparsity"]))
        if not rows:
            continue

        labels = [f"{row['output_root']}\ns={row['target_sparsity']}" for row in rows]
        times = [as_float(row["mean_pruning_time_sec"], 0.0) for row in rows]
        ppls = [as_float(row["best_ppl"], 0.0) for row in rows]

        fig, ax1 = plt.subplots(figsize=(max(9, len(rows) * 0.9), 4.8))
        x = list(range(len(rows)))
        ax1.bar(x, times, alpha=0.75, label="Mean pruning time (s)")
        ax1.set_ylabel("Pruning time (s)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(x, ppls, color="tab:red", marker="o", linewidth=1.4, label="Best PPL")
        ax2.set_ylabel("Best PPL")
        ax1.set_title(f"{model} {method} Efficiency")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"efficiency_{safe_name(model)}_{safe_name(method)}.png", dpi=200)
        plt.close(fig)


def print_summary(summary_rows):
    print("\nEfficiency summary")
    print("-" * 100)
    if not summary_rows:
        print("No result.json rows found.")
        return
    for row in summary_rows:
        print(
            f"{row['output_root']} | {row['model']} | {row['method']} | "
            f"s={row['target_sparsity']} | success={row['success']} failed={row['failed']} | "
            f"time_mean={row['mean_pruning_time_sec']} | best_ppl={row['best_ppl']} | "
            f"actual={row['best_actual_sparsity']} | hm={row['best_Hyper_m']} "
            f"la={row['best_Lamda']} alpha={row['best_Owl_alpha']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze pruning efficiency from saved result.json files.")
    parser.add_argument("--root", default="owl", help="Directory to scan.")
    parser.add_argument("--out-dir", default="owl/analysis/efficiency")
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--output-roots", nargs="*", default=[])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--sparsities", nargs="*", type=float, default=[])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    rows = collect_results(Path(args.root))
    if args.methods:
        rows = [row for row in rows if row["method"] in set(args.methods)]
    if args.output_roots:
        rows = [row for row in rows if row["output_root"] in set(args.output_roots)]
    if args.models:
        rows = [row for row in rows if row["model"] in set(args.models)]
    if args.sparsities:
        wanted = set(args.sparsities)
        rows = [row for row in rows if row["target_sparsity"] in wanted]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, best_rows = summarize(rows)

    write_csv(out_dir / "efficiency_all_results.csv", rows, [
        "output_root", "model", "sparsity_type", "method", "status",
        "target_sparsity", "actual_sparsity", "sparsity_error", "ppl_test",
        "Hyper_m", "Lamda", "Owl_alpha", "pruning_time_sec", "elapsed_sec",
        "returncode", "run_dir", "result_json", "stdout_log",
    ])
    write_csv(out_dir / "efficiency_summary.csv", summary_rows, [
        "output_root", "model", "sparsity_type", "method", "target_sparsity",
        "success", "failed", "seen", "mean_pruning_time_sec", "std_pruning_time_sec",
        "min_pruning_time_sec", "max_pruning_time_sec", "mean_actual_sparsity",
        "mean_sparsity_error", "mean_ppl", "best_ppl", "best_actual_sparsity",
        "best_pruning_time_sec", "best_Hyper_m", "best_Lamda", "best_Owl_alpha",
        "best_run_dir",
    ])
    write_csv(out_dir / "efficiency_best_runs.csv", best_rows, [
        "output_root", "model", "sparsity_type", "method", "target_sparsity",
        "actual_sparsity", "sparsity_error", "ppl_test", "Hyper_m", "Lamda",
        "Owl_alpha", "pruning_time_sec", "run_dir", "result_json", "stdout_log",
    ])

    print_summary(summary_rows)
    print(f"\nCSV written to: {out_dir}")

    if not args.no_plots:
        plot_efficiency(summary_rows, out_dir)
        print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
