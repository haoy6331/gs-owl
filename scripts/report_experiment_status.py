import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SPECIAL_EXPECTED_RUNS = {
    "nsamples_sensitivity_wsqrtg_v9": 4,
    "stability_wsqrtg_v9": 3,
    "stability_wanda_owl_v9": 3,
    "owl_v9_metric_a40": 3,
    "strict_ablation_wsqrtg_owl_v9": 3,
    "strict_ablation_zero_shot": 3,
    "layer_sensitivity_wsqrtg": 32,
    "v9_component_ablation": 8,
    "gradient_beta_ablation": 5,
    "c4_validation_gs_owl": 3,
}

OSLA_MECHANISM_REQUIRED_VARIANTS = {
    "projection": {"projection_none", "projection_uniform"},
    "boundary": {"boundary_left2", "boundary_right2"},
    "depth": {"depth_normalized"},
}


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"status": "bad_json", "error": str(exc)}


def as_float(value, default=None):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def csv_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fields})


def rel_parts(path, root):
    try:
        return path.relative_to(root).parts
    except ValueError:
        return path.parts


def infer_param_group(path, root):
    parts = rel_parts(path, root)
    output_root = parts[0] if len(parts) > 0 else ""
    model = parts[1] if len(parts) > 1 else ""
    sparsity_type = parts[2] if len(parts) > 2 else ""
    method_from_path = parts[3] if len(parts) > 3 else ""
    return output_root, model, sparsity_type, method_from_path


def collect_param_results(root):
    rows = []
    for path in root.rglob("result.json"):
        payload = load_json(path)
        if payload.get("experiment") == "osla_mechanism_ablation":
            continue
        output_root, model, sparsity_type, method_from_path = infer_param_group(path, root)
        rows.append({
            "output_root": output_root,
            "model": model,
            "sparsity_type": payload.get("sparsity_type", sparsity_type),
            "method": payload.get("method", method_from_path),
            "eval_dataset": payload.get("eval_dataset", "wikitext2"),
            "status": payload.get("status", "missing_status"),
            "target_sparsity": payload.get("target_sparsity", ""),
            "actual_sparsity": payload.get("actual_sparsity", ""),
            "ppl_test": payload.get("ppl_test", ""),
            "ppl_c4": payload.get("ppl_c4", ""),
            "Hyper_m": payload.get("Hyper_m", ""),
            "Lamda": payload.get("Lamda", ""),
            "Owl_alpha": payload.get("Owl_alpha", ""),
            "pruning_time_sec": payload.get("pruning_time_sec", ""),
            "elapsed_sec": payload.get("elapsed_sec", ""),
            "returncode": payload.get("returncode", ""),
            "result_json": str(path),
            "run_dir": payload.get("run_dir", str(path.parent)),
        })
    return rows


def collect_osla_mechanism_results(root):
    rows = []
    for path in root.rglob("result.json"):
        payload = load_json(path)
        if payload.get("experiment") != "osla_mechanism_ablation":
            continue
        rows.append({
            "suite": payload.get("suite", ""),
            "variant": payload.get("variant", ""),
            "model_key": payload.get("model_key", ""),
            "status": payload.get("status", "missing_status"),
            "target_sparsity": payload.get("target_sparsity", ""),
            "ppl_test": payload.get("ppl_test", ""),
            "returncode": payload.get("returncode", ""),
            "result_json": str(path),
            "run_dir": payload.get("run_dir", str(path.parent)),
        })
    return rows


def summarize_osla_mechanism_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["model_key"])].append(row)

    summary = []
    for (suite, model_key), group_rows in sorted(grouped.items()):
        required = OSLA_MECHANISM_REQUIRED_VARIANTS.get(suite)
        if required is None:
            required = {row["variant"] for row in group_rows}
        successful_variants = {
            row["variant"] for row in group_rows if row.get("status") == "success"
        }
        failed_variants = {
            row["variant"] for row in group_rows if row.get("status") == "failed"
        }
        summary.append({
            "kind": "osla_mechanism",
            "suite": suite,
            "model_key": model_key,
            "expected_runs": len(required),
            "success": len(required & successful_variants),
            "failed": len(required & failed_variants),
            "missing_variants": sorted(required - successful_variants),
            "completed_variants": sorted(successful_variants),
        })
    return summary


def best_param_rows(rows):
    best = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        ppl = as_float(row.get("ppl_test"))
        sparsity = as_float(row.get("target_sparsity"))
        if ppl is None or sparsity is None:
            continue
        key = (row["output_root"], row["model"], row["sparsity_type"], row["method"], sparsity)
        current = best.get(key)
        if current is None or ppl < as_float(current.get("ppl_test"), 1e18):
            best[key] = row
    return sorted(best.values(), key=lambda r: (
        r["output_root"],
        r["model"],
        r["method"],
        as_float(r.get("target_sparsity"), 0.0),
    ))


def summarize_param_rows(rows, expected_runs):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["output_root"], row["model"], row["sparsity_type"], row["method"])
        grouped[key].append(row)

    summary = []
    for key, group_rows in sorted(grouped.items()):
        output_root, model, sparsity_type, method = key
        if "baseline" in output_root.lower():
            group_expected_runs = 1 if method == "dense" else 3
        else:
            group_expected_runs = SPECIAL_EXPECTED_RUNS.get(output_root, expected_runs)
        status_counts = Counter(row.get("status", "missing_status") for row in group_rows)
        sparsity_counts = Counter(str(row.get("target_sparsity", "")) for row in group_rows)
        success_rows = [r for r in group_rows if r.get("status") == "success"]
        completed = len(success_rows)
        total_seen = len(group_rows)
        summary.append({
            "kind": "ppl",
            "output_root": output_root,
            "model": model,
            "sparsity_type": sparsity_type,
            "method": method,
            "seen_result_json": total_seen,
            "expected_runs": group_expected_runs,
            "success": completed,
            "failed": status_counts.get("failed", 0),
            "bad_json": status_counts.get("bad_json", 0),
            "other_status": sum(v for k, v in status_counts.items() if k not in {"success", "failed", "bad_json"}),
            "remaining_by_expected": max(group_expected_runs - completed, 0) if group_expected_runs else "",
            "status_counts": dict(status_counts),
            "sparsity_counts": dict(sorted(sparsity_counts.items())),
        })
    return summary


def collect_zero_shot_results(root):
    rows = []
    for path in root.rglob("zero_shot_result.json"):
        payload = load_json(path)
        parts = rel_parts(path, root)
        output_root = parts[0] if len(parts) > 0 else ""
        scores = payload.get("scores", {})
        numeric_scores = {k: as_float(v) for k, v in scores.items() if as_float(v) is not None}
        avg = payload.get("zero_shot_avg")
        if avg in (None, "") and numeric_scores:
            avg = sum(numeric_scores.values()) / len(numeric_scores)
        rows.append({
            "output_root": output_root,
            "label": payload.get("label", parts[1] if len(parts) > 1 else ""),
            "status": payload.get("status", "missing_status"),
            "target_sparsity": payload.get("target_sparsity", ""),
            "rank_in_sparsity": payload.get("rank_in_sparsity", ""),
            "zero_shot_avg": avg if avg is not None else "",
            "source_ppl": payload.get("source_ppl", ""),
            "actual_sparsity_ppl": payload.get("actual_sparsity_ppl", ""),
            "Hyper_m": payload.get("Hyper_m", ""),
            "Lamda": payload.get("Lamda", ""),
            "Owl_alpha": payload.get("Owl_alpha", ""),
            "task_id": payload.get("task_id", ""),
            "returncode": payload.get("returncode", ""),
            "result_json": str(path),
            "save_dir": payload.get("save_dir", str(path.parent)),
            "scores": numeric_scores,
        })
    return rows


def summarize_zero_shot_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["output_root"], row["label"])
        grouped[key].append(row)

    summary = []
    for key, group_rows in sorted(grouped.items()):
        output_root, label = key
        status_counts = Counter(row.get("status", "missing_status") for row in group_rows)
        sparsity_counts = Counter(str(row.get("target_sparsity", "")) for row in group_rows)
        summary.append({
            "kind": "zero_shot",
            "output_root": output_root,
            "label": label,
            "seen_result_json": len(group_rows),
            "success": status_counts.get("success", 0),
            "failed": status_counts.get("failed", 0),
            "bad_json": status_counts.get("bad_json", 0),
            "other_status": sum(v for k, v in status_counts.items() if k not in {"success", "failed", "bad_json"}),
            "status_counts": dict(status_counts),
            "sparsity_counts": dict(sorted(sparsity_counts.items())),
        })
    return summary


def best_zero_shot_rows(rows):
    best = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        avg = as_float(row.get("zero_shot_avg"))
        sparsity = as_float(row.get("target_sparsity"))
        if avg is None or sparsity is None:
            continue
        key = (row["output_root"], row["label"], sparsity)
        current = best.get(key)
        current_avg = as_float(current.get("zero_shot_avg"), -1e18) if current else -1e18
        if current is None or avg > current_avg:
            best[key] = row
    return sorted(best.values(), key=lambda r: (
        r["output_root"],
        r["label"],
        as_float(r.get("target_sparsity"), 0.0),
    ))


def format_sparsity_counts(counts):
    if not counts:
        return "-"
    pieces = []
    for sparsity, count in counts.items():
        if sparsity:
            pieces.append(f"s={sparsity}:{count}")
    return ", ".join(pieces) if pieces else "-"


def print_param_summary(summary):
    print("\nPPL parameter sweeps")
    print("-" * 80)
    if not summary:
        print("No result.json files found.")
        return
    for row in summary:
        expected = row["expected_runs"]
        missing = max(expected - row["success"], 0) if expected else 0
        tag = "[DONE]" if expected and row["success"] >= expected and row["failed"] == 0 else "[TODO]"
        print(
            f"{tag} {row['output_root']} | {row['model']} | {row['method']} | "
            f"success {row['success']}/{expected}, failed {row['failed']}, missing {missing} | "
            f"{format_sparsity_counts(row['sparsity_counts'])}"
        )


def print_best_param_rows(best_rows):
    print("\nBest PPL by sparsity")
    print("-" * 80)
    if not best_rows:
        print("No successful PPL rows found.")
        return
    for row in best_rows:
        print(
            f"{row['output_root']} | {row['model']} | s={row['target_sparsity']} | "
            f"ppl={row['ppl_test']} | hm={row['Hyper_m']} la={row['Lamda']} alpha={row['Owl_alpha']}"
        )


def print_osla_mechanism_summary(summary):
    print("\nOSLA mechanism experiments")
    print("-" * 80)
    if not summary:
        print("No OSLA mechanism result.json files found.")
        return
    for row in summary:
        done = row["success"] >= row["expected_runs"] and row["failed"] == 0
        tag = "[DONE]" if done else "[TODO]"
        missing = ",".join(row["missing_variants"]) or "-"
        print(
            f"{tag} {row['suite']} | {row['model_key']} | "
            f"success {row['success']}/{row['expected_runs']}, "
            f"failed {row['failed']} | missing: {missing}"
        )


def print_zero_shot_summary(summary, expected_zero_shot):
    print("\nZero-shot TopK runs")
    print("-" * 80)
    if not summary:
        print("No zero_shot_result.json files found.")
        return
    for row in summary:
        missing = max(expected_zero_shot - row["success"], 0) if expected_zero_shot else 0
        tag = "[DONE]" if expected_zero_shot and row["success"] >= expected_zero_shot and row["failed"] == 0 else "[TODO]"
        print(
            f"{tag} {row['output_root']} | {row['label']} | "
            f"success {row['success']}/{expected_zero_shot}, failed {row['failed']}, missing {missing} | "
            f"{format_sparsity_counts(row['sparsity_counts'])}"
        )


def print_best_zero_shot_rows(best_rows):
    print("\nBest zero-shot average by sparsity")
    print("-" * 80)
    if not best_rows:
        print("No successful zero-shot rows found.")
        return
    for row in best_rows:
        print(
            f"{row['output_root']} | {row['label']} | s={row['target_sparsity']} | "
            f"avg={row['zero_shot_avg']} | source_ppl={row['source_ppl']} | "
            f"rank={row['rank_in_sparsity']} | hm={row['Hyper_m']} la={row['Lamda']} alpha={row['Owl_alpha']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Report finished PPL sweeps and zero-shot experiments.")
    parser.add_argument("--root", default="owl", help="Directory to scan, usually owl.")
    parser.add_argument("--expected-runs", type=int, default=360, help="Expected PPL runs per full parameter sweep.")
    parser.add_argument("--expected-zero-shot", type=int, default=30, help="Expected zero-shot TopK runs per model label.")
    parser.add_argument("--out-dir", default=None, help="Directory for CSV reports. Default: <root>/experiment_status.")
    parser.add_argument("--write-csv", action="store_true", help="Write CSV reports. Default only prints status.")
    parser.add_argument("--show-best", action="store_true", help="Print best PPL and zero-shot rows.")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "experiment_status"

    param_rows = collect_param_results(root)
    param_summary = summarize_param_rows(param_rows, args.expected_runs)
    param_best = best_param_rows(param_rows)

    osla_rows = collect_osla_mechanism_results(root)
    osla_summary = summarize_osla_mechanism_rows(osla_rows)

    zero_rows = collect_zero_shot_results(root)
    zero_summary = summarize_zero_shot_rows(zero_rows)
    zero_best = best_zero_shot_rows(zero_rows)

    print_param_summary(param_summary)
    print_osla_mechanism_summary(osla_summary)
    print_zero_shot_summary(zero_summary, args.expected_zero_shot)

    if args.show_best:
        print_best_param_rows(param_best)
        print_best_zero_shot_rows(zero_best)

    if args.write_csv:
        write_csv(out_dir / "ppl_results_all.csv", param_rows, [
            "output_root", "model", "sparsity_type", "method", "eval_dataset", "status",
            "target_sparsity", "actual_sparsity", "ppl_test", "ppl_c4",
            "Hyper_m", "Lamda", "Owl_alpha", "pruning_time_sec", "elapsed_sec",
            "returncode", "result_json", "run_dir",
        ])
        write_csv(out_dir / "ppl_sweep_summary.csv", param_summary, [
            "kind", "output_root", "model", "sparsity_type", "method",
            "seen_result_json", "expected_runs", "success", "failed", "bad_json",
            "other_status", "remaining_by_expected", "status_counts", "sparsity_counts",
        ])
        write_csv(out_dir / "ppl_best_by_sparsity.csv", param_best, [
            "output_root", "model", "sparsity_type", "method", "status",
            "target_sparsity", "actual_sparsity", "ppl_test",
            "Hyper_m", "Lamda", "Owl_alpha", "pruning_time_sec",
            "result_json", "run_dir",
        ])

        write_csv(out_dir / "osla_mechanism_results.csv", osla_rows, [
            "suite", "variant", "model_key", "status", "target_sparsity",
            "ppl_test", "returncode", "result_json", "run_dir",
        ])
        write_csv(out_dir / "osla_mechanism_summary.csv", osla_summary, [
            "kind", "suite", "model_key", "expected_runs", "success",
            "failed", "missing_variants", "completed_variants",
        ])

        write_csv(out_dir / "zero_shot_results_all.csv", zero_rows, [
            "output_root", "label", "status", "target_sparsity", "rank_in_sparsity",
            "zero_shot_avg", "source_ppl", "actual_sparsity_ppl",
            "Hyper_m", "Lamda", "Owl_alpha", "task_id", "returncode",
            "scores", "result_json", "save_dir",
        ])
        write_csv(out_dir / "zero_shot_summary.csv", zero_summary, [
            "kind", "output_root", "label", "seen_result_json", "success",
            "failed", "bad_json", "other_status", "status_counts", "sparsity_counts",
        ])
        write_csv(out_dir / "zero_shot_best_by_sparsity.csv", zero_best, [
            "output_root", "label", "status", "target_sparsity", "rank_in_sparsity",
            "zero_shot_avg", "source_ppl", "actual_sparsity_ppl",
            "Hyper_m", "Lamda", "Owl_alpha", "scores", "result_json", "save_dir",
        ])
        print(f"\nCSV reports written to: {out_dir}")


if __name__ == "__main__":
    main()
