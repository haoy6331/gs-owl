#!/usr/bin/env python3
"""Print the best PLLaMA plant-MCQ parameter setting per sparsity."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize in-memory PLLaMA MCQ sweep results.")
    parser.add_argument("--root", default="owl/pllama_plant_mcq_param_sweep")
    parser.add_argument("--dense-result", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_dense_accuracy(path):
    if not path or not Path(path).is_file():
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "accuracy" in payload:
        return float(payload["accuracy"])
    evaluations = payload.get("evaluations", [])
    for item in evaluations:
        if item.get("model") == "dense":
            return float(item["accuracy"])
    return None


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.rglob("plant_mcq_result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[skip] invalid result {path}: {exc}")
            continue
        if payload.get("metric") != "multiple-choice accuracy":
            continue
        payload["result_path"] = str(path)
        rows.append(payload)

    if not rows:
        raise RuntimeError(f"No plant_mcq_result.json found under {root}")

    dense_accuracy = load_dense_accuracy(args.dense_result)
    rows.sort(key=lambda row: (float(row.get("sparsity_ratio", 0)), -float(row["accuracy"])))
    best_by_sparsity = {}
    for row in rows:
        sparsity = f"{float(row['sparsity_ratio']):.1f}"
        if sparsity not in best_by_sparsity:
            best_by_sparsity[sparsity] = row

    print("\nPLLaMA plant-science MCQ parameter sweep")
    print("=" * 108)
    print(f"{'sparsity':<10} {'Hyper_m':>8} {'Lamda':>9} {'alpha':>9} {'accuracy':>12} {'delta(pp)':>12} {'result':<40}")
    print("-" * 108)
    for sparsity, row in sorted(best_by_sparsity.items(), key=lambda item: float(item[0])):
        delta = "--"
        if dense_accuracy is not None:
            delta = f"{100 * (float(row['accuracy']) - dense_accuracy):+.3f}"
        print(
            f"{sparsity:<10} {float(row['Hyper_m']):>8.2f} {float(row['Lamda']):>9.3f} "
            f"{float(row['Owl_alpha']):>9.2f} {float(row['accuracy']):>11.4%} "
            f"{delta:>12} {row['result_path']:<40}"
        )

    print("\nAll completed candidates")
    print("-" * 108)
    for row in rows:
        print(
            f"s={float(row['sparsity_ratio']):.1f} "
            f"hm={float(row['Hyper_m']):.2f} "
            f"la={float(row['Lamda']):.3f} "
            f"alpha={float(row['Owl_alpha']):.2f} "
            f"accuracy={float(row['accuracy']):.4%} "
            f"correct={row['correct']}/{row['n_questions']}"
        )

    summary = {
        "root": str(root),
        "n_completed": len(rows),
        "dense_accuracy": dense_accuracy,
        "best_by_sparsity": best_by_sparsity,
        "all_results": rows,
    }
    output = Path(args.output) if args.output else root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary: {output}")


if __name__ == "__main__":
    main()
