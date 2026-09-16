#!/usr/bin/env python3
"""Summarize dense and baseline PLLaMA plant-MCQ results."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize PLLaMA baseline MCQ results.")
    parser.add_argument("--root", default="owl/pllama_plant_mcq_baselines")
    parser.add_argument("--dense-result", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_payload(path):
    if not path or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dense_accuracy(payload):
    if payload is None:
        return None
    if "accuracy" in payload:
        return float(payload["accuracy"])
    for item in payload.get("evaluations", []):
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
        raise RuntimeError(f"No baseline results found under {root}")

    dense = dense_accuracy(load_payload(args.dense_result))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("pruning_method", row.get("model", "unknown")),
                 f"{float(row['sparsity_ratio']):.1f}")].append(row)

    best_by_method = {}
    for key, candidates in grouped.items():
        best_by_method[key] = max(candidates, key=lambda row: float(row["accuracy"]))

    print("\nPLLaMA plant-science MCQ baselines")
    print("=" * 112)
    print(f"{'method':<14} {'sparsity':<10} {'accuracy':>12} {'delta(pp)':>12} {'correct':>12} {'result':<42}")
    print("-" * 112)
    for (method, sparsity), row in sorted(best_by_method.items(), key=lambda item: (item[0][0], float(item[0][1]))):
        delta = "--" if dense is None else f"{100 * (float(row['accuracy']) - dense):+.3f}"
        print(
            f"{method:<14} {sparsity:<10} {float(row['accuracy']):>11.4%} "
            f"{delta:>12} {str(row['correct']) + '/' + str(row['n_questions']):>12} "
            f"{row['result_path']:<42}"
        )

    output_payload = {
        "root": str(root),
        "dense_accuracy": dense,
        "n_completed": len(rows),
        "best_by_method_and_sparsity": {
            f"{method}|s={sparsity}": row
            for (method, sparsity), row in best_by_method.items()
        },
        "all_results": rows,
    }
    output = Path(args.output) if args.output else root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary: {output}")


if __name__ == "__main__":
    main()
