#!/usr/bin/env python3
"""Validate and summarize one fixed GS-OWL plant-MCQ run per sparsity."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize fixed-protocol PLLaMA plant-MCQ results.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dense-result", required=True)
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    parser.add_argument("--hyper-m", type=float, default=5.0)
    parser.add_argument("--lamda", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--require-gpu-substring", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def close(actual, expected):
    return abs(float(actual) - float(expected)) <= 1e-12


def load_dense(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "accuracy" in payload:
        return payload, payload
    evaluations = payload.get("evaluations", [])
    dense = next((item for item in evaluations if item.get("model") == "dense"), None)
    if dense is None and len(evaluations) == 1:
        dense = evaluations[0]
    if dense is None:
        raise RuntimeError(f"No dense evaluation found in {path}")
    return payload, dense


def main():
    args = parse_args()
    root = Path(args.root)
    dense_payload, dense = load_dense(args.dense_result)
    dense_accuracy = float(dense["accuracy"])
    dense_hash = dense_payload.get("question_set_sha256") or dense.get("question_set_sha256")
    dense_gpu_names = dense_payload.get("gpu_names", dense.get("gpu_names", []))
    if args.require_gpu_substring and (
        not dense_gpu_names or any(args.require_gpu_substring not in name for name in dense_gpu_names)
    ):
        raise RuntimeError(f"Dense result was not recorded on the required GPU type: {dense_gpu_names}")

    by_sparsity = {}
    for path in sorted(root.rglob("plant_mcq_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sparsity = float(payload["sparsity_ratio"])
        if not any(close(sparsity, expected) for expected in args.sparsities):
            continue
        for key, actual, expected in (
            ("Hyper_m", payload.get("Hyper_m"), args.hyper_m),
            ("Lamda", payload.get("Lamda"), args.lamda),
            ("Owl_alpha", payload.get("Owl_alpha"), args.alpha),
            ("Grad_beta", payload.get("Grad_beta"), args.beta),
        ):
            if actual is None or not close(actual, expected):
                raise RuntimeError(f"Unexpected {key}={actual!r} in {path}; expected {expected}")
        question_hash = payload.get("question_set_sha256")
        if dense_hash and question_hash != dense_hash:
            raise RuntimeError(f"Question-set hash differs between dense and sparse results: {path}")
        gpu_names = payload.get("gpu_names", [])
        if args.require_gpu_substring and (
            not gpu_names or any(args.require_gpu_substring not in name for name in gpu_names)
        ):
            raise RuntimeError(f"Result was not recorded on the required GPU type in {path}: {gpu_names}")
        key = next(expected for expected in args.sparsities if close(sparsity, expected))
        if key in by_sparsity:
            raise RuntimeError(f"More than one fixed result found for sparsity {key}: {path}")
        payload["result_path"] = str(path)
        by_sparsity[key] = payload

    missing = [sparsity for sparsity in args.sparsities if sparsity not in by_sparsity]
    if missing:
        raise RuntimeError(f"Missing fixed-protocol results for sparsities: {missing}")

    rows = []
    for sparsity in sorted(by_sparsity):
        result = by_sparsity[sparsity]
        rows.append(
            {
                "sparsity": sparsity,
                "Hyper_m": args.hyper_m,
                "Lamda": args.lamda,
                "Owl_alpha": args.alpha,
                "Grad_beta": args.beta,
                "correct": int(result["correct"]),
                "n_questions": int(result["n_questions"]),
                "accuracy": float(result["accuracy"]),
                "delta_vs_dense_pp": 100.0 * (float(result["accuracy"]) - dense_accuracy),
                "n_ties": int(result.get("n_ties", 0)),
                "n_truncated_questions": int(result.get("n_truncated_questions", 0)),
                "gpu_names": "; ".join(result.get("gpu_names", [])),
                "question_set_sha256": result.get("question_set_sha256"),
                "result_path": result["result_path"],
            }
        )

    print("\nPLLaMA plant-MCQ fixed protocol")
    print("=" * 100)
    print(
        f"Fixed parameters (H_m, lambda, alpha, beta) = "
        f"({args.hyper_m:g}, {args.lamda:g}, {args.alpha:g}, {args.beta:g})"
    )
    print(f"Dense: {dense['correct']}/{dense['n_questions']} = {dense_accuracy:.4%}")
    print("-" * 100)
    print(f"{'sparsity':<10} {'correct':>10} {'accuracy':>12} {'delta(pp)':>12} {'ties':>8} {'truncated':>11}")
    for row in rows:
        print(
            f"{row['sparsity']:<10.1f} {row['correct']:>5}/{row['n_questions']:<4} "
            f"{row['accuracy']:>11.4%} {row['delta_vs_dense_pp']:>+12.3f} "
            f"{row['n_ties']:>8} {row['n_truncated_questions']:>11}"
        )

    summary = {
        "protocol": "fixed parameters; one result per model-sparsity pair",
        "fixed_parameters": {
            "Hyper_m": args.hyper_m,
            "Lamda": args.lamda,
            "Owl_alpha": args.alpha,
            "Grad_beta": args.beta,
        },
        "question_set_sha256": dense_hash,
        "dense": dense,
        "results": rows,
    }
    output = Path(args.output) if args.output else root / "fixed_protocol_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
