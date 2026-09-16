#!/usr/bin/env python3
"""Merge plant-MCQ evaluation shards and report paired dense/pruned results."""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize PLLaMA plant-MCQ results.")
    parser.add_argument("--input", action="append", required=True, help="Evaluation JSON; repeat as needed.")
    parser.add_argument("--output", default="owl/pllama_plant_mcq/results_combined.json")
    parser.add_argument("--dense-label", default="dense")
    return parser.parse_args()


def exact_mcnemar_pvalue(dense_only, pruned_only):
    discordant = dense_only + pruned_only
    if discordant == 0:
        return 1.0
    lower = min(dense_only, pruned_only)
    tail = sum(math.comb(discordant, i) for i in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def paired_metrics(dense, pruned):
    dense_by_id = {str(item["id"]): item for item in dense["predictions"]}
    pruned_by_id = {str(item["id"]): item for item in pruned["predictions"]}
    if dense_by_id.keys() != pruned_by_id.keys():
        missing_dense = sorted(pruned_by_id.keys() - dense_by_id.keys())
        missing_pruned = sorted(dense_by_id.keys() - pruned_by_id.keys())
        raise RuntimeError(
            f"Question mismatch for {pruned['model']}: "
            f"missing_dense={missing_dense[:5]}, missing_pruned={missing_pruned[:5]}"
        )

    dense_only = 0
    pruned_only = 0
    both_correct = 0
    both_wrong = 0
    for question_id in dense_by_id:
        dense_correct = bool(dense_by_id[question_id]["correct"])
        pruned_correct = bool(pruned_by_id[question_id]["correct"])
        if dense_correct and pruned_correct:
            both_correct += 1
        elif dense_correct:
            dense_only += 1
        elif pruned_correct:
            pruned_only += 1
        else:
            both_wrong += 1

    dense_accuracy = float(dense["accuracy"])
    pruned_accuracy = float(pruned["accuracy"])
    return {
        "accuracy_delta": pruned_accuracy - dense_accuracy,
        "accuracy_delta_pp": 100.0 * (pruned_accuracy - dense_accuracy),
        "retention_ratio": pruned_accuracy / dense_accuracy if dense_accuracy else None,
        "both_correct": both_correct,
        "dense_correct_pruned_wrong": dense_only,
        "dense_wrong_pruned_correct": pruned_only,
        "both_wrong": both_wrong,
        "mcnemar_exact_pvalue": exact_mcnemar_pvalue(dense_only, pruned_only),
    }


def main():
    args = parse_args()
    evaluations = {}
    question_file = None
    n_questions = None

    for input_path in args.input:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        if question_file is None:
            question_file = payload.get("questions")
            n_questions = payload.get("n_questions")
        elif payload.get("questions") != question_file or payload.get("n_questions") != n_questions:
            raise RuntimeError(f"Evaluation set mismatch in {input_path}")
        for item in payload.get("evaluations", []):
            label = item["model"]
            if label in evaluations:
                raise RuntimeError(f"Duplicate model label: {label}")
            evaluations[label] = item

    if args.dense_label not in evaluations:
        raise RuntimeError(f"Dense reference label '{args.dense_label}' was not found")

    dense = evaluations[args.dense_label]
    paired = {
        label: paired_metrics(dense, item)
        for label, item in evaluations.items()
        if label != args.dense_label
    }
    output_payload = {
        "questions": question_file,
        "n_questions": n_questions,
        "metric": "multiple-choice accuracy",
        "dense_label": args.dense_label,
        "evaluations": list(evaluations.values()),
        "paired_vs_dense": paired,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nPLLaMA plant-science MCQ summary")
    print("-" * 88)
    print(f"{'model':<18} {'correct':>9} {'accuracy':>11} {'delta(pp)':>11} {'retention':>11} {'McNemar p':>12}")
    print("-" * 88)
    for label, item in evaluations.items():
        if label == args.dense_label:
            print(f"{label:<18} {item['correct']:>4}/{item['n_questions']:<4} {item['accuracy']:>10.4%} {'--':>11} {'--':>11} {'--':>12}")
            continue
        stats = paired[label]
        retention = stats["retention_ratio"]
        retention_text = f"{retention:.4f}" if retention is not None else "--"
        print(
            f"{label:<18} {item['correct']:>4}/{item['n_questions']:<4} {item['accuracy']:>10.4%} "
            f"{stats['accuracy_delta_pp']:>+10.3f} {retention_text:>11} "
            f"{stats['mcnemar_exact_pvalue']:>12.4g}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
