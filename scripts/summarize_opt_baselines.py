#!/usr/bin/env python3
"""Print completed OPT baseline results to the terminal."""

import argparse

from run_opt_baselines import collect_results, print_summary, write_summary


def main():
    parser = argparse.ArgumentParser(description="Summarize OPT baseline experiments.")
    parser.add_argument("--output-root", default="owl/opt_1_3b_baselines_a100")
    args = parser.parse_args()

    rows = collect_results(args.output_root)
    path = write_summary(args.output_root, rows)
    print_summary(rows)
    successes = sum(row.get("status") == "success" for row in rows)
    failures = sum(row.get("status") == "failed" for row in rows)
    print(f"Collected: {len(rows)}; success: {successes}; failed: {failures}")
    print(f"Summary CSV: {path}")


if __name__ == "__main__":
    main()
