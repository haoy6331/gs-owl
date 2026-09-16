import argparse

from run_v9_param_sweep import update_summaries


def main():
    parser = argparse.ArgumentParser(description="Summarize completed V9 Slurm array results.")
    parser.add_argument("--output-root", default="owl/owl_v9_param_sweep")
    args = parser.parse_args()

    rows, best_rows = update_summaries(args.output_root)
    successes = sum(row.get("status") == "success" for row in rows)
    failures = sum(row.get("status") == "failed" for row in rows)
    print(f"Collected results: {len(rows)}; success: {successes}; failed: {failures}")
    for row in best_rows:
        print(
            f"sparsity={row['target_sparsity']}: ppl={row.get('ppl_test')}, "
            f"Hyper_m={row.get('Hyper_m')}, Lamda={row.get('Lamda')}, "
            f"Owl_alpha={row.get('Owl_alpha')}"
        )


if __name__ == "__main__":
    main()

