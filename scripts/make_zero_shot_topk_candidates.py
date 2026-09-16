import argparse
import csv
import os
from pathlib import Path


ALL_ZERO_SHOT_TASKS = [
    "boolq",
    "rte",
    "hellaswag",
    "arc_challenge",
    "arc_easy",
    "winogrande",
    "openbookqa",
]

SOTA_FOCUS4_TASKS = {
    ("llama1_7b", 0.5): ["boolq", "rte", "arc_challenge", "winogrande"],
    ("llama1_7b", 0.6): ["boolq", "rte", "arc_challenge", "winogrande"],
    ("llama1_30b", 0.5): [
        "boolq",
        "arc_challenge",
        "arc_easy",
        "openbookqa",
    ],
    ("llama1_30b", 0.6): [
        "boolq",
        "arc_challenge",
        "winogrande",
        "openbookqa",
    ],
    ("llama2_7b", 0.5): [
        "boolq",
        "hellaswag",
        "arc_challenge",
        "winogrande",
    ],
    ("llama2_7b", 0.6): [
        "boolq",
        "rte",
        "arc_challenge",
        "openbookqa",
    ],
    ("llama2_13b", 0.5): [
        "boolq",
        "rte",
        "arc_challenge",
        "winogrande",
    ],
    ("llama2_13b", 0.6): [
        "boolq",
        "arc_challenge",
        "winogrande",
        "openbookqa",
    ],
}


def parse_job(text):
    parts = text.split("|")
    if len(parts) not in (5, 6):
        raise ValueError(
            "--job must use: label|model_path|gradient_path|summary_csv|output_root"
            "[|main_script]"
        )
    label, model_path, gradient_path, summary_csv, output_root = parts[:5]
    main_script = parts[5] if len(parts) == 6 and parts[5] else "main.py"
    return {
        "label": label,
        "model_path": model_path,
        "gradient_path": gradient_path,
        "summary_csv": Path(summary_csv),
        "output_root": output_root,
        "main_script": main_script,
    }


def parse_float(row, *keys):
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return float(value)
    raise ValueError(f"Missing numeric value for any of: {', '.join(keys)}")


def load_success_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"summary_all.csv not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "success":
                continue
            try:
                row["_target_sparsity_float"] = parse_float(row, "target_sparsity", "sparsity_ratio")
                row["_ppl_float"] = parse_float(row, "ppl_test", "ppl_stdout")
                parse_float(row, "Hyper_m")
                parse_float(row, "Lamda")
                parse_float(row, "Owl_alpha")
            except (TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def sparsity_matches(value, selected_sparsities):
    if not selected_sparsities:
        return True
    return any(abs(value - selected) < 1e-9 for selected in selected_sparsities)


def tasks_for_candidate(label, sparsity, task_profile):
    if task_profile == "all":
        return ALL_ZERO_SHOT_TASKS
    key = (label, round(float(sparsity), 6))
    if key not in SOTA_FOCUS4_TASKS:
        raise ValueError(
            f"No {task_profile} task mapping for label={label}, sparsity={sparsity}"
        )
    return SOTA_FOCUS4_TASKS[key]


def select_topk(
    job,
    rank_start,
    rank_end,
    selected_sparsities,
    task_profile,
):
    grouped = {}
    for row in load_success_rows(job["summary_csv"]):
        grouped.setdefault(row["_target_sparsity_float"], []).append(row)

    candidates = []
    for sparsity in sorted(grouped):
        if not sparsity_matches(sparsity, selected_sparsities):
            continue
        ranked = sorted(grouped[sparsity], key=lambda r: r["_ppl_float"])
        selected = ranked[rank_start - 1:rank_end]
        zero_shot_tasks = ",".join(
            tasks_for_candidate(job["label"], sparsity, task_profile)
        )
        for rank, row in enumerate(selected, start=rank_start):
            candidates.append({
                "label": job["label"],
                "model_path": job["model_path"],
                "gradient_path": job["gradient_path"],
                "output_root": job["output_root"],
                "main_script": job["main_script"],
                "zero_shot_tasks": zero_shot_tasks,
                "source_summary_csv": str(job["summary_csv"]),
                "rank_in_sparsity": rank,
                "target_sparsity": row.get("target_sparsity") or row.get("sparsity_ratio"),
                "actual_sparsity_ppl": row.get("actual_sparsity", ""),
                "ppl_test": row.get("ppl_test") or row.get("ppl_stdout"),
                "Hyper_m": row.get("Hyper_m"),
                "Lamda": row.get("Lamda"),
                "Owl_alpha": row.get("Owl_alpha"),
                "pruning_time_sec": row.get("pruning_time_sec", ""),
                "source_run_dir": row.get("run_dir", ""),
                "source_stdout_log": row.get("stdout_log", ""),
                "source_saved_log": row.get("saved_log", ""),
                "layer_density_ratios": row.get("layer_density_ratios", ""),
                "layer_sparsity_ratios": row.get("layer_sparsity_ratios", ""),
            })
    return candidates


def write_candidates(path, rows):
    fields = [
        "task_id",
        "label",
        "model_path",
        "gradient_path",
        "output_root",
        "main_script",
        "zero_shot_tasks",
        "source_summary_csv",
        "rank_in_sparsity",
        "target_sparsity",
        "actual_sparsity_ppl",
        "ppl_test",
        "Hyper_m",
        "Lamda",
        "Owl_alpha",
        "pruning_time_sec",
        "source_run_dir",
        "source_stdout_log",
        "source_saved_log",
        "layer_density_ratios",
        "layer_sparsity_ratios",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for task_id, row in enumerate(rows):
                row = dict(row)
                row["task_id"] = task_id
                writer.writerow(row)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Build zero-shot candidates from the PPL top-k rows per model and sparsity."
    )
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rank-start", type=int, default=1)
    parser.add_argument("--rank-end", type=int, default=None)
    parser.add_argument(
        "--sparsity",
        action="append",
        type=float,
        default=None,
        help="Target sparsity to include. May be repeated.",
    )
    parser.add_argument(
        "--task-profile",
        choices=["all", "sota-focus4"],
        default="all",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    rank_end = args.rank_end if args.rank_end is not None else args.top_k
    if args.rank_start < 1:
        raise ValueError("--rank-start must be at least 1")
    if rank_end < args.rank_start:
        raise ValueError("--rank-end must be greater than or equal to --rank-start")

    all_candidates = []
    for job_text in args.job:
        job = parse_job(job_text)
        selected = select_topk(
            job,
            args.rank_start,
            rank_end,
            args.sparsity,
            args.task_profile,
        )
        print(
            f"{job['label']}: selected {len(selected)} candidates "
            f"from ranks {args.rank_start}-{rank_end} in {job['summary_csv']}"
        )
        all_candidates.extend(selected)

    write_candidates(Path(args.output_csv), all_candidates)
    print(f"Wrote {len(all_candidates)} zero-shot candidates to {args.output_csv}")


if __name__ == "__main__":
    main()
