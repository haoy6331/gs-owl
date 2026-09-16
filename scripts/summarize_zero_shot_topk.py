import argparse
import csv
import json
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
USABLE_STATUSES = {"success", "partial"}


def normalize_task_list(value, fallback):
    if isinstance(value, str):
        tasks = [item.strip() for item in value.split(",") if item.strip()]
        return tasks or list(fallback)
    if isinstance(value, (list, tuple)):
        tasks = [str(item).strip() for item in value if str(item).strip()]
        return tasks or list(fallback)
    return list(fallback)


def numeric_scores(scores):
    return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}


def collect_rows(output_root):
    rows = []
    task_keys = set()
    for path in Path(output_root).rglob("zero_shot_result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        scores = numeric_scores(payload.get("scores", {}))
        task_keys.update(scores)
        values = list(scores.values())
        complete = all(task in scores for task in ALL_ZERO_SHOT_TASKS)
        evaluated_tasks = normalize_task_list(
            payload.get("evaluated_tasks"),
            list(scores),
        )
        requested_tasks = normalize_task_list(
            payload.get("requested_tasks"),
            evaluated_tasks,
        )
        row = {
            "status": payload.get("status", ""),
            "label": payload.get("label", ""),
            "target_sparsity": payload.get("target_sparsity", ""),
            "rank_in_sparsity": payload.get("rank_in_sparsity", ""),
            "requested_tasks": ",".join(requested_tasks),
            "evaluated_tasks": ",".join(evaluated_tasks),
            "is_partial": not complete,
            "evaluated_task_avg": (
                payload.get("evaluated_task_avg")
                if payload.get("evaluated_task_avg") is not None
                else (sum(values) / len(values) if values else "")
            ),
            "zero_shot_avg": (
                sum(scores[task] for task in ALL_ZERO_SHOT_TASKS)
                / len(ALL_ZERO_SHOT_TASKS)
                if complete else ""
            ),
            "source_ppl": payload.get("source_ppl", ""),
            "actual_sparsity_ppl": payload.get("actual_sparsity_ppl", ""),
            "Hyper_m": payload.get("Hyper_m", ""),
            "Lamda": payload.get("Lamda", ""),
            "Owl_alpha": payload.get("Owl_alpha", ""),
            "pruning_time_sec": payload.get("pruning_time_sec", ""),
            "save_dir": payload.get("save_dir", str(path.parent)),
            "lm_eval_json": payload.get("lm_eval_json", ""),
            "source_summary_csv": payload.get("source_summary_csv", ""),
            "source_run_dir": payload.get("source_run_dir", ""),
        }
        for key, value in scores.items():
            row[key] = value
        rows.append(row)
    return rows, sorted(task_keys)


def float_or_default(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path, rows, task_keys):
    fields = [
        "status",
        "label",
        "target_sparsity",
        "rank_in_sparsity",
        "requested_tasks",
        "evaluated_tasks",
        "is_partial",
        "evaluated_task_avg",
        "zero_shot_avg",
        "source_ppl",
        "actual_sparsity_ppl",
        "Hyper_m",
        "Lamda",
        "Owl_alpha",
        "pruning_time_sec",
    ] + task_keys + [
        "save_dir",
        "lm_eval_json",
        "source_summary_csv",
        "source_run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def best_by_model_sparsity(rows):
    best = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        if row.get("is_partial"):
            continue
        key = (row.get("label"), float_or_default(row.get("target_sparsity"), -1.0))
        current = best.get(key)
        candidate_key = (
            -float_or_default(row.get("zero_shot_avg"), -1.0),
            float_or_default(row.get("source_ppl"), 1e18),
            float_or_default(row.get("rank_in_sparsity"), 1e18),
        )
        if current is None:
            best[key] = row
            continue
        current_key = (
            -float_or_default(current.get("zero_shot_avg"), -1.0),
            float_or_default(current.get("source_ppl"), 1e18),
            float_or_default(current.get("rank_in_sparsity"), 1e18),
        )
        if candidate_key < current_key:
            best[key] = row
    return list(best.values())


def oracle_by_model_sparsity(rows):
    grouped = {}
    for row in rows:
        if row.get("status") not in USABLE_STATUSES:
            continue
        key = (
            row.get("label"),
            float_or_default(row.get("target_sparsity"), -1.0),
        )
        grouped.setdefault(key, []).append(row)

    oracle_rows = []
    for (label, sparsity), group in grouped.items():
        oracle = {
            "label": label,
            "target_sparsity": sparsity,
        }
        available_tasks = 0
        oracle_sum = 0.0
        for task in ALL_ZERO_SHOT_TASKS:
            candidates = [
                row for row in group
                if isinstance(row.get(task), (int, float))
            ]
            if not candidates:
                oracle[task] = ""
                oracle[f"{task}_rank"] = ""
                oracle[f"{task}_Hyper_m"] = ""
                oracle[f"{task}_Lamda"] = ""
                oracle[f"{task}_Owl_alpha"] = ""
                continue
            winner = min(
                candidates,
                key=lambda row: (
                    -float(row[task]),
                    float_or_default(row.get("source_ppl"), 1e18),
                    float_or_default(row.get("rank_in_sparsity"), 1e18),
                ),
            )
            score = float(winner[task])
            available_tasks += 1
            oracle_sum += score
            oracle[task] = score
            oracle[f"{task}_rank"] = winner.get("rank_in_sparsity", "")
            oracle[f"{task}_Hyper_m"] = winner.get("Hyper_m", "")
            oracle[f"{task}_Lamda"] = winner.get("Lamda", "")
            oracle[f"{task}_Owl_alpha"] = winner.get("Owl_alpha", "")
        oracle["available_tasks"] = available_tasks
        oracle["oracle_sum"] = (
            oracle_sum
            if available_tasks == len(ALL_ZERO_SHOT_TASKS)
            else ""
        )
        oracle["oracle_avg"] = (
            oracle_sum / len(ALL_ZERO_SHOT_TASKS)
            if available_tasks == len(ALL_ZERO_SHOT_TASKS)
            else ""
        )
        oracle_rows.append(oracle)
    return oracle_rows


def write_oracle_csv(path, rows):
    fields = [
        "label",
        "target_sparsity",
        "available_tasks",
        "oracle_sum",
        "oracle_avg",
    ]
    for task in ALL_ZERO_SHOT_TASKS:
        fields.extend([
            task,
            f"{task}_rank",
            f"{task}_Hyper_m",
            f"{task}_Lamda",
            f"{task}_Owl_alpha",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize TopK zero-shot results.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--best-csv", default=None)
    parser.add_argument("--oracle-csv", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows, discovered_task_keys = collect_rows(output_root)
    task_keys = [
        task for task in ALL_ZERO_SHOT_TASKS
        if task in discovered_task_keys
    ] + sorted(set(discovered_task_keys) - set(ALL_ZERO_SHOT_TASKS))
    rows = sorted(rows, key=lambda r: (
        r.get("label", ""),
        float_or_default(r.get("target_sparsity"), 0.0),
        -float_or_default(r.get("zero_shot_avg"), -1.0),
        float_or_default(r.get("source_ppl"), 1e18),
    ))
    best_rows = sorted(best_by_model_sparsity(rows), key=lambda r: (
        r.get("label", ""),
        float_or_default(r.get("target_sparsity"), 0.0),
    ))
    oracle_rows = sorted(oracle_by_model_sparsity(rows), key=lambda r: (
        r.get("label", ""),
        float_or_default(r.get("target_sparsity"), 0.0),
    ))

    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "zero_shot_topk_summary.csv"
    best_csv = Path(args.best_csv) if args.best_csv else output_root / "zero_shot_topk_best_by_sparsity.csv"
    oracle_csv = Path(args.oracle_csv) if args.oracle_csv else output_root / "zero_shot_oracle_by_sparsity.csv"
    write_csv(summary_csv, rows, task_keys)
    write_csv(best_csv, best_rows, task_keys)
    write_oracle_csv(oracle_csv, oracle_rows)

    print(f"Collected zero-shot results: {len(rows)}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote best rows: {best_csv}")
    print(f"Wrote task-wise oracle rows: {oracle_csv}")
    for row in best_rows:
        print(
            f"{row['label']} sparsity={row['target_sparsity']}: "
            f"zero_shot_avg={row['zero_shot_avg']}, source_ppl={row['source_ppl']}, "
            f"rank={row['rank_in_sparsity']}, Hyper_m={row['Hyper_m']}, "
            f"Lamda={row['Lamda']}, Owl_alpha={row['Owl_alpha']}"
        )
    for row in oracle_rows:
        print(
            f"ORACLE {row['label']} sparsity={row['target_sparsity']}: "
            f"available_tasks={row['available_tasks']}, "
            f"oracle_avg={row['oracle_avg']}"
        )


if __name__ == "__main__":
    main()
