import argparse
import csv
import json
import os
import shlex
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from run_zero_shot_best import (
    build_command,
    run_and_tee,
    summarize_lm_eval_json,
    value_tag,
)

ALL_ZERO_SHOT_TASKS = [
    "boolq",
    "rte",
    "hellaswag",
    "arc_challenge",
    "arc_easy",
    "winogrande",
    "openbookqa",
]


def load_candidates(path):
    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No candidates found in {path}")

    def sort_key(row):
        value = row.get("task_id", "")
        try:
            return int(value)
        except ValueError:
            return len(rows)

    return sorted(rows, key=sort_key)


def numeric_scores(scores):
    return {
        key: float(value)
        for key, value in scores.items()
        if isinstance(value, (int, float))
    }


def requested_tasks(candidate):
    raw_tasks = candidate.get("zero_shot_tasks", "")
    tasks = [
        task.strip()
        for task in raw_tasks.split(",")
        if task.strip()
    ] if raw_tasks else list(ALL_ZERO_SHOT_TASKS)
    invalid = sorted(set(tasks) - set(ALL_ZERO_SHOT_TASKS))
    if invalid:
        raise ValueError(f"Unsupported zero-shot tasks: {', '.join(invalid)}")
    tasks = list(dict.fromkeys(tasks))
    if not tasks:
        raise ValueError("Candidate must request at least one zero-shot task")
    return tasks


def ordered_score_tasks(scores):
    return [
        task for task in ALL_ZERO_SHOT_TASKS
        if task in scores
    ] + sorted(set(scores) - set(ALL_ZERO_SHOT_TASKS))


def load_result_payload(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}


def collect_existing_scores(result_json, lm_eval_json):
    payload = load_result_payload(result_json)
    scores = numeric_scores(payload.get("scores", {}))
    scores.update(numeric_scores(summarize_lm_eval_json(lm_eval_json)))
    return scores


def has_task_coverage(scores, tasks):
    return all(task in scores for task in tasks)


def acquire_lock(lock_path, candidate, force_unlock):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if force_unlock and lock_path.exists():
        lock_path.unlink()
    metadata = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "created_at": time.time(),
        "task_id": candidate.get("task_id", ""),
        "label": candidate.get("label", ""),
        "target_sparsity": candidate.get("target_sparsity", ""),
        "rank_in_sparsity": candidate.get("rank_in_sparsity", ""),
    }
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return True


def build_result(
    status,
    returncode,
    candidate,
    scores,
    stdout_path,
    lm_eval_json,
    save_dir,
    requested,
):
    score_dict = numeric_scores(scores)
    evaluated_tasks = ordered_score_tasks(score_dict)
    all_values = [
        score_dict[task]
        for task in ALL_ZERO_SHOT_TASKS
        if task in score_dict
    ]
    is_complete = has_task_coverage(score_dict, ALL_ZERO_SHOT_TASKS)
    result = {
        "status": status,
        "returncode": returncode,
        "task_id": int(candidate.get("task_id", -1)),
        "label": candidate["label"],
        "target_sparsity": float(candidate["target_sparsity"]),
        "rank_in_sparsity": int(candidate["rank_in_sparsity"]),
        "source_ppl": float(candidate["ppl_test"]),
        "actual_sparsity_ppl": candidate.get("actual_sparsity_ppl", ""),
        "Hyper_m": float(candidate["Hyper_m"]),
        "Lamda": float(candidate["Lamda"]),
        "Owl_alpha": float(candidate["Owl_alpha"]),
        "pruning_time_sec": candidate.get("pruning_time_sec", ""),
        "main_script": candidate.get("main_script", "main.py"),
        "requested_tasks": requested,
        "evaluated_tasks": evaluated_tasks,
        "is_partial": not is_complete,
        "evaluated_task_avg": (
            sum(all_values) / len(all_values)
            if all_values else None
        ),
        "zero_shot_avg": (
            sum(score_dict[task] for task in ALL_ZERO_SHOT_TASKS)
            / len(ALL_ZERO_SHOT_TASKS)
            if is_complete else None
        ),
        "scores": score_dict,
        "source_summary_csv": candidate.get("source_summary_csv", ""),
        "source_run_dir": candidate.get("source_run_dir", ""),
        "stdout_log": str(stdout_path),
        "lm_eval_json": str(lm_eval_json),
        "save_dir": str(save_dir),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run one zero-shot TopK candidate selected from PPL summary_all.csv."
    )
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--method", default="owl-v9-wsqrtg")
    parser.add_argument("--sparsity-type", default="unstructured")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--main", default="main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--local-datasets-path", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-unlock",
        action="store_true",
        help="Remove an existing task lock before running.",
    )
    args = parser.parse_args()

    candidates = load_candidates(Path(args.candidates_csv))
    if args.task_id < 0 or args.task_id >= len(candidates):
        raise ValueError(f"--task-id must be in [0, {len(candidates) - 1}], got {args.task_id}")

    candidate = candidates[args.task_id]
    rank_tag = f"rank_{int(candidate['rank_in_sparsity']):02d}"
    output_root = Path(candidate["output_root"])
    save_dir = (
        output_root
        / candidate["label"]
        / value_tag("s", candidate["target_sparsity"])
        / rank_tag
        / value_tag("hm", candidate["Hyper_m"])
        / value_tag("la", candidate["Lamda"])
        / value_tag("alpha", candidate["Owl_alpha"])
    )
    lm_eval_json = save_dir / f"log_lm_eval_{args.method}.json"
    result_json = save_dir / "zero_shot_result.json"
    stdout_path = save_dir / "zero_shot_stdout.log"
    lock_path = save_dir / ".zero_shot_running.lock"
    desired_tasks = requested_tasks(candidate)

    print("=" * 80)
    print(f"Zero-shot TopK task {args.task_id + 1}/{len(candidates)}")
    print(f"Label: {candidate['label']}")
    print(f"Sparsity: {candidate['target_sparsity']}")
    print(f"PPL rank: {candidate['rank_in_sparsity']}; source PPL: {candidate['ppl_test']}")
    print(
        f"Hyper_m={candidate['Hyper_m']}, "
        f"Lamda={candidate['Lamda']}, Owl_alpha={candidate['Owl_alpha']}"
    )
    print(f"Model: {candidate['model_path']}")
    print(f"Gradient: {candidate['gradient_path']}")
    print(f"Main script: {candidate.get('main_script') or args.main}")
    print(f"Requested tasks: {','.join(desired_tasks)}")
    print(f"Output: {save_dir}")
    print("=" * 80)

    existing_scores = collect_existing_scores(result_json, lm_eval_json)
    if args.resume and has_task_coverage(existing_scores, desired_tasks):
        result = build_result(
            "success",
            0,
            candidate,
            existing_scores,
            stdout_path,
            lm_eval_json,
            save_dir,
            desired_tasks,
        )
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"skip completed task coverage: {result_json}")
        return

    if not acquire_lock(lock_path, candidate, args.force_unlock):
        print(f"skip already running task: {lock_path}")
        return

    try:
        existing_scores = collect_existing_scores(result_json, lm_eval_json)
        if args.resume and has_task_coverage(existing_scores, desired_tasks):
            result = build_result(
                "success",
                0,
                candidate,
                existing_scores,
                stdout_path,
                lm_eval_json,
                save_dir,
                desired_tasks,
            )
            result_json.parent.mkdir(parents=True, exist_ok=True)
            result_json.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"skip completed after lock acquisition: {result_json}")
            return

        missing_tasks = [
            task for task in desired_tasks
            if task not in existing_scores
        ]
        print(f"Tasks to evaluate now: {','.join(missing_tasks)}")
        run_candidate = dict(candidate)
        run_candidate["zero_shot_tasks"] = ",".join(missing_tasks)
        runner_args = SimpleNamespace(
            python=args.python,
            main=candidate.get("main_script") or args.main,
            method=args.method,
            sparsity_type=args.sparsity_type,
            nsamples=args.nsamples,
            seed=args.seed,
            offline=args.offline,
            local_datasets_path=args.local_datasets_path,
        )
        job = {
            "model_path": candidate["model_path"],
            "gradient_path": candidate["gradient_path"],
        }
        command = build_command(runner_args, job, run_candidate, save_dir)
        print("Command:")
        print(" ".join(shlex.quote(str(x)) for x in command))

        if args.dry_run:
            save_dir.mkdir(parents=True, exist_ok=True)
            result = build_result(
                "dry_run",
                0,
                candidate,
                existing_scores,
                stdout_path,
                lm_eval_json,
                save_dir,
                desired_tasks,
            )
            result["missing_tasks"] = missing_tasks
            result["command"] = command
            dry_run_json = save_dir / "zero_shot_dry_run.json"
            dry_run_json.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Dry-run result: {dry_run_json}")
            return

        returncode = run_and_tee(command, stdout_path)
        new_scores = numeric_scores(summarize_lm_eval_json(lm_eval_json))
        merged_scores = dict(existing_scores)
        merged_scores.update(new_scores)
        completed_run = (
            returncode == 0
            and has_task_coverage(new_scores, missing_tasks)
        )
        if completed_run and has_task_coverage(merged_scores, desired_tasks):
            status = "success"
        elif merged_scores:
            status = "partial"
        else:
            status = "failed"
        result = build_result(
            status,
            returncode,
            candidate,
            merged_scores,
            stdout_path,
            lm_eval_json,
            save_dir,
            desired_tasks,
        )
        result["last_run_tasks"] = missing_tasks
        result_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if status != "success":
            missing_after_run = [
                task for task in desired_tasks
                if task not in merged_scores
            ]
            print(f"Missing tasks after run: {','.join(missing_after_run)}")
            sys.exit(1)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
