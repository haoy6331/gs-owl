#!/usr/bin/env python3
"""Run reproducible OPT baseline pruning experiments with resume support."""

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ZERO_SHOT_TASKS = (
    "boolq",
    "rte",
    "hellaswag",
    "arc_challenge",
    "arc_easy",
    "winogrande",
    "openbookqa",
)
METHODS = ("dense", "magnitude", "wanda", "sparsegpt", "pruner-zero")
PRINT_LOCK = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description="Run OPT baseline experiments.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gradient-path", required=True)
    parser.add_argument("--json-tree", default="data/pruner-zero_tree.json")
    parser.add_argument("--output-root", default="owl/opt_1_3b_baselines_a100")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--sparsities", nargs="+", default=["0.5", "0.6", "0.7"])
    parser.add_argument("--sparsity-type", default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default="llm_weights")
    parser.add_argument("--main", default="main_opt.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs="+", default=[])
    parser.add_argument(
        "--gpu-groups",
        nargs="+",
        default=[],
        help=(
            "Independent CUDA_VISIBLE_DEVICES groups. Use '0 1' for two single-GPU workers "
            "or '0,1 2,3' for two dual-GPU workers."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-zero-shot", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--local-datasets-path", default=None)
    return parser.parse_args()


def model_tag(model_path):
    name = os.path.basename(str(model_path).rstrip("/\\")) or "model"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def sparsity_tag(value):
    return "s_" + str(value).replace(".", "p")


def resolve_gpu_groups(args):
    if args.gpus and args.gpu_groups:
        raise ValueError("Use either --gpus or --gpu-groups, not both")
    groups = list(args.gpu_groups or args.gpus)
    allocated = set()
    for group in groups:
        ids = [item.strip() for item in str(group).split(",") if item.strip()]
        if not ids:
            raise ValueError(f"Invalid empty GPU group: {group!r}")
        overlap = allocated.intersection(ids)
        if overlap:
            raise ValueError(f"GPU ids assigned to more than one worker: {sorted(overlap)}")
        allocated.update(ids)
    return groups


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def parse_stdout(text):
    patterns = {
        "pruning_time_sec": r"pruning time:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "actual_sparsity": r"sparsity sanity check\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "ppl_test": r"wikitext perplexity\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    }
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = float(match.group(1))
    return result


def parse_saved_log(run_dir, command_method):
    log_path = run_dir / f"log_{command_method}.txt"
    if not log_path.exists():
        return {}, log_path

    header = None
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split("\t")
        if parts and parts[0] == "method":
            header = parts
        elif header and len(parts) == len(header):
            rows.append(dict(zip(header, parts)))
    if not rows:
        return {}, log_path

    row = rows[-1]
    for key in ("actual_sparsity", "ppl_test", "sparsity_ratio"):
        if key in row:
            try:
                row[key] = float(row[key])
            except ValueError:
                pass
    return row, log_path


def parse_zero_shot(run_dir, command_method):
    path = run_dir / f"log_lm_eval_{command_method}.json"
    if not path.exists():
        return {}, path
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}, path
    scores = payload.get("scores", {}) if payload.get("status") == "success" else {}
    scores = {
        task: float(value)
        for task, value in scores.items()
        if task in ZERO_SHOT_TASKS and isinstance(value, (int, float))
    }
    return scores, path


def build_tasks(args):
    tasks = []
    model_name = model_tag(args.model)
    output_root = Path(args.output_root)
    for method in args.methods:
        sparsities = ["0.0"] if method == "dense" else args.sparsities
        for sparsity in sparsities:
            run_dir = (
                output_root / model_name / args.sparsity_type / method /
                ("dense" if method == "dense" else sparsity_tag(sparsity))
            )
            tasks.append({
                "method": method,
                "command_method": "magnitude" if method == "dense" else method,
                "sparsity": float(sparsity),
                "run_dir": run_dir,
                "result_path": run_dir / "result.json",
                "stdout_path": run_dir / "stdout.log",
            })
    return tasks


def build_command(args, task):
    command = [
        args.python,
        args.main,
        "--model", args.model,
        "--cache_dir", args.cache_dir,
        "--prune_method", task["command_method"],
        "--sparsity_ratio", str(task["sparsity"]),
        "--sparsity_type", args.sparsity_type,
        "--nsamples", str(args.nsamples),
        "--seed", str(args.seed),
        "--save", str(task["run_dir"]),
    ]
    if task["method"] == "pruner-zero":
        command.extend([
            "--gradient_path", args.gradient_path,
            "--json_tree", args.json_tree,
        ])
    if args.eval_zero_shot:
        command.append("--eval_zero_shot")
        if args.offline:
            command.append("--offline")
        if args.local_datasets_path:
            command.extend(["--local_datasets_path", args.local_datasets_path])
    return command


def run_and_tee(command, stdout_path, cwd, gpu):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    prefix = f"[GPU {gpu}] " if gpu is not None else ""
    collected = []
    with stdout_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            with PRINT_LOCK:
                print(prefix + line, end="")
            log_file.write(line)
            log_file.flush()
            collected.append(line)
        return process.wait(), "".join(collected)


def task_is_complete(task, require_zero_shot):
    if not task["result_path"].exists():
        return False
    try:
        result = json.loads(task["result_path"].read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return False
    if result.get("status") != "success" or result.get("ppl_test") is None:
        return False
    if require_zero_shot:
        scores = result.get("zero_shot_scores", {})
        return all(task_name in scores for task_name in ZERO_SHOT_TASKS)
    return True


def execute_task(args, repo_root, task, gpu, index, total):
    command = build_command(args, task)
    display_command = " ".join(shlex.quote(str(part)) for part in command)
    with PRINT_LOCK:
        print("\n" + "=" * 80)
        print(
            f"[{index}/{total}] GPU={gpu if gpu is not None else 'serial'} "
            f"method={task['method']} sparsity={task['sparsity']:.1f}"
        )
        print(display_command)
        print("=" * 80)

    if args.dry_run:
        return {"status": "dry_run", "method": task["method"], "command": command}

    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()
    returncode, stdout = run_and_tee(command, task["stdout_path"], repo_root, gpu)
    parsed = parse_stdout(stdout)
    saved, saved_log = parse_saved_log(task["run_dir"], task["command_method"])
    zero_scores, zero_path = parse_zero_shot(task["run_dir"], task["command_method"])

    ppl = saved.get("ppl_test", parsed.get("ppl_test"))
    actual_sparsity = saved.get("actual_sparsity", parsed.get("actual_sparsity"))
    zero_complete = all(name in zero_scores for name in ZERO_SHOT_TASKS)
    success = returncode == 0 and ppl is not None and (not args.eval_zero_shot or zero_complete)
    result = {
        "model": model_tag(args.model),
        "method": task["method"],
        "command_method": task["command_method"],
        "target_sparsity": task["sparsity"],
        "actual_sparsity": actual_sparsity,
        "ppl_test": ppl,
        "pruning_time_sec": parsed.get("pruning_time_sec"),
        "num_samples": args.nsamples,
        "sparsity_type": args.sparsity_type,
        "seed": args.seed,
        "zero_shot_scores": zero_scores,
        "zero_shot_mean": (sum(zero_scores.values()) / len(ZERO_SHOT_TASKS)) if zero_complete else None,
        "evaluated_zero_shot": zero_complete,
        "status": "success" if success else "failed",
        "returncode": returncode,
        "gpu": gpu,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.time() - start_time,
        "run_dir": str(task["run_dir"]),
        "stdout_log": str(task["stdout_path"]),
        "saved_log": str(saved_log),
        "zero_shot_log": str(zero_path),
        "command": command,
    }
    atomic_write_json(task["result_path"], result)
    return result


def collect_results(output_root):
    rows = []
    for path in Path(output_root).rglob("result.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8", errors="replace")))
        except json.JSONDecodeError:
            continue
    order = {name: index for index, name in enumerate(METHODS)}
    return sorted(rows, key=lambda row: (order.get(row.get("method"), 99), row.get("target_sparsity", 0)))


def write_summary(output_root, rows):
    fields = [
        "model", "method", "target_sparsity", "actual_sparsity", "ppl_test",
        "zero_shot_mean", *ZERO_SHOT_TASKS, "pruning_time_sec", "num_samples",
        "sparsity_type", "seed", "status", "returncode", "gpu", "run_dir",
    ]
    path = Path(output_root) / "summary_all.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat.update(row.get("zero_shot_scores", {}))
            writer.writerow({field: flat.get(field, "") for field in fields})
    return path


def print_summary(rows):
    print("\nOPT baseline summary")
    print("-" * 78)
    print(f"{'method':<14} {'sparsity':>9} {'actual':>9} {'PPL':>12} {'zero-shot':>12} {'status':>9}")
    print("-" * 78)
    for row in rows:
        sparsity = "Dense" if row.get("method") == "dense" else f"{100 * float(row.get('target_sparsity', 0)):.0f}%"
        actual = row.get("actual_sparsity")
        ppl = row.get("ppl_test")
        mean = row.get("zero_shot_mean")
        print(
            f"{row.get('method', '-'):<14} {sparsity:>9} "
            f"{actual if actual is not None else '-':>9} "
            f"{ppl if ppl is not None else '-':>12} "
            f"{mean if mean is not None else '-':>12} "
            f"{row.get('status', '-'):>9}"
        )
    print("-" * 78)


def main():
    args = parse_args()
    gpu_groups = resolve_gpu_groups(args)
    repo_root = Path.cwd()
    if "pruner-zero" in args.methods:
        if not Path(args.gradient_path).is_file():
            raise FileNotFoundError(f"Gradient file not found: {args.gradient_path}")
        if not Path(args.json_tree).is_file():
            raise FileNotFoundError(f"Pruner-Zero tree not found: {args.json_tree}")
        json.loads(Path(args.json_tree).read_text(encoding="utf-8"))

    tasks = build_tasks(args)
    pending = []
    for index, task in enumerate(tasks, start=1):
        if args.resume and task_is_complete(task, args.eval_zero_shot):
            print(f"[{index}/{len(tasks)}] skip complete: {task['result_path']}")
        else:
            pending.append((index, task))

    print(f"Methods: {', '.join(args.methods)}")
    print(f"Total runs: {len(tasks)}; pending: {len(pending)}")
    print(f"Output root: {args.output_root}")
    print(f"Zero-shot: {'enabled' if args.eval_zero_shot else 'disabled'}")
    print(f"GPU groups: {', '.join(gpu_groups) if gpu_groups else 'inherited serial environment'}")

    failures = []
    if gpu_groups and pending:
        executors = [ThreadPoolExecutor(max_workers=1) for _ in gpu_groups]
        futures = {}
        try:
            for position, (index, task) in enumerate(pending):
                slot = position % len(gpu_groups)
                future = executors[slot].submit(
                    execute_task, args, repo_root, task, gpu_groups[slot], index, len(tasks)
                )
                futures[future] = task
            for future in as_completed(futures):
                result = future.result()
                if result.get("status") == "failed":
                    failures.append(result)
        finally:
            for executor in executors:
                executor.shutdown(wait=True)
    else:
        for index, task in pending:
            result = execute_task(args, repo_root, task, None, index, len(tasks))
            if result.get("status") == "failed":
                failures.append(result)

    rows = collect_results(args.output_root)
    summary_path = write_summary(args.output_root, rows)
    print_summary(rows)
    print(f"Completed: {sum(row.get('status') == 'success' for row in rows)}/{len(tasks)}")
    print(f"Summary CSV: {summary_path}")
    if failures:
        print(f"Failed in this run: {len(failures)}; inspect each stdout.log and rerun with --resume.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
