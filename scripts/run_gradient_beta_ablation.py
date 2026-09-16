import argparse
import csv
import json
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from run_strict_ablation import MODEL_CONFIGS, PRINT_LOCK, run_and_tee
from run_v9_param_sweep import parse_saved_log, parse_stdout, value_tag


def parse_args():
    parser = argparse.ArgumentParser(description="Run the |W| * |G|^beta metric ablation.")
    parser.add_argument("--model-keys", nargs="+", default=["llama1_7b", "llama2_7b"])
    parser.add_argument("--betas", nargs="+", default=["0", "0.25", "0.5", "0.75", "1"])
    parser.add_argument("--sparsity", default="0.6")
    parser.add_argument("--output-root", default="owl/gradient_beta_ablation")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity-type", default="unstructured")
    parser.add_argument("--cache-dir", default="llm_weights")
    parser.add_argument("--main", default="main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs="+", default=[])
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_tasks(args):
    tasks = []
    for model_key in args.model_keys:
        if model_key not in MODEL_CONFIGS:
            raise KeyError(f"Unknown model key: {model_key}")
        config = MODEL_CONFIGS[model_key]
        for beta in args.betas:
            if float(beta) < 0:
                raise ValueError(f"beta must be non-negative, got {beta}")
            run_dir = (
                Path(args.output_root)
                / model_key
                / value_tag("s", args.sparsity)
                / value_tag("beta", beta)
            )
            tasks.append({
                "model_key": model_key,
                "model": config["model"],
                "gradient": config["gradient"],
                "beta": beta,
                "run_dir": run_dir,
                "stdout_path": run_dir / "stdout.log",
                "result_path": run_dir / "result.json",
            })
    return tasks


def build_command(args, task):
    return [
        args.python,
        args.main,
        "--model", task["model"],
        "--cache_dir", args.cache_dir,
        "--prune_method", "wgradpow",
        "--gradient_path", task["gradient"],
        "--Grad_beta", task["beta"],
        "--sparsity_ratio", args.sparsity,
        "--sparsity_type", args.sparsity_type,
        "--nsamples", str(args.nsamples),
        "--seed", str(args.seed),
        "--save", str(task["run_dir"]),
    ]


def execute_task(args, task, repo_root, gpu, index, total):
    if args.resume and task["result_path"].exists():
        try:
            existing = json.loads(task["result_path"].read_text(encoding="utf-8"))
            if existing.get("status") == "success":
                return existing
        except json.JSONDecodeError:
            pass

    command = build_command(args, task)
    task["run_dir"].mkdir(parents=True, exist_ok=True)
    with PRINT_LOCK:
        print("\n" + "=" * 80)
        print(
            f"[{index}/{total}] model={task['model_key']} beta={task['beta']} "
            f"GPU={gpu if gpu is not None else 'serial'}"
        )
        print(" ".join(shlex.quote(str(part)) for part in command))
        print("=" * 80)

    started = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()
    if args.dry_run:
        returncode, stdout_text, status = 0, "", "dry_run"
    else:
        returncode, stdout_text = run_and_tee(
            command, task["stdout_path"], repo_root, gpu=gpu)
        status = "success" if returncode == 0 else "failed"

    parsed = parse_stdout(stdout_text)
    saved, saved_log = parse_saved_log(task["run_dir"], "wgradpow")
    result = {
        "experiment": "gradient_beta_ablation",
        "model_key": task["model_key"],
        "model": task["model"],
        "method": "wgradpow",
        "metric": "|W| * |G|^beta",
        "sparsity_type": args.sparsity_type,
        "Grad_beta": float(task["beta"]),
        "target_sparsity": float(args.sparsity),
        "actual_sparsity": saved.get("actual_sparsity", parsed.get("actual_sparsity_stdout")),
        "ppl_test": saved.get("ppl_test", parsed.get("ppl_stdout")),
        "pruning_time_sec": parsed.get("pruning_time_sec"),
        "nsamples": args.nsamples,
        "seed": args.seed,
        "gpu": gpu,
        "status": status,
        "returncode": returncode,
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.time() - start_time,
        "run_dir": str(task["run_dir"]),
        "stdout_log": str(task["stdout_path"]),
        "saved_log": saved_log,
        "command": command,
    }
    task["result_path"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_summary(output_root):
    rows = []
    for path in Path(output_root).rglob("result.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda row: (row.get("model_key", ""), float(row.get("Grad_beta", 0))))
    fields = [
        "experiment", "model_key", "model", "method", "metric", "Grad_beta",
        "sparsity_type", "target_sparsity", "actual_sparsity", "ppl_test", "pruning_time_sec",
        "nsamples", "seed", "gpu", "status", "returncode", "elapsed_sec",
        "run_dir", "stdout_log", "saved_log", "command",
    ]
    path = Path(output_root) / "summary_all.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), ensure_ascii=False)
                if isinstance(row.get(key), (list, dict)) else row.get(key, "")
                for key in fields
            })
    return rows, path


def main():
    args = parse_args()
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be at least 1")
    tasks = build_tasks(args)
    pending = []
    for index, task in enumerate(tasks, start=1):
        if args.resume and task["result_path"].exists():
            try:
                if json.loads(task["result_path"].read_text(encoding="utf-8")).get("status") == "success":
                    continue
            except json.JSONDecodeError:
                pass
        pending.append((index, task))

    print(f"Gradient-beta ablation tasks: {len(tasks)}; pending: {len(pending)}")
    print(f"Betas: {', '.join(args.betas)}")
    repo_root = Path.cwd()
    failures = []

    if args.gpus and pending:
        executors = [
            ThreadPoolExecutor(max_workers=args.workers_per_gpu, thread_name_prefix=f"gpu-{gpu}")
            for gpu in args.gpus
        ]
        future_map = {}
        try:
            for position, (index, task) in enumerate(pending):
                slot = position % len(args.gpus)
                future = executors[slot].submit(
                    execute_task, args, task, repo_root, args.gpus[slot], index, len(tasks))
                future_map[future] = task
            for future in as_completed(future_map):
                result = future.result()
                if result.get("status") == "failed":
                    failures.append(result)
                write_summary(args.output_root)
        finally:
            for executor in executors:
                executor.shutdown(wait=True)
    else:
        for index, task in pending:
            result = execute_task(args, task, repo_root, None, index, len(tasks))
            if result.get("status") == "failed":
                failures.append(result)
            write_summary(args.output_root)

    rows, summary = write_summary(args.output_root)
    print(f"Finished: {sum(row.get('status') == 'success' for row in rows)}/{len(tasks)} successful")
    print(f"Summary: {summary}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
