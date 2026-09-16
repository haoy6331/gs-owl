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


PRINT_LOCK = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid-search Wanda-OWL-V9 parameters and report the best setting for each sparsity."
    )
    parser.add_argument("--model", required=True, help="Model path passed to main.py --model.")
    parser.add_argument("--output-root", default="owl/owl_v9_param_sweep")
    parser.add_argument("--method", default="wanda-owl-v9")
    parser.add_argument("--sparsities", nargs="+", default=["0.5", "0.6", "0.7"])
    parser.add_argument("--owl-alphas", nargs="+", default=["0.10", "0.15", "0.20", "0.25"])
    parser.add_argument("--lamdas", nargs="+", default=[
        "0.02", "0.04", "0.06", "0.08", "0.10",
        "0.12", "0.14", "0.16", "0.18", "0.20",
    ])
    parser.add_argument("--hyper-ms", nargs="+", default=["5.0", "6.0", "7.0"])
    parser.add_argument("--sparsity-type", default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default="llm_weights")
    parser.add_argument("--main", default="main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=[],
        help="GPU ids used in parallel, for example: --gpus 0 1 2 3. Omit for serial execution.",
    )
    parser.add_argument(
        "--gpu-groups",
        nargs="+",
        default=[],
        help=(
            "Independent CUDA_VISIBLE_DEVICES groups. For example, '--gpu-groups 0 1' runs two "
            "single-GPU workers, while '--gpu-groups 0,1 2,3' runs two workers with two GPUs each."
        ),
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Concurrent experiments per GPU. Keep this at 1 unless one GPU can hold multiple models.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--defer-summary",
        action="store_true",
        help="Do not update shared summary CSV files during this process.",
    )
    parser.add_argument("--extra-args", default="")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based global task index to start from, inclusive.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Zero-based global task index to end at, inclusive. Default runs to the last task.",
    )
    return parser.parse_args()


def model_tag(model_path):
    tag = os.path.basename(str(model_path).rstrip("/\\")) or "model"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def value_tag(prefix, value):
    text = str(value).replace(".", "p").replace("-", "m")
    return f"{prefix}_{text}"


def parse_number_list(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


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


def parse_stdout(stdout_text):
    result = {}
    patterns = {
        "pruning_time_sec": r"pruning time:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "actual_sparsity_stdout": r"sparsity sanity check\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        "ppl_stdout": r"wikitext perplexity\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout_text)
        if match:
            result[key] = float(match.group(1))

    density_matches = re.findall(r"Adjusted density ratios:\s*\[(.*?)\]", stdout_text)
    if density_matches:
        result["layer_density_ratios"] = parse_number_list(density_matches[-1])

    sparsity_matches = re.findall(r"Adjusted sparsity ratios:\s*\[(.*?)\]", stdout_text)
    if sparsity_matches:
        result["layer_sparsity_ratios"] = parse_number_list(sparsity_matches[-1])

    return result


def parse_saved_log(save_dir, method):
    log_path = Path(save_dir) / f"log_{method}.txt"
    if not log_path.exists():
        return {}, str(log_path)

    header = None
    rows = []
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw_line.strip().split("\t")
        if not parts or not parts[0]:
            continue
        if parts[0] == "method":
            header = parts
        elif header and len(parts) == len(header):
            rows.append(dict(zip(header, parts)))

    if not rows:
        return {}, str(log_path)

    row = rows[-1]
    for key in ["actual_sparsity", "ppl_test", "sparsity_ratio", "Hyper_m", "Lamda", "Owl_alpha"]:
        if key in row:
            try:
                row[key] = float(row[key])
            except ValueError:
                pass
    for key in ["num_samples", "seed"]:
        if key in row:
            try:
                row[key] = int(row[key])
            except ValueError:
                pass
    return row, str(log_path)


def build_command(args, sparsity, alpha, lamda, hyper_m, run_dir):
    command = [
        args.python,
        args.main,
        "--model", args.model,
        "--cache_dir", args.cache_dir,
        "--prune_method", args.method,
        "--sparsity_ratio", str(sparsity),
        "--sparsity_type", args.sparsity_type,
        "--nsamples", str(args.nsamples),
        "--seed", str(args.seed),
        "--Hyper_m", str(hyper_m),
        "--Lamda", str(lamda),
        "--Owl_alpha", str(alpha),
        "--save", str(run_dir),
    ]
    if args.extra_args:
        command.extend(shlex.split(args.extra_args))
    return command


def run_and_tee(command, stdout_path, cwd, gpu=None):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    collected = []
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    prefix = f"[GPU {gpu}] " if gpu is not None else ""
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
        returncode = process.wait()
    return returncode, "".join(collected)


def csv_value(value):
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def collect_results(output_root):
    rows = []
    for path in Path(output_root).rglob("result.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8", errors="replace")))
        except json.JSONDecodeError:
            continue
    return rows


def write_csv(path, rows):
    fields = [
        "method", "target_sparsity", "actual_sparsity", "ppl_test",
        "Hyper_m", "Lamda", "Owl_alpha", "pruning_time_sec",
        "num_samples", "sparsity_type", "seed", "gpu", "status", "returncode",
        "run_dir", "stdout_log", "saved_log", "layer_density_ratios",
        "layer_sparsity_ratios", "command",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, "")) for k in fields})


def update_summaries(output_root):
    rows = collect_results(output_root)
    rows = sorted(rows, key=lambda r: (
        float(r.get("target_sparsity", 0)),
        float(r.get("ppl_test", 1e18)) if r.get("ppl_test") is not None else 1e18,
        float(r.get("Owl_alpha", 0)),
        float(r.get("Lamda", 0)),
        float(r.get("Hyper_m", 0)),
    ))
    write_csv(Path(output_root) / "summary_all.csv", rows)

    best_rows = []
    for sparsity in sorted({float(r["target_sparsity"]) for r in rows if r.get("status") == "success"}):
        candidates = [
            r for r in rows
            if r.get("status") == "success"
            and float(r.get("target_sparsity")) == sparsity
            and r.get("ppl_test") is not None
        ]
        if candidates:
            best_rows.append(min(candidates, key=lambda r: float(r["ppl_test"])))
    write_csv(Path(output_root) / "best_by_sparsity.csv", best_rows)
    return rows, best_rows


def build_tasks(args, output_root, model_name):
    tasks = []
    for sparsity in args.sparsities:
        for hyper_m in args.hyper_ms:
            for lamda in args.lamdas:
                for alpha in args.owl_alphas:
                    run_dir = (
                        output_root / model_name / args.sparsity_type / args.method /
                        value_tag("s", sparsity) / value_tag("hm", hyper_m) /
                        value_tag("la", lamda) / value_tag("alpha", alpha)
                    )
                    tasks.append({
                        "sparsity": sparsity,
                        "hyper_m": hyper_m,
                        "lamda": lamda,
                        "alpha": alpha,
                        "run_dir": run_dir,
                        "result_path": run_dir / "result.json",
                        "stdout_path": run_dir / "stdout.log",
                    })
    return tasks


def execute_task(args, repo_root, task, gpu, run_index, total):
    sparsity = task["sparsity"]
    hyper_m = task["hyper_m"]
    lamda = task["lamda"]
    alpha = task["alpha"]
    run_dir = task["run_dir"]
    result_path = task["result_path"]
    stdout_path = task["stdout_path"]
    command = build_command(args, sparsity, alpha, lamda, hyper_m, run_dir)

    with PRINT_LOCK:
        print("\n" + "=" * 80)
        print(
            f"[{run_index}/{total}] GPU={gpu if gpu is not None else 'serial'} "
            f"sparsity={sparsity}, Hyper_m={hyper_m}, Lamda={lamda}, Owl_alpha={alpha}"
        )
        print(" ".join(shlex.quote(str(x)) for x in command))
        print("=" * 80)

    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()

    if args.dry_run:
        result = {
            "method": args.method,
            "target_sparsity": float(sparsity),
            "Hyper_m": float(hyper_m),
            "Lamda": float(lamda),
            "Owl_alpha": float(alpha),
            "gpu": gpu,
            "status": "dry_run",
            "run_dir": str(run_dir),
            "stdout_log": str(stdout_path),
            "command": command,
            "returncode": None,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    returncode, stdout_text = run_and_tee(command, stdout_path, cwd=repo_root, gpu=gpu)
    parsed_stdout = parse_stdout(stdout_text)
    saved, saved_log = parse_saved_log(run_dir, args.method)

    result = {
        "method": args.method,
        "target_sparsity": float(sparsity),
        "actual_sparsity": saved.get("actual_sparsity", parsed_stdout.get("actual_sparsity_stdout")),
        "ppl_test": saved.get("ppl_test", parsed_stdout.get("ppl_stdout")),
        "Hyper_m": saved.get("Hyper_m", float(hyper_m)),
        "Lamda": saved.get("Lamda", float(lamda)),
        "Owl_alpha": saved.get("Owl_alpha", float(alpha)),
        "pruning_time_sec": parsed_stdout.get("pruning_time_sec"),
        "num_samples": saved.get("num_samples", args.nsamples),
        "sparsity_type": saved.get("sparsity_type", args.sparsity_type),
        "seed": saved.get("seed", args.seed),
        "gpu": gpu,
        "layer_density_ratios": parsed_stdout.get("layer_density_ratios", []),
        "layer_sparsity_ratios": parsed_stdout.get("layer_sparsity_ratios", []),
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.time() - start_time,
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
        "saved_log": saved_log,
        "command": command,
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def print_best(best_rows):
    if not best_rows:
        return
    print("Current best by sparsity:")
    for row in best_rows:
        print(
            f"  s={row['target_sparsity']}: ppl={row.get('ppl_test')} "
            f"alpha={row.get('Owl_alpha')} Lamda={row.get('Lamda')} Hyper_m={row.get('Hyper_m')}"
        )


def main():
    args = parse_args()
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be at least 1")
    gpu_groups = resolve_gpu_groups(args)

    repo_root = Path.cwd()
    output_root = Path(args.output_root)
    model_name = model_tag(args.model)
    all_tasks = build_tasks(args, output_root, model_name)
    total = len(all_tasks)
    start_index = max(args.start_index, 0)
    end_index = total - 1 if args.end_index is None else min(args.end_index, total - 1)
    if start_index > end_index:
        raise ValueError(
            f"Invalid task range: start_index={start_index}, end_index={end_index}, total={total}"
        )
    pending = []

    for run_index, task in enumerate(all_tasks, start=1):
        zero_based_index = run_index - 1
        if zero_based_index < start_index or zero_based_index > end_index:
            continue
        result_path = task["result_path"]
        if args.resume and result_path.exists():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8", errors="replace"))
                if existing.get("status") == "success":
                    print(f"[{run_index}/{total}] skip complete: {result_path}")
                    continue
            except json.JSONDecodeError:
                pass
        pending.append((run_index, task))

    print(f"Method: {args.method}")
    print(f"Total runs: {total}; selected range: {start_index}-{end_index}; pending: {len(pending)}")
    print(f"Output root: {output_root}")
    if gpu_groups:
        print(f"Parallel GPU groups: {', '.join(gpu_groups)}; workers per group: {args.workers_per_gpu}")
    else:
        print("Execution mode: serial")

    failures = []
    if gpu_groups and pending:
        executors = [
            ThreadPoolExecutor(max_workers=args.workers_per_gpu, thread_name_prefix=f"gpu-{gpu}")
            for gpu in gpu_groups
        ]
        future_map = {}
        try:
            for position, (run_index, task) in enumerate(pending):
                gpu_slot = position % len(gpu_groups)
                gpu = gpu_groups[gpu_slot]
                future = executors[gpu_slot].submit(
                    execute_task, args, repo_root, task, gpu, run_index, total
                )
                future_map[future] = task

            for future in as_completed(future_map):
                result = future.result()
                with PRINT_LOCK:
                    print(f"Saved result: {result.get('run_dir')}/result.json")
                    if not args.defer_summary:
                        _, best_rows = update_summaries(output_root)
                        print_best(best_rows)
                if result.get("status") == "failed":
                    failures.append(result)
        finally:
            for executor in executors:
                executor.shutdown(wait=True)
    else:
        for run_index, task in pending:
            result = execute_task(args, repo_root, task, None, run_index, total)
            print(f"Saved result: {result.get('run_dir')}/result.json")
            if not args.defer_summary:
                _, best_rows = update_summaries(output_root)
                print_best(best_rows)
            if result.get("status") == "failed":
                failures.append(result)

    print("\nSweep finished.")
    if args.defer_summary:
        print("Summary generation deferred. Run scripts/summarize_v9_param_sweep.py after all workers finish.")
    else:
        update_summaries(output_root)
        print(f"All results: {output_root / 'summary_all.csv'}")
        print(f"Best params: {output_root / 'best_by_sparsity.csv'}")
    if failures:
        print(f"Failed runs: {len(failures)}. Rerun with --resume after checking their stdout.log files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
