import argparse
import csv
import json
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from run_strict_ablation import (
    MODEL_CONFIGS,
    PRINT_LOCK,
    as_float,
    load_fixed_params,
    parse_best_root_overrides,
    run_and_tee,
)
from run_v9_param_sweep import parse_saved_log, parse_stdout, value_tag


VALID_COMPONENTS = ["count", "severity", "anchored", "core", "full"]
COMPONENTS = ["count", "anchored", "core", "full"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the five-stage OWL-V9 component ablation.")
    parser.add_argument("--model-keys", nargs="+", default=["llama1_7b", "llama2_7b"])
    parser.add_argument("--sparsities", nargs="+", default=["0.6", "0.7"])
    parser.add_argument("--components", nargs="+", default=COMPONENTS)
    parser.add_argument("--output-root", default="owl/v9_component_ablation")
    parser.add_argument("--best-root", action="append", default=[], metavar="MODEL_KEY=PATH")
    parser.add_argument("--core-start", type=int, default=4)
    parser.add_argument("--core-end", type=int, default=10)
    parser.add_argument("--tail-start", type=int, default=16)
    parser.add_argument("--tail-end", type=int, default=30)
    parser.add_argument(
        "--projection",
        choices=["none", "uniform", "protected"],
        default="protected",
    )
    parser.add_argument(
        "--interval-mode",
        choices=["absolute", "normalized"],
        default="absolute",
    )
    parser.add_argument("--reference-layers", type=int, default=32)
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
    overrides = parse_best_root_overrides(args.best_root)
    tasks = []
    for model_key in args.model_keys:
        if model_key not in MODEL_CONFIGS:
            raise KeyError(f"Unknown model key: {model_key}")
        config = MODEL_CONFIGS[model_key]
        for sparsity in args.sparsities:
            params = load_fixed_params(model_key, sparsity, overrides)
            for component in args.components:
                if component not in VALID_COMPONENTS:
                    raise ValueError(f"Unknown component: {component}")
                run_dir = (
                    Path(args.output_root)
                    / model_key
                    / component
                    / value_tag("s", sparsity)
                )
                tasks.append({
                    "model_key": model_key,
                    "model": config["model"],
                    "gradient": config["gradient"],
                    "sparsity": sparsity,
                    "component": component,
                    **params,
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
        "--prune_method", "owl-v9-wsqrtg",
        "--gradient_path", task["gradient"],
        "--sparsity_ratio", task["sparsity"],
        "--sparsity_type", args.sparsity_type,
        "--nsamples", str(args.nsamples),
        "--seed", str(args.seed),
        "--Hyper_m", str(task["Hyper_m"]),
        "--Lamda", str(task["Lamda"]),
        "--Owl_alpha", str(task["Owl_alpha"]),
        "--v9_component", task["component"],
        "--v9_core_start", str(args.core_start),
        "--v9_core_end", str(args.core_end),
        "--v9_tail_start", str(args.tail_start),
        "--v9_tail_end", str(args.tail_end),
        "--v9_projection", args.projection,
        "--v9_interval_mode", args.interval_mode,
        "--v9_reference_layers", str(args.reference_layers),
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
            f"[{index}/{total}] model={task['model_key']} sparsity={task['sparsity']} "
            f"component={task['component']} GPU={gpu if gpu is not None else 'serial'}"
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
    saved, saved_log = parse_saved_log(task["run_dir"], "owl-v9-wsqrtg")
    diagnostics_path = task["run_dir"] / "v9_layer_diagnostics.json"
    diagnostics = {}
    if diagnostics_path.exists():
        try:
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            diagnostics = {}
    result = {
        "experiment": "v9_component_ablation",
        "model_key": task["model_key"],
        "model": task["model"],
        "method": "owl-v9-wsqrtg",
        "sparsity_type": args.sparsity_type,
        "v9_component": task["component"],
        "target_sparsity": float(task["sparsity"]),
        "actual_sparsity": saved.get("actual_sparsity", parsed.get("actual_sparsity_stdout")),
        "ppl_test": saved.get("ppl_test", parsed.get("ppl_stdout")),
        "Hyper_m": float(task["Hyper_m"]),
        "Lamda": float(task["Lamda"]),
        "Owl_alpha": float(task["Owl_alpha"]),
        "core_start": args.core_start,
        "core_end": args.core_end,
        "tail_start": args.tail_start,
        "tail_end": args.tail_end,
        "projection_mode": args.projection,
        "interval_mode": args.interval_mode,
        "reference_layers": args.reference_layers,
        "source_final_ppl": task["source_ppl"],
        "source_params_csv": task["source_csv"],
        "pruning_time_sec": parsed.get("pruning_time_sec"),
        "layer_density_ratios": parsed.get("layer_density_ratios", []),
        "layer_sparsity_ratios": parsed.get("layer_sparsity_ratios", []),
        "budget_diagnostics": diagnostics.get("budget", {}),
        "constraint_diagnostics": diagnostics.get("constraints", {}),
        "resolved_intervals": diagnostics.get("resolved_intervals", {}),
        "diagnostics_json": str(diagnostics_path),
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
    rows.sort(key=lambda row: (
        row.get("model_key", ""),
        float(row.get("target_sparsity", 0)),
        VALID_COMPONENTS.index(row.get("v9_component", "full")),
    ))
    fields = [
        "experiment", "model_key", "model", "method", "v9_component",
        "sparsity_type", "target_sparsity", "actual_sparsity", "ppl_test", "Hyper_m", "Lamda",
        "Owl_alpha", "core_start", "core_end", "tail_start", "tail_end",
        "projection_mode", "interval_mode", "reference_layers",
        "source_final_ppl", "source_params_csv", "pruning_time_sec", "nsamples",
        "seed", "gpu", "status", "returncode", "elapsed_sec", "run_dir",
        "stdout_log", "saved_log", "layer_density_ratios",
        "layer_sparsity_ratios", "budget_diagnostics", "constraint_diagnostics",
        "resolved_intervals", "diagnostics_json", "command",
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

    print(f"V9 component ablation tasks: {len(tasks)}; pending: {len(pending)}")
    print(f"Components: {', '.join(args.components)}")
    print(f"Output root: {args.output_root}")
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
