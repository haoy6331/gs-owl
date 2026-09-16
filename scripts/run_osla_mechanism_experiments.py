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
    load_fixed_params,
    parse_best_root_overrides,
    run_and_tee,
)
from run_v9_param_sweep import parse_saved_log, parse_stdout, value_tag


DEFAULT_MODELS = {
    "progressive": ["llama1_7b", "llama2_7b"],
    "projection": ["llama1_7b", "llama2_7b"],
    "boundary": ["llama1_7b", "llama2_7b"],
    "depth": ["llama1_13b", "llama1_30b"],
}

DEFAULT_SPARSITIES = {
    "progressive": ["0.6", "0.7"],
    "projection": ["0.7"],
    "boundary": ["0.7"],
    "depth": ["0.7"],
}

SUITE_VARIANTS = {
    "progressive": {
        "count": {"component": "count"},
        "severity_fusion": {"component": "anchored"},
        "core_protection": {"component": "core"},
        "full_osla": {"component": "full"},
    },
    "projection": {
        "projection_none": {"component": "full", "projection": "none"},
        "projection_uniform": {"component": "full", "projection": "uniform"},
        "projection_protected": {"component": "full", "projection": "protected"},
    },
    "boundary": {
        "boundary_none": {"component": "anchored"},
        "boundary_default": {"component": "full"},
        "boundary_left2": {
            "component": "full",
            "core_start_delta": -2,
            "core_end_delta": -2,
            "tail_start_delta": -2,
            "tail_end_delta": -2,
        },
        "boundary_right2": {
            "component": "full",
            "core_start_delta": 2,
            "core_end_delta": 2,
            "tail_start_delta": 2,
            "tail_end_delta": 2,
        },
    },
    "depth": {
        "depth_absolute": {"component": "full", "interval_mode": "absolute"},
        "depth_normalized": {"component": "full", "interval_mode": "normalized"},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OSLA progressive, projection, boundary, and depth ablations."
    )
    parser.add_argument("--suite", choices=sorted(SUITE_VARIANTS), required=True)
    parser.add_argument("--model-keys", nargs="+", default=None)
    parser.add_argument("--sparsities", nargs="+", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset of suite variants. Defaults to every variant in the selected suite.",
    )
    parser.add_argument("--output-root", default="owl/osla_mechanism_ablation")
    parser.add_argument("--best-root", action="append", default=[], metavar="MODEL_KEY=PATH")
    parser.add_argument("--core-start", type=int, default=4)
    parser.add_argument("--core-end", type=int, default=10)
    parser.add_argument("--tail-start", type=int, default=16)
    parser.add_argument("--tail-end", type=int, default=30)
    parser.add_argument("--reference-layers", type=int, default=32)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity-type", default="unstructured")
    parser.add_argument("--cache-dir", default="llm_weights")
    parser.add_argument("--main", default="main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=[],
        help="Independent GPU slots, e.g. --gpus 0 1 or --gpus 0,1 2,3.",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_variant(args, variant_name):
    definition = dict(SUITE_VARIANTS[args.suite][variant_name])
    return {
        "component": definition.get("component", "full"),
        "projection": definition.get("projection", "protected"),
        "interval_mode": definition.get("interval_mode", "absolute"),
        "core_start": args.core_start + definition.get("core_start_delta", 0),
        "core_end": args.core_end + definition.get("core_end_delta", 0),
        "tail_start": args.tail_start + definition.get("tail_start_delta", 0),
        "tail_end": args.tail_end + definition.get("tail_end_delta", 0),
    }


def build_tasks(args):
    model_keys = args.model_keys or DEFAULT_MODELS[args.suite]
    sparsities = args.sparsities or DEFAULT_SPARSITIES[args.suite]
    variants = args.variants or list(SUITE_VARIANTS[args.suite])
    unknown_variants = [name for name in variants if name not in SUITE_VARIANTS[args.suite]]
    if unknown_variants:
        raise ValueError(
            f"Unknown {args.suite} variants: {', '.join(unknown_variants)}. "
            f"Available: {', '.join(SUITE_VARIANTS[args.suite])}"
        )

    overrides = parse_best_root_overrides(args.best_root)
    tasks = []
    for model_key in model_keys:
        if model_key not in MODEL_CONFIGS:
            raise KeyError(f"Unknown model key: {model_key}")
        config = MODEL_CONFIGS[model_key]
        for sparsity in sparsities:
            params = load_fixed_params(model_key, sparsity, overrides)
            for variant_name in variants:
                variant = resolve_variant(args, variant_name)
                run_dir = (
                    Path(args.output_root)
                    / args.suite
                    / model_key
                    / value_tag("s", sparsity)
                    / variant_name
                )
                tasks.append({
                    "suite": args.suite,
                    "variant": variant_name,
                    "model_key": model_key,
                    "model": config["model"],
                    "gradient": config["gradient"],
                    "sparsity": str(sparsity),
                    **params,
                    **variant,
                    "run_dir": run_dir,
                    "stdout_path": run_dir / "stdout.log",
                    "result_path": run_dir / "result.json",
                    "diagnostics_path": run_dir / "v9_layer_diagnostics.json",
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
        "--v9_projection", task["projection"],
        "--v9_interval_mode", task["interval_mode"],
        "--v9_reference_layers", str(args.reference_layers),
        "--v9_core_start", str(task["core_start"]),
        "--v9_core_end", str(task["core_end"]),
        "--v9_tail_start", str(task["tail_start"]),
        "--v9_tail_end", str(task["tail_end"]),
        "--save", str(task["run_dir"]),
    ]


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}


def execute_task(args, task, repo_root, gpu, index, total):
    if args.resume and task["result_path"].exists():
        existing = load_json(task["result_path"])
        if existing.get("status") == "success":
            with PRINT_LOCK:
                print(f"[{index}/{total}] skip completed: {task['result_path']}")
            return existing

    command = build_command(args, task)
    task["run_dir"].mkdir(parents=True, exist_ok=True)
    with PRINT_LOCK:
        print("\n" + "=" * 88)
        print(
            f"[{index}/{total}] suite={task['suite']} variant={task['variant']} "
            f"model={task['model_key']} sparsity={task['sparsity']} "
            f"GPU={gpu if gpu is not None else 'serial'}"
        )
        print(" ".join(shlex.quote(str(part)) for part in command))
        print("=" * 88)

    started = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()
    if args.dry_run:
        returncode, stdout_text, status = 0, "", "dry_run"
    else:
        returncode, stdout_text = run_and_tee(
            command, task["stdout_path"], repo_root, gpu=gpu
        )
        status = "success" if returncode == 0 else "failed"

    parsed = parse_stdout(stdout_text)
    saved, saved_log = parse_saved_log(task["run_dir"], "owl-v9-wsqrtg")
    diagnostics = load_json(task["diagnostics_path"])
    ppl = saved.get("ppl_test", parsed.get("ppl_stdout"))
    if not args.dry_run and ppl is None:
        status = "failed"

    budget = diagnostics.get("budget", {})
    constraints = diagnostics.get("constraints", {})
    intervals = diagnostics.get("resolved_intervals", {})
    result = {
        "experiment": "osla_mechanism_ablation",
        "suite": task["suite"],
        "variant": task["variant"],
        "model_key": task["model_key"],
        "model": task["model"],
        "method": "owl-v9-wsqrtg",
        "pruning_metric": "wsqrtg",
        "target_sparsity": float(task["sparsity"]),
        "actual_sparsity": saved.get(
            "actual_sparsity", parsed.get("actual_sparsity_stdout")
        ),
        "ppl_test": ppl,
        "Hyper_m": float(task["Hyper_m"]),
        "Lamda": float(task["Lamda"]),
        "Owl_alpha": float(task["Owl_alpha"]),
        "v9_component": task["component"],
        "projection_mode": task["projection"],
        "interval_mode": task["interval_mode"],
        "reference_layers": args.reference_layers,
        "core_start": task["core_start"],
        "core_end": task["core_end"],
        "tail_start": task["tail_start"],
        "tail_end": task["tail_end"],
        "resolved_core_start": intervals.get("core_start"),
        "resolved_core_end": intervals.get("core_end"),
        "resolved_tail_start": intervals.get("tail_start"),
        "resolved_tail_end": intervals.get("tail_end"),
        "pre_projection_budget_error": budget.get("pre_projection_error"),
        "post_projection_budget_error": budget.get("post_projection_error"),
        "projection_displacement": budget.get("mean_projection_displacement"),
        "active_core_layers": constraints.get("active_core_layers"),
        "active_tail_layers": constraints.get("active_tail_layers"),
        "source_final_ppl": task["source_ppl"],
        "source_params_csv": task["source_csv"],
        "pruning_time_sec": parsed.get("pruning_time_sec"),
        "layer_density_ratios": parsed.get("layer_density_ratios", []),
        "layer_sparsity_ratios": parsed.get("layer_sparsity_ratios", []),
        "nsamples": args.nsamples,
        "seed": args.seed,
        "sparsity_type": args.sparsity_type,
        "gpu": gpu,
        "status": status,
        "returncode": returncode,
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.time() - start_time,
        "run_dir": str(task["run_dir"]),
        "stdout_log": str(task["stdout_path"]),
        "saved_log": saved_log,
        "diagnostics_json": str(task["diagnostics_path"]),
        "command": command,
    }
    task["result_path"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


SUMMARY_FIELDS = [
    "experiment", "suite", "variant", "model_key", "model", "method",
    "pruning_metric", "target_sparsity", "actual_sparsity", "ppl_test",
    "Hyper_m", "Lamda", "Owl_alpha", "v9_component", "projection_mode",
    "interval_mode", "reference_layers", "core_start", "core_end",
    "tail_start", "tail_end", "resolved_core_start", "resolved_core_end",
    "resolved_tail_start", "resolved_tail_end", "pre_projection_budget_error",
    "post_projection_budget_error", "projection_displacement",
    "active_core_layers", "active_tail_layers", "source_final_ppl",
    "source_params_csv", "pruning_time_sec", "nsamples", "seed",
    "sparsity_type", "gpu", "status", "returncode", "elapsed_sec", "run_dir",
    "stdout_log", "saved_log", "diagnostics_json", "layer_density_ratios",
    "layer_sparsity_ratios", "command",
]


def write_summary(output_root):
    rows = []
    for path in Path(output_root).rglob("result.json"):
        row = load_json(path)
        if row.get("experiment") == "osla_mechanism_ablation":
            rows.append(row)
    rows.sort(key=lambda row: (
        row.get("suite", ""),
        row.get("model_key", ""),
        float(row.get("target_sparsity", 0)),
        row.get("variant", ""),
    ))
    summary_path = Path(output_root) / "summary_all.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row.get(field), ensure_ascii=False)
                if isinstance(row.get(field), (list, dict)) else row.get(field, "")
                for field in SUMMARY_FIELDS
            })
    return rows, summary_path


def main():
    args = parse_args()
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be at least 1")
    tasks = build_tasks(args)
    pending = []
    for index, task in enumerate(tasks, start=1):
        existing = load_json(task["result_path"]) if task["result_path"].exists() else {}
        if args.resume and existing.get("status") == "success":
            continue
        pending.append((index, task))

    print(f"OSLA mechanism suite: {args.suite}")
    print(f"Tasks: {len(tasks)}; pending: {len(pending)}")
    print(f"Output root: {args.output_root}")
    repo_root = Path.cwd()
    failures = []

    if args.gpus and pending:
        executors = [
            ThreadPoolExecutor(
                max_workers=args.workers_per_gpu,
                thread_name_prefix=f"gpu-{gpu}",
            )
            for gpu in args.gpus
        ]
        future_map = {}
        try:
            for position, (index, task) in enumerate(pending):
                slot = position % len(args.gpus)
                future = executors[slot].submit(
                    execute_task,
                    args,
                    task,
                    repo_root,
                    args.gpus[slot],
                    index,
                    len(tasks),
                )
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

    rows, summary_path = write_summary(args.output_root)
    successful = sum(row.get("status") == "success" for row in rows)
    print(f"Recorded successful runs: {successful}/{len(rows)}")
    print(f"Summary: {summary_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
