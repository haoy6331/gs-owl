import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from run_v9_param_sweep import parse_saved_log, parse_stdout, value_tag


PRINT_LOCK = threading.Lock()

ZERO_SHOT_TASKS = [
    "boolq",
    "rte",
    "hellaswag",
    "arc_challenge",
    "arc_easy",
    "winogrande",
    "openbookqa",
]


MODEL_CONFIGS = {
    "llama1_7b": {
        "model": "/home/yh114/workdir/models/decapoda-research-llama-7B-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-7B-hf_128_0_fixed.pth",
        "best_roots": [
            "owl/owl_v9_wsqrtg_param_a40",
            "owl/owl_v9_wsqrtg_param_llama1_7b_a100",
        ],
    },
    "llama2_7b": {
        "model": "/home/yh114/workdir/DSnoT/models/Llama-2-7b-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-7b-hf_128_0.pth",
        "best_roots": [
            "owl/owl_v9_wsqrtg_param_llama2_7b_l20",
            "owl/owl_v9_wsqrtg_param_llama2_7b_a100",
        ],
    },
    "llama1_13b": {
        "model": "/home/yh114/workdir/DSnoT/models/decapoda-research-llama-13B-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-13B-hf_128_0.pth",
        "best_roots": ["owl/owl_v9_wsqrtg_param_13b_l20"],
    },
    "llama2_13b": {
        "model": "/home/yh114/workdir/DSnoT/models/Llama-2-13b-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-13b-hf_128_0.pth",
        "best_roots": ["owl/owl_v9_wsqrtg_param_llama2_13b_a40_2gpu_final"],
    },
    "llama1_30b": {
        "model": "/home/yh114/workdir/DSnoT/models/decapoda-research-llama-30B-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_decapoda-research-llama-30B-hf_128_0.pth",
        "best_roots": ["owl/owl_v9_wsqrtg_param_llama1_30b_l20_4gpu"],
    },
}


METHOD_INFO = {
    "wanda": {"metric": "wanda", "allocation": "uniform", "gradient": False},
    "wanda-owl": {"metric": "wanda", "allocation": "original_owl", "gradient": False},
    "wanda-owl-v9": {"metric": "wanda", "allocation": "owl_v9", "gradient": False},
    "wsqrtg": {"metric": "wsqrtg", "allocation": "uniform", "gradient": True},
    "owl-wsqrtg": {"metric": "wsqrtg", "allocation": "original_owl", "gradient": True},
    "owl-v9-wsqrtg": {"metric": "wsqrtg", "allocation": "owl_v9", "gradient": True},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the strict 2-metric x 3-allocation OWL-V9 ablation."
    )
    parser.add_argument("--model-keys", nargs="+", default=["llama1_7b", "llama2_7b"])
    parser.add_argument("--methods", nargs="+", default=list(METHOD_INFO))
    parser.add_argument("--sparsities", nargs="+", default=["0.5", "0.6", "0.7"])
    parser.add_argument("--output-root", default="owl/strict_ablation_wsqrtg_owl_v9")
    parser.add_argument(
        "--best-root",
        action="append",
        default=[],
        metavar="MODEL_KEY=PATH",
        help="Override the final-method sweep root used to obtain fixed V9 parameters.",
    )
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
        help="Independent GPU slots, e.g. --gpus 0 1 or --gpus 0,1 2,3.",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--eval-zero-shot", action="store_true")
    parser.add_argument(
        "--zero-shot-tasks",
        default=",".join(ZERO_SHOT_TASKS),
        help="Comma-separated zero-shot tasks used with --eval-zero-shot.",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--local-datasets-path", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_best_root_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--best-root must use MODEL_KEY=PATH, got: {item}")
        model_key, path = item.split("=", 1)
        if not model_key or not path:
            raise ValueError(f"--best-root must use MODEL_KEY=PATH, got: {item}")
        overrides[model_key] = [path]
    return overrides


def candidate_summary_paths(root):
    root = Path(root)
    direct = [root / "best_by_sparsity.csv", root / "summary_all.csv"]
    existing = [path for path in direct if path.exists()]
    if existing:
        return existing
    return sorted(root.rglob("best_by_sparsity.csv")) + sorted(root.rglob("summary_all.csv"))


def load_fixed_params(model_key, sparsity, root_overrides):
    roots = root_overrides.get(model_key, MODEL_CONFIGS[model_key]["best_roots"])
    candidates = []
    searched = []
    for root in roots:
        paths = candidate_summary_paths(root)
        searched.extend(str(path) for path in paths)
        for path in paths:
            for row in read_csv_rows(path):
                if row.get("status", "success") != "success":
                    continue
                method = row.get("method", "owl-v9-wsqrtg")
                if method and method != "owl-v9-wsqrtg":
                    continue
                row_sparsity = as_float(row.get("target_sparsity") or row.get("sparsity_ratio"))
                ppl = as_float(row.get("ppl_test") or row.get("ppl_stdout"))
                if row_sparsity is None or ppl is None:
                    continue
                if abs(row_sparsity - float(sparsity)) > 1e-6:
                    continue
                candidates.append((ppl, row, path))

    if not candidates:
        roots_text = ", ".join(roots)
        searched_text = ", ".join(searched) if searched else "no summary CSV found"
        raise FileNotFoundError(
            f"No successful OWL-V9+wsqrtg parameters for model={model_key}, sparsity={sparsity}. "
            f"Roots: {roots_text}. Searched: {searched_text}"
        )

    ppl, row, source = min(candidates, key=lambda item: item[0])
    return {
        "Hyper_m": row.get("Hyper_m") or row.get("hyper_m") or "5",
        "Lamda": row.get("Lamda") or row.get("lamda") or "0.08",
        "Owl_alpha": row.get("Owl_alpha") or row.get("owl_alpha") or "0.2",
        "source_ppl": ppl,
        "source_csv": str(source),
    }


def build_tasks(args):
    root_overrides = parse_best_root_overrides(args.best_root)
    output_root = Path(args.output_root)
    tasks = []

    for model_key in args.model_keys:
        if model_key not in MODEL_CONFIGS:
            raise KeyError(f"Unknown model key '{model_key}'. Supported: {', '.join(MODEL_CONFIGS)}")
        config = MODEL_CONFIGS[model_key]
        for sparsity in args.sparsities:
            params = load_fixed_params(model_key, sparsity, root_overrides)
            for method in args.methods:
                if method not in METHOD_INFO:
                    raise KeyError(f"Unknown method '{method}'. Supported: {', '.join(METHOD_INFO)}")
                info = METHOD_INFO[method]
                run_dir = (
                    output_root / model_key / method / value_tag("s", sparsity) /
                    value_tag("hm", params["Hyper_m"]) /
                    value_tag("la", params["Lamda"]) /
                    value_tag("alpha", params["Owl_alpha"])
                )
                tasks.append({
                    "model_key": model_key,
                    "model": config["model"],
                    "gradient": config["gradient"],
                    "method": method,
                    "metric": info["metric"],
                    "allocation": info["allocation"],
                    "needs_gradient": info["gradient"],
                    "sparsity": sparsity,
                    **params,
                    "run_dir": run_dir,
                    "stdout_path": run_dir / "stdout.log",
                    "result_path": run_dir / "result.json",
                })
    return tasks


def build_command(args, task):
    command = [
        args.python,
        args.main,
        "--model", task["model"],
        "--cache_dir", args.cache_dir,
        "--prune_method", task["method"],
        "--sparsity_ratio", str(task["sparsity"]),
        "--sparsity_type", args.sparsity_type,
        "--nsamples", str(args.nsamples),
        "--seed", str(args.seed),
        "--Hyper_m", str(task["Hyper_m"]),
        "--Lamda", str(task["Lamda"]),
        "--Owl_alpha", str(task["Owl_alpha"]),
        "--save", str(task["run_dir"]),
    ]
    if task["needs_gradient"]:
        command.extend(["--gradient_path", task["gradient"]])
    if args.eval_zero_shot:
        command.extend([
            "--eval_zero_shot",
            "--zero_shot_tasks", args.zero_shot_tasks,
        ])
        if args.offline:
            command.append("--offline")
        if args.local_datasets_path:
            command.extend(["--local_datasets_path", args.local_datasets_path])
    if args.extra_args:
        command.extend(shlex.split(args.extra_args))
    return command


def load_zero_shot_result(run_dir, method):
    path = Path(run_dir) / f"log_lm_eval_{method}.json"
    if not path.exists():
        return {}, "", path
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}, "invalid_json", path
    scores = {
        key: float(value)
        for key, value in payload.get("scores", {}).items()
        if isinstance(value, (int, float))
    }
    return scores, payload.get("status", ""), path


def run_and_tee(command, stdout_path, cwd, gpu=None):
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


def execute_task(args, task, repo_root, gpu, index, total):
    if args.resume and task["result_path"].exists():
        try:
            existing = json.loads(task["result_path"].read_text(encoding="utf-8", errors="replace"))
            if existing.get("status") == "success":
                with PRINT_LOCK:
                    print(f"[{index}/{total}] skip completed: {task['result_path']}")
                return existing
        except json.JSONDecodeError:
            pass

    command = build_command(args, task)
    task["run_dir"].mkdir(parents=True, exist_ok=True)
    with PRINT_LOCK:
        print("\n" + "=" * 80)
        print(
            f"[{index}/{total}] model={task['model_key']} metric={task['metric']} "
            f"allocation={task['allocation']} method={task['method']} "
            f"sparsity={task['sparsity']} GPU={gpu if gpu is not None else 'serial'}"
        )
        print(" ".join(shlex.quote(str(part)) for part in command))
        print("=" * 80)

    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()
    if args.dry_run:
        returncode, stdout_text, status = 0, "", "dry_run"
    else:
        returncode, stdout_text = run_and_tee(command, task["stdout_path"], repo_root, gpu=gpu)
        status = "success" if returncode == 0 else "failed"

    parsed = parse_stdout(stdout_text)
    saved, saved_log = parse_saved_log(task["run_dir"], task["method"])
    zero_shot_scores, zero_shot_status, zero_shot_path = load_zero_shot_result(
        task["run_dir"], task["method"]
    )
    requested_tasks = [
        item.strip()
        for item in args.zero_shot_tasks.split(",")
        if item.strip()
    ]
    if args.eval_zero_shot and not args.dry_run:
        coverage_ok = all(task_name in zero_shot_scores for task_name in requested_tasks)
        if zero_shot_status != "success" or not coverage_ok:
            status = "failed"
    zero_shot_avg = (
        sum(zero_shot_scores[name] for name in requested_tasks) / len(requested_tasks)
        if requested_tasks and all(name in zero_shot_scores for name in requested_tasks)
        else None
    )
    result = {
        "experiment": "strict_metric_allocation_ablation",
        "model_key": task["model_key"],
        "model": task["model"],
        "method": task["method"],
        "pruning_metric": task["metric"],
        "allocation_method": task["allocation"],
        "target_sparsity": float(task["sparsity"]),
        "actual_sparsity": saved.get("actual_sparsity", parsed.get("actual_sparsity_stdout")),
        "ppl_test": saved.get("ppl_test", parsed.get("ppl_stdout")),
        "Hyper_m": float(task["Hyper_m"]),
        "Lamda": float(task["Lamda"]),
        "Owl_alpha": float(task["Owl_alpha"]),
        "source_final_ppl": task["source_ppl"],
        "source_params_csv": task["source_csv"],
        "nsamples": args.nsamples,
        "seed": args.seed,
        "sparsity_type": args.sparsity_type,
        "pruning_time_sec": parsed.get("pruning_time_sec"),
        "eval_zero_shot": args.eval_zero_shot,
        "evaluated_tasks": requested_tasks if args.eval_zero_shot else [],
        "zero_shot_scores": zero_shot_scores,
        "zero_shot_avg": zero_shot_avg,
        "zero_shot_status": zero_shot_status,
        "zero_shot_json": str(zero_shot_path),
        "layer_density_ratios": parsed.get("layer_density_ratios", []),
        "layer_sparsity_ratios": parsed.get("layer_sparsity_ratios", []),
        "gpu": gpu,
        "status": status,
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.time() - start_time,
        "run_dir": str(task["run_dir"]),
        "stdout_log": str(task["stdout_path"]),
        "saved_log": saved_log,
        "command": command,
    }
    task["result_path"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def write_summary(output_root):
    rows = []
    for path in Path(output_root).rglob("result.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8", errors="replace")))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda row: (
        row.get("model_key", ""),
        float(row.get("target_sparsity", 0)),
        row.get("pruning_metric", ""),
        row.get("allocation_method", ""),
    ))
    fields = [
        "experiment", "model_key", "model", "method", "pruning_metric", "allocation_method",
        "target_sparsity", "actual_sparsity", "ppl_test", "Hyper_m", "Lamda", "Owl_alpha",
        "source_final_ppl", "source_params_csv", "nsamples", "seed", "sparsity_type",
        "pruning_time_sec", "eval_zero_shot", "zero_shot_avg", "zero_shot_status",
        *ZERO_SHOT_TASKS,
        "status", "returncode", "gpu", "elapsed_sec", "run_dir",
        "stdout_log", "saved_log", "layer_density_ratios", "layer_sparsity_ratios", "command",
    ]
    summary_path = Path(output_root) / "summary_all.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output_row = {
                field: json.dumps(row.get(field), ensure_ascii=False)
                if isinstance(row.get(field), (list, dict)) else row.get(field, "")
                for field in fields
            }
            for task_name in ZERO_SHOT_TASKS:
                output_row[task_name] = row.get("zero_shot_scores", {}).get(task_name, "")
            writer.writerow(output_row)
    return rows, summary_path


def write_effect_table(output_root, rows):
    successful = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        ppl = as_float(row.get("ppl_test"))
        sparsity = as_float(row.get("target_sparsity"))
        if ppl is None or sparsity is None:
            continue
        key = (
            row.get("model_key", ""),
            sparsity,
            row.get("pruning_metric", ""),
            row.get("allocation_method", ""),
        )
        successful[key] = ppl

    table_rows = []
    groups = sorted({(key[0], key[1]) for key in successful})
    for model_key, sparsity in groups:
        values = {}
        for metric in ("wanda", "wsqrtg"):
            for allocation in ("uniform", "original_owl", "owl_v9"):
                values[f"{metric}_{allocation}_ppl"] = successful.get(
                    (model_key, sparsity, metric, allocation), ""
                )

        def improvement(left, right):
            left_value = as_float(values.get(left))
            right_value = as_float(values.get(right))
            if left_value is None or right_value is None:
                return ""
            return left_value - right_value

        table_rows.append({
            "model_key": model_key,
            "target_sparsity": sparsity,
            **values,
            "v9_gain_with_wanda": improvement("wanda_uniform_ppl", "wanda_owl_v9_ppl"),
            "v9_gain_with_wsqrtg": improvement("wsqrtg_uniform_ppl", "wsqrtg_owl_v9_ppl"),
            "wsqrtg_gain_with_uniform": improvement("wanda_uniform_ppl", "wsqrtg_uniform_ppl"),
            "wsqrtg_gain_with_original_owl": improvement(
                "wanda_original_owl_ppl", "wsqrtg_original_owl_ppl"
            ),
            "wsqrtg_gain_with_v9": improvement("wanda_owl_v9_ppl", "wsqrtg_owl_v9_ppl"),
        })

    fields = [
        "model_key", "target_sparsity",
        "wanda_uniform_ppl", "wanda_original_owl_ppl", "wanda_owl_v9_ppl",
        "wsqrtg_uniform_ppl", "wsqrtg_original_owl_ppl", "wsqrtg_owl_v9_ppl",
        "v9_gain_with_wanda", "v9_gain_with_wsqrtg",
        "wsqrtg_gain_with_uniform", "wsqrtg_gain_with_original_owl", "wsqrtg_gain_with_v9",
    ]
    path = Path(output_root) / "ablation_effects.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)
    return path


def main():
    args = parse_args()
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be at least 1")
    tasks = build_tasks(args)
    repo_root = Path.cwd()
    pending = []
    for index, task in enumerate(tasks, start=1):
        if args.resume and task["result_path"].exists():
            try:
                existing = json.loads(task["result_path"].read_text(encoding="utf-8", errors="replace"))
                if existing.get("status") == "success":
                    print(f"[{index}/{len(tasks)}] skip completed: {task['result_path']}")
                    continue
            except json.JSONDecodeError:
                pass
        pending.append((index, task))

    print("Strict ablation: 2 pruning metrics x 3 layer-allocation methods")
    print(f"Models: {', '.join(args.model_keys)}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Sparsities: {', '.join(args.sparsities)}")
    print(f"Total tasks: {len(tasks)}; pending: {len(pending)}")
    print(f"Output root: {args.output_root}")
    print(f"GPU slots: {', '.join(args.gpus) if args.gpus else 'serial'}")

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
                gpu = args.gpus[slot]
                future = executors[slot].submit(
                    execute_task, args, task, repo_root, gpu, index, len(tasks)
                )
                future_map[future] = task
            for future in as_completed(future_map):
                result = future.result()
                write_summary(args.output_root)
                with PRINT_LOCK:
                    print(f"Saved: {result.get('run_dir')}/result.json ({result.get('status')})")
                if result.get("status") == "failed":
                    failures.append(result)
        finally:
            for executor in executors:
                executor.shutdown(wait=True)
    else:
        for index, task in pending:
            result = execute_task(args, task, repo_root, None, index, len(tasks))
            write_summary(args.output_root)
            print(f"Saved: {result.get('run_dir')}/result.json ({result.get('status')})")
            if result.get("status") == "failed":
                failures.append(result)

    rows, summary_path = write_summary(args.output_root)
    effects_path = write_effect_table(args.output_root, rows)
    success_count = sum(row.get("status") == "success" for row in rows)
    failed_count = sum(row.get("status") == "failed" for row in rows)
    print("\nStrict ablation finished.")
    print(f"Successful results: {success_count}/{len(tasks)}; failed: {failed_count}")
    print(f"Summary: {summary_path}")
    print(f"Ablation effects: {effects_path}")
    if failures:
        print("Rerun the same command with --resume after checking failed stdout.log files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
