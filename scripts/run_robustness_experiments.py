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

from run_v9_param_sweep import model_tag, parse_saved_log, parse_stdout, value_tag


PRINT_LOCK = threading.Lock()


MODEL_CONFIGS = {
    "llama1_7b": {
        "model": "/home/yh114/workdir/models/decapoda-research-llama-7B-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-7B-hf_128_0_fixed.pth",
        "best_roots": [
            "owl/owl_v9_wsqrtg_param_llama1_7b_a100",
            "owl/owl_v9_wsqrtg_param_a40",
        ],
    },
    "llama2_7b": {
        "model": "/home/yh114/workdir/DSnoT/models/Llama-2-7b-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-7b-hf_128_0.pth",
        "best_roots": [
            "owl/owl_v9_wsqrtg_param_llama2_7b_a100",
            "owl/owl_v9_wsqrtg_param_llama2_7b_l20",
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
        "best_roots": [
            "owl/owl_v9_wsqrtg_param_llama2_13b_l20_2gpu",
            "owl/owl_v9_wsqrtg_param_llama2_13b_a100",
        ],
    },
    "qwen2_5_7b": {
        "model": "/home/yh114/workdir/DSnoT/models/Qwen2.5-7B",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/qwen2/gradients_l2_Qwen2.5-7B_128_0.pth",
        "best_roots": [
            "owl/owl_v9_wsqrtg_param_qwen2_5_7b_l20_2gpu",
            "owl/owl_v9_wsqrtg_param_qwen2_5_7b_a100",
        ],
    },
    "mistral_7b": {
        "model": "/home/yh114/workdir/DSnoT/models/Mistral-7B",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/mistral/gradients_l2_Mistral-7B_128_0.pth",
        "best_roots": ["owl/owl_v9_wsqrtg_param_mistral_7b_l20_2gpu"],
    },
}

BASELINE_BEST_ROOTS = {
    ("llama1_7b", "wanda-owl-v9"): [
        "owl/owl_v9_param_sweep",
        "owl/strict_ablation_wsqrtg_owl_v9",
    ],
    ("llama2_7b", "wanda-owl-v9"): [
        "owl/strict_ablation_wsqrtg_owl_v9",
    ],
}

GRADIENT_METHODS = {
    "owl-v9-wsqrtg",
    "owl-v9-tanhwg",
    "owl-v9-w3gx",
    "owl-v9-w2mmsg",
    "owl-v9-w2xg",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run stability or calibration-sample sensitivity experiments.")
    parser.add_argument("--experiment", choices=["stability", "nsamples"], required=True)
    parser.add_argument("--model-keys", nargs="+", default=["llama1_7b", "llama2_7b"])
    parser.add_argument("--methods", nargs="+", default=["owl-v9-wsqrtg"])
    parser.add_argument("--sparsity", default="0.6")
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--nsamples-list", nargs="+", default=["32", "64", "128", "256"])
    parser.add_argument("--default-nsamples", default="128")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--sparsity-type", default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--cache-dir", default="llm_weights")
    parser.add_argument("--main", default="main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs="+", default=[], help="GPU slots, e.g. --gpus 0 1 or --gpus 0,1.")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def candidate_best_csvs(model_key, method):
    roots = BASELINE_BEST_ROOTS.get((model_key, method))
    if roots is None:
        roots = MODEL_CONFIGS[model_key]["best_roots"]

    paths = []
    for root in roots:
        paths.extend([
            Path(root) / "best_by_sparsity.csv",
            Path(root) / "summary_all.csv",
        ])
    return paths


def load_best_params(model_key, method, sparsity):
    best_rows = []
    for path in candidate_best_csvs(model_key, method):
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            if row.get("status", "success") != "success":
                continue
            if row.get("method") and row.get("method") != method:
                continue
            row_sparsity = as_float(row.get("target_sparsity") or row.get("sparsity_ratio"))
            row_ppl = as_float(row.get("ppl_test") or row.get("ppl_stdout"))
            if row_sparsity is None or row_ppl is None:
                continue
            if abs(row_sparsity - float(sparsity)) > 1e-6:
                continue
            row["_source_csv"] = str(path)
            row["_ppl"] = row_ppl
            best_rows.append(row)

    if not best_rows:
        raise FileNotFoundError(
            f"No successful best params found for model_key={model_key}, method={method}, sparsity={sparsity}. "
            "Run the 360-grid sweep first or provide the correct best_by_sparsity.csv."
        )

    best = min(best_rows, key=lambda row: row["_ppl"])
    return {
        "Hyper_m": best.get("Hyper_m") or best.get("hyper_m") or "5",
        "Lamda": best.get("Lamda") or best.get("lamda") or "0.08",
        "Owl_alpha": best.get("Owl_alpha") or best.get("owl_alpha") or "0.2",
        "source_ppl": best.get("ppl_test") or best.get("ppl_stdout"),
        "source_csv": best["_source_csv"],
    }


def build_tasks(args):
    tasks = []
    output_root = Path(args.output_root or f"owl/{args.experiment}_wsqrtg_v9")

    for model_key in args.model_keys:
        if model_key not in MODEL_CONFIGS:
            raise KeyError(f"Unknown model key: {model_key}. Supported: {', '.join(MODEL_CONFIGS)}")

        config = MODEL_CONFIGS[model_key]
        for method in args.methods:
            params = load_best_params(model_key, method, args.sparsity)

            if args.experiment == "stability":
                variants = [{"seed": seed, "nsamples": args.default_nsamples, "variant": value_tag("seed", seed)}
                            for seed in args.seeds]
            else:
                variants = [{"seed": "0", "nsamples": nsamples, "variant": value_tag("ns", nsamples)}
                            for nsamples in args.nsamples_list]

            for variant in variants:
                run_dir = (
                    output_root
                    / model_key
                    / model_tag(config["model"])
                    / args.sparsity_type
                    / method
                    / value_tag("s", args.sparsity)
                    / variant["variant"]
                    / value_tag("hm", params["Hyper_m"])
                    / value_tag("la", params["Lamda"])
                    / value_tag("alpha", params["Owl_alpha"])
                )
                tasks.append({
                    "model_key": model_key,
                    "model": config["model"],
                    "gradient": config["gradient"],
                    "method": method,
                    "sparsity": args.sparsity,
                    "seed": variant["seed"],
                    "nsamples": variant["nsamples"],
                    "Hyper_m": params["Hyper_m"],
                    "Lamda": params["Lamda"],
                    "Owl_alpha": params["Owl_alpha"],
                    "source_ppl": params["source_ppl"],
                    "source_csv": params["source_csv"],
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
        "--nsamples", str(task["nsamples"]),
        "--seed", str(task["seed"]),
        "--Hyper_m", str(task["Hyper_m"]),
        "--Lamda", str(task["Lamda"]),
        "--Owl_alpha", str(task["Owl_alpha"]),
        "--save", str(task["run_dir"]),
    ]
    if task["method"] in GRADIENT_METHODS:
        command.extend(["--gradient_path", task["gradient"]])
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


def execute_task(args, task, repo_root, gpu, index, total):
    if args.resume and task["result_path"].exists():
        try:
            existing = json.loads(task["result_path"].read_text(encoding="utf-8", errors="replace"))
            if existing.get("status") == "success":
                print(f"[{index}/{total}] skip completed: {task['result_path']}")
                return existing
        except json.JSONDecodeError:
            pass

    command = build_command(args, task)
    with PRINT_LOCK:
        print("\n" + "=" * 80)
        print(
            f"[{index}/{total}] {args.experiment} | model={task['model_key']} | "
            f"method={task['method']} | s={task['sparsity']} | seed={task['seed']} | "
            f"nsamples={task['nsamples']} | GPU={gpu if gpu is not None else 'serial'}"
        )
        print(" ".join(shlex.quote(str(x)) for x in command))
        print("=" * 80)

    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()
    task["run_dir"].mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        returncode, stdout_text = 0, ""
        status = "dry_run"
    else:
        returncode, stdout_text = run_and_tee(command, task["stdout_path"], repo_root, gpu=gpu)
        status = "success" if returncode == 0 else "failed"

    parsed_stdout = parse_stdout(stdout_text)
    saved, saved_log = parse_saved_log(task["run_dir"], task["method"])
    result = {
        "experiment": args.experiment,
        "model_key": task["model_key"],
        "model": task["model"],
        "method": task["method"],
        "sparsity_type": args.sparsity_type,
        "target_sparsity": float(task["sparsity"]),
        "actual_sparsity": saved.get("actual_sparsity", parsed_stdout.get("actual_sparsity_stdout")),
        "ppl_test": saved.get("ppl_test", parsed_stdout.get("ppl_stdout")),
        "source_ppl": task["source_ppl"],
        "source_csv": task["source_csv"],
        "Hyper_m": float(task["Hyper_m"]),
        "Lamda": float(task["Lamda"]),
        "Owl_alpha": float(task["Owl_alpha"]),
        "nsamples": int(task["nsamples"]),
        "seed": int(task["seed"]),
        "pruning_time_sec": parsed_stdout.get("pruning_time_sec"),
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


def write_summary(output_root, tasks):
    rows = []
    for task in tasks:
        path = task["result_path"]
        if not path.exists():
            continue
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8", errors="replace")))
        except json.JSONDecodeError:
            continue
    if not rows:
        return

    fields = [
        "experiment", "model_key", "model", "method", "sparsity_type", "target_sparsity",
        "actual_sparsity", "ppl_test", "source_ppl", "Hyper_m", "Lamda",
        "Owl_alpha", "nsamples", "seed", "pruning_time_sec", "gpu",
        "status", "returncode", "run_dir", "stdout_log", "source_csv",
    ]
    summary_path = Path(output_root) / "summary_all.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (
            r.get("model_key", ""),
            r.get("method", ""),
            int(r.get("seed", 0)),
            int(r.get("nsamples", 0)),
        )):
            writer.writerow(row)
    print(f"Summary: {summary_path}")


def main():
    args = parse_args()
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be at least 1")

    repo_root = Path.cwd()
    tasks = build_tasks(args)
    output_root = Path(args.output_root or f"owl/{args.experiment}_wsqrtg_v9")
    total = len(tasks)

    print(f"Experiment: {args.experiment}")
    print(f"Total runs: {total}")
    print(f"Output root: {output_root}")
    if args.gpus:
        print(f"GPU slots: {', '.join(args.gpus)}")
    else:
        print("Execution mode: serial")

    failures = []
    if args.gpus:
        executors = [
            ThreadPoolExecutor(max_workers=args.workers_per_gpu, thread_name_prefix=f"gpu-{gpu}")
            for gpu in args.gpus
        ]
        future_map = {}
        try:
            for position, task in enumerate(tasks):
                gpu_slot = position % len(args.gpus)
                gpu = args.gpus[gpu_slot]
                future = executors[gpu_slot].submit(
                    execute_task, args, task, repo_root, gpu, position + 1, total
                )
                future_map[future] = task
            for future in as_completed(future_map):
                result = future.result()
                if result.get("status") == "failed":
                    failures.append(result)
                write_summary(output_root, tasks)
        finally:
            for executor in executors:
                executor.shutdown(wait=True)
    else:
        for index, task in enumerate(tasks, start=1):
            result = execute_task(args, task, repo_root, None, index, total)
            if result.get("status") == "failed":
                failures.append(result)
            write_summary(output_root, tasks)

    write_summary(output_root, tasks)
    if failures:
        print(f"Failed runs: {len(failures)}. Rerun with --resume after checking stdout.log.")
        sys.exit(1)


if __name__ == "__main__":
    main()
