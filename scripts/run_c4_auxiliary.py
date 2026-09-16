import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


MODEL_CONFIGS = {
    "llama1_7b": {
        "model": "/home/yh114/workdir/DSnoT/models/decapoda-research-llama-7B-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-7B-hf_128_0_fixed.pth",
        "sweep_root": "owl/owl_v9_wsqrtg_param_a40",
    },
    "llama1_13b": {
        "model": "/home/yh114/workdir/DSnoT/models/decapoda-research-llama-13B-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-13B-hf_128_0.pth",
        "sweep_root": "owl/owl_v9_wsqrtg_param_13b_l20",
    },
    "llama1_30b": {
        "model": "/home/yh114/workdir/DSnoT/models/decapoda-research-llama-30B-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_l2_decapoda-research-llama-30B-hf_128_0.pth",
        "sweep_root": "owl/owl_v9_wsqrtg_param_llama1_30b_l20_4gpu",
    },
    "llama2_7b": {
        "model": "/home/yh114/workdir/DSnoT/models/Llama-2-7b-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-7b-hf_128_0.pth",
        "sweep_root": "owl/owl_v9_wsqrtg_param_llama2_7b_l20",
    },
    "llama2_13b": {
        "model": "/home/yh114/workdir/DSnoT/models/Llama-2-13b-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-13b-hf_128_0.pth",
        "sweep_root": "owl/owl_v9_wsqrtg_param_llama2_13b_a40_2gpu_final",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the fixed GS-OWL configurations selected by the main WikiText2 "
            "sweep on the C4 validation split."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_CONFIGS),
        default=sorted(MODEL_CONFIGS),
    )
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    parser.add_argument("--output-root", default="owl/c4_validation_gs_owl")
    parser.add_argument("--main", default="main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cache-dir", default="llm_weights")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-args", default="")
    return parser.parse_args()


def as_float(row, *keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def load_sweep_rows(sweep_root):
    root = Path(sweep_root)
    candidates = [root / "best_by_sparsity.csv", root / "summary_all.csv"]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return path, list(csv.DictReader(handle))

    rows = []
    for result_path in root.rglob("result.json") if root.exists() else []:
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["_source_result_json"] = str(result_path)
            rows.append(payload)
    if rows:
        return root, rows

    raise FileNotFoundError(
        "No best_by_sparsity.csv, summary_all.csv, or readable result.json "
        f"files found under {root}"
    )


def load_fixed_rows(sweep_root, requested_sparsities):
    source_path, rows = load_sweep_rows(sweep_root)

    selected = {}
    for target in requested_sparsities:
        candidates = []
        for row in rows:
            sparsity = as_float(row, "target_sparsity", "sparsity_ratio")
            ppl = as_float(row, "ppl_test", "ppl_stdout")
            status = row.get("status", "success")
            if sparsity is None or abs(sparsity - target) > 1e-8:
                continue
            if status not in ("", "success") or ppl is None:
                continue
            candidates.append((ppl, row))
        if not candidates:
            raise ValueError(
                f"No successful fixed configuration for sparsity={target} in {source_path}"
            )
        selected[target] = min(candidates, key=lambda item: item[0])[1]
    return source_path, selected


def value_tag(prefix, value):
    return f"{prefix}_{str(value).replace('.', 'p').replace('-', 'm')}"


def build_tasks(args):
    tasks = []
    for model_key in args.models:
        config = MODEL_CONFIGS[model_key]
        source_path, selected = load_fixed_rows(config["sweep_root"], args.sparsities)
        for sparsity in args.sparsities:
            row = selected[sparsity]
            hyper_m = as_float(row, "Hyper_m")
            lamda = as_float(row, "Lamda")
            alpha = as_float(row, "Owl_alpha")
            if None in (hyper_m, lamda, alpha):
                raise ValueError(f"Missing GS-OWL parameters in {source_path}: {row}")

            run_dir = (
                Path(args.output_root)
                / model_key
                / value_tag("s", sparsity)
                / value_tag("hm", hyper_m)
                / value_tag("la", lamda)
                / value_tag("alpha", alpha)
            )
            tasks.append(
                {
                    "model_key": model_key,
                    "model": config["model"],
                    "gradient": config["gradient"],
                    "source_results": str(source_path),
                    "source_ppl": as_float(row, "ppl_test", "ppl_stdout"),
                    "sparsity": sparsity,
                    "Hyper_m": hyper_m,
                    "Lamda": lamda,
                    "Owl_alpha": alpha,
                    "run_dir": run_dir,
                    "result_path": run_dir / "result.json",
                    "stdout_path": run_dir / "stdout.log",
                }
            )
    return tasks


def build_command(args, task):
    command = [
        args.python,
        args.main,
        "--model", task["model"],
        "--cache_dir", args.cache_dir,
        "--prune_method", "owl-v9-wsqrtg",
        "--sparsity_ratio", str(task["sparsity"]),
        "--sparsity_type", "unstructured",
        "--nsamples", str(args.nsamples),
        "--seed", str(args.seed),
        "--Hyper_m", str(task["Hyper_m"]),
        "--Lamda", str(task["Lamda"]),
        "--Owl_alpha", str(task["Owl_alpha"]),
        "--gradient_path", task["gradient"],
        "--eval_dataset", "c4",
        "--save", str(task["run_dir"]),
    ]
    if args.extra_args:
        command.extend(shlex.split(args.extra_args))
    return command


def run_and_tee(command, stdout_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    collected = []
    with stdout_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
            collected.append(line)
    return process.wait(), "".join(collected)


def parse_stdout(stdout_text):
    def match_float(pattern):
        match = re.search(pattern, stdout_text)
        return float(match.group(1)) if match else None

    return {
        "actual_sparsity": match_float(
            r"sparsity sanity check\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
        "ppl_c4": match_float(
            r"c4 perplexity\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
        "pruning_time_sec": match_float(
            r"pruning time:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
    }


def write_summary(output_root):
    rows = []
    for path in Path(output_root).rglob("result.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    rows.sort(key=lambda row: (row.get("model_key", ""), row.get("target_sparsity", 0)))
    fieldnames = [
        "model_key", "target_sparsity", "actual_sparsity", "source_wikitext2_ppl",
        "ppl_c4", "Hyper_m", "Lamda", "Owl_alpha", "pruning_time_sec", "status",
        "source_results", "run_dir", "stdout_log",
    ]
    summary_path = Path(output_root) / "summary_c4.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main():
    args = parse_args()
    tasks = build_tasks(args)
    end_index = len(tasks) - 1 if args.end_index is None else min(args.end_index, len(tasks) - 1)
    if args.start_index < 0 or args.start_index > end_index:
        raise ValueError(
            f"Invalid task range {args.start_index}-{end_index} for {len(tasks)} tasks"
        )

    for index, task in enumerate(tasks):
        if index < args.start_index or index > end_index:
            continue
        result_path = task["result_path"]
        if args.resume and result_path.exists():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            if existing.get("status") == "success" and existing.get("ppl_c4") is not None:
                print(f"[{index + 1}/{len(tasks)}] skip completed {task['model_key']} s={task['sparsity']}")
                continue

        command = build_command(args, task)
        print("\n" + "=" * 80)
        print(
            f"[{index + 1}/{len(tasks)}] {task['model_key']} "
            f"sparsity={task['sparsity']} -> C4 validation"
        )
        print(" ".join(shlex.quote(str(part)) for part in command))
        print("=" * 80)

        task["run_dir"].mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            returncode, stdout_text = None, ""
            parsed = {"actual_sparsity": None, "ppl_c4": None, "pruning_time_sec": None}
            status = "dry_run"
        else:
            started = time.time()
            returncode, stdout_text = run_and_tee(command, task["stdout_path"])
            parsed = parse_stdout(stdout_text)
            status = "success" if returncode == 0 and parsed["ppl_c4"] is not None else "failed"

        result = {
            "experiment": "c4_cross_dataset_validation",
            "selection_protocol": "fixed_from_main_wikitext2_sweep",
            "method": "owl-v9-wsqrtg",
            "sparsity_type": "unstructured",
            "eval_dataset": "c4",
            "model_key": task["model_key"],
            "target_sparsity": task["sparsity"],
            "actual_sparsity": parsed["actual_sparsity"],
            "source_wikitext2_ppl": task["source_ppl"],
            "ppl_c4": parsed["ppl_c4"],
            "Hyper_m": task["Hyper_m"],
            "Lamda": task["Lamda"],
            "Owl_alpha": task["Owl_alpha"],
            "pruning_time_sec": parsed["pruning_time_sec"],
            "status": status,
            "returncode": returncode,
            "source_results": task["source_results"],
            "run_dir": str(task["run_dir"]),
            "stdout_log": str(task["stdout_path"]),
            "command": command,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        if not args.dry_run:
            result["elapsed_sec"] = time.time() - started
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved result: {result_path}")

    summary_path = write_summary(args.output_root)
    print(f"C4 summary: {summary_path}")


if __name__ == "__main__":
    main()
