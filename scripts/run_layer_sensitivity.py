import argparse
import ast
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_v9_param_sweep import parse_saved_log, parse_stdout


MODEL_CONFIGS = {
    "llama1_7b": {
        "model": "/home/yh114/workdir/models/decapoda-research-llama-7B-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-7B-hf_128_0_fixed.pth",
        "layers": 32,
        "v9_roots": [
            "owl/owl_v9_wsqrtg_param_a40",
            "owl/owl_v9_wsqrtg_param_llama1_7b_a100",
        ],
    },
    "llama2_7b": {
        "model": "/home/yh114/workdir/DSnoT/models/Llama-2-7b-hf",
        "gradient": "/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-7b-hf_128_0.pth",
        "layers": 32,
        "v9_roots": [
            "owl/owl_v9_wsqrtg_param_llama2_7b_l20",
            "owl/owl_v9_wsqrtg_param_llama2_7b_a100",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure ground-truth transformer-layer pruning sensitivity."
    )
    parser.add_argument("--model-key", default="llama2_7b", choices=MODEL_CONFIGS)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--local-sparsity", type=float, default=0.5)
    parser.add_argument("--dense-ppl", type=float, default=None)
    parser.add_argument("--v9-log", default=None)
    parser.add_argument("--v9-reference-sparsity", type=float, default=0.6)
    parser.add_argument("--output-root", default="owl/layer_sensitivity_wsqrtg")
    parser.add_argument("--gpus", nargs="+", default=[])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--main", default="main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def rank_values(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + end - 1) / 2.0 + 1.0
        for offset in range(position, end):
            ranks[order[offset]] = average_rank
        position = end
    return ranks


def pearson(left, right):
    if len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    left_scale = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_scale = sum((y - right_mean) ** 2 for y in right) ** 0.5
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def spearman(left, right):
    return pearson(rank_values(left), rank_values(right))


def parse_v9_log(path):
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = {}
    pattern = re.compile(
        r"layer\s+(\d+)\s+V9 raw ratios:\s+count=([0-9.eE+-]+)%,\s+"
        r"severity_ratio=([0-9.eE+-]+)%"
    )
    for match in pattern.finditer(text):
        layer = int(match.group(1))
        rows[layer] = {
            "v9_count_ratio": float(match.group(2)),
            "v9_severity_ratio": float(match.group(3)),
        }
    density_match = re.search(r"Adjusted density ratios:\s*(\[[^\n]+\])", text)
    if density_match:
        try:
            density = ast.literal_eval(density_match.group(1))
        except (SyntaxError, ValueError):
            density = []
        for layer, value in enumerate(density):
            rows.setdefault(layer, {})["v9_density"] = float(value)
    return rows


def resolve_v9_log(args, config):
    if args.v9_log:
        return args.v9_log
    candidates = []
    for root in config.get("v9_roots", []):
        for filename in ("best_by_sparsity.csv", "summary_all.csv"):
            path = Path(root) / filename
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("status", "success") != "success":
                        continue
                    try:
                        sparsity = float(
                            row.get("target_sparsity") or row.get("sparsity_ratio")
                        )
                        ppl = float(row.get("ppl_test") or row.get("ppl_stdout"))
                    except (TypeError, ValueError):
                        continue
                    if abs(sparsity - args.v9_reference_sparsity) > 1e-6:
                        continue
                    stdout_log = row.get("stdout_log")
                    if not stdout_log and row.get("run_dir"):
                        stdout_log = str(Path(row["run_dir"]) / "stdout.log")
                    if stdout_log and Path(stdout_log).exists():
                        candidates.append((ppl, stdout_log))
    if not candidates:
        print(
            "No V9 reference log was found automatically; "
            "the layer PPL results will still run, but V9 correlations will be empty."
        )
        return None
    _, selected = min(candidates, key=lambda item: item[0])
    print(f"Auto-selected V9 reference log: {selected}")
    return selected


def build_command(args, config, layer, run_dir):
    return [
        args.python,
        args.main,
        "--model", config["model"],
        "--gradient_path", config["gradient"],
        "--prune_method", "layer-sensitivity-wsqrtg",
        "--target_layer", str(layer),
        "--sparsity_ratio", str(args.local_sparsity),
        "--sparsity_type", "unstructured",
        "--nsamples", str(args.nsamples),
        "--seed", str(args.seed),
        "--save", str(run_dir),
    ]


def run_task(args, config, layer, gpu):
    run_dir = Path(args.output_root) / args.model_key / f"layer_{layer:02d}"
    result_path = run_dir / "result.json"
    stdout_path = run_dir / "stdout.log"
    if args.resume and result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("status") == "success":
                return payload
        except json.JSONDecodeError:
            pass

    command = build_command(args, config, layer, run_dir)
    print(f"[layer {layer:02d}] GPU={gpu or 'serial'}")
    print(" ".join(shlex.quote(str(item)) for item in command))
    if args.dry_run:
        return {
            "status": "dry_run",
            "model_key": args.model_key,
            "target_layer": layer,
            "command": command,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        lines = []
        for line in process.stdout:
            print(f"[layer {layer:02d}] {line}", end="")
            log.write(line)
            log.flush()
            lines.append(line)
        returncode = process.wait()
    stdout_text = "".join(lines)
    parsed = parse_stdout(stdout_text)
    saved, saved_log = parse_saved_log(run_dir, "layer-sensitivity-wsqrtg")
    ppl = saved.get("ppl_test", parsed.get("ppl_stdout"))
    result = {
        "experiment": "ground_truth_layer_sensitivity",
        "model_key": args.model_key,
        "model": config["model"],
        "method": "layer-sensitivity-wsqrtg",
        "sparsity_type": "unstructured",
        "target_layer": layer,
        "target_sparsity": args.local_sparsity,
        "local_sparsity": args.local_sparsity,
        "ppl_test": ppl,
        "dense_ppl": args.dense_ppl,
        "delta_ppl": (
            float(ppl) - args.dense_ppl
            if ppl is not None and args.dense_ppl is not None
            else None
        ),
        "actual_global_sparsity": saved.get(
            "actual_sparsity", parsed.get("actual_sparsity_stdout")
        ),
        "actual_sparsity": saved.get(
            "actual_sparsity", parsed.get("actual_sparsity_stdout")
        ),
        "gpu": gpu,
        "status": "success" if returncode == 0 and ppl is not None else "failed",
        "returncode": returncode,
        "stdout_log": str(stdout_path),
        "saved_log": saved_log,
        "run_dir": str(run_dir),
        "command": command,
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def write_summary(args, config):
    v9_rows = parse_v9_log(args.v9_log)
    rows = []
    root = Path(args.output_root) / args.model_key
    for path in root.rglob("result.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        layer = int(row["target_layer"])
        if args.dense_ppl is not None and row.get("ppl_test") is not None:
            row["dense_ppl"] = args.dense_ppl
            row["delta_ppl"] = float(row["ppl_test"]) - args.dense_ppl
        row.update(v9_rows.get(layer, {}))
        rows.append(row)
    rows.sort(key=lambda row: int(row["target_layer"]))

    fields = [
        "model_key", "method", "target_layer", "local_sparsity", "ppl_test", "dense_ppl",
        "delta_ppl", "actual_global_sparsity", "v9_count_ratio",
        "v9_severity_ratio", "v9_density", "status", "gpu", "run_dir",
    ]
    summary_path = Path(args.output_root) / f"{args.model_key}_layer_sensitivity.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    correlations = {}
    target_field = "delta_ppl" if args.dense_ppl is not None else "ppl_test"
    valid = [row for row in rows if row.get(target_field) is not None]
    for metric in ("v9_count_ratio", "v9_severity_ratio", "v9_density"):
        paired = [
            (float(row[metric]), float(row[target_field]))
            for row in valid
            if row.get(metric) is not None
        ]
        correlations[metric] = (
            spearman(
                [item[0] for item in paired],
                [item[1] for item in paired],
            )
            if paired else None
        )
    correlation_path = Path(args.output_root) / f"{args.model_key}_correlations.json"
    correlation_path.write_text(
        json.dumps(
            {
                "model_key": args.model_key,
                "dense_ppl": args.dense_ppl,
                "v9_log": args.v9_log,
                "correlation_target": target_field,
                "spearman_with_layer_sensitivity": correlations,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")
    print(f"Correlations: {correlation_path}")


def main():
    args = parse_args()
    config = MODEL_CONFIGS[args.model_key]
    args.v9_log = resolve_v9_log(args, config)
    last_layer = config["layers"] - 1 if args.layer_end is None else args.layer_end
    if args.layer_start < 0 or last_layer >= config["layers"] or last_layer < args.layer_start:
        raise ValueError(
            f"Layer range must be within 0-{config['layers'] - 1}"
        )
    layers = list(range(args.layer_start, last_layer + 1))
    if not args.summarize_only:
        if args.gpus:
            executors = [
                ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gpu-{gpu}")
                for gpu in args.gpus
            ]
            futures = []
            try:
                for index, layer in enumerate(layers):
                    slot = index % len(args.gpus)
                    futures.append(
                        executors[slot].submit(
                            run_task, args, config, layer, args.gpus[slot]
                        )
                    )
                for future in as_completed(futures):
                    result = future.result()
                    print(
                        f"layer={result.get('target_layer')} "
                        f"status={result.get('status')} ppl={result.get('ppl_test')}"
                    )
            finally:
                for executor in executors:
                    executor.shutdown(wait=True)
        else:
            for layer in layers:
                run_task(args, config, layer, None)
    if not args.dry_run:
        write_summary(args, config)


if __name__ == "__main__":
    main()
