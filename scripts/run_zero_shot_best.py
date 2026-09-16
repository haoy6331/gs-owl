import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


def value_tag(prefix, value):
    return f"{prefix}_{str(value).replace('.', 'p').replace('-', 'm')}"


def parse_job(text):
    parts = text.split("|")
    if len(parts) != 5:
        raise ValueError(
            "--job must use: label|model_path|gradient_path|best_csv|output_root"
        )
    label, model_path, gradient_path, best_csv, output_root = parts
    return {
        "label": label,
        "model_path": model_path,
        "gradient_path": gradient_path,
        "best_csv": Path(best_csv),
        "output_root": Path(output_root),
    }


def load_best_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"best_by_sparsity.csv not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "success") != "success":
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No successful best rows found in {path}")
    return sorted(rows, key=lambda r: float(r["target_sparsity"]))


def build_command(args, job, row, save_dir):
    command = [
        args.python,
        args.main,
        "--model", job["model_path"],
        "--gradient_path", job["gradient_path"],
        "--prune_method", args.method,
        "--sparsity_ratio", str(row["target_sparsity"]),
        "--sparsity_type", args.sparsity_type,
        "--nsamples", str(args.nsamples),
        "--seed", str(args.seed),
        "--Hyper_m", str(row["Hyper_m"]),
        "--Lamda", str(row["Lamda"]),
        "--Owl_alpha", str(row["Owl_alpha"]),
        "--save", str(save_dir),
        "--eval_zero_shot",
    ]
    zero_shot_tasks = row.get("zero_shot_tasks", "")
    if zero_shot_tasks:
        command.extend(["--zero_shot_tasks", str(zero_shot_tasks)])
    if args.offline:
        command.append("--offline")
    if args.local_datasets_path:
        command.extend(["--local_datasets_path", args.local_datasets_path])
    return command


def run_and_tee(command, stdout_path):
    stdout_path = Path(stdout_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    log_file = None
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    def reopen_log(mode="a"):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        return stdout_path.open(mode, encoding="utf-8", errors="replace")

    try:
        log_file = reopen_log("w")
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            try:
                log_file.write(line)
                log_file.flush()
            except OSError as exc:
                print(
                    f"[warning] failed to write log {stdout_path}: {exc}; reopening log",
                    file=sys.stderr,
                )
                try:
                    log_file.close()
                except OSError:
                    pass
                log_file = reopen_log("a")
                log_file.write(f"[warning] log file reopened after write failure: {exc}\n")
                log_file.write(line)
                log_file.flush()
        return process.wait()
    finally:
        if log_file is not None:
            try:
                log_file.close()
            except OSError:
                pass
        if process.poll() is None:
            process.wait()


def summarize_lm_eval_json(path):
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return payload.get("scores", {})


def main():
    parser = argparse.ArgumentParser(
        description="Run zero-shot evaluation for the best OWL-V9 wsqrtg rows."
    )
    parser.add_argument(
        "--job",
        action="append",
        required=True,
        help="label|model_path|gradient_path|best_csv|output_root",
    )
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
    args = parser.parse_args()

    jobs = [parse_job(x) for x in args.job]
    failures = []

    for job in jobs:
        print("=" * 80)
        print(f"Zero-shot job: {job['label']}")
        print(f"Model: {job['model_path']}")
        print(f"Gradient: {job['gradient_path']}")
        print(f"Best CSV: {job['best_csv']}")
        print("=" * 80)
        rows = load_best_rows(job["best_csv"])

        for row in rows:
            sparsity = row["target_sparsity"]
            save_dir = (
                job["output_root"]
                / job["label"]
                / value_tag("s", sparsity)
                / value_tag("hm", row["Hyper_m"])
                / value_tag("la", row["Lamda"])
                / value_tag("alpha", row["Owl_alpha"])
            )
            lm_eval_json = save_dir / f"log_lm_eval_{args.method}.json"
            result_json = save_dir / "zero_shot_result.json"
            if args.resume and lm_eval_json.exists():
                scores = summarize_lm_eval_json(lm_eval_json)
                result_json.write_text(
                    json.dumps(
                        {
                            "status": "success",
                            "label": job["label"],
                            "target_sparsity": float(sparsity),
                            "scores": scores,
                            "lm_eval_json": str(lm_eval_json),
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                print(f"skip completed: {lm_eval_json}")
                continue

            command = build_command(args, job, row, save_dir)
            stdout_path = save_dir / "zero_shot_stdout.log"
            print("Command:")
            print(" ".join(shlex.quote(str(x)) for x in command))
            if args.dry_run:
                save_dir.mkdir(parents=True, exist_ok=True)
                result_json.write_text(
                    json.dumps(
                        {
                            "status": "dry_run",
                            "label": job["label"],
                            "target_sparsity": float(sparsity),
                            "command": command,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                continue

            returncode = run_and_tee(command, stdout_path)
            scores = summarize_lm_eval_json(lm_eval_json)
            result = {
                "status": "success" if returncode == 0 and lm_eval_json.exists() else "failed",
                "returncode": returncode,
                "label": job["label"],
                "target_sparsity": float(sparsity),
                "Hyper_m": float(row["Hyper_m"]),
                "Lamda": float(row["Lamda"]),
                "Owl_alpha": float(row["Owl_alpha"]),
                "scores": scores,
                "stdout_log": str(stdout_path),
                "lm_eval_json": str(lm_eval_json),
                "save_dir": str(save_dir),
            }
            result_json.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if result["status"] == "failed":
                failures.append(result)
                break

        if failures:
            break

    if failures:
        print(f"Failed zero-shot runs: {len(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
