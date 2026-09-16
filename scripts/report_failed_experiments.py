import argparse
import json
from collections import Counter
from pathlib import Path


ERROR_MARKERS = [
    "traceback",
    "runtimeerror",
    "torch.outofmemoryerror",
    "cuda out of memory",
    "out of memory",
    "filenotfounderror",
    "no such file",
    "does not exist",
    "keyerror",
    "hfvalidationerror",
    "incorrect path_or_model_id",
    "oserror",
    "valueerror",
    "modulenotfounderror",
    "importerror",
    "killed",
    "failed",
    "error",
]


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"status": "bad_json", "error": str(exc)}


def as_path(path_text, cwd):
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return cwd / path


def rel_parts(path, root):
    try:
        return path.relative_to(root).parts
    except ValueError:
        return path.parts


def first_existing(paths):
    for path in paths:
        if path and path.exists():
            return path
    return None


def find_log_path(result_path, payload, kind, cwd):
    candidates = []
    if payload.get("stdout_log"):
        candidates.append(as_path(payload["stdout_log"], cwd))
    if kind == "ppl":
        candidates.extend([
            result_path.parent / "stdout.log",
            result_path.parent / "log.txt",
        ])
    else:
        candidates.extend([
            result_path.parent / "zero_shot_stdout.log",
            result_path.parent / "stdout.log",
        ])
    return first_existing(candidates)


def infer_ppl_identity(path, root, payload):
    parts = rel_parts(path, root)
    output_root = parts[0] if len(parts) > 0 else ""
    model = parts[1] if len(parts) > 1 else ""
    method = payload.get("method", parts[3] if len(parts) > 3 else "")
    return output_root, model, method


def infer_zero_identity(path, root, payload):
    parts = rel_parts(path, root)
    output_root = parts[0] if len(parts) > 0 else ""
    label = payload.get("label", parts[1] if len(parts) > 1 else "")
    return output_root, label, "zero-shot"


def extract_reason(log_path, tail_lines):
    if not log_path:
        return ["log file not found"]
    if not log_path.exists():
        return [f"log file not found: {log_path}"]

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return ["log file is empty"]

    lowered = [line.lower() for line in lines]
    marker_indices = [
        index for index, line in enumerate(lowered)
        if any(marker in line for marker in ERROR_MARKERS)
    ]

    if marker_indices:
        start = marker_indices[-1]
        if "traceback" in lowered[start]:
            end = min(len(lines), start + tail_lines)
            return lines[start:end]
        start = max(0, start - 5)
        end = min(len(lines), marker_indices[-1] + tail_lines)
        return lines[start:end]

    nonempty = [line for line in lines if line.strip()]
    return nonempty[-tail_lines:] if nonempty else ["log file has no non-empty lines"]


def read_log_text(log_path):
    if not log_path:
        return ""
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")


def classify_reason(log_path):
    text = read_log_text(log_path).lower()
    if not text:
        return "log_missing_or_empty"
    if "cuda out of memory" in text or "torch.outofmemoryerror" in text:
        return "cuda_oom"
    if "keyerror: 'qwen2'" in text or 'keyerror: "qwen2"' in text:
        return "transformers_too_old_for_qwen2"
    if "incorrect path_or_model_id" in text or "hfvalidationerror" in text:
        return "model_path_error"
    if "gradient file does not exist" in text:
        return "gradient_path_missing"
    if "model directory does not exist" in text:
        return "model_path_missing"
    if "filenotfounderror" in text or "no such file or directory" in text:
        return "file_missing"
    if "killed" in text:
        return "process_killed"
    if "modulenotfounderror" in text or "importerror" in text:
        return "python_import_error"
    if "traceback" in text:
        return "python_traceback_other"
    return "unknown"


def collect_failures(root, kind_filter, cwd):
    failures = []

    if kind_filter in {"all", "ppl"}:
        for path in root.rglob("result.json"):
            payload = load_json(path)
            if payload.get("status") == "success":
                continue
            output_root, model, method = infer_ppl_identity(path, root, payload)
            failures.append({
                "kind": "ppl",
                "output_root": output_root,
                "model": model,
                "method": method,
                "status": payload.get("status", "missing_status"),
                "returncode": payload.get("returncode", ""),
                "target_sparsity": payload.get("target_sparsity", ""),
                "Hyper_m": payload.get("Hyper_m", ""),
                "Lamda": payload.get("Lamda", ""),
                "Owl_alpha": payload.get("Owl_alpha", ""),
                "result_path": path,
                "log_path": find_log_path(path, payload, "ppl", cwd),
            })

    if kind_filter in {"all", "zero"}:
        for path in root.rglob("zero_shot_result.json"):
            payload = load_json(path)
            if payload.get("status") == "success":
                continue
            output_root, label, method = infer_zero_identity(path, root, payload)
            failures.append({
                "kind": "zero",
                "output_root": output_root,
                "model": label,
                "method": method,
                "status": payload.get("status", "missing_status"),
                "returncode": payload.get("returncode", ""),
                "target_sparsity": payload.get("target_sparsity", ""),
                "Hyper_m": payload.get("Hyper_m", ""),
                "Lamda": payload.get("Lamda", ""),
                "Owl_alpha": payload.get("Owl_alpha", ""),
                "rank_in_sparsity": payload.get("rank_in_sparsity", ""),
                "task_id": payload.get("task_id", ""),
                "result_path": path,
                "log_path": find_log_path(path, payload, "zero", cwd),
            })

    for item in failures:
        item["reason"] = classify_reason(item["log_path"])

    return sorted(failures, key=lambda item: (
        item["kind"],
        item["output_root"],
        item["model"],
        str(item.get("target_sparsity", "")),
        str(item.get("task_id", "")),
        str(item["result_path"]),
    ))


def print_failure(item, index, total, tail_lines):
    params = (
        f"s={item.get('target_sparsity')} "
        f"hm={item.get('Hyper_m')} "
        f"la={item.get('Lamda')} "
        f"alpha={item.get('Owl_alpha')}"
    )
    extra = ""
    if item["kind"] == "zero":
        extra = f" task_id={item.get('task_id')} rank={item.get('rank_in_sparsity')}"

    print("\n" + "=" * 100)
    print(
        f"[{index}/{total}] {item['kind']} | {item['output_root']} | "
        f"{item['model']} | {item['method']} | status={item['status']} "
        f"returncode={item['returncode']} | reason={item.get('reason')} | {params}{extra}"
    )
    print(f"result: {item['result_path']}")
    print(f"log:    {item['log_path'] if item['log_path'] else 'not found'}")
    print("-" * 100)
    for line in extract_reason(item["log_path"], tail_lines):
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Print failed PPL/zero-shot experiments and log excerpts.")
    parser.add_argument("--root", default="owl", help="Directory to scan, usually owl.")
    parser.add_argument("--kind", choices=["all", "ppl", "zero"], default="all")
    parser.add_argument("--limit", type=int, default=40, help="Maximum failures to print. Use 0 for all.")
    parser.add_argument("--tail-lines", type=int, default=35, help="How many lines to print around the detected error.")
    parser.add_argument("--summary-only", action="store_true", help="Only print grouped failure counts.")
    args = parser.parse_args()

    cwd = Path.cwd()
    root = Path(args.root)
    failures = collect_failures(root, args.kind, cwd)
    total = len(failures)

    print(f"Found failed/non-success results: {total}")
    if not failures:
        return

    print("\nFailure reason summary")
    print("-" * 100)
    for reason, count in Counter(item["reason"] for item in failures).most_common():
        print(f"{reason}: {count}")

    print("\nFailure directory summary")
    print("-" * 100)
    for (kind, output_root, model, reason), count in Counter(
        (item["kind"], item["output_root"], item["model"], item["reason"])
        for item in failures
    ).most_common():
        print(f"{count:4d} | {kind} | {output_root} | {model} | {reason}")

    if args.summary_only:
        return

    shown = failures if args.limit == 0 else failures[:args.limit]
    for index, item in enumerate(shown, start=1):
        print_failure(item, index, total, args.tail_lines)

    if len(shown) < total:
        print("\n" + "-" * 100)
        print(f"Only showed {len(shown)} of {total}. Use --limit 0 to print all failures.")


if __name__ == "__main__":
    main()
