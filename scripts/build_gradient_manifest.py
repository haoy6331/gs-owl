#!/usr/bin/env python3
"""Create or update a reproducibility manifest for saved gradient tensors."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def package_versions():
    versions = {}
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda_runtime"] = torch.version.cuda
    except ImportError:
        versions["torch"] = None
        versions["cuda_runtime"] = None

    for package in ("transformers", "accelerate"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", None)
        except ImportError:
            versions[package] = None
    return versions


def inspect_checkpoint(path):
    try:
        import torch
    except ImportError:
        return {
            "status": "not_inspected",
            "error": "PyTorch is not installed in the current environment.",
        }

    load_attempts = (
        {"map_location": "cpu", "weights_only": True, "mmap": True},
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu"},
    )
    errors = []
    checkpoint = None
    for load_kwargs in load_attempts:
        try:
            checkpoint = torch.load(path, **load_kwargs)
            break
        except (TypeError, RuntimeError, ValueError) as exc:
            errors.append(f"{load_kwargs}: {exc}")
    if checkpoint is None:
        return {"status": "failed", "error": " | ".join(errors)}

    if not isinstance(checkpoint, dict):
        return {
            "status": "failed",
            "error": f"Expected a dictionary, got {type(checkpoint).__name__}.",
        }

    tensors = []
    for name, value in checkpoint.items():
        if torch.is_tensor(value):
            tensors.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).replace("torch.", ""),
                    "numel": value.numel(),
                }
            )

    return {
        "status": "verified",
        "tensor_count": len(tensors),
        "total_numel": sum(item["numel"] for item in tensors),
        "dtypes": sorted({item["dtype"] for item in tensors}),
        "tensors": tensors,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create or update a JSON manifest for one gradient .pth file."
    )
    parser.add_argument("--gradient", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--nsamples", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--seqlen", required=True, type=int)
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--accumulation-steps", required=True, type=int)
    parser.add_argument("--model-dtype", default="float16")
    parser.add_argument("--accumulator-dtype", default="float32_cpu")
    parser.add_argument("--saved-dtype", default="float16")
    parser.add_argument("--generation-script")
    parser.add_argument("--generation-command")
    parser.add_argument("--source-log")
    parser.add_argument("--gpu-description")
    parser.add_argument(
        "--metadata-status",
        choices=("verified_from_log", "reconstructed_from_script", "author_confirmed"),
        default="reconstructed_from_script",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--skip-tensor-inspection",
        action="store_true",
        help="Record file metadata and hash without opening the checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gradient = args.gradient.expanduser().resolve()
    output = args.output.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()

    if not gradient.is_file():
        raise FileNotFoundError(f"Gradient file does not exist: {gradient}")
    if gradient.suffix != ".pth":
        raise ValueError(f"Expected a .pth gradient file: {gradient}")

    stat = gradient.stat()
    now = datetime.now(timezone.utc).isoformat()
    checkpoint = (
        {"status": "skipped"}
        if args.skip_tensor_inspection
        else inspect_checkpoint(gradient)
    )

    entry = {
        "model_label": args.model_label,
        "model_path": args.model_path,
        "gradient_path": str(gradient),
        "gradient_file": {
            "size_bytes": stat.st_size,
            "modified_at_utc": utc_timestamp(stat.st_mtime),
            "sha256": sha256_file(gradient),
        },
        "calibration_data": {
            "dataset": args.dataset,
            "split": args.split,
            "nsamples": args.nsamples,
            "seed": args.seed,
            "sequence_length": args.seqlen,
        },
        "gradient_computation": {
            "scale": args.scale,
            "accumulation_steps": args.accumulation_steps,
            "aggregation": (
                "sqrt(sum_steps((scale * accumulated_gradient)^2)); "
                "loss is divided by accumulation_steps before backward"
            ),
            "model_dtype": args.model_dtype,
            "accumulator_dtype": args.accumulator_dtype,
            "saved_dtype": args.saved_dtype,
        },
        "provenance": {
            "metadata_status": args.metadata_status,
            "generation_script": args.generation_script,
            "generation_command": args.generation_command,
            "source_log": args.source_log,
            "gpu_description": args.gpu_description,
            "code_commit": git_commit(repo_root),
            "software": package_versions(),
        },
        "checkpoint_inspection": checkpoint,
        "manifest_entry_updated_at_utc": now,
    }

    if output.exists():
        manifest = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
            raise ValueError(f"Invalid existing manifest: {output}")
    else:
        manifest = {
            "schema_version": 1,
            "description": "Reproducibility records for gradient tensors used in pruning experiments.",
            "created_at_utc": now,
            "entries": [],
        }

    entries = [
        item
        for item in manifest["entries"]
        if item.get("gradient_path") != str(gradient)
    ]
    entries.append(entry)
    entries.sort(key=lambda item: (item.get("model_label", ""), item.get("gradient_path", "")))
    manifest["entries"] = entries
    manifest["updated_at_utc"] = now

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + os.linesep,
        encoding="utf-8",
    )
    temporary.replace(output)

    print(f"Manifest updated: {output}")
    print(f"Model: {args.model_label}")
    print(f"Gradient: {gradient}")
    print(f"SHA-256: {entry['gradient_file']['sha256']}")
    print(f"Checkpoint inspection: {checkpoint['status']}")
    print(f"Manifest entries: {len(entries)}")


if __name__ == "__main__":
    main()
