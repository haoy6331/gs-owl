import argparse
import json
import sys
from pathlib import Path

from run_v9_param_sweep import build_tasks, execute_task, model_tag


def parse_args():
    parser = argparse.ArgumentParser(description="Run one shard of the V9 parameter sweep.")
    parser.add_argument("--shard-id", type=int, required=True, help="Zero-based shard id.")
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--model", required=True)
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-args", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"--shard-id must be in [0, {args.num_shards - 1}]")

    output_root = Path(args.output_root)
    all_tasks = build_tasks(args, output_root, model_tag(args.model))
    shard_tasks = [
        (index, task)
        for index, task in enumerate(all_tasks)
        if index % args.num_shards == args.shard_id
    ]

    print(
        f"Shard {args.shard_id}/{args.num_shards - 1}: "
        f"{len(shard_tasks)} of {len(all_tasks)} total experiments"
    )

    failures = 0
    for local_index, (global_index, task) in enumerate(shard_tasks, start=1):
        result_path = task["result_path"]
        if args.resume and result_path.exists():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8", errors="replace"))
                if existing.get("status") == "success":
                    print(
                        f"[shard {local_index}/{len(shard_tasks)}] "
                        f"skip completed global task {global_index}: {result_path}"
                    )
                    continue
            except json.JSONDecodeError:
                pass

        result = execute_task(
            args=args,
            repo_root=Path.cwd(),
            task=task,
            gpu=None,
            run_index=global_index + 1,
            total=len(all_tasks),
        )
        print(
            f"[shard {local_index}/{len(shard_tasks)}] global task {global_index} "
            f"finished with status={result.get('status')}"
        )
        if result.get("status") == "failed":
            failures += 1
            break

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
