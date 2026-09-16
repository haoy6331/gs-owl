import argparse
import shutil
from pathlib import Path

from report_failed_experiments import collect_failures


def matches_filter(value, filters):
    if not filters:
        return True
    text = str(value)
    return any(item in text for item in filters)


def has_success_result(path):
    for name in ("result.json", "zero_shot_result.json"):
        for result_path in path.rglob(name):
            try:
                import json

                payload = json.loads(result_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            if payload.get("status") == "success":
                return True
    return False


def safe_leaf_dir(result_path, root):
    root = root.resolve()
    target = result_path.parent.resolve()
    if target == root:
        raise ValueError(f"Refuse to delete root directory: {target}")
    if root not in target.parents:
        raise ValueError(f"Refuse to delete outside root: {target}")
    return target


def main():
    parser = argparse.ArgumentParser(
        description="Dry-run or delete failed experiment directories under an owl result root."
    )
    parser.add_argument("--root", default="owl", help="Directory to scan, usually owl.")
    parser.add_argument("--kind", choices=["all", "ppl", "zero"], default="all")
    parser.add_argument(
        "--output-root",
        action="append",
        default=[],
        help="Keep only failures whose top-level output root contains this text. Can be repeated.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Keep only failures whose model/label contains this text. Can be repeated.",
    )
    parser.add_argument(
        "--reason",
        action="append",
        default=[],
        help="Keep only failures whose classified reason contains this text. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of directories. 0 means no limit.")
    parser.add_argument("--delete", action="store_true", help="Actually delete. Omit for dry-run.")
    parser.add_argument(
        "--allow-success-inside",
        action="store_true",
        help="Allow deleting a directory that contains a nested successful result. Off by default.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    failures = collect_failures(root, args.kind, cwd)
    selected = []
    seen = set()

    for item in failures:
        if not matches_filter(item["output_root"], args.output_root):
            continue
        if not matches_filter(item["model"], args.model):
            continue
        if not matches_filter(item["reason"], args.reason):
            continue

        target = safe_leaf_dir(Path(item["result_path"]), root)
        if target in seen:
            continue
        seen.add(target)
        selected.append((target, item))

    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    mode = "DELETE" if args.delete else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Root: {root}")
    print(f"Matched failed directories: {len(selected)}")

    skipped_success = 0
    deleted = 0
    for index, (target, item) in enumerate(selected, start=1):
        rel_target = target.relative_to(root.resolve())
        prefix = (
            f"[{index}/{len(selected)}] {item['kind']} | {item['output_root']} | "
            f"{item['model']} | reason={item['reason']} | {rel_target}"
        )

        if not args.allow_success_inside and has_success_result(target):
            skipped_success += 1
            print(f"SKIP success-inside: {prefix}")
            continue

        if args.delete:
            shutil.rmtree(target)
            deleted += 1
            print(f"DELETED: {prefix}")
        else:
            print(f"WOULD DELETE: {prefix}")

    if args.delete:
        print(f"Deleted directories: {deleted}")
        print(f"Skipped because success result was nested inside: {skipped_success}")
    else:
        print("No files were deleted. Add --delete to actually remove these directories.")


if __name__ == "__main__":
    main()
