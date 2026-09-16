#!/usr/bin/env python3
"""Prepare and audit a reproducible plant-science MCQ evaluation set.

The source file is never modified. Invalid or placeholder questions are
rejected, harmless whitespace is normalized, and every decision is recorded
in a sidecar audit report.
"""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path


PLACEHOLDER_TEXT = {
    "?",
    "n/a",
    "na",
    "none",
    "null",
    "placeholder",
    "tbd",
    "test",
    "todo",
    "x",
    "xx",
    "xxx",
}
MAX_OPTIONS = 26
LONG_PROMPT_CHAR_THRESHOLD = 6000
OPTION_LENGTH_RATIO_THRESHOLD = 5.0


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare plant-science multiple-choice questions.")
    parser.add_argument("--output", default="data/plant_mcq/mobiplant_expert_clean_v2.jsonl")
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Number of questions to sample (100-1000); 0 keeps every eligible question.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset", default="manufernandezbur/MoBiPlant")
    parser.add_argument("--split", default="train")
    parser.add_argument("--all-mobiplant", action="store_true", help="Allow synthetic MoBiPlant questions too.")
    parser.add_argument("--local-json", default=None, help="Use a local JSON/JSONL file instead of Hugging Face.")
    parser.add_argument(
        "--preserve-ids",
        action="store_true",
        help="Keep source IDs. Use this when cleaning an existing evaluation JSONL.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split())


def canonical_text(value):
    return normalize_text(value).casefold()


def is_placeholder(value):
    return canonical_text(value) in PLACEHOLDER_TEXT


def word_count(value):
    return max(1, len(re.findall(r"\b\w+\b", value, flags=re.UNICODE)))


def read_rows(args):
    if args.local_json:
        path = Path(args.local_json)
        if not path.is_file():
            raise FileNotFoundError(f"Local question file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            rows = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
                rows = payload["data"]
            else:
                raise ValueError(f"Expected a JSON list or a mapping with a 'data' list: {path}")
        return rows, str(path), sha256_file(path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets or pass --local-json.") from exc
    return list(load_dataset(args.dataset, split=args.split)), args.dataset, None


def rejected_entry(source_index, row, reason, detail=None):
    entry = {
        "source_index": source_index,
        "source_id": row.get("id") if isinstance(row, dict) else None,
        "reason": reason,
    }
    if isinstance(row, dict):
        entry["question_preview"] = normalize_text(row.get("question"))[:160]
    if detail:
        entry["detail"] = detail
    return entry


def normalize_rows(raw_rows, args):
    accepted = []
    rejected = []
    seen_samples = {}
    seen_ids = {}
    whitespace_fields_normalized = 0

    for source_index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            rejected.append(rejected_entry(source_index, raw_row, "row_not_object"))
            continue
        row = raw_row

        if not args.all_mobiplant and row.get("is_expert") is not True:
            rejected.append(rejected_entry(source_index, row, "non_expert"))
            continue

        original_question = "" if row.get("question") is None else str(row.get("question"))
        question = normalize_text(original_question)
        if question != original_question.strip():
            whitespace_fields_normalized += 1
        if not question:
            rejected.append(rejected_entry(source_index, row, "empty_question"))
            continue
        if is_placeholder(question):
            rejected.append(rejected_entry(source_index, row, "placeholder_question"))
            continue

        raw_options = row.get("options")
        if not isinstance(raw_options, (list, tuple)) or not 2 <= len(raw_options) <= MAX_OPTIONS:
            rejected.append(
                rejected_entry(
                    source_index,
                    row,
                    "invalid_option_count",
                    f"expected 2-{MAX_OPTIONS} options",
                )
            )
            continue

        options = []
        for option in raw_options:
            original_option = "" if option is None else str(option)
            normalized_option = normalize_text(original_option)
            if normalized_option != original_option.strip():
                whitespace_fields_normalized += 1
            options.append(normalized_option)
        if any(not option for option in options):
            rejected.append(rejected_entry(source_index, row, "empty_option"))
            continue

        canonical_options = [canonical_text(option) for option in options]
        if len(set(canonical_options)) != len(canonical_options):
            rejected.append(rejected_entry(source_index, row, "duplicate_options"))
            continue

        answer = row.get("answer")
        if isinstance(answer, bool):
            rejected.append(rejected_entry(source_index, row, "invalid_answer"))
            continue
        try:
            answer = int(answer)
        except (TypeError, ValueError):
            rejected.append(rejected_entry(source_index, row, "invalid_answer"))
            continue
        if answer < 0 or answer >= len(options):
            rejected.append(rejected_entry(source_index, row, "answer_out_of_range"))
            continue

        source_id = normalize_text(row.get("id"))
        if args.preserve_ids and not source_id:
            rejected.append(rejected_entry(source_index, row, "missing_id"))
            continue
        if source_id:
            if source_id in seen_ids:
                rejected.append(
                    rejected_entry(
                        source_index,
                        row,
                        "duplicate_id",
                        f"first seen at source index {seen_ids[source_id]}",
                    )
                )
                continue
            seen_ids[source_id] = source_index

        sample_key = (canonical_text(question), tuple(canonical_options))
        if sample_key in seen_samples:
            rejected.append(
                rejected_entry(
                    source_index,
                    row,
                    "duplicate_sample",
                    f"first seen at source index {seen_samples[sample_key]}",
                )
            )
            continue
        seen_samples[sample_key] = source_index

        accepted.append(
            {
                "id": source_id,
                "question": question,
                "options": options,
                "answer": answer,
                "area": normalize_text(row.get("normalized_area", row.get("area", "unknown"))) or "unknown",
                "plant_species": normalize_text(
                    row.get("normalized_plant_species", row.get("plant_species", ""))
                ),
                "source": normalize_text(row.get("source", "MoBiPlant")) or "MoBiPlant",
                "doi": normalize_text(row.get("doi", "")),
                "is_expert": bool(row.get("is_expert", True)),
                "source_dataset": normalize_text(row.get("source_dataset", "")),
                "_source_index": source_index,
            }
        )

    return accepted, rejected, whitespace_fields_normalized


def build_warnings(rows):
    warnings = []
    for row in rows:
        option_lengths = [word_count(option) for option in row["options"]]
        length_ratio = max(option_lengths) / min(option_lengths)
        if length_ratio >= OPTION_LENGTH_RATIO_THRESHOLD:
            warnings.append(
                {
                    "id": row["id"],
                    "type": "option_length_imbalance",
                    "word_counts": option_lengths,
                    "max_min_ratio": round(length_ratio, 3),
                }
            )

        prompt_lines = [
            "Answer the following plant science multiple-choice question.",
            "Select exactly one option.",
            "",
            f"Question: {row['question']}",
        ]
        for option_index, option in enumerate(row["options"]):
            prompt_lines.append(f"{chr(ord('A') + option_index)}. {option}")
        prompt_lines.append("Answer:")
        prompt = "\n".join(prompt_lines)
        max_scored_chars = len(prompt) + 1 + max(len(option) for option in row["options"])
        if max_scored_chars >= LONG_PROMPT_CHAR_THRESHOLD:
            warnings.append(
                {
                    "id": row["id"],
                    "type": "long_prompt",
                    "max_scored_text_chars": max_scored_chars,
                }
            )

        if not row.get("doi"):
            warnings.append({"id": row["id"], "type": "missing_doi"})
    return warnings


def length_bias_diagnostics(rows):
    expected_correct = 0.0
    unique_longest_correct = 0
    unique_longest_questions = 0
    tied_longest_questions = 0
    for row in rows:
        lengths = [word_count(option) for option in row["options"]]
        longest = max(lengths)
        candidates = [index for index, length in enumerate(lengths) if length == longest]
        if len(candidates) == 1:
            unique_longest_questions += 1
            unique_longest_correct += int(candidates[0] == row["answer"])
        else:
            tied_longest_questions += 1
        if row["answer"] in candidates:
            expected_correct += 1.0 / len(candidates)
    return {
        "heuristic": "choose the option with the largest whitespace-delimited word count",
        "expected_accuracy_with_uniform_tie_break": expected_correct / len(rows),
        "unique_longest_questions": unique_longest_questions,
        "unique_longest_accuracy": (
            unique_longest_correct / unique_longest_questions if unique_longest_questions else None
        ),
        "tied_longest_questions": tied_longest_questions,
    }


def main():
    args = parse_args()
    if args.n != 0 and not 100 <= args.n <= 1000:
        raise ValueError("--n must be 0 or an integer from 100 to 1000")

    output = Path(args.output)
    if args.local_json and Path(args.local_json).resolve() == output.resolve():
        raise ValueError("Refusing to overwrite the source question file; choose a new --output path.")

    raw_rows, source_label, input_sha256 = read_rows(args)
    rows, rejected, whitespace_fields_normalized = normalize_rows(raw_rows, args)
    if args.n and len(rows) < args.n:
        raise RuntimeError(f"Only {len(rows)} valid questions available, but --n={args.n} was requested.")

    rng = random.Random(args.seed)
    selected = list(rows) if args.n == 0 else rng.sample(rows, args.n)
    if args.preserve_ids:
        selected.sort(key=lambda row: row["_source_index"])
    else:
        selected.sort(key=lambda row: (str(row.get("area", "")), row["question"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for output_index, selected_row in enumerate(selected):
        row = {key: value for key, value in selected_row.items() if not key.startswith("_")}
        if not args.preserve_ids:
            row["id"] = f"plant-mcq-{output_index:04d}"
        row["source_dataset"] = row.get("source_dataset") or source_label
        row["sampling_seed"] = args.seed
        output_rows.append(row)

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_sha256 = sha256_file(output)
    warnings = build_warnings(output_rows)
    rejection_counts = Counter(item["reason"] for item in rejected)
    warning_counts = Counter(item["type"] for item in warnings)

    manifest = {
        "cleaning_version": 2,
        "dataset": source_label,
        "split": args.split,
        "input_rows": len(raw_rows),
        "valid_rows_before_sampling": len(rows),
        "n_questions": len(output_rows),
        "rejected_rows": len(rejected),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "seed": args.seed,
        "expert_only": not args.all_mobiplant,
        "preserve_ids": args.preserve_ids,
        "output": str(output),
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "answer_index_convention": "0-based",
        "evaluation": "conditional log-likelihood over answer options",
    }
    audit = {
        "input": source_label,
        "output": str(output),
        "input_rows": len(raw_rows),
        "output_rows": len(output_rows),
        "whitespace_fields_normalized": whitespace_fields_normalized,
        "rejected": rejected,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "warnings": warnings,
        "warning_counts": dict(sorted(warning_counts.items())),
        "length_bias_diagnostics": length_bias_diagnostics(output_rows),
        "answer_position_counts": dict(sorted(Counter(row["answer"] for row in output_rows).items())),
        "area_counts": dict(sorted(Counter(row["area"] for row in output_rows).items())),
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
    }

    manifest_path = output.with_suffix(".manifest.json")
    audit_path = output.with_suffix(".audit.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input rows: {len(raw_rows)}")
    print(f"Rejected rows: {len(rejected)} {dict(sorted(rejection_counts.items()))}")
    print(f"Wrote {len(output_rows)} questions to {output}")
    print(f"Output SHA256: {output_sha256}")
    print(f"Manifest: {manifest_path}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
