"""In-memory plant-science multiple-choice evaluation utilities."""

import hashlib
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path

try:
    import torch
except ImportError:  # Dataset validation should also work on lightweight CPU-only machines.
    torch = None


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLACEHOLDER_TEXT = {
    "?", "n/a", "na", "none", "null", "placeholder", "tbd", "test", "todo", "x", "xx", "xxx"
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_text(value):
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split()).casefold()


def _validate_questions(rows, path):
    seen_ids = {}
    seen_samples = {}
    for index, row in enumerate(rows, start=1):
        location = f"{path}:{index}"
        if not isinstance(row, dict):
            raise ValueError(f"Question row must be a JSON object at {location}")

        question_id = row.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Question ID is missing at {location}")
        if question_id in seen_ids:
            raise ValueError(
                f"Duplicate question ID {question_id!r} at {location}; first seen on line {seen_ids[question_id]}"
            )
        seen_ids[question_id] = index

        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question text is empty for {question_id!r} at {location}")
        if _canonical_text(question) in PLACEHOLDER_TEXT:
            raise ValueError(f"Placeholder question detected for {question_id!r} at {location}")

        options = row.get("options")
        if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
            raise ValueError(f"Invalid option list for {question_id!r} at {location}")
        if any(not isinstance(option, str) or not option.strip() for option in options):
            raise ValueError(f"Empty option detected for {question_id!r} at {location}")
        canonical_options = tuple(_canonical_text(option) for option in options)
        if len(set(canonical_options)) != len(canonical_options):
            raise ValueError(f"Duplicate options detected for {question_id!r} at {location}")

        answer = row.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(options):
            raise ValueError(f"Invalid answer index for {question_id!r} at {location}")

        sample_key = (_canonical_text(question), canonical_options)
        if sample_key in seen_samples:
            raise ValueError(
                f"Duplicate question content for {question_id!r} at {location}; "
                f"first seen on line {seen_samples[sample_key]}"
            )
        seen_samples[sample_key] = index


def load_questions(path, limit=None):
    question_path = Path(path)
    rows = []
    for line_number, line in enumerate(question_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {question_path}: {exc}") from exc
    _validate_questions(rows, question_path)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"No questions found in {path}")
    return rows


def format_prompt(row):
    lines = [
        "Answer the following plant science multiple-choice question.",
        "Select exactly one option.",
        "",
        f"Question: {row['question']}",
    ]
    for index, option in enumerate(row["options"]):
        lines.append(f"{LETTERS[index]}. {option}")
    lines.append("Answer:")
    return "\n".join(lines)


def get_input_device(model):
    embeddings = model.get_input_embeddings()
    if embeddings is not None and hasattr(embeddings, "weight"):
        return embeddings.weight.device
    return next(model.parameters()).device


def select_prediction(scores):
    if not scores or any(not math.isfinite(score) for score in scores):
        raise ValueError(f"Option scores must be finite and non-empty: {scores}")
    best_score = max(scores)
    tied_options = [item for item, score in enumerate(scores) if score == best_score]
    return (None if len(tied_options) > 1 else tied_options[0]), tied_options


def option_score(model, tokenizer, prompt, option, max_length=2048):
    if torch is None:
        raise RuntimeError("PyTorch is required for model scoring but is not installed in this Python environment")
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids[0]
    option_ids = tokenizer(
        " " + option,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]
    if option_ids.numel() == 0:
        return -float("inf")

    original_prompt_tokens = prompt_ids.numel()
    if prompt_ids.numel() + option_ids.numel() > max_length:
        keep = max_length - option_ids.numel()
        if keep <= 0:
            raise RuntimeError("--plant_mcq_max_length is too small for an answer option")
        prompt_ids = prompt_ids[-keep:]

    input_ids = torch.cat([prompt_ids, option_ids], dim=0).unsqueeze(0)
    input_ids = input_ids.to(get_input_device(model))
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :]

    start = prompt_ids.numel() - 1
    predicted = logits[:, start:start + option_ids.numel(), :]
    target = input_ids[:, start + 1:].to(predicted.device)
    log_probs = torch.log_softmax(predicted.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.mean().item()), original_prompt_tokens - prompt_ids.numel()


def evaluate_model_instance(label, model, tokenizer, questions, max_length=2048, model_path=None):
    model.eval()
    predictions = []
    for index, row in enumerate(questions, start=1):
        prompt = format_prompt(row)
        scored_options = [
            option_score(model, tokenizer, prompt, option, max_length)
            for option in row["options"]
        ]
        scores = [item[0] for item in scored_options]
        prompt_tokens_truncated = [item[1] for item in scored_options]
        try:
            prediction, tied_options = select_prediction(scores)
        except ValueError as exc:
            raise RuntimeError(f"Invalid option scores for question {row.get('id', index)!r}") from exc
        tie = len(tied_options) > 1
        predictions.append({
            "id": row.get("id", index),
            "area": row.get("area", "unknown"),
            "gold": int(row["answer"]),
            "prediction": prediction,
            "correct": not tie and prediction == int(row["answer"]),
            "tie": tie,
            "tied_options": tied_options if tie else [],
            "prompt_tokens_truncated": prompt_tokens_truncated,
            "option_scores": scores,
        })
        if index % 25 == 0 or index == len(questions):
            accuracy = sum(item["correct"] for item in predictions) / len(predictions)
            print(f"[{label}] {index}/{len(questions)} accuracy={accuracy:.4f}", flush=True)

    by_area = defaultdict(list)
    for item in predictions:
        by_area[item["area"]].append(item["correct"])

    correct = sum(item["correct"] for item in predictions)
    ties = sum(item["tie"] for item in predictions)
    truncated_questions = sum(any(count > 0 for count in item["prompt_tokens_truncated"]) for item in predictions)
    max_truncated_tokens = max(
        (max(item["prompt_tokens_truncated"]) for item in predictions),
        default=0,
    )
    return {
        "model": label,
        "model_path": model_path,
        "n_questions": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "n_ties": ties,
        "tie_rate": ties / len(predictions),
        "tie_policy": "exact top-score ties are counted as incorrect",
        "n_truncated_questions": truncated_questions,
        "max_prompt_tokens_truncated": max_truncated_tokens,
        "area_accuracy": {
            area: sum(values) / len(values)
            for area, values in sorted(by_area.items())
        },
        "predictions": predictions,
    }
