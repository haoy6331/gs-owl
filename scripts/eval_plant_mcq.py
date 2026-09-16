#!/usr/bin/env python3
"""Evaluate one or more causal LMs on a plant-science MCQ JSONL file.

Each answer is selected by conditional log-likelihood, avoiding fragile
post-processing of free-form generated letters. The same question file must
be used for base and pruned checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.plant_mcq import evaluate_model_instance, file_sha256, load_questions


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PLLaMA or another causal LM on plant MCQs.")
    parser.add_argument("--model", action="append", required=True, help="Can be repeated: --model label=path")
    parser.add_argument("--questions", default="data/plant_mcq/mobiplant_expert_clean_v2.jsonl")
    parser.add_argument("--output", default="owl/plant_mcq_eval/results.json")
    parser.add_argument("--cache-dir", default="llm_weights")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_model_spec(spec):
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, path
    path = spec.rstrip("/\\")
    return Path(path).name, path


def evaluate_model(label, model_path, questions, args):
    print(f"Loading {label}: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map=args.device_map if torch.cuda.is_available() else None,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    evaluation = evaluate_model_instance(
        label,
        model,
        tokenizer,
        questions,
        max_length=args.max_length,
        model_path=model_path,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return evaluation


def main():
    args = parse_args()
    questions = load_questions(args.questions, args.max_questions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    evaluations = []
    for spec in args.model:
        label, path = parse_model_spec(spec)
        evaluations.append(evaluate_model(label, path, questions, args))

    payload = {
        "questions": str(args.questions),
        "question_set_sha256": file_sha256(args.questions),
        "device_map": args.device_map,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "n_questions": len(questions),
        "metric": "multiple-choice accuracy",
        "scoring": "mean conditional log-likelihood of option tokens",
        "evaluations": evaluations,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nPlant MCQ results")
    print("-" * 72)
    for item in evaluations:
        print(f"{item['model']}: accuracy={item['accuracy']:.4%} n={item['n_questions']}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
