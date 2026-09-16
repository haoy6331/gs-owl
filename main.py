import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
import argparse
import os
import time
import numpy as np
import json
import torch
import tempfile
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import (prune_wanda, prune_wanda_owl, prune_wanda_owl_v1, prune_wanda_owl_v2, prune_wanda_owl_v3, prune_wanda_owl_v4, prune_wanda_owl_v5, prune_wanda_owl_v6, prune_wanda_owl_v7, prune_wanda_owl_v8, prune_wanda_owl_v9, prune_metric_uniform, prune_metric_owl, prune_metric_owl_v9, prune_single_layer_wsqrtg, prune_magnitude, prune_sparsegpt, prune_ablate,
                       check_sparsity, find_layers, prune_TSP, prune_pruner_zero,
                       prune_ria, prune_ria_owl)  # 添加RIA导入
from lib.eval import eval_ppl, eval_zero_shot, eval_gsm8k  # 添加eval_gsm8k导入
from lib.plant_mcq import evaluate_model_instance, file_sha256, load_questions

from lib.gptree import GPTree


V9_METRIC_METHODS = {
    "owl-v9-wsqrtg": "wsqrtg",
    "owl-v9-tanhwg": "tanhwg",
    "owl-v9-w3gx": "w3gx",
    "owl-v9-w2mmsg": "w2mmsg",
    "owl-v9-w2xg": "w2xg",
}

UNIFORM_METRIC_METHODS = {
    "wsqrtg": "wsqrtg",
    "wgradpow": "wgradpow",
}

ORIGINAL_OWL_METRIC_METHODS = {
    "owl-wsqrtg": "wsqrtg",
}

PRUNING_METRIC_METHODS = {
    **UNIFORM_METRIC_METHODS,
    **ORIGINAL_OWL_METRIC_METHODS,
    **V9_METRIC_METHODS,
}

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    device_map = os.environ.get("MODEL_DEVICE_MAP", "auto")
    print(f"model device map strategy: {device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map=device_map
    )

    model.seqlen = min(model.config.max_position_embeddings, 8192)
    return model


def resolve_model_device(model, preferred_keys=("lm_head", "model.norm", "model.embed_tokens")):
    device_map = getattr(model, "hf_device_map", {}) or {}

    def as_cuda_device(value):
        if isinstance(value, int):
            return torch.device(f"cuda:{value}")
        if isinstance(value, torch.device):
            return value if value.type == "cuda" else None
        text = str(value)
        if text.isdigit():
            return torch.device(f"cuda:{text}")
        if text.startswith("cuda"):
            return torch.device(text)
        return None

    for key in preferred_keys:
        if key in device_map:
            resolved = as_cuda_device(device_map[key])
            if resolved is not None:
                print(f"Resolved model device from {key}: {resolved}")
                return resolved

    for key, value in device_map.items():
        resolved = as_cuda_device(value)
        if resolved is not None:
            print(f"Resolved model device from fallback map entry {key!r}: {resolved}")
            return resolved

    print("No CUDA entry found in hf_device_map; falling back to cuda:0")
    return torch.device("cuda:0")


def print_device_debug(model):
    if not hasattr(model, "hf_device_map"):
        print("DEBUG_DEVICE_MAP: model has no hf_device_map")
        return

    device_map = model.hf_device_map
    print("DEBUG_DEVICE_MAP: begin")
    device_counts = {}
    for module_name, module_device in device_map.items():
        device_counts[str(module_device)] = device_counts.get(str(module_device), 0) + 1
    print(f"DEBUG_DEVICE_MAP: device counts = {device_counts}")

    layer_items = [
        (name, dev)
        for name, dev in device_map.items()
        if name.startswith("model.layers.") or name.startswith("model.decoder.layers.")
    ]
    print(f"DEBUG_DEVICE_MAP: layer entries = {len(layer_items)}")
    for name, dev in layer_items[:8]:
        print(f"DEBUG_DEVICE_MAP: {name} -> {dev}")
    if len(layer_items) > 16:
        print("DEBUG_DEVICE_MAP: ...")
    for name, dev in layer_items[-8:]:
        print(f"DEBUG_DEVICE_MAP: {name} -> {dev}")

    for key in ("model.embed_tokens", "model.norm", "lm_head"):
        if key in device_map:
            print(f"DEBUG_DEVICE_MAP: {key} -> {device_map[key]}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(
                f"DEBUG_CUDA_MEM: gpu={i} allocated={allocated:.2f}GB "
                f"reserved={reserved:.2f}GB total={total:.2f}GB"
            )
    print("DEBUG_DEVICE_MAP: end")


def maybe_empty_cuda_cache(context):
    if not torch.cuda.is_available():
        return
    if os.environ.get("SKIP_CUDA_EMPTY_CACHE", "0") == "1":
        print(f"SKIP_CUDA_EMPTY_CACHE: skip torch.cuda.empty_cache() {context}")
        return
    torch.cuda.empty_cache()


def main():
    # ============= 设置临时目录到空间充足的位置 =============
    TEMP_DIR = "/home/yh114/tmp"
    os.makedirs(TEMP_DIR, exist_ok=True)
    tempfile.tempdir = TEMP_DIR
    os.environ['TMPDIR'] = TEMP_DIR
    print(f"临时目录已设置为: {TEMP_DIR}")
    # ====================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=[
        "magnitude", "wanda", "wanda-owl", "wanda-owl-v1", "wanda-owl-v2", "wanda-owl-v3", "wanda-owl-v4", "wanda-owl-v5", "wanda-owl-v6", "wanda-owl-v7", "wanda-owl-v8", "wanda-owl-v9",
        "wsqrtg", "wgradpow", "owl-wsqrtg",
        "owl-v9-wsqrtg", "owl-v9-tanhwg", "owl-v9-w3gx", "owl-v9-w2mmsg", "owl-v9-w2xg", "sparsegpt",
        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter",
        "ablate_wanda_iter", "search", "pruner-zero",
        "ablate_prunerzero_seq", "ablate_prunerzero_iter",
        "TSP",
        "layer-sensitivity-wsqrtg",
        # ========== 新增RIA方法 ==========
        "ria",  # 基础RIA剪枝
        "ria-owl"  # RIA + OWL层级稀疏度分配
    ])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true",
                        help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4"],
        help="Dataset used for perplexity evaluation. Defaults to WikiText2.",
    )
    parser.add_argument(
        "--skip_ppl_eval",
        action="store_true",
        help="Skip perplexity evaluation after pruning (useful when saving a checkpoint for another benchmark).",
    )
    parser.add_argument(
        "--eval_plant_mcq",
        action="store_true",
        help="Evaluate the in-memory pruned model on a fixed plant-science MCQ JSONL file.",
    )
    parser.add_argument(
        "--plant_mcq_questions",
        type=str,
        default="data/plant_mcq/mobiplant_expert_clean_v2.jsonl",
        help="Fixed plant-science MCQ JSONL file.",
    )
    parser.add_argument(
        "--plant_mcq_output",
        type=str,
        default=None,
        help="JSON path for the in-memory plant-MCQ result.",
    )
    parser.add_argument(
        "--plant_mcq_label",
        type=str,
        default="pruned",
        help="Label written into the plant-MCQ result.",
    )
    parser.add_argument(
        "--plant_mcq_max_questions",
        type=int,
        default=None,
        help="Optional prefix size for a quick plant-MCQ smoke test.",
    )
    parser.add_argument(
        "--plant_mcq_max_length",
        type=int,
        default=2048,
        help="Maximum prompt plus option length used by plant-MCQ scoring.",
    )
    # gradient_path
    parser.add_argument("--gradient_path", type=str, default=None, help="Path to save the gradient.")
    parser.add_argument(
        "--target_layer",
        type=int,
        default=-1,
        help="Transformer block index used by single-layer sensitivity experiments.",
    )
    parser.add_argument("--json_tree", type=str, default="data/best_tree.json", help="Path to load the json tree.")
    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument(
        "--zero_shot_tasks",
        type=str,
        default=None,
        help="Comma-separated zero-shot tasks. Defaults to the standard seven tasks.",
    )

    # ========== 新增GSM8K评估参数 ==========
    parser.add_argument("--eval_gsm8k", action="store_true",
                        help="Evaluate on GSM8K dataset with 8-shot Chain-of-Thought")
    parser.add_argument("--gsm8k_fewshot", type=int, default=8,
                        help="Number of few-shot examples for GSM8K (default: 8, as in Pruner-Zero paper)")

    # OWL相关参数
    parser.add_argument("--Hyper_m", type=float, default=5,
                        help="OWL outlier threshold multiplier")
    parser.add_argument("--Lamda", type=float, default=0.08,
                        help="OWL lambda for sparsity adjustment")
    parser.add_argument("--Owl_alpha", type=float, default=0.2,
                        help="V8: blend weight for auxiliary severity-based OWL density")
    parser.add_argument(
        "--Grad_beta",
        type=float,
        default=0.5,
        help="Gradient exponent beta for the |W| * |G|^beta ablation.",
    )
    parser.add_argument(
        "--v9_component",
        choices=["count", "severity", "anchored", "core", "full"],
        default="full",
        help="OWL-V9 density component used by the internal ablation.",
    )
    parser.add_argument("--v9_core_start", type=int, default=4)
    parser.add_argument("--v9_core_end", type=int, default=10)
    parser.add_argument("--v9_tail_start", type=int, default=16)
    parser.add_argument("--v9_tail_end", type=int, default=30)
    parser.add_argument(
        "--v9_projection",
        choices=["none", "uniform", "protected"],
        default="protected",
        help=(
            "Layer-density budget projection used after OSLA constraints. "
            "The default 'protected' preserves the current behavior."
        ),
    )
    parser.add_argument(
        "--v9_interval_mode",
        choices=["absolute", "normalized"],
        default="absolute",
        help=(
            "Interpret OSLA core/tail intervals as absolute layer indices or "
            "scale them from --v9_reference_layers to the current model depth."
        ),
    )
    parser.add_argument(
        "--v9_reference_layers",
        type=int,
        default=32,
        help="Reference depth used by normalized OSLA intervals.",
    )

    # ========== 新增RIA相关参数 ==========
    parser.add_argument("--a", type=float, default=0.5,
                        help="RIA: exponent for activation scaling (default: 0.5)")
    parser.add_argument("--per_outneuron", action="store_true",
                        help="RIA: pruning per output neuron (Wanda's tactic)")

    # 离线评估相关参数
    parser.add_argument('--offline', action="store_true", help="Use offline mode for evaluation")
    parser.add_argument('--local_datasets_path', type=str, default=None,
                        help='Local datasets path for offline evaluation')

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if os.environ.get("DEBUG_DEVICE_MAP", "0") == "1":
        print_device_debug(model)

    device = torch.device("cuda:0")
    model_name_lower = args.model.lower()
    if "13b" in model_name_lower or "30b" in model_name_lower or "65b" in model_name_lower or "70b" in model_name_lower or "33b" in model_name_lower:
        device = resolve_model_device(model)
        print(f"Large model detected, execution device: {device}")
        maybe_empty_cuda_cache("after large-model device resolution")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("Using single GPU")
    print("use device ", device)

    start_time = time.time()
    if args.sparsity_ratio != 0:
        print("pruning starts")
        print(f"Method: {args.prune_method}, Sparsity: {args.sparsity_ratio}")

        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda-owl":
            print(f"Wanda-OWL True OWL params: Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_wanda_owl(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda-owl-v1":
            print(f"Wanda-OWL-V1 params: Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_wanda_owl_v1(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda-owl-v2":
            print(f"Wanda-OWL-V2 params: Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_wanda_owl_v2(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda-owl-v3":
            print(f"Wanda-OWL-V3 params: Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_wanda_owl_v3(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda-owl-v4":
            print(f"Wanda-OWL-V4 params: Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_wanda_owl_v4(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda-owl-v5":
            print(f"Wanda-OWL-V5 params: Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_wanda_owl_v5(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda-owl-v6":
            print(f"Wanda-OWL-V6 params: Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_wanda_owl_v6(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda-owl-v7":
            print(f"Wanda-OWL-V7 params: Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_wanda_owl_v7(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda-owl-v8":
            print(f"Wanda-OWL-V8 params: Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_wanda_owl_v8(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda-owl-v9":
            print(f"Wanda-OWL-V9 params: Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_wanda_owl_v9(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method in UNIFORM_METRIC_METHODS:
            metric_name = UNIFORM_METRIC_METHODS[args.prune_method]
            print(
                f"Uniform metric ablation: metric={metric_name}, "
                f"sparsity={args.sparsity_ratio}, Grad_beta={args.Grad_beta}"
            )
            prune_metric_uniform(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                metric_name=metric_name)
        elif args.prune_method in ORIGINAL_OWL_METRIC_METHODS:
            metric_name = ORIGINAL_OWL_METRIC_METHODS[args.prune_method]
            print(f"Original OWL metric ablation: metric={metric_name}, "
                  f"Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_metric_owl(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                metric_name=metric_name)
        elif args.prune_method in V9_METRIC_METHODS:
            metric_name = V9_METRIC_METHODS[args.prune_method]
            print(f"OWL-V9 metric experiment: metric={metric_name}, Hyper_m={args.Hyper_m}, "
                  f"Lamda={args.Lamda}, Owl_alpha={args.Owl_alpha}, "
                  f"component={args.v9_component}")
            prune_metric_owl_v9(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                metric_name=metric_name)
        elif args.prune_method == "layer-sensitivity-wsqrtg":
            prune_single_layer_wsqrtg(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
            )
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "TSP":
            engine = GPTree.load_tree(args.json_tree)
            prune_TSP(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, engine=engine)
        elif "pruner-zero" in args.prune_method:
            engine = GPTree.load_tree(args.json_tree)
            prune_pruner_zero(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, engine=engine)
        # ========== 新增RIA方法调用 ==========
        elif args.prune_method == "ria":
            print(f"RIA参数: a={args.a}, per_outneuron={args.per_outneuron}")
            prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "ria-owl":
            print(f"RIA-OWL参数: a={args.a}, Hyper_m={args.Hyper_m}, Lamda={args.Lamda}")
            prune_ria_owl(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    end_time = time.time()
    print("pruning time: ", end_time - start_time)
    if os.environ.get("DEBUG_CUDA_SYNC", "0") == "1" and torch.cuda.is_available():
        print("DEBUG_CUDA_SYNC: synchronizing after pruning")
        torch.cuda.synchronize()
        print("DEBUG_CUDA_SYNC: synchronize after pruning complete")

    ################################################################
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)
    ################################################################
    ppl_test = None
    if args.skip_ppl_eval:
        print("Skipping perplexity evaluation (--skip_ppl_eval).")
    else:
        ppl_test = eval_ppl(args, model, tokenizer, device)
        print(f"{args.eval_dataset} perplexity {ppl_test}")

    if args.eval_plant_mcq:
        print("Starting in-memory plant-science MCQ evaluation...")
        print(f"Question file: {args.plant_mcq_questions}")
        questions = load_questions(
            args.plant_mcq_questions,
            limit=args.plant_mcq_max_questions,
        )
        plant_result = evaluate_model_instance(
            args.plant_mcq_label,
            model,
            tokenizer,
            questions,
            max_length=args.plant_mcq_max_length,
            model_path=args.model,
        )
        plant_result.update({
            "metric": "multiple-choice accuracy",
            "scoring": "mean conditional log-likelihood of option tokens",
            "questions": args.plant_mcq_questions,
            "question_set_sha256": file_sha256(args.plant_mcq_questions),
            "sparsity_ratio": args.sparsity_ratio,
            "actual_sparsity": sparsity_ratio,
            "Hyper_m": args.Hyper_m,
            "Lamda": args.Lamda,
            "Owl_alpha": args.Owl_alpha,
            "Grad_beta": args.Grad_beta,
            "calibration_samples": args.nsamples,
            "calibration_seed": args.seed,
            "plant_mcq_max_length": args.plant_mcq_max_length,
            "gradient_path": args.gradient_path,
            "cuda_available": torch.cuda.is_available(),
            "visible_gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "model_device_map": os.environ.get("MODEL_DEVICE_MAP", "unset"),
            "pruning_method": args.prune_method,
            "ppl_evaluation": "skipped" if ppl_test is None else args.eval_dataset,
        })
        plant_output = args.plant_mcq_output
        if plant_output is None:
            plant_output = os.path.join(
                args.save or ".",
                "plant_mcq_result.json",
            )
        os.makedirs(os.path.dirname(os.path.abspath(plant_output)), exist_ok=True)
        with open(plant_output, "w", encoding="utf-8") as handle:
            json.dump(plant_result, handle, indent=2, ensure_ascii=False)
        print(
            f"Plant MCQ accuracy: {plant_result['correct']}/{plant_result['n_questions']} "
            f"({plant_result['accuracy']:.4%})"
        )
        print(f"Plant MCQ result saved to: {plant_output}")

    # 保存困惑度结果
    if args.save:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
        with open(save_filepath, "a+") as f:
            result_data = {
                'method': args.prune_method,
                'actual_sparsity': f"{sparsity_ratio:.4f}",
                'eval_dataset': args.eval_dataset,
                'num_samples': str(args.nsamples),
                'sparsity_ratio': f"{args.sparsity_ratio:.4f}",
                'sparsity_type': args.sparsity_type,
                'seed': str(args.seed)
            }
            if ppl_test is None:
                result_data['ppl_eval'] = "skipped"
            elif args.eval_dataset == "wikitext2":
                result_data['ppl_test'] = f"{ppl_test:.4f}"
            else:
                result_data[f'ppl_{args.eval_dataset}'] = f"{ppl_test:.4f}"

            # 添加OWL特定参数
            if hasattr(args, 'Hyper_m'):
                result_data['Hyper_m'] = str(args.Hyper_m)
            if hasattr(args, 'Lamda'):
                result_data['Lamda'] = str(args.Lamda)
            if hasattr(args, 'Owl_alpha'):
                result_data['Owl_alpha'] = str(args.Owl_alpha)
            if args.prune_method == "wgradpow":
                result_data['Grad_beta'] = str(args.Grad_beta)
            if args.prune_method in V9_METRIC_METHODS or args.prune_method == "wanda-owl-v9":
                result_data['v9_component'] = str(args.v9_component)
                result_data['v9_core_start'] = str(args.v9_core_start)
                result_data['v9_core_end'] = str(args.v9_core_end)
                result_data['v9_tail_start'] = str(args.v9_tail_start)
                result_data['v9_tail_end'] = str(args.v9_tail_end)
                result_data['v9_projection'] = str(args.v9_projection)
                result_data['v9_interval_mode'] = str(args.v9_interval_mode)
                result_data['v9_reference_layers'] = str(args.v9_reference_layers)
            if args.prune_method in PRUNING_METRIC_METHODS:
                result_data['pruning_metric'] = PRUNING_METRIC_METHODS[args.prune_method]
            if args.prune_method == "layer-sensitivity-wsqrtg":
                result_data["target_layer"] = str(args.target_layer)
                result_data["pruning_metric"] = "wsqrtg"

            # ========== 新增RIA特定参数 ==========
            if args.prune_method in ["ria", "ria-owl"]:
                result_data['a'] = str(args.a)
                result_data['per_outneuron'] = str(args.per_outneuron)

            print("\t".join(result_data.keys()), file=f, flush=True)
            print("\t".join(result_data.values()), file=f, flush=True)

    # ==================== 零样本评估或GSM8K评估 ====================
    if args.eval_zero_shot or args.eval_gsm8k:
        temp_model_path = None
        try:
            if args.save_model:
                model_path_for_eval = args.save_model
                print(f"Saving pruned model to specified path: {model_path_for_eval}")
                if not os.path.exists(model_path_for_eval):
                    os.makedirs(model_path_for_eval)
            else:
                temp_model_path = tempfile.mkdtemp(prefix="pruned_model_")
                model_path_for_eval = temp_model_path
                print(f"Saving pruned model to temporary path: {model_path_for_eval}")

            model.save_pretrained(model_path_for_eval)
            tokenizer.save_pretrained(model_path_for_eval)
            print("Pruned model saved successfully!")

            del model
            import gc
            gc.collect()
            maybe_empty_cuda_cache("before zero-shot/GSM8K evaluation")

            # ========== 零样本评估 ==========
            if args.eval_zero_shot:
                print("Starting zero-shot evaluation on pruned model...")

                default_task_list = [
                    "boolq",
                    "rte",
                    "hellaswag",
                    "arc_challenge",
                    "arc_easy",
                    "winogrande",
                    "openbookqa",
                ]
                if args.zero_shot_tasks:
                    requested_tasks = [
                        task.strip()
                        for task in args.zero_shot_tasks.split(",")
                        if task.strip()
                    ]
                    invalid_tasks = sorted(set(requested_tasks) - set(default_task_list))
                    if invalid_tasks:
                        raise ValueError(
                            "Unsupported zero-shot tasks: "
                            + ", ".join(invalid_tasks)
                        )
                    task_list = list(dict.fromkeys(requested_tasks))
                    if not task_list:
                        raise ValueError("--zero_shot_tasks must include at least one task")
                else:
                    task_list = default_task_list
                print(f"Zero-shot tasks: {','.join(task_list)}")
                num_shot = 0

                results = eval_zero_shot(
                    model_path_for_eval,
                    task_list,
                    num_shot,
                    use_accelerate=True,
                    offline=args.offline,
                    local_datasets_path=args.local_datasets_path,
                    tokenizer_path=args.model
                )
                results["evaluated_tasks"] = task_list

                print("Zero-shot evaluation results:")
                print(results)

                if args.save and results.get("status") == "success":
                    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
                    with open(save_filepath, "a") as f:
                        scores = results.get("scores", {})
                        score_str = "\t".join([
                            f"{task}:{scores.get(task, 'N/A'):.4f}" if isinstance(scores.get(task), (int, float))
                            else f"{task}:{scores.get(task, 'N/A')}"
                            for task in task_list
                        ])
                        print(f"zero_shot_results\t{score_str}", file=f, flush=True)

                    json_filepath = os.path.join(args.save, f"log_lm_eval_{args.prune_method}.json")
                    results_json = json.dumps(results, indent=4)
                    with open(json_filepath, "w") as file:
                        file.write(results_json)
                    print(f"JSON results saved to {json_filepath}")

            # ========== GSM8K评估 ==========
            if args.eval_gsm8k:
                print("\n" + "=" * 60)
                print("Starting GSM8K evaluation on pruned model...")
                print(f"Using {args.gsm8k_fewshot}-shot Chain-of-Thought")
                print("=" * 60)

                gsm8k_results = eval_gsm8k(
                    model_path_for_eval,
                    num_fewshot=args.gsm8k_fewshot,
                    offline=args.offline,
                    local_datasets_path=args.local_datasets_path
                )

                print("\nGSM8K evaluation results:")
                print(gsm8k_results)

                # 保存GSM8K结果
                if args.save and gsm8k_results.get("status") == "success":
                    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
                    with open(save_filepath, "a") as f:
                        gsm8k_acc = gsm8k_results.get("gsm8k_accuracy", "N/A")
                        if isinstance(gsm8k_acc, (int, float)):
                            print(f"gsm8k_accuracy\t{gsm8k_acc:.4f}", file=f, flush=True)
                        else:
                            print(f"gsm8k_accuracy\t{gsm8k_acc}", file=f, flush=True)

                    # 保存完整JSON结果
                    json_filepath = os.path.join(args.save, f"log_gsm8k_{args.prune_method}.json")
                    results_json = json.dumps(gsm8k_results, indent=4)
                    with open(json_filepath, "w") as file:
                        file.write(results_json)
                    print(f"GSM8K JSON results saved to {json_filepath}")

        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if temp_model_path and os.path.exists(temp_model_path):
                try:
                    shutil.rmtree(temp_model_path)
                    print(f"Cleaned up temporary directory: {temp_model_path}")
                except Exception as e:
                    print(f"Warning: Failed to clean up temporary directory {temp_model_path}: {e}")

    else:
        if args.save_model:
            print(f"Saving pruned model to: {args.save_model}")
            if not os.path.exists(args.save_model):
                os.makedirs(args.save_model)
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)

        import gc
        del model
        gc.collect()
        maybe_empty_cuda_cache("after evaluation")


if __name__ == '__main__':
    main()
