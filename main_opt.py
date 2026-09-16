import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
import argparse
import os
import numpy as np
import torch
import tempfile
import shutil
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune_opt import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers, \
    prune_pruner_zero, prune_TSP_opt, prune_metric_owl_v9_opt
from lib.eval import eval_ppl, eval_zero_shot

from lib.gptree import GPTree

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


OPT_V9_METRIC_METHODS = {
    "owl-v9-wsqrtg": "wsqrtg",
}


def get_llm(model_name, cache_dir="llm_weights"):
    device_map = os.environ.get("MODEL_DEVICE_MAP", "auto").strip() or "auto"
    print(f"model device_map strategy: {device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    model.seqlen = model.config.max_position_embeddings
    return model


def normalize_cuda_device(device):
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def get_eval_device(model):
    device_map = getattr(model, "hf_device_map", {})
    if "lm_head" in device_map:
        return normalize_cuda_device(device_map["lm_head"])
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.device
    return next(model.parameters()).device


def allocated_cuda_devices(model):
    devices = set()
    for value in getattr(model, "hf_device_map", {}).values():
        if isinstance(value, int):
            devices.add(value)
        elif isinstance(value, str) and value.startswith("cuda:"):
            devices.add(int(value.split(":", 1)[1]))
    return sorted(devices)


def main():
    # ============= 设置临时目录到空间充足的位置 =============
    TEMP_DIR = "/home/yh114/tmp"
    os.makedirs(TEMP_DIR, exist_ok=True)
    tempfile.tempdir = TEMP_DIR
    os.environ['TMPDIR'] = TEMP_DIR
    print(f"临时目录已设置为: {TEMP_DIR}")
    # ====================================================

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='OPT model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt",
                                                             "owl-v9-wsqrtg",
                                                             "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter",
                                                             "ablate_wanda_iter",
                                                             "search", "pruner-zero", "TSP-opt"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true",
                        help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument("--gradient_path", type=str, default=None, help="Path to save the gradient.")
    parser.add_argument("--json_tree", type=str, default="data/best_tree.json", help="Path to load the json tree.")

    # 添加 OWL相关参数
    parser.add_argument("--Hyper_m", type=float, default=5,
                        help="OWL outlier threshold multiplier")
    parser.add_argument("--Lamda", type=float, default=0.08,
                        help="OWL lambda for sparsity adjustment")
    parser.add_argument("--Owl_alpha", type=float, default=0.2,
                        help="OWL-V9 anchor blend weight")

    # 零样本评估参数
    parser.add_argument("--eval_zero_shot", action="store_true")

    # 添加离线评估相关参数
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

    device = get_eval_device(model)
    allocated_devices = allocated_cuda_devices(model)
    if len(allocated_devices) > 1:
        print(f"Multi-GPU OPT model detected, allocated CUDA devices: {allocated_devices}")
    else:
        print("Using single-GPU OPT model placement")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    print("use device ", device)

    start_time = time.time()
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method in OPT_V9_METRIC_METHODS:
            prune_metric_owl_v9_opt(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
                metric_name=OPT_V9_METRIC_METHODS[args.prune_method])
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "TSP-opt":
            engine = GPTree.load_tree(args.json_tree)
            prune_TSP_opt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, engine=engine)
        elif "pruner-zero" in args.prune_method:
            engine = GPTree.load_tree(args.json_tree)
            prune_pruner_zero(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, engine=engine)

    end_time = time.time()
    print("pruning time: ", end_time - start_time)

    ################################################################
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    # 保存困惑度结果
    if args.save:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
        with open(save_filepath, "a+") as f:
            # 创建参数字典，便于扩展
            result_data = {
                'method': args.prune_method,
                'actual_sparsity': f"{sparsity_ratio:.4f}",
                'ppl_test': f"{ppl_test:.4f}",
                'num_samples': str(args.nsamples),
                'sparsity_ratio': f"{args.sparsity_ratio:.4f}",
                'sparsity_type': args.sparsity_type,
                'seed': str(args.seed)
            }

            # 添加 OWL特定参数
            if hasattr(args, 'Hyper_m'):
                result_data['Hyper_m'] = str(args.Hyper_m)
            if hasattr(args, 'Lamda'):
                result_data['Lamda'] = str(args.Lamda)
            if hasattr(args, 'Owl_alpha'):
                result_data['Owl_alpha'] = str(args.Owl_alpha)

            # 写入头部和数据
            print("\t".join(result_data.keys()), file=f, flush=True)
            print("\t".join(result_data.values()), file=f, flush=True)

    # 零样本评估 - 使用与main.py相同的方法
    if args.eval_zero_shot:
        temp_model_path = None
        try:
            # 如果用户指定了保存路径，直接使用；否则创建临时路径
            if args.save_model:
                model_path_for_eval = args.save_model
                print(f"Saving pruned model to specified path: {model_path_for_eval}")
                if not os.path.exists(model_path_for_eval):
                    os.makedirs(model_path_for_eval)
            else:
                # 创建临时目录保存剪枝模型
                temp_model_path = tempfile.mkdtemp(prefix="pruned_model_opt_")
                model_path_for_eval = temp_model_path
                print(f"Saving pruned model to temporary path: {model_path_for_eval}")

            # 保存剪枝后的模型和tokenizer
            model.save_pretrained(model_path_for_eval)
            tokenizer.save_pretrained(model_path_for_eval)
            print("Pruned model saved successfully!")

            # 清理GPU内存
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # 使用保存的剪枝模型进行零样本评估
            print("Starting zero-shot evaluation on pruned model...")

            # 任务列表
            task_list = ["boolq", "rte", "hellaswag", "arc_challenge", "arc_easy", "winogrande", "openbookqa"]
            num_shot = 0

            # 调用新的eval_zero_shot函数（基于lm_eval命令行）
            results = eval_zero_shot(
                model_path_for_eval,
                task_list,
                num_shot,
                use_accelerate=True,
                offline=args.offline,
                local_datasets_path=args.local_datasets_path
            )

            print("Zero-shot evaluation results:")
            print(results)

            # 保存零样本结果
            if args.save and results.get("status") == "success":
                # 保存到原始日志文件
                save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
                with open(save_filepath, "a") as f:
                    # 添加零样本结果的表头（如果是第一次写入）
                    scores = results.get("scores", {})
                    score_str = "\t".join([
                        f"{task}:{scores.get(task, 'N/A'):.4f}" if isinstance(scores.get(task), (int, float))
                        else f"{task}:{scores.get(task, 'N/A')}"
                        for task in task_list
                    ])
                    print(f"zero_shot_results\t{score_str}", file=f, flush=True)

                # 同时保存完整的JSON结果
                json_filepath = os.path.join(args.save, f"log_lm_eval_{args.prune_method}.json")
                results_json = json.dumps(results, indent=4)
                with open(json_filepath, "w") as file:
                    file.write(results_json)
                print(f"JSON results saved to {json_filepath}")

        except Exception as e:
            print(f"Error during zero-shot evaluation: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 清理临时目录
            if temp_model_path and os.path.exists(temp_model_path):
                try:
                    shutil.rmtree(temp_model_path)
                    print(f"Cleaned up temporary directory: {temp_model_path}")
                except Exception as e:
                    print(f"Warning: Failed to clean up temporary directory {temp_model_path}: {e}")

    else:
        # 如果用户指定了保存模型但不进行零样本评估
        if args.save_model:
            print(f"Saving pruned model to: {args.save_model}")
            if not os.path.exists(args.save_model):
                os.makedirs(args.save_model)
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)

        # 清理GPU内存
        import gc
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
