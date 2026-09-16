# Import necessary modules
import time
import torch
import torch.nn as nn
import subprocess
import os
import sys
import re
import json

# Import get_loaders function from data module within the same directory
from .data import get_loaders

from collections import defaultdict
import fnmatch


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    dataset = getattr(args, "eval_dataset", "wikitext2")

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer
    )

    if testloader is None:
        raise RuntimeError(f"No evaluation split was returned for dataset: {dataset}")

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()


# ==================== GSM8K 评估函数 ====================

def eval_gsm8k(model_name, num_fewshot=8, offline=False, local_datasets_path=None):
    """
    评估模型在GSM8K数据集上的In-Context Learning准确率

    按照Pruner-Zero论文的设置:
    - 使用8-shot Chain-of-Thought
    - 使用LLaMA2-13B模型

    参数:
        model_name: 模型路径（剪枝后保存的模型路径）
        num_fewshot: few-shot数量，默认8（论文设置）
        offline: 是否使用离线模式
        local_datasets_path: 本地数据集路径

    返回:
        dict: 包含评估结果的字典
    """
    import subprocess
    import os
    import time
    import json

    print("=" * 60)
    print("GSM8K In-Context Learning 评估")
    print("=" * 60)
    print(f"模型路径: {model_name}")
    print(f"Few-shot数量: {num_fewshot}")
    print(f"离线模式: {offline}")

    # 检测GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")

    # 设置环境变量
    env = os.environ.copy()

    # 设置HF镜像（如果在中国大陆）
    if 'HF_ENDPOINT' not in env:
        env['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"设置HF镜像: {env['HF_ENDPOINT']}")

    if offline:
        print("设置离线模式环境变量...")
        env['HF_DATASETS_OFFLINE'] = '1'
        env['HF_HUB_OFFLINE'] = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'

        if local_datasets_path:
            env['HF_DATASETS_CACHE'] = os.path.abspath(local_datasets_path)
            env['HF_HOME'] = os.path.abspath(local_datasets_path)
            print(f"设置HF_DATASETS_CACHE为: {env['HF_DATASETS_CACHE']}")
    else:
        env['HF_DATASETS_OFFLINE'] = '0'
        env['HF_HUB_OFFLINE'] = '0'
        env['TRANSFORMERS_OFFLINE'] = '0'

    # 模型参数
    model_args = f"pretrained={model_name},trust_remote_code=True,device_map=auto"

    # 如果有多个GPU，添加并行参数
    if num_gpus > 1:
        model_args += ",parallelize=True"

    print(f"模型参数: {model_args}")

    # 构建lm_eval命令
    # GSM8K任务在lm_eval中的名称是 "gsm8k"
    # 使用 Chain-of-Thought 版本: "gsm8k_cot" 或直接 "gsm8k"
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", "gsm8k_cot",  # 使用CoT版本的GSM8K
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "1",
        "--device", "cuda",
        "--output_path", "./eval_results_gsm8k",
        "--log_samples"
    ]

    # 如果指定了本地数据集路径，设置环境变量
    if local_datasets_path:
        # 设置HF缓存目录为本地路径的父目录
        local_path = os.path.abspath(local_datasets_path)
        env['HF_DATASETS_CACHE'] = local_path
        env['HF_HOME'] = local_path
        # 如果路径直接指向gsm8k目录，使用其父目录
        if 'gsm8k' in local_path:
            parent_path = os.path.dirname(local_path)
            env['HF_DATASETS_CACHE'] = parent_path
            env['HF_HOME'] = parent_path
        print(f"本地数据集路径: {env['HF_DATASETS_CACHE']}")

    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 60)

    try:
        start_time = time.time()

        # 执行命令
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )

        output_lines = []
        print("开始GSM8K评估...")
        print("-" * 40)

        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                output_lines.append(line)
                # 显示进度和结果
                if any(keyword in line.lower() for keyword in ['gsm8k', 'acc', 'exact_match', 'flexible_extract']):
                    print(f"[GSM8K] {line}")
                elif any(keyword in line.lower() for keyword in ['loading', 'evaluating', 'running', 'finished']):
                    print(f"[INFO] {line}")

        process.wait()
        output = '\n'.join(output_lines)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print("-" * 40)
        print(f"GSM8K评估完成，耗时: {elapsed_time:.2f}秒")

        if process.returncode == 0:
            print("GSM8K评估成功完成!")

            # 解析结果
            results = parse_gsm8k_results(output)

            # 尝试从JSON文件读取更准确的结果
            json_results_dir = "./eval_results_gsm8k"
            if os.path.exists(json_results_dir):
                # 查找最新的results.json文件
                for root, dirs, files in os.walk(json_results_dir):
                    for file in files:
                        if file == "results.json":
                            json_path = os.path.join(root, file)
                            try:
                                with open(json_path, "r") as f:
                                    json_data = json.load(f)
                                    # 从JSON中提取GSM8K结果
                                    if "results" in json_data:
                                        for task_name, task_results in json_data["results"].items():
                                            if "gsm8k" in task_name.lower():
                                                # 尝试获取不同的指标
                                                if "exact_match,strict-match" in task_results:
                                                    score = task_results["exact_match,strict-match"]
                                                    results["scores"]["gsm8k_strict"] = score
                                                if "exact_match,flexible-extract" in task_results:
                                                    score = task_results["exact_match,flexible-extract"]
                                                    results["scores"]["gsm8k_flexible"] = score
                                                if "acc" in task_results:
                                                    score = task_results["acc"]
                                                    results["scores"]["gsm8k_acc"] = score
                                                # 保存主要分数
                                                if results["scores"]:
                                                    # 优先使用flexible-extract
                                                    if "gsm8k_flexible" in results["scores"]:
                                                        results["gsm8k_accuracy"] = results["scores"]["gsm8k_flexible"]
                                                    elif "gsm8k_strict" in results["scores"]:
                                                        results["gsm8k_accuracy"] = results["scores"]["gsm8k_strict"]
                                                    elif "gsm8k_acc" in results["scores"]:
                                                        results["gsm8k_accuracy"] = results["scores"]["gsm8k_acc"]
                            except Exception as e:
                                print(f"读取JSON结果时出错: {e}")

            # 显示结果摘要
            print("\n" + "=" * 40)
            print("GSM8K 评估结果摘要")
            print("=" * 40)
            if "gsm8k_accuracy" in results:
                acc = results["gsm8k_accuracy"]
                if acc <= 1.0:
                    print(f"GSM8K准确率: {acc:.4f} ({acc * 100:.2f}%)")
                else:
                    print(f"GSM8K准确率: {acc:.2f}%")
            if results.get("scores"):
                for metric, score in results["scores"].items():
                    if score <= 1.0:
                        print(f"  {metric}: {score:.4f}")
                    else:
                        print(f"  {metric}: {score:.2f}%")
            print("=" * 40)

            return results
        else:
            print(f"GSM8K评估失败，返回码: {process.returncode}")
            return {"status": "failed", "error": output}

    except FileNotFoundError:
        print("错误：找不到 lm_eval 命令")
        print("请确保已安装 lm-evaluation-harness:")
        print("  pip install lm-eval")
        return {"status": "error", "error": "lm_eval command not found"}
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def parse_gsm8k_results(output):
    """解析GSM8K评估输出结果"""
    results = {"status": "success", "scores": {}}

    lines = output.split('\n')

    for line in lines:
        line_lower = line.lower()

        # 查找GSM8K相关的结果行
        if 'gsm8k' in line_lower:
            # 尝试匹配 exact_match 分数
            match = re.search(r'exact_match[,\s]*(?:flexible-extract|strict-match)?[:\s|]*(\d+\.?\d*)', line_lower)
            if match:
                score = float(match.group(1))
                results["scores"]["gsm8k"] = score
                results["gsm8k_accuracy"] = score
                continue

            # 尝试匹配 acc 分数
            match = re.search(r'\|acc\s*\|[↑↓]?\s*\|?(\d+\.?\d*)', line)
            if match:
                score = float(match.group(1))
                results["scores"]["gsm8k_acc"] = score
                if "gsm8k_accuracy" not in results:
                    results["gsm8k_accuracy"] = score
                continue

            # 尝试匹配表格格式
            match = re.search(r'(\d+\.\d+)\s*$', line)
            if match and 'gsm8k' in line_lower:
                score = float(match.group(1))
                results["scores"]["gsm8k"] = score
                if "gsm8k_accuracy" not in results:
                    results["gsm8k_accuracy"] = score

    return results


# ==================== 原有的零样本评估函数 ====================

def eval_zero_shot(model_name,
                   task_list=["boolq", "rte", "hellaswag", "arc_challenge", "arc_easy", "winogrande", "openbookqa"],
                   num_fewshot=0, use_accelerate=True, add_special_tokens=False,
                   offline=False, local_datasets_path=None, tokenizer_path=None):
    """
    评估模型的零样本性能
    """
    import subprocess
    import os
    import sys
    import time
    import re
    import json

    print(f"开始零样本评估模型: {model_name}")
    print(f"评估任务: {task_list}")
    print(f"离线模式: {offline}")

    # 检测GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")

    if offline and local_datasets_path:
        print(f"使用本地数据集路径: {local_datasets_path}")

    tasks_str = ",".join(task_list)

    # 设置环境变量
    env = os.environ.copy()

    if offline:
        print("设置离线模式环境变量...")
        env['HF_DATASETS_OFFLINE'] = '1'
        env['HF_HUB_OFFLINE'] = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'

        if local_datasets_path:
            env['HF_DATASETS_CACHE'] = os.path.abspath(local_datasets_path)
            env['HF_HOME'] = os.path.abspath(local_datasets_path)
            print(f"设置HF_DATASETS_CACHE为: {env['HF_DATASETS_CACHE']}")
    else:
        env['HF_DATASETS_OFFLINE'] = '0'
        env['HF_HUB_OFFLINE'] = '0'
        env['TRANSFORMERS_OFFLINE'] = '0'

    # ============ 修复：移除重复的torch_dtype参数 ============
    # 只保留必要的参数，torch_dtype由lm_eval内部处理
    model_args = f"pretrained={model_name},trust_remote_code=True,device_map=auto"
    if tokenizer_path:
        model_args += f",tokenizer={tokenizer_path},use_fast_tokenizer=False"

    # 如果有多个GPU，添加并行参数
    if num_gpus > 1:
        model_args += ",parallelize=True"

    print(f"模型参数: {model_args}")
    # ===================================================

    # 构建命令
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks_str,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "1",
        "--device", "cuda",
        "--output_path", "./eval_results",
        "--log_samples"
    ]

    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 60)

    # ... 其余代码保持不变 ...
    try:
        # 记录开始时间
        start_time = time.time()

        # 执行命令，实时显示输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )

        # 实时打印输出
        output_lines = []
        print("开始评估...")
        print("-" * 40)

        # 用于存储最后一个进度条的位置
        last_progress_line = None

        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                output_lines.append(line)

                # 特别关注包含指标的行
                if any(metric in line.lower() for metric in ['acc', 'acc_norm', 'em', 'f1']):
                    print(f"[METRIC] {line}")
                    last_progress_line = None
                # 处理进度条 - 只保留最新的进度条
                elif "Running loglikelihood requests:" in line:
                    # 使用\r回车符覆盖上一行
                    print(f"\r[INFO] {line}", end='', flush=True)
                    last_progress_line = line
                # 显示结果行
                elif any(task in line.lower() for task in task_list) and ('|' in line):
                    # 如果之前有进度条，先换行
                    if last_progress_line:
                        print()  # 换行
                        last_progress_line = None
                    print(f"[RESULT] {line}")
                # 显示其他重要信息
                elif any(keyword in line.lower() for keyword in [
                    'loading', 'evaluating', 'finished', 'completed', 'task',
                    'error', 'warning'
                ]) and "Running loglikelihood requests:" not in line:
                    # 如果之前有进度条，先换行
                    if last_progress_line:
                        print()  # 换行
                        last_progress_line = None
                    print(f"[INFO] {line}")

        # 确保最后的进度条后面有换行
        if last_progress_line:
            print()

        process.wait()
        output = '\n'.join(output_lines)

        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("-" * 40)
        print(f"评估完成，耗时: {elapsed_time:.2f}秒")

        if process.returncode == 0:
            print("零样本评估成功完成!")

            # 解析结果
            results = parse_results(output)

            # 尝试从JSON文件读取更准确的结果
            json_results_path = "./eval_results/results.json"
            if os.path.exists(json_results_path):
                try:
                    with open(json_results_path, "r") as f:
                        json_data = json.load(f)
                        # 从JSON中提取结果（静默）
                        for task in task_list:
                            if task in json_data.get("results", {}):
                                task_results = json_data["results"][task]
                                # hellaswag和openbookqa使用acc_norm
                                if task in ["hellaswag", "openbookqa"] and "acc_norm" in task_results:
                                    score = task_results["acc_norm"] * 100
                                    results["scores"][task] = score
                                elif "acc" in task_results:
                                    score = task_results["acc"] * 100
                                    results["scores"][task] = score
                except Exception as e:
                    print(f"读取JSON结果时出错: {e}")

            # 显示结果摘要
            if results.get("status") == "success" and results.get("scores"):
                print("\n结果摘要:")
                print("-" * 30)
                for task, score in results["scores"].items():
                    if isinstance(score, (int, float)):
                        print(f"{task:15}: {score:.2f}%")
                    else:
                        print(f"{task:15}: {score}")

            return results
        else:
            print(f"评估失败，返回码: {process.returncode}")

            # 检查常见错误
            if "ConnectionError" in output or "Couldn't reach" in output:
                print("\n检测到网络连接问题。解决方案：")
                print("1. 确认数据集已下载到:", local_datasets_path)
                print("2. 尝试手动指定数据集路径")
                print("3. 检查 lm_eval 版本是否支持离线模式")

            elif "unrecognized arguments" in output:
                print("\n命令参数错误。请检查 lm_eval 版本和支持的参数。")
                print("运行 'lm_eval --help' 查看支持的参数。")

            return {"status": "failed", "error": output}

    except subprocess.TimeoutExpired:
        print(f"评估超时（超过3600秒）")
        return {"status": "timeout", "error": "Evaluation timed out after 3600 seconds"}
    except FileNotFoundError:
        print("错误：找不到 lm_eval 命令")
        print("请确保已安装 lm-evaluation-harness:")
        print("  pip install lm-eval")
        return {"status": "error", "error": "lm_eval command not found"}
    except Exception as e:
        print(f"执行出错: {e}")
        return {"status": "error", "error": str(e)}


def parse_results(output):
    """解析 lm_eval 的输出结果 - 支持acc_norm指标"""
    import re

    results = {"status": "success", "scores": {}}

    # 改进的解析逻辑 - 处理表格格式的输出
    lines = output.split('\n')

    for i, line in enumerate(lines):
        # Hellaswag 特殊处理 - 使用 acc_norm
        if "hellaswag" in line.lower():
            # 首先尝试查找 acc_norm（这是主要指标）
            match = re.search(r'\|acc_norm\s*\|[↑↓]?\s*\|(\d+\.\d+)\|', line)
            if not match:
                match = re.search(r'acc_norm[:\s]+(\d+\.\d+)', line.lower())
            if match:
                score = float(match.group(1))
                if score <= 1.0:
                    score = score * 100
                results["scores"]["hellaswag"] = score
                continue
            # 如果没有acc_norm，尝试查找普通acc
            match = re.search(r'\|acc\s*\|[↑↓]?\s*\|(\d+\.\d+)\|', line)
            if not match:
                match = re.search(r'acc[:\s]+(\d+\.\d+)', line.lower())
            if match:
                score = float(match.group(1))
                if score <= 1.0:
                    score = score * 100
                results["scores"]["hellaswag"] = score
                continue

        # OpenBookQA 处理 - 优先使用 acc_norm
        elif "openbookqa" in line.lower():
            # 优先查找 acc_norm
            match = re.search(r'\|acc_norm\s*\|[↑↓]?\s*\|(\d+\.\d+)\|', line)
            if not match:
                match = re.search(r'acc_norm[:\s]+(\d+\.\d+)', line.lower())
            if match:
                score = float(match.group(1))
                if score <= 1.0:
                    score = score * 100
                results["scores"]["openbookqa"] = score
                continue
            # 如果没有 acc_norm，尝试 acc
            match = re.search(r'\|acc\s*\|[↑↓]?\s*\|(\d+\.\d+)\|', line)
            if not match:
                match = re.search(r'acc[:\s]+(\d+\.\d+)', line.lower())
            if match:
                score = float(match.group(1))
                if score <= 1.0:
                    score = score * 100
                results["scores"]["openbookqa"] = score
                continue

        # 其他任务继续使用 acc
        else:
            for task in ["boolq", "rte", "arc_challenge", "arc_easy", "winogrande"]:
                if task in line.lower():
                    # 尝试匹配表格格式
                    match = re.search(r'\|acc\s*\|[↑↓]?\s*\|(\d+\.\d+)\|', line)
                    if match:
                        score = float(match.group(1))
                        if score <= 1.0:
                            score = score * 100
                        results["scores"][task] = score
                        break

                    # 尝试匹配另一种格式 acc: 0.xxxx
                    match = re.search(r'acc[:\s]+(\d+\.\d+)', line.lower())
                    if match:
                        score = float(match.group(1))
                        if score <= 1.0:
                            score = score * 100
                        results["scores"][task] = score
                        break

    # 如果没有找到任何分数，尝试更宽松的解析（静默模式）
    if not results["scores"]:
        for line in lines:
            parts = line.split('|')
            if len(parts) >= 5:  # 确保有足够的列
                for task in ["boolq", "rte", "hellaswag", "arc_challenge", "arc_easy", "winogrande", "openbookqa"]:
                    if task in parts[0].lower():
                        # 查找包含数值的列
                        for j, part in enumerate(parts):
                            part = part.strip()
                            # 检查是否是acc_norm列（对于hellaswag和openbookqa）
                            if j > 0 and j < len(parts) - 1:
                                prev_part = parts[j - 1].strip().lower()
                                if task in ["hellaswag", "openbookqa"] and "acc_norm" in prev_part:
                                    if re.match(r'^\d+\.\d+$', part):
                                        score = float(part)
                                        if score <= 1.0:
                                            score = score * 100
                                        results["scores"][task] = score
                                        break
                                elif "acc" in prev_part and "norm" not in prev_part:
                                    if re.match(r'^\d+\.\d+$', part):
                                        score = float(part)
                                        if score <= 1.0:
                                            score = score * 100
                                        results["scores"][task] = score
                                        break

    return results
