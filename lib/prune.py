import time
import heapq
import gc
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
from .data import get_loaders

from .ablate import AblateGPT


def safe_to_device(tensor, device):
    """安全地将张量移动到设备，处理None情况"""
    return tensor.to(device) if tensor is not None else None


def safe_empty_cache(context=""):
    if not torch.cuda.is_available():
        return
    if os.environ.get("SKIP_CUDA_EMPTY_CACHE", "0") == "1":
        print(f"SKIP_CUDA_EMPTY_CACHE: skip torch.cuda.empty_cache() {context}")
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        if os.environ.get("IGNORE_CUDA_EMPTY_CACHE_ERROR", "1") == "1":
            print(f"WARNING: torch.cuda.empty_cache() failed {context}: {exc}")
            print("WARNING: continuing because IGNORE_CUDA_EMPTY_CACHE_ERROR=1")
            return
        raise


def synchronize_all_cuda_devices(context=""):
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)
    if context:
        print(f"DEBUG_CUDA_SYNC: {context} complete on all visible GPUs")


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()
            sub_count += (W == 0).sum().item()
            sub_params += W.numel()
        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
    model.config.use_cache = use_cache
    return float(count) / total_params


def prepare_calibration_input(model, dataloader, device, nsamples=128):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    if hasattr(model, 'hf_device_map') and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            if "OPT" not in model.__class__.__name__:
                cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def owl_density_from_outlier_ratios(outlier_ratios, target_sparsity, lamda):
    D = np.asarray(outlier_ratios, dtype=float)
    if D.max() - D.min() > 1e-12:
        z = (D - D.min()) / (D.max() - D.min())
    else:
        z = np.zeros_like(D)

    target_sparsity = float(target_sparsity)
    lamda = float(lamda)
    min_s = max(0.01, target_sparsity - lamda)
    max_s = min(0.99, target_sparsity + lamda)

    layer_sparsity = target_sparsity + (1 - z - 0.5) * (2 * lamda)
    layer_sparsity = np.clip(layer_sparsity, min_s, max_s)
    layer_sparsity = np.clip(layer_sparsity - (layer_sparsity.mean() - target_sparsity), min_s, max_s)
    layer_density = 1 - layer_sparsity
    return layer_density, layer_sparsity


def renormalize_density_with_bounds(layer_density, target_sparsity, lamda, protected_mask=None):
    density = np.asarray(layer_density, dtype=float).copy()
    target_sparsity = float(target_sparsity)
    lamda = float(lamda)
    target_total = (1.0 - target_sparsity) * density.size
    min_s = max(0.01, target_sparsity - lamda)
    max_s = min(0.99, target_sparsity + lamda)
    min_density = 1.0 - max_s
    max_density = 1.0 - min_s
    density = np.clip(density, min_density, max_density)

    if protected_mask is None:
        protected_mask = np.zeros_like(density, dtype=bool)
    else:
        protected_mask = np.asarray(protected_mask, dtype=bool)

    for _ in range(64):
        diff = density.sum() - target_total
        if abs(diff) < 1e-10:
            break

        if diff > 0:
            candidates = (~protected_mask) & (density > min_density + 1e-12)
            if not np.any(candidates):
                candidates = density > min_density + 1e-12
            capacity = density[candidates] - min_density
            if capacity.sum() <= 1e-12:
                break
            reduction = np.minimum(diff * capacity / capacity.sum(), capacity)
            density[candidates] -= reduction
        else:
            candidates = density < max_density - 1e-12
            capacity = max_density - density[candidates]
            if capacity.sum() <= 1e-12:
                break
            increase = np.minimum((-diff) * capacity / capacity.sum(), capacity)
            density[candidates] += increase

    layer_sparsity = 1.0 - density
    return density, layer_sparsity


def project_density_with_bounds(
        layer_density, target_sparsity, lamda, mode="protected", protected_mask=None):
    """Apply the selected OSLA budget projection without changing legacy defaults."""
    if mode not in {"none", "uniform", "protected"}:
        raise ValueError(f"Unknown OSLA projection mode: {mode}")

    density = np.asarray(layer_density, dtype=float).copy()
    target_sparsity = float(target_sparsity)
    lamda = float(lamda)
    min_s = max(0.01, target_sparsity - lamda)
    max_s = min(0.99, target_sparsity + lamda)
    density = np.clip(density, 1.0 - max_s, 1.0 - min_s)

    if mode == "none":
        return density, 1.0 - density
    if mode == "uniform":
        protected_mask = None
    return renormalize_density_with_bounds(
        density,
        target_sparsity,
        lamda,
        protected_mask=protected_mask,
    )


def resolve_osla_intervals(
        n_layers, core_start, core_end, tail_start, tail_end,
        mode="absolute", reference_layers=32):
    """Resolve inclusive OSLA intervals for the current transformer depth."""
    if mode not in {"absolute", "normalized"}:
        raise ValueError(f"Unknown OSLA interval mode: {mode}")
    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    if reference_layers < 2:
        raise ValueError("v9_reference_layers must be at least 2")

    values = [int(core_start), int(core_end), int(tail_start), int(tail_end)]
    if mode == "normalized" and n_layers != reference_layers:
        scale = (n_layers - 1) / float(reference_layers - 1)
        values = [int(round(value * scale)) for value in values]

    values = [min(max(value, 0), n_layers - 1) for value in values]
    resolved_core_start, resolved_core_end, resolved_tail_start, resolved_tail_end = values
    if resolved_core_start > resolved_core_end:
        resolved_core_start, resolved_core_end = resolved_core_end, resolved_core_start
    if resolved_tail_start > resolved_tail_end:
        resolved_tail_start, resolved_tail_end = resolved_tail_end, resolved_tail_start
    return (
        resolved_core_start,
        resolved_core_end,
        resolved_tail_start,
        resolved_tail_end,
    )


def write_osla_diagnostics(args, payload):
    """Atomically persist machine-readable layer-allocation diagnostics."""
    save_dir = getattr(args, "save", None)
    if not save_dir:
        return None
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v9_layer_diagnostics.json"
    temporary_path = output_dir / "v9_layer_diagnostics.json.tmp"
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(f"OSLA diagnostics saved to: {output_path}")
    return output_path


def build_wanda_mask(W_metric, sparsity_ratio, use_variant=False, prune_n=0, prune_m=0):
    W_mask = (torch.zeros_like(W_metric) == 1)
    if prune_n != 0:
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii:(ii + prune_m)].float()
                W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
    elif sparsity_ratio > 0:
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        if use_variant:
            tmp_metric = torch.cumsum(sort_res[0], dim=1)
            sum_before = W_metric.sum(dim=1)
            alpha = 0.4
            alpha_hist = [0., 0.8]
            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
            while (torch.abs(cur_sparsity - sparsity_ratio) > 0.001) and (
                    alpha_hist[1] - alpha_hist[0] >= 0.001):
                if cur_sparsity > sparsity_ratio:
                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                    alpha_hist[1] = alpha
                else:
                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                    alpha_hist[0] = alpha
                alpha = alpha_new
                W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
        else:
            indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
            W_mask.scatter_(1, indices, True)
    return W_mask


# ==================== RIA 剪枝方法 ====================

def prune_ria(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    RIA (Relative Importance and Activations) 剪枝方法

    核心公式: W_metric = (|W|/sum(|W|, dim=0) + |W|/sum(|W|, dim=1)) * (sqrt(scaler_row))^a

    参数:
        args: 包含以下属性
            - sparsity_ratio: 目标稀疏度
            - nsamples: 校准样本数量
            - seed: 随机种子
            - a: 激活指数 (默认0.5)
            - per_outneuron: 是否按输出神经元剪枝
            - use_variant: 是否使用wanda变体
        model: 要剪枝的模型
        tokenizer: 分词器
        device: 计算设备
        prune_n, prune_m: N:M结构化稀疏参数
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # 获取激活指数，默认为0.5
    activation_exp = getattr(args, 'a', 0.5)
    per_outneuron = getattr(args, 'per_outneuron', False)

    print(f"RIA剪枝开始 - 激活指数a={activation_exp}, per_outneuron={per_outneuron}")

    # 加载校准数据
    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    # 准备校准输入
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    # 获取模型层
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    # 逐层处理
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # 处理多GPU情况
        if hasattr(model, 'hf_device_map') and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        # 包装层以收集激活统计
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        # 注册前向钩子
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # 前向传播收集激活
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        # 移除钩子
        for h in handles:
            h.remove()

        # 对每个子层进行剪枝
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()

            # ========== RIA核心度量计算 ==========
            # 相对重要性：行方向 + 列方向的归一化
            W_abs = torch.abs(W)
            # 列方向归一化: |W| / sum(|W|, dim=0)
            col_norm = W_abs / (torch.sum(W_abs, dim=0) + 1e-10)
            # 行方向归一化: |W| / sum(|W|, dim=1)
            row_norm = W_abs / (torch.sum(W_abs, dim=1, keepdim=True) + 1e-10)
            # 相对重要性
            relative_importance = col_norm + row_norm

            # 激活感知: sqrt(scaler_row)^a
            activation_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            activation_scale = torch.pow(activation_scale, activation_exp)

            # 最终RIA度量
            W_metric = relative_importance * activation_scale
            # =====================================

            # 初始化掩码
            W_mask = (torch.zeros_like(W_metric) == 1)

            if prune_n != 0:
                # N:M结构化稀疏
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                # 非结构化稀疏
                if per_outneuron:
                    # 按输出神经元剪枝 (Wanda的策略)
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                else:
                    if getattr(args, 'use_variant', False):
                        # Wanda变体
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and \
                                (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha
                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # 全局阈值剪枝
                        thresh = torch.sort(W_metric.flatten().cuda())[0][
                            int(W.shape[0] * W.shape[1] * args.sparsity_ratio)].cpu()
                        W_mask = (W_metric <= thresh)

            # 应用掩码
            subset[name].weight.data[W_mask] = 0

        # 更新到下一层
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("RIA剪枝完成!")


def prune_ria_owl(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    RIA + OWL: 结合RIA度量和OWL的层级稀疏度分配

    第一阶段: 使用RIA度量计算每层的异常值比例
    第二阶段: 使用OWL算法调整每层的稀疏度
    第三阶段: 使用调整后的稀疏度进行RIA剪枝
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    activation_exp = getattr(args, 'a', 0.5)
    Hyper_m = getattr(args, 'Hyper_m', 5)
    Lamda = getattr(args, 'Lamda', 0.08)

    print("=" * 50)
    print(f"RIA-OWL剪枝开始")
    print(f"激活指数a={activation_exp}, Hyper_m={Hyper_m}, Lambda={Lamda}")
    print("=" * 50)

    # ========== 第一阶段：计算每层的异常值比例 ==========
    print("\n第一阶段：计算LOD（层级异常值分布）...")

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    all_layer_ratio = []

    # 第一次遍历：计算异常值
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if hasattr(model, 'hf_device_map') and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        # 收集当层的RIA metric
        layer_wmetric = []
        for name in subset:
            W = subset[name].weight.data
            W_abs = torch.abs(W)
            col_norm = W_abs / (torch.sum(W_abs, dim=0) + 1e-10)
            row_norm = W_abs / (torch.sum(W_abs, dim=1, keepdim=True) + 1e-10)
            relative_importance = col_norm + row_norm
            activation_scale = torch.pow(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))), activation_exp)
            W_metric = relative_importance * activation_scale
            layer_wmetric.append(W_metric.cpu())

        # 前向传播到下一层
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

        # 合并当前层的所有W_metric
        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric])

        # 计算异常值比例
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_ratio = (layer_wmetric > threshold).sum().item() / layer_wmetric.numel() * 100

        print(f"层 {i} 异常值比例: {outlier_ratio:.2f}%")
        all_layer_ratio.append(outlier_ratio)

    # ========== 第二阶段：调整稀疏率 ==========
    print("\n调整前的异常值比例:", [f"{x:.2f}" for x in all_layer_ratio])

    all_layer_ratio = np.array(all_layer_ratio)

    # OWL的归一化和调整
    if all_layer_ratio.max() > all_layer_ratio.min():
        all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) *
                           (1 / (all_layer_ratio.max() - all_layer_ratio.min()) * Lamda * 2))
    else:
        all_layer_ratio = np.ones_like(all_layer_ratio) * Lamda

    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    all_layer_ratio = np.clip(all_layer_ratio, 0.01, 0.99)

    print(f"调整后 - 均值: {np.mean(all_layer_ratio):.3f}, "
          f"最大: {np.max(all_layer_ratio):.3f}, 最小: {np.min(all_layer_ratio):.3f}")
    print("调整后的密度比例:", [f"{x:.3f}" for x in all_layer_ratio])

    # ========== 第三阶段：使用调整后的稀疏率进行RIA剪枝 ==========
    print("\n第三阶段：开始RIA剪枝...")
    print("=" * 50)

    # 重新初始化
    model.config.use_cache = False
    torch.cuda.empty_cache()

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    per_outneuron = getattr(args, 'per_outneuron', False)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # 获取当前层的稀疏率
        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = np.clip(layer_sparsity_ratio, 0.0, 0.99)

        print(f"\n层 {i}: 密度={all_layer_ratio[i]:.3f}, 稀疏率={layer_sparsity_ratio:.3f}")

        if hasattr(model, 'hf_device_map') and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        # 对每个子模块进行剪枝
        for name in subset:
            print(f"  修剪 {name}")
            W = subset[name].weight.data.clone()

            # RIA度量计算
            W_abs = torch.abs(W)
            col_norm = W_abs / (torch.sum(W_abs, dim=0) + 1e-10)
            row_norm = W_abs / (torch.sum(W_abs, dim=1, keepdim=True) + 1e-10)
            relative_importance = col_norm + row_norm
            activation_scale = torch.pow(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))), activation_exp)
            W_metric = relative_importance * activation_scale

            W_mask = (torch.zeros_like(W_metric) == 1)

            if prune_n != 0:
                # N:M结构化稀疏
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                # 非结构化稀疏 - 使用OWL计算的层级稀疏度
                if layer_sparsity_ratio > 0:
                    if per_outneuron:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)
                        indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)
                    else:
                        thresh = torch.sort(W_metric.flatten().cuda())[0][
                            int(W.numel() * layer_sparsity_ratio)].cpu()
                        W_mask = (W_metric <= thresh)

            subset[name].weight.data[W_mask] = 0

            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    实际稀疏率: {actual_sparsity:.3f}")

        # 更新到下一层
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("RIA-OWL剪枝完成!")
    print("=" * 50)


# ==================== 原有的剪枝方法 ====================

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity_ratio)].cpu()
                W_mask = (W_metric <= thresh)
            W[W_mask] = 0


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device,
                                                                             nsamples=args.nsamples)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)
                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (
                            alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def _prune_original_owl(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                        pruning_metric="wanda"):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if pruning_metric not in V9_PRUNING_METRIC_LABELS:
        raise ValueError(f"Unknown original-OWL pruning metric: {pruning_metric}")

    gradients = None
    if pruning_metric != "wanda":
        if args.gradient_path is None:
            raise ValueError(f"--gradient_path is required for original-OWL metric '{pruning_metric}'")
        print(f"loading gradients from {args.gradient_path}")
        gradients = torch.load(args.gradient_path, map_location=torch.device('cpu'))
        if not isinstance(gradients, dict):
            raise ValueError("The gradient file must contain a dictionary of layer tensors")
        print(f"loaded {len(gradients)} gradient tensors")

    Hyper_m = getattr(args, 'Hyper_m', 5)
    Lamda = getattr(args, 'Lamda', 0.08)

    print("=" * 50)
    print("Original OWL pruning starts (True OWL LOD score)")
    print(f"Stage-2 pruning metric: {pruning_metric} = {V9_PRUNING_METRIC_LABELS[pruning_metric]}")
    print(f"Hyper_m(outlier multiplier): {Hyper_m}, Lambda: {Lamda}")
    print("=" * 50)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    all_layer_ratio = []

    print("Stage 1: collecting true OWL LOD ratios...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_abs = torch.abs(subset[name].weight.data)
            X_l2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            outlier_score = W_abs * X_l2
            layer_wmetric.append(outlier_score.cpu())

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric])
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_ratio = (layer_wmetric > threshold).sum().item() / layer_wmetric.numel() * 100
        print(f"layer {i} true OWL LOD ratio: {outlier_ratio:.2f}% "
              f"(mean={mean_val.item():.6e}, threshold={threshold.item():.6e})")
        all_layer_ratio.append(outlier_ratio)

    safe_empty_cache("after original OWL stage 1")

    print("\nRaw true OWL LOD ratios:", [f"{x:.2f}" for x in all_layer_ratio])
    all_layer_ratio, layer_sparsity_ratios = owl_density_from_outlier_ratios(
        all_layer_ratio, args.sparsity_ratio, Lamda)

    print(f"Adjusted sparsity - mean: {np.mean(layer_sparsity_ratios):.3f}, "
          f"max: {np.max(layer_sparsity_ratios):.3f}, min: {np.min(layer_sparsity_ratios):.3f}")
    print("Adjusted sparsity ratios:", [f"{x:.3f}" for x in layer_sparsity_ratios])
    print("Adjusted density ratios:", [f"{x:.3f}" for x in all_layer_ratio])

    print(f"\nStage 2: applying {pruning_metric} pruning with layer-wise original OWL sparsity...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = float(np.clip(layer_sparsity_ratio, 0.0, 0.99))
        print(f"\nlayer {i}: density={all_layer_ratio[i]:.3f}, sparsity={layer_sparsity_ratio:.3f}")

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"  pruning {name}")
            gradient = None
            gradient_key = None
            if gradients is not None:
                gradient, gradient_key = _get_v9_gradient(gradients, i, name)
            W_metric = build_v9_pruning_metric(
                pruning_metric,
                subset[name].weight.data,
                gradient,
                wrapped_layers[name].scaler_row,
            )
            if gradient_key is not None:
                print(f"    gradient key: {gradient_key}")
            W_mask = build_wanda_mask(
                W_metric,
                layer_sparsity_ratio,
                use_variant=getattr(args, 'use_variant', False),
                prune_n=prune_n,
                prune_m=prune_m)

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    actual sparsity: {actual_sparsity:.3f}")
            del gradient, W_metric, W_mask

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    safe_empty_cache("after original OWL pruning")
    print(f"Original OWL pruning complete! metric={pruning_metric}")


def prune_wanda_owl(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    return _prune_original_owl(
        args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
        pruning_metric="wanda")


def prune_metric_owl(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                     metric_name=None):
    if metric_name is None:
        raise ValueError("metric_name must be provided for prune_metric_owl")
    return _prune_original_owl(
        args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
        pruning_metric=metric_name)


def prune_wanda_owl_v1(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    Hyper_m = getattr(args, 'Hyper_m', 5)
    Lamda = getattr(args, 'Lamda', 0.08)

    print("=" * 50)
    print("Wanda-OWL-V1 pruning starts (Loss-Energy outlier score)")
    print(f"Hyper_m(outlier multiplier): {Hyper_m}, Lambda: {Lamda}")
    print("=" * 50)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    all_layer_ratio = []

    print("Stage 1: collecting outlier ratios with Loss-Energy score...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_abs = torch.abs(subset[name].weight.data)
            X_energy = wrapped_layers[name].scaler_row.reshape((1, -1))
            outlier_score = W_abs.pow(2) * X_energy
            layer_wmetric.append(outlier_score.cpu())

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric])
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_ratio = (layer_wmetric > threshold).sum().item() / layer_wmetric.numel() * 100
        print(f"layer {i} Loss-Energy outlier ratio: {outlier_ratio:.2f}% "
              f"(mean={mean_val.item():.6e}, threshold={threshold.item():.6e})")
        all_layer_ratio.append(outlier_ratio)

    torch.cuda.empty_cache()

    print("\nRaw Loss-Energy outlier ratios:", [f"{x:.2f}" for x in all_layer_ratio])
    all_layer_ratio, layer_sparsity_ratios = owl_density_from_outlier_ratios(
        all_layer_ratio, args.sparsity_ratio, Lamda)

    print(f"Adjusted sparsity - mean: {np.mean(layer_sparsity_ratios):.3f}, "
          f"max: {np.max(layer_sparsity_ratios):.3f}, min: {np.min(layer_sparsity_ratios):.3f}")
    print("Adjusted sparsity ratios:", [f"{x:.3f}" for x in layer_sparsity_ratios])
    print("Adjusted density ratios:", [f"{x:.3f}" for x in all_layer_ratio])

    print("\nStage 2: applying Wanda pruning with layer-wise OWL sparsity...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = float(np.clip(layer_sparsity_ratio, 0.0, 0.99))
        print(f"\nlayer {i}: density={all_layer_ratio[i]:.3f}, sparsity={layer_sparsity_ratio:.3f}")

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"  pruning {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                if layer_sparsity_ratio > 0:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    if getattr(args, 'use_variant', False):
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)
                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (
                                alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > layer_sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha
                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"    alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    actual sparsity: {actual_sparsity:.3f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Wanda-OWL-V1 pruning complete!")


def prune_wanda_owl_v2(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    Hyper_m = getattr(args, 'Hyper_m', 5)
    Lamda = getattr(args, 'Lamda', 0.08)

    print("=" * 50)
    print("Wanda-OWL-V2 pruning starts (Channel-Tail outlier score)")
    print(f"Hyper_m(outlier multiplier): {Hyper_m}, Lambda: {Lamda}")
    print("=" * 50)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    all_layer_ratio = []

    print("Stage 1: collecting channel-level outlier ratios...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_abs = torch.abs(subset[name].weight.data)
            channel_weight = torch.mean(W_abs, dim=0)
            X_l2 = torch.sqrt(wrapped_layers[name].scaler_row)
            outlier_score = channel_weight * X_l2
            layer_wmetric.append(outlier_score.cpu())

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric])
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_ratio = (layer_wmetric > threshold).sum().item() / layer_wmetric.numel() * 100
        print(f"layer {i} Channel-Tail outlier ratio: {outlier_ratio:.2f}% "
              f"(mean={mean_val.item():.6e}, threshold={threshold.item():.6e})")
        all_layer_ratio.append(outlier_ratio)

    torch.cuda.empty_cache()

    print("\nRaw Channel-Tail outlier ratios:", [f"{x:.2f}" for x in all_layer_ratio])
    all_layer_ratio, layer_sparsity_ratios = owl_density_from_outlier_ratios(
        all_layer_ratio, args.sparsity_ratio, Lamda)

    print(f"Adjusted sparsity - mean: {np.mean(layer_sparsity_ratios):.3f}, "
          f"max: {np.max(layer_sparsity_ratios):.3f}, min: {np.min(layer_sparsity_ratios):.3f}")
    print("Adjusted sparsity ratios:", [f"{x:.3f}" for x in layer_sparsity_ratios])
    print("Adjusted density ratios:", [f"{x:.3f}" for x in all_layer_ratio])

    print("\nStage 2: applying Wanda pruning with layer-wise OWL sparsity...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = float(np.clip(layer_sparsity_ratio, 0.0, 0.99))
        print(f"\nlayer {i}: density={all_layer_ratio[i]:.3f}, sparsity={layer_sparsity_ratio:.3f}")

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"  pruning {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                if layer_sparsity_ratio > 0:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    if getattr(args, 'use_variant', False):
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)
                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (
                                alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > layer_sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha
                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"    alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    actual sparsity: {actual_sparsity:.3f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Wanda-OWL-V2 pruning complete!")


def prune_wanda_owl_v3(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    Hyper_m = getattr(args, 'Hyper_m', 5)
    Lamda = getattr(args, 'Lamda', 0.08)

    print("=" * 50)
    print("Wanda-OWL-V3 pruning starts (Tail-Mass outlier ratio)")
    print(f"Hyper_m(outlier multiplier): {Hyper_m}, Lambda: {Lamda}")
    print("=" * 50)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    all_layer_ratio = []

    print("Stage 1: collecting tail-mass outlier ratios with true OWL score...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_abs = torch.abs(subset[name].weight.data)
            X_l2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            outlier_score = W_abs * X_l2
            layer_wmetric.append(outlier_score.cpu())

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric])
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_mask = layer_wmetric > threshold
        count_ratio = outlier_mask.sum().item() / layer_wmetric.numel() * 100
        total_mass = torch.sum(layer_wmetric)
        if torch.isfinite(total_mass).item() and total_mass.item() > 0:
            outlier_ratio = (torch.sum(layer_wmetric[outlier_mask]) / total_mass).item() * 100
        else:
            outlier_ratio = 0.0
        print(f"layer {i} Tail-Mass outlier ratio: {outlier_ratio:.2f}% "
              f"(count={count_ratio:.2f}%, mean={mean_val.item():.6e}, threshold={threshold.item():.6e})")
        all_layer_ratio.append(outlier_ratio)

    torch.cuda.empty_cache()

    print("\nRaw Tail-Mass outlier ratios:", [f"{x:.2f}" for x in all_layer_ratio])
    all_layer_ratio, layer_sparsity_ratios = owl_density_from_outlier_ratios(
        all_layer_ratio, args.sparsity_ratio, Lamda)

    print(f"Adjusted sparsity - mean: {np.mean(layer_sparsity_ratios):.3f}, "
          f"max: {np.max(layer_sparsity_ratios):.3f}, min: {np.min(layer_sparsity_ratios):.3f}")
    print("Adjusted sparsity ratios:", [f"{x:.3f}" for x in layer_sparsity_ratios])
    print("Adjusted density ratios:", [f"{x:.3f}" for x in all_layer_ratio])

    print("\nStage 2: applying Wanda pruning with layer-wise OWL sparsity...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = float(np.clip(layer_sparsity_ratio, 0.0, 0.99))
        print(f"\nlayer {i}: density={all_layer_ratio[i]:.3f}, sparsity={layer_sparsity_ratio:.3f}")

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"  pruning {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                if layer_sparsity_ratio > 0:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    if getattr(args, 'use_variant', False):
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)
                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (
                                alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > layer_sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha
                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"    alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    actual sparsity: {actual_sparsity:.3f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Wanda-OWL-V3 pruning complete!")


def prune_wanda_owl_v4(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    Hyper_m = getattr(args, 'Hyper_m', 5)
    Lamda = getattr(args, 'Lamda', 0.08)
    hybrid_alpha = 0.5

    print("=" * 50)
    print("Wanda-OWL-V4 pruning starts (Hybrid Count-Mass outlier ratio)")
    print(f"Hyper_m(outlier multiplier): {Hyper_m}, Lambda: {Lamda}, alpha: {hybrid_alpha}")
    print("=" * 50)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    all_layer_ratio = []

    print("Stage 1: collecting hybrid count-mass outlier ratios with true OWL score...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_abs = torch.abs(subset[name].weight.data)
            X_l2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            outlier_score = W_abs * X_l2
            layer_wmetric.append(outlier_score.cpu())

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric])
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_mask = layer_wmetric > threshold

        count_ratio = outlier_mask.sum().item() / layer_wmetric.numel() * 100
        total_mass = torch.sum(layer_wmetric)
        if torch.isfinite(total_mass).item() and total_mass.item() > 0:
            mass_ratio = (torch.sum(layer_wmetric[outlier_mask]) / total_mass).item() * 100
        else:
            mass_ratio = 0.0
        outlier_ratio = hybrid_alpha * count_ratio + (1 - hybrid_alpha) * mass_ratio

        print(f"layer {i} Hybrid Count-Mass outlier ratio: {outlier_ratio:.2f}% "
              f"(count={count_ratio:.2f}%, mass={mass_ratio:.2f}%, "
              f"mean={mean_val.item():.6e}, threshold={threshold.item():.6e})")
        all_layer_ratio.append(outlier_ratio)

    torch.cuda.empty_cache()

    print("\nRaw Hybrid Count-Mass outlier ratios:", [f"{x:.2f}" for x in all_layer_ratio])
    all_layer_ratio, layer_sparsity_ratios = owl_density_from_outlier_ratios(
        all_layer_ratio, args.sparsity_ratio, Lamda)

    print(f"Adjusted sparsity - mean: {np.mean(layer_sparsity_ratios):.3f}, "
          f"max: {np.max(layer_sparsity_ratios):.3f}, min: {np.min(layer_sparsity_ratios):.3f}")
    print("Adjusted sparsity ratios:", [f"{x:.3f}" for x in layer_sparsity_ratios])
    print("Adjusted density ratios:", [f"{x:.3f}" for x in all_layer_ratio])

    print("\nStage 2: applying Wanda pruning with layer-wise OWL sparsity...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = float(np.clip(layer_sparsity_ratio, 0.0, 0.99))
        print(f"\nlayer {i}: density={all_layer_ratio[i]:.3f}, sparsity={layer_sparsity_ratio:.3f}")

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"  pruning {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                if layer_sparsity_ratio > 0:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    if getattr(args, 'use_variant', False):
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)
                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (
                                alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > layer_sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha
                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"    alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    actual sparsity: {actual_sparsity:.3f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Wanda-OWL-V4 pruning complete!")


def prune_wanda_owl_v5(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    Hyper_m = getattr(args, 'Hyper_m', 5)
    Lamda = getattr(args, 'Lamda', 0.08)
    sensitivity_alpha = 0.5

    print("=" * 50)
    print("Wanda-OWL-V5 pruning starts (OWL + layer reconstruction sensitivity)")
    print(f"Hyper_m(outlier multiplier): {Hyper_m}, Lambda: {Lamda}, alpha: {sensitivity_alpha}")
    print("=" * 50)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    owl_ratios = []
    sensitivity_scores = []

    print("Stage 1: collecting true OWL ratios and layer reconstruction sensitivities...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_abs = torch.abs(subset[name].weight.data)
            X_l2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            outlier_score = W_abs * X_l2
            layer_wmetric.append(outlier_score.cpu())

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric])
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        owl_ratio = (layer_wmetric > threshold).sum().item() / layer_wmetric.numel() * 100

        original_weights = {}
        try:
            for name in subset:
                original_weights[name] = subset[name].weight.data.clone()
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_mask = build_wanda_mask(
                    W_metric,
                    args.sparsity_ratio,
                    use_variant=getattr(args, 'use_variant', False),
                    prune_n=prune_n,
                    prune_m=prune_m)
                subset[name].weight.data[W_mask] = 0

            err_num = 0.0
            err_den = 0.0
            for j in range(args.nsamples):
                with torch.no_grad():
                    if "OPT" in model.__class__.__name__:
                        pruned_out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                    else:
                        pruned_out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                           position_ids=position_ids)[0]
                    dense_out = outs[j].unsqueeze(0)
                    diff = pruned_out.float() - dense_out.float()
                    err_num += torch.sum(diff * diff).item()
                    err_den += torch.sum(dense_out.float() * dense_out.float()).item()
        finally:
            for name, weight in original_weights.items():
                subset[name].weight.data.copy_(weight)
                del weight

        sensitivity = err_num / (err_den + 1e-12)
        print(f"layer {i} OWL ratio: {owl_ratio:.2f}% "
              f"(mean={mean_val.item():.6e}, threshold={threshold.item():.6e}), "
              f"sensitivity={sensitivity:.6e}")
        owl_ratios.append(owl_ratio)
        sensitivity_scores.append(sensitivity)

        inps, outs = outs, inps

    torch.cuda.empty_cache()

    owl_arr = np.asarray(owl_ratios, dtype=float)
    sens_arr = np.asarray(sensitivity_scores, dtype=float)

    def normalize_metric(values):
        if values.max() - values.min() > 1e-12:
            return (values - values.min()) / (values.max() - values.min())
        return np.zeros_like(values)

    owl_norm = normalize_metric(owl_arr)
    sens_norm = normalize_metric(sens_arr)
    combined_importance = sensitivity_alpha * owl_norm + (1 - sensitivity_alpha) * sens_norm

    print("\nRaw true OWL ratios:", [f"{x:.2f}" for x in owl_arr])
    print("Raw sensitivity scores:", [f"{x:.6e}" for x in sens_arr])
    print("Combined OWL-sensitivity importance:", [f"{x:.3f}" for x in combined_importance])

    all_layer_ratio, layer_sparsity_ratios = owl_density_from_outlier_ratios(
        combined_importance, args.sparsity_ratio, Lamda)

    print(f"Adjusted sparsity - mean: {np.mean(layer_sparsity_ratios):.3f}, "
          f"max: {np.max(layer_sparsity_ratios):.3f}, min: {np.min(layer_sparsity_ratios):.3f}")
    print("Adjusted sparsity ratios:", [f"{x:.3f}" for x in layer_sparsity_ratios])
    print("Adjusted density ratios:", [f"{x:.3f}" for x in all_layer_ratio])

    print("\nStage 2: applying Wanda pruning with OWL-sensitivity layer sparsity...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = float(np.clip(layer_sparsity_ratio, 0.0, 0.99))
        print(f"\nlayer {i}: density={all_layer_ratio[i]:.3f}, sparsity={layer_sparsity_ratio:.3f}")

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"  pruning {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_mask = build_wanda_mask(
                W_metric,
                layer_sparsity_ratio,
                use_variant=getattr(args, 'use_variant', False),
                prune_n=prune_n,
                prune_m=prune_m)

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    actual sparsity: {actual_sparsity:.3f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Wanda-OWL-V5 pruning complete!")


def prune_wanda_owl_v6(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    Hyper_m = getattr(args, 'Hyper_m', 5)
    Lamda = getattr(args, 'Lamda', 0.08)
    eps = 1e-12

    print("=" * 50)
    print("Wanda-OWL-V6 pruning starts (Normalized Tail-Excess OWL)")
    print(f"Hyper_m(outlier multiplier): {Hyper_m}, Lambda: {Lamda}")
    print("=" * 50)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    all_layer_ratio = []

    print("Stage 1: collecting V6 normalized tail-excess OWL ratios...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_abs = torch.abs(subset[name].weight.data)
            X_l2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            outlier_score = W_abs * X_l2
            layer_wmetric.append(outlier_score.cpu())

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric])
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        count_ratio = (layer_wmetric > threshold).sum().item() / layer_wmetric.numel() * 100
        tail_mass_ratio = layer_wmetric[layer_wmetric > threshold].sum() / (layer_wmetric.sum() + eps) * 100
        tail_excess = torch.clamp(layer_wmetric - threshold, min=0)
        outlier_ratio = tail_excess.sum() / (layer_wmetric.sum() + eps) * 100
        outlier_ratio = outlier_ratio.item()
        print(f"layer {i} V6 tail-excess ratio: {outlier_ratio:.2f}% "
              f"(mean={mean_val.item():.6e}, threshold={threshold.item():.6e}, "
              f"count={count_ratio:.2f}%, tail_mass={tail_mass_ratio.item():.2f}%)")
        all_layer_ratio.append(outlier_ratio)

    torch.cuda.empty_cache()

    print("\nRaw V6 normalized tail-excess OWL ratios:", [f"{x:.2f}" for x in all_layer_ratio])
    all_layer_ratio, layer_sparsity_ratios = owl_density_from_outlier_ratios(
        all_layer_ratio, args.sparsity_ratio, Lamda)

    print(f"Adjusted sparsity - mean: {np.mean(layer_sparsity_ratios):.3f}, "
          f"max: {np.max(layer_sparsity_ratios):.3f}, min: {np.min(layer_sparsity_ratios):.3f}")
    print("Adjusted sparsity ratios:", [f"{x:.3f}" for x in layer_sparsity_ratios])
    print("Adjusted density ratios:", [f"{x:.3f}" for x in all_layer_ratio])

    print("\nStage 2: applying standard Wanda pruning with V6 OWL layer sparsity...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = float(np.clip(layer_sparsity_ratio, 0.0, 0.99))
        print(f"\nlayer {i}: density={all_layer_ratio[i]:.3f}, sparsity={layer_sparsity_ratio:.3f}")

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"  pruning {name}")
            scaler_row = wrapped_layers[name].scaler_row
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(scaler_row.reshape((1, -1)))
            W_mask = build_wanda_mask(
                W_metric,
                layer_sparsity_ratio,
                use_variant=getattr(args, 'use_variant', False),
                prune_n=prune_n,
                prune_m=prune_m)

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    actual sparsity: {actual_sparsity:.3f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Wanda-OWL-V6 pruning complete!")


def prune_wanda_owl_v7(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    Hyper_m = getattr(args, 'Hyper_m', 5)
    Lamda = getattr(args, 'Lamda', 0.08)
    eps = 1e-12

    print("=" * 50)
    print("Wanda-OWL-V7 pruning starts (Severity-Weighted Count OWL)")
    print(f"Hyper_m(outlier multiplier): {Hyper_m}, Lambda: {Lamda}")
    print("=" * 50)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    all_layer_ratio = []

    print("Stage 1: collecting V7 severity-weighted OWL ratios...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_abs = torch.abs(subset[name].weight.data)
            X_l2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            outlier_score = W_abs * X_l2
            layer_wmetric.append(outlier_score.cpu())

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric]).float()
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_mask = layer_wmetric > threshold
        outlier_count = outlier_mask.sum().item()
        count_ratio = outlier_count / layer_wmetric.numel() * 100
        tail_mass_ratio = layer_wmetric[outlier_mask].sum() / (layer_wmetric.sum() + eps) * 100

        if outlier_count > 0:
            relative_excess = (layer_wmetric[outlier_mask] - threshold) / (threshold + eps)
            severity = torch.log1p(relative_excess.mean()).item()
            outlier_ratio = count_ratio * (1.0 + severity)
        else:
            severity = 0.0
            outlier_ratio = 0.0

        print(f"layer {i} V7 severity-weighted ratio: {outlier_ratio:.2f}% "
              f"(mean={mean_val.item():.6e}, threshold={threshold.item():.6e}, "
              f"count={count_ratio:.2f}%, severity={severity:.4f}, "
              f"tail_mass={tail_mass_ratio.item():.2f}%)")
        all_layer_ratio.append(outlier_ratio)

    torch.cuda.empty_cache()

    print("\nRaw V7 severity-weighted OWL ratios:", [f"{x:.2f}" for x in all_layer_ratio])
    all_layer_ratio, layer_sparsity_ratios = owl_density_from_outlier_ratios(
        all_layer_ratio, args.sparsity_ratio, Lamda)

    print(f"Adjusted sparsity - mean: {np.mean(layer_sparsity_ratios):.3f}, "
          f"max: {np.max(layer_sparsity_ratios):.3f}, min: {np.min(layer_sparsity_ratios):.3f}")
    print("Adjusted sparsity ratios:", [f"{x:.3f}" for x in layer_sparsity_ratios])
    print("Adjusted density ratios:", [f"{x:.3f}" for x in all_layer_ratio])

    print("\nStage 2: applying standard Wanda pruning with V7 OWL layer sparsity...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = float(np.clip(layer_sparsity_ratio, 0.0, 0.99))
        print(f"\nlayer {i}: density={all_layer_ratio[i]:.3f}, sparsity={layer_sparsity_ratio:.3f}")

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"  pruning {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_mask = build_wanda_mask(
                W_metric,
                layer_sparsity_ratio,
                use_variant=getattr(args, 'use_variant', False),
                prune_n=prune_n,
                prune_m=prune_m)

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    actual sparsity: {actual_sparsity:.3f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Wanda-OWL-V7 pruning complete!")


def prune_wanda_owl_v8(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    Hyper_m = getattr(args, 'Hyper_m', 5)
    Lamda = getattr(args, 'Lamda', 0.08)
    anchor_alpha = float(np.clip(getattr(args, 'Owl_alpha', 0.2), 0.0, 1.0))
    eps = 1e-12

    print("=" * 50)
    print("Wanda-OWL-V8 pruning starts (Anchored Severity OWL)")
    print(f"Hyper_m(outlier multiplier): {Hyper_m}, Lambda: {Lamda}, anchor_alpha: {anchor_alpha}")
    print("=" * 50)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    count_ratios = []
    severity_ratios = []

    print("Stage 1: collecting V8 anchored OWL ratios...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_abs = torch.abs(subset[name].weight.data)
            X_l2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            outlier_score = W_abs * X_l2
            layer_wmetric.append(outlier_score.cpu())

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric]).float()
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_mask = layer_wmetric > threshold
        outlier_count = outlier_mask.sum().item()
        count_ratio = outlier_count / layer_wmetric.numel() * 100
        tail_mass_ratio = layer_wmetric[outlier_mask].sum() / (layer_wmetric.sum() + eps) * 100

        if outlier_count > 0:
            relative_excess = (layer_wmetric[outlier_mask] - threshold) / (threshold + eps)
            severity = torch.log1p(relative_excess.mean()).item()
            severity_ratio = count_ratio * (1.0 + severity)
        else:
            severity = 0.0
            severity_ratio = 0.0

        print(f"layer {i} V8 raw ratios: count={count_ratio:.2f}%, "
              f"severity_ratio={severity_ratio:.2f}% "
              f"(mean={mean_val.item():.6e}, threshold={threshold.item():.6e}, "
              f"severity={severity:.4f}, tail_mass={tail_mass_ratio.item():.2f}%)")
        count_ratios.append(count_ratio)
        severity_ratios.append(severity_ratio)

    torch.cuda.empty_cache()

    print("\nRaw V8 true OWL count ratios:", [f"{x:.2f}" for x in count_ratios])
    print("Raw V8 severity-weighted ratios:", [f"{x:.2f}" for x in severity_ratios])

    count_density, count_sparsity = owl_density_from_outlier_ratios(
        count_ratios, args.sparsity_ratio, Lamda)
    severity_density, severity_sparsity = owl_density_from_outlier_ratios(
        severity_ratios, args.sparsity_ratio, Lamda)

    all_layer_ratio = (1.0 - anchor_alpha) * count_density + anchor_alpha * severity_density
    layer_sparsity_ratios = 1.0 - all_layer_ratio

    print(f"Count-OWL sparsity - mean: {np.mean(count_sparsity):.3f}, "
          f"max: {np.max(count_sparsity):.3f}, min: {np.min(count_sparsity):.3f}")
    print(f"Severity-OWL sparsity - mean: {np.mean(severity_sparsity):.3f}, "
          f"max: {np.max(severity_sparsity):.3f}, min: {np.min(severity_sparsity):.3f}")
    print(f"Adjusted sparsity - mean: {np.mean(layer_sparsity_ratios):.3f}, "
          f"max: {np.max(layer_sparsity_ratios):.3f}, min: {np.min(layer_sparsity_ratios):.3f}")
    print("Count-OWL density ratios:", [f"{x:.3f}" for x in count_density])
    print("Severity-OWL density ratios:", [f"{x:.3f}" for x in severity_density])
    print("Adjusted sparsity ratios:", [f"{x:.3f}" for x in layer_sparsity_ratios])
    print("Adjusted density ratios:", [f"{x:.3f}" for x in all_layer_ratio])

    print("\nStage 2: applying standard Wanda pruning with V8 OWL layer sparsity...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = float(np.clip(layer_sparsity_ratio, 0.0, 0.99))
        print(f"\nlayer {i}: density={all_layer_ratio[i]:.3f}, sparsity={layer_sparsity_ratio:.3f}")

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"  pruning {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_mask = build_wanda_mask(
                W_metric,
                layer_sparsity_ratio,
                use_variant=getattr(args, 'use_variant', False),
                prune_n=prune_n,
                prune_m=prune_m)

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    actual sparsity: {actual_sparsity:.3f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Wanda-OWL-V8 pruning complete!")


V9_PRUNING_METRIC_LABELS = {
    "wanda": "|W| * sqrt(X)",
    "wsqrtg": "|W| * sqrt(|G|)",
    "wgradpow": "|W| * |G|^beta",
    "tanhwg": "tanh(|W|) * |W| * |G|",
    "w3gx": "|W|^3 * |G| * X",
    "w2mmsg": "|W|^2 * mms(|G|)",
    "w2xg": "|W|^2 * X * |G|",
}


def build_v9_pruning_metric(metric_name, weight, gradient, scaler_row, gradient_beta=0.5):
    """Build a non-negative element-wise pruning metric for OWL-V9 stage 2."""
    if metric_name not in V9_PRUNING_METRIC_LABELS:
        raise ValueError(f"Unknown OWL-V9 pruning metric: {metric_name}")

    W_abs = torch.abs(weight).to(dtype=torch.float32)
    X = None
    if metric_name in {"wanda", "w3gx", "w2xg"}:
        if scaler_row is None:
            raise ValueError(f"Activation scaler_row is required by metric '{metric_name}'")
        X = scaler_row.reshape((1, -1)).to(
            device=weight.device, dtype=torch.float32).clamp_min(0)

    if metric_name == "wanda":
        metric = W_abs * torch.sqrt(X)
    else:
        if gradient is None:
            raise ValueError(f"Gradient is required by OWL-V9 metric '{metric_name}'")
        G_abs = torch.abs(gradient).to(device=weight.device, dtype=torch.float32)
        if G_abs.shape != W_abs.shape:
            raise ValueError(
                f"Gradient shape {tuple(G_abs.shape)} does not match weight shape {tuple(W_abs.shape)}")

        if metric_name == "wsqrtg":
            metric = W_abs * torch.sqrt(G_abs)
        elif metric_name == "wgradpow":
            gradient_beta = float(gradient_beta)
            if gradient_beta < 0:
                raise ValueError(f"gradient_beta must be non-negative, got {gradient_beta}")
            metric = W_abs * G_abs.pow(gradient_beta)
        elif metric_name == "tanhwg":
            metric = torch.tanh(W_abs) * W_abs * G_abs
        elif metric_name == "w3gx":
            metric = W_abs.pow(3) * G_abs * X
        elif metric_name == "w2mmsg":
            g_min = G_abs.amin()
            g_range = G_abs.amax() - g_min
            if g_range.item() < 1e-8:
                G_scaled = torch.full_like(G_abs, 0.5)
            else:
                G_scaled = (G_abs - g_min) / g_range
            metric = W_abs.pow(2) * G_scaled
        else:  # w2xg
            metric = W_abs.pow(2) * X * G_abs

    max_value = torch.finfo(metric.dtype).max
    return torch.nan_to_num(metric, nan=0.0, posinf=max_value, neginf=0.0).clamp_min(0)


def _get_v9_gradient(gradients, layer_index, module_name):
    candidate_keys = (
        f"{module_name}_layer_{layer_index}",
        f"model.layers.{layer_index}.{module_name}",
        f"model.decoder.layers.{layer_index}.{module_name}",
    )
    for key in candidate_keys:
        if key in gradients:
            return gradients[key], key
    raise ValueError(
        f"No gradient found for layer {layer_index}, module '{module_name}'. "
        f"Tried keys: {candidate_keys}")


def prune_metric_uniform(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                         metric_name=None):
    """Prune every layer at the same sparsity using an element-wise metric."""
    del tokenizer, device
    if metric_name is None:
        raise ValueError("metric_name must be provided for prune_metric_uniform")
    if metric_name not in V9_PRUNING_METRIC_LABELS:
        raise ValueError(f"Unknown uniform pruning metric: {metric_name}")
    if metric_name == "wanda":
        raise ValueError("Use prune_wanda for the uniform Wanda metric")
    if args.gradient_path is None:
        raise ValueError(f"--gradient_path is required for uniform metric '{metric_name}'")

    print(f"loading gradients from {args.gradient_path}")
    gradients = torch.load(args.gradient_path, map_location=torch.device('cpu'))
    if not isinstance(gradients, dict):
        raise ValueError("The gradient file must contain a dictionary of layer tensors")
    print(f"loaded {len(gradients)} gradient tensors")
    print("=" * 50)
    print("Uniform metric pruning starts")
    print(f"Pruning metric: {metric_name} = {V9_PRUNING_METRIC_LABELS[metric_name]}")
    print(f"Uniform sparsity ratio: {args.sparsity_ratio}")
    print("=" * 50)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    try:
        for i, layer in enumerate(layers):
            subset = find_layers(layer)
            layer_key = layer_key_template.format(i)
            print(f"\nlayer {i}: uniform sparsity={args.sparsity_ratio:.3f}")
            if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
                print(f"  device={model.hf_device_map[layer_key]}")

            for name in subset:
                gradient, gradient_key = _get_v9_gradient(gradients, i, name)
                print(f"  pruning {name}; gradient key: {gradient_key}")
                W_metric = build_v9_pruning_metric(
                    metric_name,
                    subset[name].weight.data,
                    gradient,
                    scaler_row=None,
                    gradient_beta=getattr(args, 'Grad_beta', 0.5),
                )
                W_mask = build_wanda_mask(
                    W_metric,
                    args.sparsity_ratio,
                    use_variant=getattr(args, 'use_variant', False),
                    prune_n=prune_n,
                    prune_m=prune_m)
                subset[name].weight.data[W_mask] = 0
                actual_sparsity = W_mask.sum().item() / W_mask.numel()
                print(f"    actual sparsity: {actual_sparsity:.3f}")
                del gradient, W_metric, W_mask
    finally:
        model.config.use_cache = use_cache

    safe_empty_cache("after uniform metric pruning")
    print(f"Uniform metric pruning complete! metric={metric_name}")


def prune_single_layer_wsqrtg(args, model, tokenizer, device=torch.device("cuda:0"),
                              prune_n=0, prune_m=0):
    """Prune one transformer block with |W| * sqrt(|G|) for sensitivity analysis."""
    del tokenizer, device
    if args.gradient_path is None:
        raise ValueError("--gradient_path is required for layer-sensitivity-wsqrtg")
    target_layer = int(getattr(args, "target_layer", -1))

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    if target_layer < 0 or target_layer >= len(layers):
        raise ValueError(
            f"--target_layer must be in [0, {len(layers) - 1}], got {target_layer}"
        )

    print(f"loading gradients from {args.gradient_path}")
    gradients = torch.load(args.gradient_path, map_location=torch.device("cpu"))
    if not isinstance(gradients, dict):
        raise ValueError("The gradient file must contain a dictionary of layer tensors")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layer = layers[target_layer]
    subset = find_layers(layer)
    print("=" * 50)
    print("Single-layer sensitivity pruning")
    print(f"Target layer: {target_layer}")
    print(f"Local sparsity: {args.sparsity_ratio}")
    print("Metric: |W| * sqrt(|G|)")
    print("=" * 50)

    try:
        for name in subset:
            gradient, gradient_key = _get_v9_gradient(gradients, target_layer, name)
            print(f"  pruning {name}; gradient key: {gradient_key}")
            W_metric = build_v9_pruning_metric(
                "wsqrtg",
                subset[name].weight.data,
                gradient,
                scaler_row=None,
            )
            W_mask = build_wanda_mask(
                W_metric,
                args.sparsity_ratio,
                use_variant=getattr(args, "use_variant", False),
                prune_n=prune_n,
                prune_m=prune_m,
            )
            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    actual local sparsity: {actual_sparsity:.3f}")
            del gradient, W_metric, W_mask
    finally:
        model.config.use_cache = use_cache

    safe_empty_cache("after single-layer sensitivity pruning")
    print(f"Single-layer sensitivity pruning complete! target_layer={target_layer}")


def _prune_owl_v9(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                  pruning_metric="wanda"):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if pruning_metric not in V9_PRUNING_METRIC_LABELS:
        raise ValueError(f"Unknown OWL-V9 pruning metric: {pruning_metric}")

    gradients = None
    if pruning_metric != "wanda":
        if args.gradient_path is None:
            raise ValueError(f"--gradient_path is required for OWL-V9 metric '{pruning_metric}'")
        print(f"loading gradients from {args.gradient_path}")
        gradients = torch.load(args.gradient_path, map_location=torch.device('cpu'))
        if not isinstance(gradients, dict):
            raise ValueError("The gradient file must contain a dictionary of layer tensors")
        print(f"loaded {len(gradients)} gradient tensors")

    Hyper_m = getattr(args, 'Hyper_m', 5)
    Lamda = getattr(args, 'Lamda', 0.08)
    anchor_alpha = float(np.clip(getattr(args, 'Owl_alpha', 0.2), 0.0, 1.0))
    component = getattr(args, 'v9_component', 'full')
    core_start = int(getattr(args, 'v9_core_start', 4))
    core_end = int(getattr(args, 'v9_core_end', 10))
    tail_start = int(getattr(args, 'v9_tail_start', 16))
    tail_end = int(getattr(args, 'v9_tail_end', 30))
    projection_mode = getattr(args, 'v9_projection', 'protected')
    interval_mode = getattr(args, 'v9_interval_mode', 'absolute')
    reference_layers = int(getattr(args, 'v9_reference_layers', 32))
    eps = 1e-12

    print("=" * 50)
    print("OWL-V9 pruning starts (Core-Preserving Anchored OWL)")
    print(f"Stage-2 pruning metric: {pruning_metric} = {V9_PRUNING_METRIC_LABELS[pruning_metric]}")
    print(f"Hyper_m(outlier multiplier): {Hyper_m}, Lambda: {Lamda}, "
          f"anchor_alpha: {anchor_alpha}, core_layers: {core_start}-{core_end}, "
          f"tail_cap_layers: {tail_start}-{tail_end}, component: {component}, "
          f"projection: {projection_mode}, interval_mode: {interval_mode}")
    print("=" * 50)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    count_ratios = []
    severity_ratios = []
    severity_values = []
    tail_mass_ratios = []

    print("Stage 1: collecting V9 core-preserving OWL ratios...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_abs = torch.abs(subset[name].weight.data)
            X_l2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            outlier_score = W_abs * X_l2
            layer_wmetric.append(outlier_score.cpu())

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric]).float()
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_mask = layer_wmetric > threshold
        outlier_count = outlier_mask.sum().item()
        count_ratio = outlier_count / layer_wmetric.numel() * 100
        tail_mass_ratio = layer_wmetric[outlier_mask].sum() / (layer_wmetric.sum() + eps) * 100

        if outlier_count > 0:
            relative_excess = (layer_wmetric[outlier_mask] - threshold) / (threshold + eps)
            severity = torch.log1p(relative_excess.mean()).item()
            severity_ratio = count_ratio * (1.0 + severity)
        else:
            severity = 0.0
            severity_ratio = 0.0

        print(f"layer {i} V9 raw ratios: count={count_ratio:.2f}%, "
              f"severity_ratio={severity_ratio:.2f}% "
              f"(mean={mean_val.item():.6e}, threshold={threshold.item():.6e}, "
              f"severity={severity:.4f}, tail_mass={tail_mass_ratio.item():.2f}%)")
        count_ratios.append(count_ratio)
        severity_ratios.append(severity_ratio)
        severity_values.append(severity)
        tail_mass_ratios.append(tail_mass_ratio.item())
        if os.environ.get("DEBUG_CUDA_SYNC", "0") == "1" and torch.cuda.is_available():
            print(f"DEBUG_CUDA_SYNC: synchronizing after V9 stage1 layer {i}")
            synchronize_all_cuda_devices(f"stage1 layer {i}")

    safe_empty_cache("after V9 stage1 ratio collection")

    print("\nRaw V9 true OWL count ratios:", [f"{x:.2f}" for x in count_ratios])
    print("Raw V9 severity-weighted ratios:", [f"{x:.2f}" for x in severity_ratios])

    count_density, count_sparsity = owl_density_from_outlier_ratios(
        count_ratios, args.sparsity_ratio, Lamda)
    severity_density, severity_sparsity = owl_density_from_outlier_ratios(
        severity_ratios, args.sparsity_ratio, Lamda)

    anchored_density = (1.0 - anchor_alpha) * count_density + anchor_alpha * severity_density
    if component == "count":
        constrained_density = count_density.copy()
    elif component == "severity":
        constrained_density = severity_density.copy()
    else:
        constrained_density = anchored_density.copy()
    n_layers = constrained_density.size

    resolved_core_start, resolved_core_end, resolved_tail_start, resolved_tail_end = (
        resolve_osla_intervals(
            n_layers,
            core_start,
            core_end,
            tail_start,
            tail_end,
            mode=interval_mode,
            reference_layers=reference_layers,
        )
    )
    print(
        "Resolved OSLA intervals: "
        f"core={resolved_core_start}-{resolved_core_end}, "
        f"tail={resolved_tail_start}-{resolved_tail_end}, "
        f"layers={n_layers}"
    )

    core_mask = np.zeros(n_layers, dtype=bool)
    lo = resolved_core_start
    hi = resolved_core_end + 1
    use_core_constraint = component in {"core", "full"}
    core_changed_mask = np.zeros(n_layers, dtype=bool)
    if use_core_constraint:
        core_mask[lo:hi] = True
        core_before = constrained_density.copy()
        constrained_density[core_mask] = np.maximum(
            constrained_density[core_mask], count_density[core_mask])
        core_changed_mask = np.abs(constrained_density - core_before) > 1e-12

    tail_lo = resolved_tail_start
    tail_hi = resolved_tail_end + 1
    use_tail_constraint = component == "full"
    tail_mask = np.zeros(n_layers, dtype=bool)
    tail_changed_mask = np.zeros(n_layers, dtype=bool)
    if use_tail_constraint and tail_lo < tail_hi:
        tail_mask[tail_lo:tail_hi] = True
        tail_before = constrained_density.copy()
        constrained_density[tail_lo:tail_hi] = np.minimum(
            constrained_density[tail_lo:tail_hi], count_density[tail_lo:tail_hi])
        tail_changed_mask = np.abs(constrained_density - tail_before) > 1e-12

    density_before_projection = constrained_density.copy()
    all_layer_ratio, layer_sparsity_ratios = project_density_with_bounds(
        constrained_density,
        args.sparsity_ratio,
        Lamda,
        mode=projection_mode,
        protected_mask=core_mask if use_core_constraint else None,
    )
    target_density = 1.0 - float(args.sparsity_ratio)
    pre_projection_budget_error = float(np.mean(density_before_projection) - target_density)
    post_projection_budget_error = float(np.mean(all_layer_ratio) - target_density)
    projection_displacement = float(np.mean(np.abs(all_layer_ratio - density_before_projection)))

    print(f"Count-OWL sparsity - mean: {np.mean(count_sparsity):.3f}, "
          f"max: {np.max(count_sparsity):.3f}, min: {np.min(count_sparsity):.3f}")
    print(f"Severity-OWL sparsity - mean: {np.mean(severity_sparsity):.3f}, "
          f"max: {np.max(severity_sparsity):.3f}, min: {np.min(severity_sparsity):.3f}")
    print(f"Adjusted sparsity - mean: {np.mean(layer_sparsity_ratios):.3f}, "
          f"max: {np.max(layer_sparsity_ratios):.3f}, min: {np.min(layer_sparsity_ratios):.3f}")
    print("Count-OWL density ratios:", [f"{x:.3f}" for x in count_density])
    print("Severity-OWL density ratios:", [f"{x:.3f}" for x in severity_density])
    print("Anchored density ratios before constraints:", [f"{x:.3f}" for x in anchored_density])
    print(
        f"V9 component profile: {component}; "
        f"core_constraint={use_core_constraint}; tail_constraint={use_tail_constraint}; "
        f"projection={projection_mode}; interval_mode={interval_mode}"
    )
    print(
        "OSLA budget diagnostics: "
        f"pre_error={pre_projection_budget_error:+.8f}, "
        f"post_error={post_projection_budget_error:+.8f}, "
        f"mean_displacement={projection_displacement:.8f}, "
        f"active_core={int(core_changed_mask.sum())}, "
        f"active_tail={int(tail_changed_mask.sum())}"
    )
    print("Adjusted sparsity ratios:", [f"{x:.3f}" for x in layer_sparsity_ratios])
    print("Adjusted density ratios:", [f"{x:.3f}" for x in all_layer_ratio])

    diagnostic_payload = {
        "schema_version": 1,
        "status": "allocation_ready",
        "model": getattr(args, "model", ""),
        "pruning_method": getattr(args, "prune_method", ""),
        "pruning_metric": pruning_metric,
        "target_sparsity": float(args.sparsity_ratio),
        "target_density": target_density,
        "Hyper_m": float(Hyper_m),
        "Lamda": float(Lamda),
        "Owl_alpha": float(anchor_alpha),
        "component": component,
        "projection_mode": projection_mode,
        "interval_mode": interval_mode,
        "reference_layers": reference_layers,
        "num_layers": int(n_layers),
        "requested_intervals": {
            "core_start": core_start,
            "core_end": core_end,
            "tail_start": tail_start,
            "tail_end": tail_end,
        },
        "resolved_intervals": {
            "core_start": resolved_core_start,
            "core_end": resolved_core_end,
            "tail_start": resolved_tail_start,
            "tail_end": resolved_tail_end,
        },
        "constraints": {
            "core_enabled": use_core_constraint,
            "tail_enabled": use_tail_constraint,
            "core_mask": core_mask.tolist(),
            "tail_mask": tail_mask.tolist(),
            "core_changed_mask": core_changed_mask.tolist(),
            "tail_changed_mask": tail_changed_mask.tolist(),
            "active_core_layers": int(core_changed_mask.sum()),
            "active_tail_layers": int(tail_changed_mask.sum()),
        },
        "budget": {
            "pre_projection_error": pre_projection_budget_error,
            "post_projection_error": post_projection_budget_error,
            "mean_projection_displacement": projection_displacement,
            "pre_projection_density_sum": float(density_before_projection.sum()),
            "post_projection_density_sum": float(all_layer_ratio.sum()),
            "target_density_sum": float(target_density * n_layers),
        },
        "layers": [
            {
                "layer": int(layer_index),
                "count_ratio": float(count_ratios[layer_index]),
                "severity_ratio": float(severity_ratios[layer_index]),
                "severity": float(severity_values[layer_index]),
                "tail_mass_ratio": float(tail_mass_ratios[layer_index]),
                "count_density": float(count_density[layer_index]),
                "severity_density": float(severity_density[layer_index]),
                "anchored_density": float(anchored_density[layer_index]),
                "density_before_projection": float(density_before_projection[layer_index]),
                "target_density_after_projection": float(all_layer_ratio[layer_index]),
                "target_sparsity_after_projection": float(layer_sparsity_ratios[layer_index]),
                "in_core_interval": bool(core_mask[layer_index]),
                "in_tail_interval": bool(tail_mask[layer_index]),
                "core_constraint_active": bool(core_changed_mask[layer_index]),
                "tail_constraint_active": bool(tail_changed_mask[layer_index]),
            }
            for layer_index in range(n_layers)
        ],
    }
    write_osla_diagnostics(args, diagnostic_payload)

    # Stage 2 rebuilds calibration activations. Release Stage 1 tensors first to
    # avoid doubling multi-GB activation buffers on 13B/30B models.
    del inps, outs, attention_mask, position_ids
    gc.collect()
    safe_empty_cache("before V9 stage2 calibration")

    print(f"\nStage 2: applying {V9_PRUNING_METRIC_LABELS[pruning_metric]} "
          "with V9 OWL layer sparsity...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
        layer_key_template = "model.decoder.layers.{}"
    else:
        layers = model.model.layers
        layer_key_template = "model.layers.{}"

    actual_layer_sparsities = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = float(np.clip(layer_sparsity_ratio, 0.0, 0.99))
        print(f"\nlayer {i}: density={all_layer_ratio[i]:.3f}, sparsity={layer_sparsity_ratio:.3f}")

        if hasattr(model, 'hf_device_map') and layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        layer_pruned_weights = 0
        layer_total_weights = 0
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"  pruning {name}")
            gradient = None
            gradient_key = None
            if gradients is not None:
                gradient, gradient_key = _get_v9_gradient(gradients, i, name)
            W_metric = build_v9_pruning_metric(
                pruning_metric,
                subset[name].weight.data,
                gradient,
                wrapped_layers[name].scaler_row,
                gradient_beta=getattr(args, 'Grad_beta', 0.5),
            )
            if gradient_key is not None:
                print(f"    gradient key: {gradient_key}")
            W_mask = build_wanda_mask(
                W_metric,
                layer_sparsity_ratio,
                use_variant=getattr(args, 'use_variant', False),
                prune_n=prune_n,
                prune_m=prune_m)

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            layer_pruned_weights += int(W_mask.sum().item())
            layer_total_weights += int(W_mask.numel())
            print(f"    actual sparsity: {actual_sparsity:.3f}")

        actual_layer_sparsity = (
            layer_pruned_weights / layer_total_weights
            if layer_total_weights else 0.0
        )
        actual_layer_sparsities.append(actual_layer_sparsity)
        print(f"  layer {i} aggregate actual sparsity: {actual_layer_sparsity:.6f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                    position_ids=position_ids)[0]
        inps, outs = outs, inps
        if os.environ.get("DEBUG_CUDA_SYNC", "0") == "1" and torch.cuda.is_available():
            print(f"DEBUG_CUDA_SYNC: synchronizing after V9 stage2 layer {i}")
            synchronize_all_cuda_devices(f"stage2 layer {i}")

    model.config.use_cache = use_cache
    diagnostic_payload["status"] = "pruning_complete"
    diagnostic_payload["actual_layer_sparsities"] = [
        float(value) for value in actual_layer_sparsities
    ]
    for layer_index, actual_sparsity in enumerate(actual_layer_sparsities):
        diagnostic_payload["layers"][layer_index]["actual_sparsity"] = float(actual_sparsity)
        diagnostic_payload["layers"][layer_index]["actual_density"] = float(1.0 - actual_sparsity)
    write_osla_diagnostics(args, diagnostic_payload)
    safe_empty_cache("after OWL-V9 pruning")
    print(f"OWL-V9 pruning complete! metric={pruning_metric}")


def prune_wanda_owl_v9(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    return _prune_owl_v9(
        args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
        pruning_metric="wanda")


def prune_metric_owl_v9(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                        metric_name=None):
    if metric_name is None:
        raise ValueError("metric_name must be provided for prune_metric_owl_v9")
    return _prune_owl_v9(
        args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
        pruning_metric=metric_name)


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    print('Starting ...')
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        subset = find_layers(layer)
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    print('Starting ...')
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "OPT" in model.__class__.__name__:
                cache['position_ids'] = None
            else:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        subset = find_layers(layer)
        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name], gradient_path=args.gradient_path)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_prunerzero_seq":
                indexed_name = f'{name}_layer_{i}'
                prune_mask = gpts[name].get_prunerzero_mask(args.sparsity_ratio, prune_n, prune_m, indexed_name)
            elif "iter" in args.prune_method:
                prune_mask = None

            indexed_name = f'{name}_layer_{i}'
            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m,
                                   percdamp=0.01, blocksize=128, indexed_name=indexed_name)
            gpts[name].free()

        for j in range(args.nsamples):
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_pruner_zero(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, engine=None):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if args.gradient_path is None:
        raise ValueError("gradient_path参数未指定，请提供有效路径")

    try:
        with open(args.gradient_path, 'rb') as file:
            gradients = torch.load(file, map_location=torch.device('cpu'))
    except Exception as e:
        raise RuntimeError(f"加载梯度文件时发生错误: {e}")

    print("加载校准数据")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("数据集加载完成")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device,
                                                                             nsamples=args.nsamples)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            indexed_name = f'{name}_layer_{i}'
            print(f"正在修剪第{i}层，名称: {name}")
            W = torch.abs(subset[name].weight.data)
            X = wrapped_layers[name].scaler_row.reshape((1, -1))
            G = gradients.get(indexed_name, None)

            if G is None:
                raise ValueError(f"没有找到梯度信息: {indexed_name}")

            W_metric = engine.forward(
                W.to(dtype=torch.float32),
                G.to(device=W.device, dtype=torch.float32),
                X.to(device=W.device, dtype=torch.float32),
            )
            assert W_metric is not None

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)
                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (
                            alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"找到的alpha: {alpha} 稀疏度: {cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_TSP(args, model, tokenizer, device=torch.device("cuda:0"),
                          prune_n=0, prune_m=0, engine=None):
    """OWL with correct outlier calculation (using Wanda metric)"""
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("=" * 50)
    print("OWL剪枝开始（使用正确的离群值计算）")
    print("=" * 50)

    print("第一阶段：计算LOD（层级异常值分布）...")

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers
    all_layer_ratio = []

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W = torch.abs(subset[name].weight.data)
            X = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            outlier_score = W * X
            layer_wmetric.append(outlier_score.cpu())

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric])
        Hyper_m = getattr(args, 'Hyper_m', 5)
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_ratio = ((layer_wmetric > threshold).sum().item() / layer_wmetric.numel() * 100)

        print(f"层 {i} 异常值比例: {outlier_ratio:.2f}%")
        all_layer_ratio.append(outlier_ratio)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("\n调整前的异常值比例:", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)
    Lamda = getattr(args, 'Lamda', 0.08)

    print(f"Hyper_m: {Hyper_m}, Lambda: {Lamda}")

    if all_layer_ratio.max() > all_layer_ratio.min():
        all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) *
                           (1 / (all_layer_ratio.max() - all_layer_ratio.min()) * Lamda * 2))
    else:
        all_layer_ratio = np.ones_like(all_layer_ratio) * Lamda

    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    all_layer_ratio = np.clip(all_layer_ratio, 0.01, 0.99)

    print(f"调整后 - 均值: {np.mean(all_layer_ratio):.3f}, "
          f"最大: {np.max(all_layer_ratio):.3f}, 最小: {np.min(all_layer_ratio):.3f}")
    print("调整后的密度比例:", [f"{x:.3f}" for x in all_layer_ratio])

    print("\n第三阶段：开始剪枝...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = np.clip(layer_sparsity_ratio, 0.0, 0.99)

        print(f"\n层 {i}: 密度={all_layer_ratio[i]:.3f}, 稀疏率={layer_sparsity_ratio:.3f}")

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = safe_to_device(attention_mask, dev)
            position_ids = safe_to_device(position_ids, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            indexed_name = f'{name}_layer_{i}'
            print(f"  正在修剪 {name}")

            W = torch.abs(subset[name].weight.data)
            X = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            if engine is not None and args.gradient_path is not None:
                G = gradients.get(indexed_name, None) if 'gradients' in locals() else None
                if G is None:
                    try:
                        with open(args.gradient_path, 'rb') as file:
                            gradients = torch.load(file, map_location=torch.device('cpu'))
                        G = gradients.get(indexed_name, torch.ones_like(W))
                    except:
                        G = torch.ones_like(W)

                W_metric = engine.forward(
                    W.to(dtype=torch.float32),
                    G.to(device=W.device, dtype=torch.float32),
                    X.to(device=W.device, dtype=torch.float32),
                )
            else:
                W_metric = W * X

            W_mask = (torch.zeros_like(W_metric) == 1)

            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                if layer_sparsity_ratio > 0:
                    if args.use_variant:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)

                        while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and \
                                (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > layer_sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha
                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    else:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)
                        indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    实际稀疏率: {actual_sparsity:.3f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("OWL剪枝完成！")
    print("=" * 50)
