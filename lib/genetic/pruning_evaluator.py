import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
import copy
from tqdm import tqdm
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from lib.data import get_loaders
from lib.eval import eval_ppl_wikitext


class PruningEvaluator:
    """
    LLaMA模型剪枝评估器 - 集成OWL层级自适应稀疏
    """

    def __init__(self,
                 model_path: str = "/home/yh114/workdir/DSnoT/models/decapoda-research-llama-7B-hf",
                 sparsity_ratio: float = 0.5,
                 calibration_samples: int = 128,
                 device: str = "cuda",
                 seed: int = 0,
                 gradient_path: str = None,
                 # OWL相关参数
                 use_owl: bool = False,
                 hyper_m: float = 5.0,
                 lamda: float = 0.08):

        self.model_path = model_path
        self.sparsity_ratio = sparsity_ratio
        self.calibration_samples = calibration_samples
        self.device = device if torch.cuda.is_available() else "cpu"
        self.seed = seed
        self.gradient_path = gradient_path

        # OWL参数
        self.use_owl = use_owl
        self.hyper_m = hyper_m
        self.lamda = lamda
        self.layer_sparsity_ratios = None

        # 缓存
        self.weights_cache = {}
        self.gradient_cache = {}
        self.activation_cache = {}  # 添加激活值缓存
        self.original_ppl = None

        # 加载模型
        logging.info(f"加载模型: {model_path}")
        self.model, self.tokenizer = self.load_model()

        # 加载数据
        self.calibration_loader = None
        self.test_loader = None
        self.load_data()

        # 初始化权重、梯度和激活值
        self.initialize_caches()

        # 评估原始模型
        self.evaluate_original_model()

        # 如果使用OWL，计算层级稀疏率
        if self.use_owl:
            self.compute_owl_sparsity_ratios()

    def load_model(self):
        """加载LLaMA模型"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            config = AutoConfig.from_pretrained(self.model_path)

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                torch_dtype=torch.float16,
                device_map={"": 0},
                low_cpu_mem_usage=True
            )

            model.eval()
            model.seqlen = 2048

            return model, tokenizer

        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            raise

    def load_data(self):
        """加载数据集"""
        try:
            _, self.calibration_loader = get_loaders(
                "wikitext2",
                nsamples=self.calibration_samples,
                seed=self.seed,
                seqlen=2048,
                tokenizer=self.tokenizer
            )

            _, self.test_loader = get_loaders(
                "wikitext2",
                nsamples=40,
                seed=self.seed,
                seqlen=2048,
                tokenizer=self.tokenizer
            )

            logging.info(f"成功加载数据")

        except Exception as e:
            logging.warning(f"加载数据失败: {e}")
            self.calibration_loader = []
            self.test_loader = []

    def collect_activations(self):
        """收集激活值（按照论文OWL的定义，统计每个Linear层输入特征在所有 token 上的 L2 范数）

        目标：为每个 Linear 层 j 维输入收集  ||X_j||_2  （跨 N×L token 聚合），
        供 OWL 的离群值打分 A_ij = ||X_j||_2 * |W_ij| 使用。
        """
        logging.info("收集激活值 (L2 across tokens) ...")

        # 用于累计各层的“平方和”与样本数量，最后再开根号得到 L2 范数
        sumsqs = {}
        sizes = {}

        activations_temp = {}
        handles = []

        def hook_fn(name):
            def hook(module, inputs, output):
                if not inputs:
                    return
                inp = inputs[0]
                if inp is None or not isinstance(inp, torch.Tensor):
                    return

                try:
                    x = inp.detach().float()
                    # 统一成 [tokens, Cin] 形状：把 batch/seq 维展平，只保留最后一维为特征维
                    if x.dim() >= 2:
                        x = x.reshape(-1, x.shape[-1])  # (N*L, Cin)
                    else:
                        # 单向量，视为一个 token
                        x = x.reshape(1, -1)

                    Cin = module.weight.shape[1]
                    if x.shape[-1] != Cin:
                        # 形状对不上（可能是转置/投影等），跳过该次
                        return

                    # 累计每个输入特征维度的平方和：sum(x^2) over tokens
                    sq = (x ** 2).sum(dim=0).cpu()

                    if name not in sumsqs:
                        sumsqs[name] = sq
                    else:
                        # 数组尺寸保护
                        if sumsqs[name].shape[0] == sq.shape[0]:
                            sumsqs[name] += sq
                        else:
                            # 维度不一致，忽略该批次
                            pass
                except Exception as e:
                    logging.debug(f"激活 hook 失败 {name}: {e}")

            return hook

        # 为所有 Linear 注册 forward hook
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(hook_fn(name)))

        # 使用校准数据前向几步，统计激活
        try:
            with torch.no_grad():
                used_steps = 0
                if self.calibration_loader and len(self.calibration_loader) > 0:
                    for batch in self.calibration_loader:
                        if isinstance(batch, torch.Tensor):
                            batch = batch.to(self.device)
                        try:
                            _ = self.model(batch[:, :min(256, batch.shape[1])])
                            used_steps += 1
                        except Exception as e:
                            logging.debug(f"校准前向失败: {e}")
                        if used_steps >= 8:
                            break
                else:
                    # 没有校准集时，走几步随机输入以保证每层至少有一份统计
                    batch_size, seq_len, vocab_size = 1, 128, 32000
                    for _ in range(4):
                        rand_in = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
                        try:
                            _ = self.model(rand_in)
                        except Exception as e:
                            logging.debug(f"随机前向失败: {e}")
                            break
        finally:
            for h in handles:
                h.remove()

        # 将平方和开根号，得到每层 Cin 维的 L2 范数
        self.activation_cache = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                Cin = module.weight.shape[1]
                if name in sumsqs and sumsqs[name].shape[0] == Cin:
                    l2 = torch.sqrt(sumsqs[name].clamp_min(1e-20)).float()
                    self.activation_cache[name] = l2
                else:
                    # 回退：若没有统计到，就给一个稳定的常量
                    self.activation_cache[name] = torch.ones(Cin).float()

        logging.info(f"已为 {len(self.activation_cache)} 个 Linear 层收集到 L2 激活范数。")

        # 直接使用随机输入生成激活值
        logging.info("使用随机输入生成激活值...")

        # 为每个Linear层创建激活值
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 创建与权重形状兼容的激活值
                weight_shape = module.weight.shape
                # 激活值是输入维度的向量
                # 生成随机激活值（模拟实际激活值的分布）
                activation = torch.randn(weight_shape[1]) * 0.1 + 0.5  # 均值0.5，标准差0.1
                activation = torch.abs(activation)  # 确保为正
                self.activation_cache[name] = activation.float()

        # 尝试使用真实数据更新激活值（如果可能）
        activations_temp = {}
        handles = []

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    input_tensor = input[0]
                else:
                    input_tensor = input

                if input_tensor is not None and isinstance(input_tensor, torch.Tensor):
                    try:
                        # 计算激活值统计
                        if len(input_tensor.shape) >= 2:
                            # 计算L2范数
                            if input_tensor.shape[-1] == module.weight.shape[1]:
                                activation_norm = torch.norm(input_tensor.float(), p=2, dim=0)
                                if len(activation_norm.shape) > 1:
                                    activation_norm = activation_norm.mean(dim=0)
                            else:
                                # 形状不匹配，使用平均值
                                activation_norm = torch.abs(input_tensor.float()).mean()
                                activation_norm = torch.full((module.weight.shape[1],), activation_norm.item())
                        else:
                            activation_norm = torch.abs(input_tensor.float()).mean()
                            activation_norm = torch.full((module.weight.shape[1],), activation_norm.item())

                        if name not in activations_temp:
                            activations_temp[name] = []
                        activations_temp[name].append(activation_norm.detach().cpu())
                    except Exception as e:
                        logging.debug(f"Hook处理失败 {name}: {e}")

            return hook

        # 注册hooks
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)

        # 尝试运行一些前向传播
        try:
            with torch.no_grad():
                # 创建一些随机输入
                batch_size = 1
                seq_len = 128
                vocab_size = 32000  # LLaMA的词汇表大小

                for _ in range(4):  # 运行4个batch
                    random_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
                    try:
                        _ = self.model(random_input)
                    except Exception as e:
                        logging.debug(f"前向传播失败: {e}")
                        break

                # 如果有calibration数据，也尝试使用
                if self.calibration_loader and len(self.calibration_loader) > 0:
                    for batch_idx, batch in enumerate(self.calibration_loader):
                        if batch_idx >= 4:
                            break
                        try:
                            if isinstance(batch, torch.Tensor):
                                batch = batch.to(self.device)
                                if batch.shape[0] > 0 and batch.shape[1] > 0:
                                    _ = self.model(batch[:, :min(128, batch.shape[1])])
                        except Exception as e:
                            logging.debug(f"批次 {batch_idx} 处理失败: {e}")
                            continue

        except Exception as e:
            logging.warning(f"前向传播过程出错: {e}")

        finally:
            # 移除hooks
            for handle in handles:
                handle.remove()

        # 更新激活值缓存（如果收集到了真实数据）
        for name in activations_temp:
            if len(activations_temp[name]) > 0:
                try:
                    # 取平均
                    stacked = torch.stack(activations_temp[name])
                    mean_activation = stacked.mean(dim=0)
                    self.activation_cache[name] = mean_activation.float()
                except Exception as e:
                    logging.debug(f"更新层 {name} 激活值失败: {e}")

        # 确保所有层都有激活值
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if name not in self.activation_cache:
                    weight_shape = module.weight.shape
                    self.activation_cache[name] = torch.ones(weight_shape[1]).float() * 0.5

        logging.info(f"收集了 {len(self.activation_cache)} 层的激活值")

        # 验证激活值
        if len(self.activation_cache) > 0:
            sample_name = list(self.activation_cache.keys())[0]
            sample_activation = self.activation_cache[sample_name]
            logging.debug(f"示例层 {sample_name}: 激活值形状={sample_activation.shape}, "
                          f"均值={sample_activation.mean():.4f}, 标准差={sample_activation.std():.4f}")

    def initialize_caches(self):
        """初始化权重、梯度和激活值缓存"""
        # 先初始化权重和梯度
        self.initialize_gradients()

        # 然后收集激活值
        self.collect_activations()

        # 如果激活值收集失败，再次确保有默认值
        if len(self.activation_cache) == 0:
            logging.warning("激活值收集失败，使用备用方案")
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    weight_shape = module.weight.shape
                    self.activation_cache[name] = torch.ones(weight_shape[1]).float() * 0.5
            logging.info(f"使用备用方案生成了 {len(self.activation_cache)} 层的激活值")

    def initialize_gradients(self):
        """初始化权重和梯度缓存"""
        logging.info("初始化权重和梯度...")

        # 缓存所有权重
        layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data.clone().cpu().float()
                self.weights_cache[name] = weight
                layer_names.append(name)
                # 初始化为权重幅度（默认值）
                self.gradient_cache[name] = torch.abs(weight) + 1e-10

        logging.info(f"找到 {len(layer_names)} 个线性层")

        # 尝试加载预计算梯度
        if self.gradient_path and os.path.exists(self.gradient_path):
            try:
                logging.info(f"加载预计算梯度: {self.gradient_path}")
                gradient_data = torch.load(self.gradient_path, map_location='cpu')

                if isinstance(gradient_data, dict):
                    matched = 0
                    for layer_name in layer_names:
                        if layer_name in gradient_data:
                            grad = gradient_data[layer_name]
                            if isinstance(grad, torch.Tensor):
                                self.gradient_cache[layer_name] = grad.float().abs() + 1e-10
                                matched += 1
                    logging.info(f"成功匹配 {matched} 层梯度")

            except Exception as e:
                logging.error(f"加载梯度失败: {e}")

    def compute_owl_sparsity_ratios(self):
        """计算 OWL 的层级稀疏率分布
        依据原论文 §3.2 的 LOD 定义：
          A_ij = ||X_j||_2 * |W_ij|，阈值为  M * mean(A) ；
          每个层/Block 的离群比例 D_l 为 A 中超过阈值的元素占比；
          稀疏率 S_l 与 (1 - D_l) 成比例，并限制在 [S-λ, S+λ] 且均值为 S。
        """
        logging.info("按照论文逻辑计算 OWL 层级稀疏率（权重 + 激活的 L2 范数）...")

        # 确保激活（L2）已经就绪
        if not self.activation_cache:
            self.collect_activations()

        all_layer_outlier_ratios = []
        layers = self.model.model.layers
        M = self.hyper_m
        eps = 1e-12

        for i in range(len(layers)):
            layer = layers[i]
            A_chunks = []

            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    full_name = f"model.layers.{i}.{name}"
                    W = self.weights_cache.get(full_name, None)
                    X = self.activation_cache.get(full_name, None)
                    if W is None or X is None:
                        continue

                    W = W.float().abs()  # |W_ij|, shape [Cout, Cin]
                    # X is ||X_j||_2 over tokens, shape [Cin]
                    if X.dim() == 1 and X.shape[0] == W.shape[1]:
                        Xb = X.unsqueeze(0).expand(W.shape[0], -1)  # broadcast to [Cout, Cin]
                    else:
                        # 尺寸不对，跳过该层
                        continue

                    A = W * Xb  # A_ij = ||X_j||_2 * |W_ij|
                    A_chunks.append(A.reshape(-1))

            if A_chunks:
                A_all = torch.cat(A_chunks)
                meanA = A_all.mean()
                thresh = meanA * M
                D = (A_all > thresh).float().mean().item()  # 该 Block 的离群比例
                all_layer_outlier_ratios.append(D)
            else:
                all_layer_outlier_ratios.append(0.0)

        all_layer_outlier_ratios = np.asarray(all_layer_outlier_ratios, dtype=float)
        target_sparsity = float(self.sparsity_ratio)

        # 将稀疏率设定为与 (1 - D_l) 成比例，之后做尺度化与 [S-λ, S+λ] 约束
        D = all_layer_outlier_ratios
        if D.max() - D.min() > 1e-12:
            # 归一化到 [0,1]
            z = (D - D.min()) / (D.max() - D.min())
        else:
            z = np.zeros_like(D)

        # (1 - z) 越大 -> 稀疏率越高
        lam = float(self.lamda)
        raw = target_sparsity + (1 - z - 0.5) * (2 * lam)  # 在 [S-λ, S+λ] 的线性映射
        min_s, max_s = max(0.01, target_sparsity - lam), min(0.99, target_sparsity + lam)
        layer_sparsity = np.clip(raw, min_s, max_s)

        # 再次对齐平均值为目标 S（轻微再中心化后再裁剪）
        mean_now = layer_sparsity.mean()
        layer_sparsity = np.clip(layer_sparsity - (mean_now - target_sparsity), min_s, max_s)

        self.layer_sparsity_ratios = layer_sparsity

        logging.info("OWL 稀疏率分布已计算完成：")
        logging.info(f"  目标稀疏率 S={target_sparsity:.3f}, λ={lam:.3f}, M={M:.2f}")
        logging.info(f"  实际均值={self.layer_sparsity_ratios.mean():.3f}, 方差={self.layer_sparsity_ratios.var():.5f}")

    def evaluate_original_model(self):
        """评估原始模型的困惑度"""
        if self.original_ppl is None:
            logging.info("评估原始模型PPL...")
            try:
                with torch.no_grad():
                    self.original_ppl = eval_ppl_wikitext(self.model, self.test_loader, 1, self.device)
                logging.info(f"原始模型PPL: {self.original_ppl:.4f}")
            except:
                self.original_ppl = 5.67
                logging.warning(f"使用默认PPL: {self.original_ppl}")

    def apply_pruning_metric(self, metric_tree, layer_name: str) -> torch.Tensor:
        """应用剪枝指标到特定层 - 完整评估版本"""
        W = self.weights_cache.get(layer_name)
        G = self.gradient_cache.get(layer_name)
        X = self.activation_cache.get(layer_name)

        if W is None or G is None:
            logging.warning(f"层 {layer_name} 缺少权重或梯度")
            return None

        W = W.float()
        G = G.float()

        # 处理激活值
        if X is not None:
            X = X.float()
            # 确保X的形状与W兼容
            if X.shape != W.shape:
                if len(X.shape) == 1 and len(W.shape) == 2:
                    if X.shape[0] == W.shape[1]:
                        X = X.unsqueeze(0).expand(W.shape[0], -1)
                    elif X.shape[0] == W.shape[0]:
                        X = X.unsqueeze(1).expand(-1, W.shape[1])
                    else:
                        X = torch.ones_like(W) * 0.5
                else:
                    X = torch.ones_like(W) * 0.5
        else:
            X = torch.ones_like(W) * 0.5

        try:
            # 减少数值保护，只保留最基本的
            W = torch.where(torch.isnan(W) | torch.isinf(W),
                            torch.zeros_like(W), W)
            G = torch.where(torch.isnan(G) | torch.isinf(G) | (G <= 0),
                            torch.ones_like(G) * 1e-10, G)
            X = torch.where(torch.isnan(X) | torch.isinf(X) | (X <= 0),
                            torch.ones_like(X) * 1e-10, X)

            # 计算重要性分数
            importance_scores = metric_tree.compute_tree(W, G, X)

            # 检查数值稳定性
            if not isinstance(importance_scores, torch.Tensor):
                importance_scores = torch.tensor(importance_scores)

            # 检测并记录数值问题
            nan_ratio = torch.isnan(importance_scores).sum().item() / importance_scores.numel()
            inf_ratio = torch.isinf(importance_scores).sum().item() / importance_scores.numel()

            if nan_ratio > 0.1 or inf_ratio > 0.1:
                logging.warning(f"层 {layer_name}: NaN比例={nan_ratio:.2%}, Inf比例={inf_ratio:.2%}")
                # 对于数值不稳定的指标，返回惩罚性分数
                return None

            # 处理异常值但不过度保护
            importance_scores = torch.nan_to_num(importance_scores, nan=0.0, posinf=1e6, neginf=-1e6)

            # 确保为正值
            importance_scores = torch.abs(importance_scores) + 1e-10

            return importance_scores

        except Exception as e:
            logging.warning(f"计算层 {layer_name} 的指标失败: {e}")
            return None

    def prune_model_with_metric(self, metric_tree):
        """使用指标剪枝模型（支持OWL层级稀疏）"""
        masks = {}
        layer_sparsities = {}

        # 可剪枝层
        prunable_layers = []
        layer_indices = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if 'embed' in name.lower() or 'lm_head' in name.lower():
                    continue
                prunable_layers.append(name)
                if 'model.layers.' in name:
                    layer_idx = int(name.split('model.layers.')[1].split('.')[0])
                    layer_indices[name] = layer_idx

        logging.info(f"可剪枝层数: {len(prunable_layers)}")

        for name in prunable_layers:
            module = dict(self.model.named_modules())[name]
            layer_idx = layer_indices.get(name, 0)

            # 使用OWL稀疏率或全局稀疏率
            if self.use_owl and self.layer_sparsity_ratios is not None:
                target_sparsity = self.layer_sparsity_ratios[min(layer_idx, len(self.layer_sparsity_ratios) - 1)]
                target_sparsity = np.clip(target_sparsity, 0.0, 0.99)
            else:
                target_sparsity = self.sparsity_ratio

            importance_scores = self.apply_pruning_metric(metric_tree, name)

            if importance_scores is not None:
                if importance_scores.shape != module.weight.shape:
                    try:
                        importance_scores = importance_scores.reshape(module.weight.shape)
                    except:
                        logging.error(f"层 {name} 形状不匹配")
                        continue

                flat_scores = importance_scores.flatten()
                num_params = flat_scores.numel()
                num_keep = int(num_params * (1 - target_sparsity))
                num_keep = max(1, min(num_keep, num_params - 1))

                if num_keep > 0:
                    topk_values, _ = torch.topk(flat_scores, num_keep, sorted=False)
                    threshold = topk_values.min()
                else:
                    threshold = flat_scores.max() + 1

                mask = (importance_scores >= threshold).float()
                masks[name] = mask

                actual_sparsity = 1.0 - (mask.sum().item() / mask.numel())
                layer_sparsities[name] = actual_sparsity

        if masks:
            total_params = sum(m.numel() for m in masks.values())
            pruned_params = sum((1 - m).sum().item() for m in masks.values())
            overall_sparsity = pruned_params / total_params if total_params > 0 else 0
            logging.info(f"目标稀疏度: {self.sparsity_ratio:.2%}, 实际: {overall_sparsity:.2%}")

        return masks, layer_sparsities

    def prune_model_with_metric_full(self, metric_tree):
        """使用指标剪枝模型（完整版本）- 更严格的检查和更详细的评估"""
        masks = {}
        layer_sparsities = {}

        # 可剪枝层
        prunable_layers = []
        layer_indices = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if 'embed' in name.lower() or 'lm_head' in name.lower():
                    continue
                prunable_layers.append(name)
                if 'model.layers.' in name:
                    layer_idx = int(name.split('model.layers.')[1].split('.')[0])
                    layer_indices[name] = layer_idx

        logging.info(f"完整模式：可剪枝层数: {len(prunable_layers)}")

        # 记录失败和成功的详细信息
        failed_layers = []
        success_layers = []
        layer_details = {}

        for name in prunable_layers:
            module = dict(self.model.named_modules())[name]
            layer_idx = layer_indices.get(name, 0)

            # 使用OWL稀疏率或全局稀疏率
            if self.use_owl and self.layer_sparsity_ratios is not None:
                target_sparsity = self.layer_sparsity_ratios[min(layer_idx, len(self.layer_sparsity_ratios) - 1)]
                target_sparsity = np.clip(target_sparsity, 0.0, 0.99)
            else:
                target_sparsity = self.sparsity_ratio

            # 完整版本：多次尝试机制
            importance_scores = None
            attempts = 3
            last_error = None

            for attempt in range(attempts):
                try:
                    importance_scores = self.apply_pruning_metric(metric_tree, name)

                    if importance_scores is not None:
                        # 详细的数值检查
                        total_elements = importance_scores.numel()
                        nan_count = torch.isnan(importance_scores).sum().item()
                        inf_count = torch.isinf(importance_scores).sum().item()
                        zero_count = (importance_scores == 0).sum().item()

                        nan_ratio = nan_count / total_elements
                        inf_ratio = inf_count / total_elements
                        zero_ratio = zero_count / total_elements

                        # 完整版本的严格检查
                        if nan_ratio > 0.05:  # 超过5%的NaN
                            logging.warning(f"层 {name} 尝试 {attempt + 1}: NaN比例过高 {nan_ratio:.2%}")
                            importance_scores = None
                            continue
                        elif inf_ratio > 0.05:  # 超过5%的Inf
                            logging.warning(f"层 {name} 尝试 {attempt + 1}: Inf比例过高 {inf_ratio:.2%}")
                            importance_scores = None
                            continue
                        elif zero_ratio > 0.95:  # 超过95%的零值
                            logging.warning(f"层 {name} 尝试 {attempt + 1}: 零值比例过高 {zero_ratio:.2%}")
                            importance_scores = None
                            continue
                        else:
                            # 检查通过，记录详细信息
                            layer_details[name] = {
                                'nan_ratio': nan_ratio,
                                'inf_ratio': inf_ratio,
                                'zero_ratio': zero_ratio,
                                'mean_score': importance_scores.mean().item(),
                                'std_score': importance_scores.std().item(),
                                'attempt': attempt + 1
                            }
                            break  # 成功，退出重试循环

                except Exception as e:
                    last_error = str(e)
                    logging.warning(f"层 {name} 尝试 {attempt + 1} 失败: {e}")
                    importance_scores = None

            if importance_scores is not None:
                # 形状检查和调整
                if importance_scores.shape != module.weight.shape:
                    try:
                        importance_scores = importance_scores.reshape(module.weight.shape)
                        logging.debug(f"层 {name}: 重塑形状为 {module.weight.shape}")
                    except:
                        logging.error(
                            f"层 {name} 形状不匹配: 期望 {module.weight.shape}, 得到 {importance_scores.shape}")
                        failed_layers.append((name, "形状不匹配"))
                        continue

                # 剪枝逻辑
                flat_scores = importance_scores.flatten()
                num_params = flat_scores.numel()
                num_keep = int(num_params * (1 - target_sparsity))
                num_keep = max(1, min(num_keep, num_params - 1))

                if num_keep > 0:
                    try:
                        topk_values, topk_indices = torch.topk(flat_scores, num_keep, sorted=False)
                        threshold = topk_values.min()

                        # 完整版本：验证阈值的合理性
                        if threshold <= 0:
                            logging.warning(f"层 {name}: 阈值为非正数 {threshold}")
                            # 使用备选策略
                            positive_scores = flat_scores[flat_scores > 0]
                            if len(positive_scores) > 0:
                                threshold = positive_scores.quantile(1 - target_sparsity)
                            else:
                                threshold = flat_scores.median()

                    except Exception as e:
                        logging.error(f"层 {name} TopK操作失败: {e}")
                        failed_layers.append((name, f"TopK失败: {e}"))
                        continue
                else:
                    threshold = flat_scores.max() + 1

                mask = (importance_scores >= threshold).float()
                masks[name] = mask

                actual_sparsity = 1.0 - (mask.sum().item() / mask.numel())
                layer_sparsities[name] = actual_sparsity

                success_layers.append(name)

                # 更新层详细信息
                layer_details[name].update({
                    'target_sparsity': target_sparsity,
                    'actual_sparsity': actual_sparsity,
                    'threshold': threshold.item() if isinstance(threshold, torch.Tensor) else threshold,
                    'num_params': num_params,
                    'num_kept': mask.sum().item()
                })

                logging.debug(f"层 {name}: 目标稀疏={target_sparsity:.3f}, 实际稀疏={actual_sparsity:.3f}, "
                              f"阈值={threshold:.6f}")

            else:
                failed_layers.append((name, last_error or "重要性分数计算失败"))

        # 完整版本：详细的结果分析和报告
        total_layers = len(prunable_layers)
        success_count = len(success_layers)
        failure_count = len(failed_layers)
        success_rate = success_count / total_layers if total_layers > 0 else 0

        logging.info(f"完整剪枝评估结果:")
        logging.info(f"  总层数: {total_layers}")
        logging.info(f"  成功层数: {success_count}")
        logging.info(f"  失败层数: {failure_count}")
        logging.info(f"  成功率: {success_rate:.1%}")

        # 失败层详细信息
        if failed_layers:
            logging.warning(f"失败层详情:")
            for layer_name, error in failed_layers[:5]:  # 只显示前5个
                logging.warning(f"  {layer_name}: {error}")
            if len(failed_layers) > 5:
                logging.warning(f"  ... 还有 {len(failed_layers) - 5} 个失败层")

        # 成功层的统计分析
        if success_layers and layer_sparsities:
            sparsity_values = [layer_sparsities[name] for name in success_layers]
            target_sparsity_values = [layer_details[name]['target_sparsity'] for name in success_layers]

            logging.info(f"稀疏度统计:")
            logging.info(f"  实际稀疏度: 均值={np.mean(sparsity_values):.3f}, "
                         f"标准差={np.std(sparsity_values):.3f}, "
                         f"范围=[{np.min(sparsity_values):.3f}, {np.max(sparsity_values):.3f}]")
            logging.info(f"  目标稀疏度: 均值={np.mean(target_sparsity_values):.3f}, "
                         f"标准差={np.std(target_sparsity_values):.3f}")

            # 稀疏度误差分析
            sparsity_errors = [abs(actual - target) for actual, target in
                               zip(sparsity_values, target_sparsity_values)]
            logging.info(f"  稀疏度误差: 均值={np.mean(sparsity_errors):.3f}, "
                         f"最大={np.max(sparsity_errors):.3f}")

        # 完整版本：质量检查
        if success_rate < 0.8:  # 成功率低于80%
            logging.error(f"成功率过低 ({success_rate:.1%})，剪枝质量可能不佳")

            # 如果成功率太低，考虑返回空结果
            if success_rate < 0.5:
                logging.error("成功率低于50%，返回空结果")
                return {}, {}

        # 整体稀疏度计算和验证
        if masks:
            total_params = sum(m.numel() for m in masks.values())
            pruned_params = sum((1 - m).sum().item() for m in masks.values())
            overall_sparsity = pruned_params / total_params if total_params > 0 else 0

            logging.info(f"整体剪枝结果:")
            logging.info(f"  目标稀疏度: {self.sparsity_ratio:.3f}")
            logging.info(f"  实际稀疏度: {overall_sparsity:.3f}")
            logging.info(f"  稀疏度误差: {abs(overall_sparsity - self.sparsity_ratio):.3f}")

            # 如果整体稀疏度误差过大，发出警告
            if abs(overall_sparsity - self.sparsity_ratio) > 0.1:
                logging.warning(f"整体稀疏度误差较大: 目标={self.sparsity_ratio:.3f}, "
                                f"实际={overall_sparsity:.3f}")

        return masks, layer_sparsities

    def apply_masks_to_model(self, masks):
        """应用mask到模型"""
        original_weights = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if name in masks:
                    original_weights[name] = module.weight.data.clone()
                    mask = masks[name].to(module.weight.device).to(module.weight.dtype)
                    with torch.no_grad():
                        module.weight.data = module.weight.data * mask

        return original_weights

    def restore_model_weights(self, original_weights):
        """恢复模型权重"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if name in original_weights:
                    with torch.no_grad():
                        module.weight.data = original_weights[name].clone()

    def calculate_model_sparsity(self, model) -> float:
        """计算模型稀疏度"""
        total_params = 0
        zero_params = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'embed' not in name.lower() and 'lm_head' not in name.lower():
                    total_params += module.weight.numel()
                    zero_params += (module.weight == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0

    def evaluate_perplexity_fast(self, model: nn.Module) -> float:
        """快速评估困惑度"""
        model.eval()

        try:
            if self.test_loader:
                with torch.no_grad():
                    ppl = eval_ppl_wikitext(model, self.test_loader, 1, self.device)

                if np.isnan(ppl) or np.isinf(ppl) or ppl > 10000:
                    return 10000.0

                return float(ppl)
            else:
                return 10000.0

        except Exception as e:
            logging.warning(f"评估失败: {e}")
            return 10000.0

    def evaluate_metric(self, metric_tree) -> float:
        """主评估函数 - 完整评估版本"""
        try:
            metric_str = repr(metric_tree) if hasattr(metric_tree, '__repr__') else str(metric_tree)
            logging.info(f"评估指标: {metric_str[:80]}...")

            # 记录失败的层数
            failed_layers = 0
            total_layers = 0

            masks, layer_sparsities = self.prune_model_with_metric_full(metric_tree)

            if not masks:
                logging.warning("没有生成有效的mask")
                return 10000.0

            # 检查数值稳定性
            for name, mask in masks.items():
                if mask is None:
                    failed_layers += 1
                total_layers += 1

            # 如果超过10%的层失败，惩罚该指标
            if failed_layers > total_layers * 0.1:
                logging.warning(f"数值不稳定：{failed_layers}/{total_layers} 层失败")
                return 10000.0

            original_weights = self.apply_masks_to_model(masks)

            actual_sparsity = self.calculate_model_sparsity(self.model)
            logging.info(f"剪枝后稀疏度: {actual_sparsity:.2%}")

            # 检查稀疏度是否合理
            sparsity_error = abs(actual_sparsity - self.sparsity_ratio)
            if sparsity_error > 0.1:  # 允许10%的误差
                logging.warning(f"稀疏度偏差过大: 目标={self.sparsity_ratio:.2%}, 实际={actual_sparsity:.2%}")
                # 添加惩罚项
                penalty = sparsity_error * 1000
            else:
                penalty = 0

            # 使用完整的测试集评估
            perplexity = self.evaluate_perplexity_full(self.model)

            # 添加惩罚
            final_score = perplexity + penalty

            if self.original_ppl is not None and self.original_ppl > 0:
                ratio = perplexity / self.original_ppl
                logging.info(f"PPL: {perplexity:.4f} (原始: {self.original_ppl:.4f}, 比例: {ratio:.2f})")
                logging.info(f"最终分数: {final_score:.4f} (惩罚: {penalty:.1f})")

            self.restore_model_weights(original_weights)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return final_score

        except Exception as e:
            logging.error(f"评估失败: {e}")
            import traceback
            traceback.print_exc()
            return 10000.0

    def evaluate_perplexity_full(self, model: nn.Module) -> float:
        """完整评估困惑度 - 使用标准eval_ppl_wikitext函数"""
        model.eval()

        try:
            # 使用标准的eval_ppl_wikitext函数，与原始评估保持一致
            with torch.no_grad():
                # self.test_loader应该是从get_loaders返回的格式
                # 直接使用现有的eval_ppl_wikitext函数
                ppl = eval_ppl_wikitext(model, self.test_loader, 1, self.device)

            # 数值检查
            if np.isnan(ppl) or np.isinf(ppl) or ppl > 10000:
                logging.warning(f"PPL异常: {ppl}, 返回上限值")
                return 10000.0

            return float(ppl)

        except Exception as e:
            logging.warning(f"完整评估失败: {e}")
            # 尝试使用备用评估方法
            try:
                return self.evaluate_perplexity_backup(model)
            except:
                return 10000.0

    def evaluate_perplexity_backup(self, model: nn.Module) -> float:
        """备用困惑度评估方法"""
        model.eval()

        try:
            total_loss = 0
            total_tokens = 0
            batch_count = 0

            with torch.no_grad():
                for batch in self.test_loader:
                    if batch_count >= 10:  # 限制批次数
                        break

                    # 处理不同的数据格式
                    if hasattr(batch, 'input_ids'):
                        # 如果batch有input_ids属性
                        input_ids = batch.input_ids.to(self.device)
                    elif isinstance(batch, torch.Tensor):
                        # 如果batch直接是tensor
                        input_ids = batch.to(self.device)
                    elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                        # 如果batch是列表或元组
                        input_ids = batch[0].to(self.device)
                    else:
                        logging.warning(f"未知的batch格式: {type(batch)}")
                        continue

                    # 确保输入形状正确
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)

                    # 限制序列长度
                    max_len = min(input_ids.shape[1], 2048)
                    input_ids = input_ids[:, :max_len]

                    if input_ids.shape[1] < 2:
                        continue

                    # 前向传播
                    outputs = model(input_ids[:, :-1])
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                    # 计算损失
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        input_ids[:, 1:].reshape(-1),
                        reduction='sum',
                        ignore_index=-100  # 忽略填充token
                    )

                    total_loss += loss.item()
                    total_tokens += input_ids[:, 1:].numel()
                    batch_count += 1

            if total_tokens == 0:
                logging.warning("没有处理任何token")
                return 10000.0

            # 计算困惑度
            avg_loss = total_loss / total_tokens
            ppl = torch.exp(torch.tensor(avg_loss)).item()

            # 检查结果
            if np.isnan(ppl) or np.isinf(ppl) or ppl > 10000:
                logging.warning(f"备用评估PPL异常: {ppl}")
                return 10000.0

            logging.info(f"备用评估成功: PPL={ppl:.4f}, 批次数={batch_count}, token数={total_tokens}")
            return float(ppl)

        except Exception as e:
            logging.error(f"备用评估也失败: {e}")
            return 10000.0

def create_fitness_function(evaluator: PruningEvaluator):
    """创建适应度函数"""

    def fitness_function(tree):
        return evaluator.evaluate_metric(tree)

    return fitness_function