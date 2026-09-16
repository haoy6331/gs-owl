#!/usr/bin/env python3
"""
调试Pruner-Zero树公式的计算过程
用法: python debug_tree_formula.py
"""

import sys
import os
import torch
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gptree import GPTree


def load_debug_data(debug_file="pruner_zero_debug.pkl"):
    """加载保存的调试数据"""
    if not os.path.exists(debug_file):
        print(f"错误: 未找到调试文件 {debug_file}")
        print("请先运行带debug=True的剪枝过程")
        return None

    with open(debug_file, 'rb') as f:
        debug_info = pickle.load(f)

    print(f"加载了 {len(debug_info)} 层的调试信息")
    return debug_info


def analyze_layer_computation(layer_name, layer_data, tree_path="data/best_tree.json"):
    """分析单层的计算过程"""
    print("\n" + "=" * 70)
    print(f"分析层: {layer_name}")
    print("=" * 70)

    W = layer_data['W']
    G = layer_data['G']
    X = layer_data['X']
    W_metric = layer_data['W_metric']
    W_metric_theory = layer_data['W_metric_theory']
    manual_result = layer_data['manual_result']

    # 统计信息
    print("\n1. 输入数据统计:")
    print(f"   W: shape={W.shape}, mean={W.mean():.4f}, std={W.std():.4f}")
    print(f"   G: shape={G.shape}, mean={G.mean():.4f}, std={G.std():.4f}")
    print(f"   X: shape={X.shape}, mean={X.mean():.4f}, std={X.std():.4f}")

    print("\n2. 计算结果统计:")
    print(f"   GPTree结果: mean={W_metric.mean():.4f}, std={W_metric.std():.4f}")
    print(f"   理论值(W²): mean={W_metric_theory.mean():.4f}, std={W_metric_theory.std():.4f}")
    print(f"   手动计算: mean={manual_result.mean():.4f}, std={manual_result.std():.4f}")

    print("\n3. 差异分析:")
    diff_gp_th = torch.abs(W_metric - W_metric_theory)
    diff_gp_mn = torch.abs(W_metric - manual_result)
    diff_th_mn = torch.abs(W_metric_theory - manual_result)

    print(f"   GPTree vs 理论值: mean_diff={diff_gp_th.mean():.6f}, max_diff={diff_gp_th.max():.6f}")
    print(f"   GPTree vs 手动: mean_diff={diff_gp_mn.mean():.6f}, max_diff={diff_gp_mn.max():.6f}")
    print(f"   理论值 vs 手动: mean_diff={diff_th_mn.mean():.6f}, max_diff={diff_th_mn.max():.6f}")

    print("\n4. 剪枝掩码重叠度:")
    print(f"   GPTree vs 理论值: {layer_data['overlap_gp_th']:.2%}")
    print(f"   GPTree vs 手动: {layer_data['overlap_gp_mn']:.2%}")
    print(f"   理论值 vs 手动: {layer_data['overlap_th_mn']:.2%}")

    # 检查异常值
    print("\n5. 异常值检查:")
    threshold = W_metric.mean() + 3 * W_metric.std()
    outliers_gp = (W_metric > threshold).sum().item()
    outliers_th = (W_metric_theory > threshold).sum().item()

    print(f"   GPTree异常值 (>mean+3std): {outliers_gp}/{W_metric.numel()}")
    print(f"   理论值异常值: {outliers_th}/{W_metric_theory.numel()}")

    # 检查G接近0的影响
    small_g_mask = torch.abs(G) < 1e-6
    if small_g_mask.any():
        print(f"\n6. 小梯度值的影响:")
        print(f"   G<1e-6的位置数: {small_g_mask.sum().item()}")
        print(f"   对应的W_metric均值: {W_metric[small_g_mask].mean():.6f}")
        print(f"   对应的理论值均值: {W_metric_theory[small_g_mask].mean():.6f}")

    return {
        'mean_diff_gp_th': diff_gp_th.mean().item(),
        'max_diff_gp_th': diff_gp_th.max().item(),
        'overlap_gp_th': layer_data['overlap_gp_th']
    }


def visualize_differences(debug_info, num_layers=5):
    """可视化前几层的差异"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    layer_names = list(debug_info.keys())[:num_layers]

    for idx, layer_name in enumerate(layer_names):
        if idx >= 6:
            break

        row = idx // 3
        col = idx % 3
        ax = axes[row, col] if len(axes.shape) > 1 else axes[idx]

        layer_data = debug_info[layer_name]
        W_metric = layer_data['W_metric'].flatten()
        W_metric_theory = layer_data['W_metric_theory'].flatten()

        # 绘制散点图
        sample_size = min(1000, W_metric.numel())
        indices = torch.randperm(W_metric.numel())[:sample_size]

        ax.scatter(W_metric_theory[indices], W_metric[indices],
                   alpha=0.5, s=1)
        ax.plot([W_metric_theory.min(), W_metric_theory.max()],
                [W_metric_theory.min(), W_metric_theory.max()],
                'r--', label='y=x')

        ax.set_xlabel('理论值 (W²)')
        ax.set_ylabel('GPTree结果')
        ax.set_title(f'{layer_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pruner_zero_comparison.png', dpi=150)
    print(f"\n可视化图已保存到 pruner_zero_comparison.png")


def main():
    """主函数"""
    print("Pruner-Zero 树公式调试工具")
    print("=" * 70)

    # 加载调试数据
    debug_info = load_debug_data()
    if debug_info is None:
        return

    # 分析每一层
    all_stats = {}
    for layer_name in list(debug_info.keys())[:10]:  # 只分析前10层
        stats = analyze_layer_computation(layer_name, debug_info[layer_name])
        all_stats[layer_name] = stats

    # 汇总统计
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)

    mean_diffs = [s['mean_diff_gp_th'] for s in all_stats.values()]
    max_diffs = [s['max_diff_gp_th'] for s in all_stats.values()]
    overlaps = [s['overlap_gp_th'] for s in all_stats.values()]

    print(f"\n所有层的平均差异:")
    print(f"  平均差异: {np.mean(mean_diffs):.6f} ± {np.std(mean_diffs):.6f}")
    print(f"  最大差异: {np.mean(max_diffs):.6f} ± {np.std(max_diffs):.6f}")
    print(f"  掩码重叠度: {np.mean(overlaps):.2%} ± {np.std(overlaps):.2%}")

    # 生成可视化
    visualize_differences(debug_info)

    print("\n分析完成!")


if __name__ == "__main__":
    main()