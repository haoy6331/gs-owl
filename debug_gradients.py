#!/usr/bin/env python
"""
调试梯度文件格式
运行: python debug_gradients.py
"""

import torch
import sys
import os


def inspect_gradient_file(filepath):
    """检查梯度文件的格式和内容"""
    print("=" * 60)
    print(f"检查梯度文件: {filepath}")
    print("=" * 60)

    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return

    # 加载梯度文件
    data = torch.load(filepath, map_location='cpu')

    print(f"数据类型: {type(data)}")

    if isinstance(data, dict):
        print(f"字典包含 {len(data)} 个键")
        print("\n前10个键:")
        for i, key in enumerate(list(data.keys())[:10]):
            value = data[key]
            if isinstance(value, torch.Tensor):
                print(f"  {i}: {key} -> Tensor{list(value.shape)}")
            else:
                print(f"  {i}: {key} -> {type(value)}")

        # 检查是否有特殊的键
        if 'gradients' in data:
            print("\n找到 'gradients' 键!")
            grad_data = data['gradients']
            print(f"  类型: {type(grad_data)}")
            if isinstance(grad_data, dict):
                print(f"  包含 {len(grad_data)} 个子键")

        if 'model' in data:
            print("\n找到 'model' 键!")
            model_data = data['model']
            print(f"  类型: {type(model_data)}")

    elif isinstance(data, list):
        print(f"列表包含 {len(data)} 个元素")
        print("\n前10个元素:")
        for i, item in enumerate(data[:10]):
            if isinstance(item, torch.Tensor):
                print(f"  {i}: Tensor{list(item.shape)}")
            else:
                print(f"  {i}: {type(item)}")

    elif isinstance(data, torch.Tensor):
        print(f"单个张量，形状: {list(data.shape)}")
        print(f"数据类型: {data.dtype}")
        print(f"设备: {data.device}")
        print(f"最小值: {data.min().item():.6f}")
        print(f"最大值: {data.max().item():.6f}")
        print(f"均值: {data.mean().item():.6f}")

    else:
        print(f"未知格式: {type(data)}")

    print("\n" + "=" * 60)

    # 尝试提取实际的梯度
    print("尝试提取梯度...")

    gradients = None

    # 方法1：直接是梯度
    if isinstance(data, (list, torch.Tensor)):
        gradients = data
        print("梯度直接存储在文件中")

    # 方法2：在字典的某个键下
    elif isinstance(data, dict):
        # 常见的键名
        possible_keys = ['gradients', 'grads', 'gradient', 'model', 'state_dict', 'weights']
        for key in possible_keys:
            if key in data:
                gradients = data[key]
                print(f"在键 '{key}' 下找到梯度")
                break

        # 如果还没找到，可能键名就是层名
        if gradients is None:
            # 检查是否所有值都是张量
            all_tensors = all(isinstance(v, torch.Tensor) for v in data.values())
            if all_tensors:
                gradients = data
                print("梯度存储为字典，键是层名")

    if gradients is not None:
        print("\n梯度信息:")
        if isinstance(gradients, dict):
            print(f"  字典格式，{len(gradients)} 层")
            # 显示一个样本
            sample_key = list(gradients.keys())[0]
            sample_grad = gradients[sample_key]
            if isinstance(sample_grad, torch.Tensor):
                print(f"  样本层 '{sample_key}':")
                print(f"    形状: {list(sample_grad.shape)}")
                print(f"    范数: {torch.norm(sample_grad).item():.6f}")
        elif isinstance(gradients, list):
            print(f"  列表格式，{len(gradients)} 个元素")
            if len(gradients) > 0 and isinstance(gradients[0], torch.Tensor):
                print(f"  第一个元素:")
                print(f"    形状: {list(gradients[0].shape)}")
                print(f"    范数: {torch.norm(gradients[0]).item():.6f}")
        elif isinstance(gradients, torch.Tensor):
            print(f"  张量格式，形状: {list(gradients.shape)}")
            print(f"  范数: {torch.norm(gradients).item():.6f}")
    else:
        print("无法识别梯度格式")


def main():
    # 梯度文件路径
    gradient_files = [
        "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-7B-hf_128_0.pth",
        # 可以添加其他文件
    ]

    for filepath in gradient_files:
        if os.path.exists(filepath):
            inspect_gradient_file(filepath)
        else:
            print(f"文件不存在: {filepath}")


if __name__ == "__main__":
    main()