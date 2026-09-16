#!/usr/bin/env python
"""
修复梯度文件中的键名格式，使其与模型层名匹配
"""

import torch
import argparse
import re
from pathlib import Path


def convert_gradient_key_to_model_key(grad_key):
    """
    将梯度键名转换为模型键名格式
    例如: 'self_attn.q_proj_layer_0' -> 'model.layers.0.self_attn.q_proj'
    """
    # 提取层号
    match = re.search(r'layer_(\d+)$', grad_key)
    if not match:
        return None

    layer_num = match.group(1)
    # 移除 _layer_X 后缀
    base_key = re.sub(r'_layer_\d+$', '', grad_key)

    # 构建新的键名
    new_key = f'model.layers.{layer_num}.{base_key}'
    return new_key


def fix_gradient_file(input_path, output_path=None):
    """
    修复梯度文件中的键名
    """
    print(f"加载梯度文件: {input_path}")
    gradients = torch.load(input_path, map_location='cpu')

    if not isinstance(gradients, dict):
        print(f"错误: 梯度文件不是字典格式")
        return False

    print(f"原始键数量: {len(gradients)}")
    print("\n前10个原始键:")
    for i, key in enumerate(list(gradients.keys())[:10]):
        print(f"  {i}: {key}")

    # 转换键名
    new_gradients = {}
    conversion_count = 0
    failed_keys = []

    for old_key, value in gradients.items():
        new_key = convert_gradient_key_to_model_key(old_key)
        if new_key:
            new_gradients[new_key] = value
            conversion_count += 1
            if conversion_count <= 5:
                print(f"\n转换: '{old_key}' -> '{new_key}'")
        else:
            # 保留无法转换的键（可能是特殊层）
            new_gradients[old_key] = value
            failed_keys.append(old_key)

    print(f"\n成功转换 {conversion_count} 个键")
    if failed_keys:
        print(f"未转换的键 ({len(failed_keys)} 个):")
        for key in failed_keys[:5]:
            print(f"  - {key}")
        if len(failed_keys) > 5:
            print(f"  ... 还有 {len(failed_keys) - 5} 个")

    print(f"\n新键数量: {len(new_gradients)}")
    print("\n前10个新键:")
    for i, key in enumerate(list(new_gradients.keys())[:10]):
        print(f"  {i}: {key}")

    # 保存修复后的文件
    if output_path is None:
        # 在原文件名基础上添加 _fixed 后缀
        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"

    print(f"\n保存修复后的文件到: {output_path}")
    torch.save(new_gradients, output_path)
    print("完成!")

    return True


def check_model_layer_names(model_path):
    """
    检查模型中的实际层名（可选）
    """
    try:
        from transformers import AutoModel
        print(f"\n检查模型层名: {model_path}")
        model = AutoModel.from_pretrained(model_path)

        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers.append(name)

        print(f"找到 {len(linear_layers)} 个线性层")
        print("前10个层名:")
        for i, name in enumerate(linear_layers[:10]):
            print(f"  {i}: {name}")

        return linear_layers
    except Exception as e:
        print(f"无法加载模型: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='修复梯度文件键名格式')
    parser.add_argument('--input', '-i', type=str,
                        default='/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-7B-hf_128_0.pth',
                        help='输入梯度文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出文件路径（默认为原文件名_fixed）')
    parser.add_argument('--check-model', type=str, default=None,
                        help='检查模型层名（提供模型路径）')

    args = parser.parse_args()

    # 可选：检查模型层名
    if args.check_model:
        check_model_layer_names(args.check_model)

    # 修复梯度文件
    fix_gradient_file(args.input, args.output)


if __name__ == '__main__':
    main()