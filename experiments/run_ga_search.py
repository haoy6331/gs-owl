#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pruner-Zero遗传算法搜索主程序 - 集成OWL层级自适应稀疏
用于搜索LLaMA-7B的最优符号剪枝指标
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import logging
import time
from datetime import datetime
import torch
import numpy as np
import random

# 导入遗传算法相关模块
from lib.genetic import (
    PrunerZeroGA,
    GAConfig,
    PruningEvaluator,
    setup_logging,
    save_results,
    create_experiment_summary
)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    """主函数"""

    # 加载配置
    if args.config:
        config_dict = load_config(args.config)
    else:
        config_dict = {}

    # 命令行参数覆盖配置文件
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config_dict[key] = value

    # 设置实验名称
    owl_suffix = "_owl" if config_dict.get('use_owl', False) else ""
    experiment_name = config_dict.get('experiment_name',
                                      f"ga_llama7b{owl_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # 设置日志
    log_file = setup_logging("logs", experiment_name)

    logging.info("=" * 80)
    logging.info("Pruner-Zero 遗传算法搜索")
    if config_dict.get('use_owl', False):
        logging.info("🦉 启用OWL层级自适应稀疏")
    logging.info("=" * 80)
    logging.info(f"实验名称: {experiment_name}")
    logging.info(f"配置: {config_dict}")

    # 设置随机种子
    seed = config_dict.get('seed', 42)
    set_seed(seed)
    logging.info(f"随机种子: {seed}")

    # 创建GA配置 - 包含OWL参数
    ga_config = GAConfig(
        population_size=config_dict.get('population_size', 50),
        max_iterations=config_dict.get('max_iterations', 300),
        sample_ratio=config_dict.get('sample_ratio', 0.4),
        top_k=config_dict.get('top_k', 10),
        mutation_prob=config_dict.get('mutation_prob', 0.5),
        min_depth=config_dict.get('min_depth', 3),
        max_depth=config_dict.get('max_depth', 5),
        calibration_samples=config_dict.get('calibration_samples', 128),
        sparsity_ratio=config_dict.get('sparsity_ratio', 0.5),
        model_path=config_dict.get('model_path',
                                   "/home/yh114/workdir/DSnoT/models/decapoda-research-llama-7B-hf"),
        device=config_dict.get('device', 'cuda'),
        seed=seed,
        checkpoint_freq=config_dict.get('checkpoint_freq', 10),
        # OWL参数
        use_owl=config_dict.get('use_owl', False),
        hyper_m=config_dict.get('hyper_m', 5.0),
        lamda=config_dict.get('lamda', 0.08)
    )

    # 确定梯度文件路径
    gradient_file = None
    if "llama-7B" in ga_config.model_path:
        gradient_file = "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l2_model_decapoda-research-llama-7B-hf_128_0_fixed.pth"
    elif "llama-13B" in ga_config.model_path:
        gradient_file = "/home/yh114/workdir/Pruner-Zero/gradients/llama1/gradients_aggregrate_norm_l1_model_decapoda-research-llama-13B-hf_128_0.pth"

    if gradient_file and os.path.exists(gradient_file):
        logging.info(f"使用预计算梯度: {gradient_file}")
    else:
        logging.warning("未找到预计算梯度文件，将实时计算")

    # 初始化评估器 - 传入OWL参数
    logging.info("初始化剪枝评估器...")
    evaluator = PruningEvaluator(
        model_path=ga_config.model_path,
        sparsity_ratio=ga_config.sparsity_ratio,
        calibration_samples=ga_config.calibration_samples,
        device=ga_config.device,
        seed=ga_config.seed,
        gradient_path=gradient_file,
        # OWL参数
        use_owl=ga_config.use_owl,
        hyper_m=ga_config.hyper_m,
        lamda=ga_config.lamda
    )

    # 如果使用OWL，显示层级稀疏率信息
    if ga_config.use_owl and hasattr(evaluator, 'layer_sparsity_ratios'):
        if evaluator.layer_sparsity_ratios is not None:
            logging.info("=" * 80)
            logging.info("OWL层级稀疏率分布:")
            num_layers = len(evaluator.layer_sparsity_ratios)

            # 显示前10层和后5层
            for i in range(min(10, num_layers)):
                logging.info(f"  层 {i:2d}: {evaluator.layer_sparsity_ratios[i]:.3f}")

            if num_layers > 15:
                logging.info("  ...")
                for i in range(num_layers - 5, num_layers):
                    logging.info(f"  层 {i:2d}: {evaluator.layer_sparsity_ratios[i]:.3f}")

            # 显示统计信息
            logging.info(f"  均值: {np.mean(evaluator.layer_sparsity_ratios):.3f}")
            logging.info(f"  标准差: {np.std(evaluator.layer_sparsity_ratios):.3f}")
            logging.info(f"  最小值: {np.min(evaluator.layer_sparsity_ratios):.3f}")
            logging.info(f"  最大值: {np.max(evaluator.layer_sparsity_ratios):.3f}")
            logging.info("=" * 80)

    # 初始化遗传算法
    logging.info("初始化遗传算法...")
    ga = PrunerZeroGA(ga_config, evaluator)

    # 如果指定了检查点，加载它
    if args.checkpoint:
        logging.info(f"从检查点恢复: {args.checkpoint}")
        ga.load_checkpoint(args.checkpoint)

    # 开始进化
    logging.info("开始遗传算法搜索...")
    start_time = time.time()

    try:
        # 运行遗传算法
        best_individual = ga.evolve()

        # 记录结果
        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info("=" * 80)
        logging.info("搜索完成！")
        logging.info(f"总用时: {elapsed_time / 3600:.2f} 小时")
        logging.info(f"最佳符号剪枝指标: {repr(best_individual.tree)}")
        logging.info(f"最佳Perplexity: {best_individual.fitness:.4f}")

        if ga_config.use_owl:
            logging.info(f"OWL配置: Hyper_m={ga_config.hyper_m}, Lambda={ga_config.lamda}")

        # 分析最佳解
        leaf_counts = best_individual.tree.aggregate_leaf()
        ops_counts = best_individual.tree.aggregate_ops()

        logging.info("最佳解分析:")
        logging.info(f"  - 树大小: {best_individual.tree.size()} 节点")
        logging.info(f"  - 叶子节点: W={leaf_counts['W']}, G={leaf_counts['G']}, X={leaf_counts['X']}")
        logging.info(f"  - 操作统计: {ops_counts}")

        # 创建实验总结
        summary = create_experiment_summary(ga, evaluator, best_individual)
        summary['experiment_name'] = experiment_name
        summary['elapsed_time_hours'] = elapsed_time / 3600
        summary['use_owl'] = ga_config.use_owl

        if ga_config.use_owl:
            summary['owl_config'] = {
                'hyper_m': ga_config.hyper_m,
                'lamda': ga_config.lamda,
                'layer_sparsity_ratios': evaluator.layer_sparsity_ratios.tolist() if evaluator.layer_sparsity_ratios is not None else None
            }

        # 保存结果
        result_file = save_results(summary, f"results/ga_search/{experiment_name}")

        # 保存最佳树
        best_tree_file = f"results/ga_search/{experiment_name}/best_tree.json"
        best_individual.tree.save_tree(best_tree_file)
        logging.info(f"最佳树保存到: {best_tree_file}")

        # 如果论文中的公式是: ||W| × |W|| × σ(|G|)
        # 检查我们是否找到了类似的公式
        logging.info("\n与论文公式的比较:")
        logging.info("论文公式: ||W| × |W|| × σ(|G|)")
        logging.info(f"搜索结果: {repr(best_individual.tree)}")

        # 如果使用OWL，保存对比信息
        if ga_config.use_owl:
            owl_comparison_file = f"results/ga_search/{experiment_name}/owl_comparison.txt"
            with open(owl_comparison_file, 'w') as f:
                f.write("OWL层级自适应稀疏搜索结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"最佳指标: {repr(best_individual.tree)}\n")
                f.write(f"最佳PPL: {best_individual.fitness:.4f}\n")
                f.write(f"Hyper_m: {ga_config.hyper_m}\n")
                f.write(f"Lambda: {ga_config.lamda}\n")
                f.write(f"整体稀疏率: {ga_config.sparsity_ratio}\n")
                f.write("\n层级稀疏率分布:\n")
                if evaluator.layer_sparsity_ratios is not None:
                    for i, ratio in enumerate(evaluator.layer_sparsity_ratios):
                        f.write(f"  层 {i}: {ratio:.3f}\n")
            logging.info(f"OWL对比信息保存到: {owl_comparison_file}")

        return best_individual

    except KeyboardInterrupt:
        logging.info("用户中断，保存当前进度...")
        checkpoint_file = f"checkpoints/interrupted_{experiment_name}.json"
        ga.save_checkpoint(checkpoint_file)
        logging.info(f"检查点保存到: {checkpoint_file}")
        raise

    except Exception as e:
        logging.error(f"发生错误: {e}")
        logging.error("保存错误检查点...")
        checkpoint_file = f"checkpoints/error_{experiment_name}.json"
        ga.save_checkpoint(checkpoint_file)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pruner-Zero遗传算法搜索")

    # 基础参数
    parser.add_argument('--config', type=str, default='experiments/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='实验名称')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='从检查点恢复')

    # GA参数
    parser.add_argument('--population-size', type=int, default=None,
                        help='种群大小')
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='最大迭代次数')
    parser.add_argument('--mutation-prob', type=float, default=None,
                        help='突变概率')
    parser.add_argument('--top-k', type=int, default=None,
                        help='锦标赛选择的top-k')

    # 模型参数
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型路径')
    parser.add_argument('--sparsity-ratio', type=float, default=None,
                        help='剪枝比例')
    parser.add_argument('--calibration-samples', type=int, default=None,
                        help='校准样本数')

    # OWL参数
    parser.add_argument('--use-owl', action='store_true',
                        help='使用OWL层级自适应稀疏')
    parser.add_argument('--hyper-m', type=float, default=None,
                        help='OWL异常值检测阈值倍数')
    parser.add_argument('--lamda', type=float, default=None,
                        help='OWL归一化参数')

    # 其他参数
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')

    args = parser.parse_args()

    # 运行主程序
    main(args)