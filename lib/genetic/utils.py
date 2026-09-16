import os
import json
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def setup_logging(log_dir: str = "logs", experiment_name: str = None):
    """
    设置日志

    Args:
        log_dir: 日志目录
        experiment_name: 实验名称
    """
    if experiment_name is None:
        experiment_name = f"ga_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"日志保存到: {log_file}")
    return log_file


def save_results(results: Dict, save_dir: str = "results/ga_search"):
    """
    保存结果

    Args:
        results: 结果字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存JSON结果
    result_file = os.path.join(save_dir, f"results_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"结果保存到: {result_file}")

    # 生成并保存图表
    if 'generation_history' in results:
        plot_evolution_history(results['generation_history'], save_dir, timestamp)

    return result_file


def load_checkpoint(checkpoint_path: str) -> Dict:
    """
    加载检查点

    Args:
        checkpoint_path: 检查点文件路径

    Returns:
        检查点字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    logging.info(f"加载检查点: {checkpoint_path}")
    return checkpoint


def plot_evolution_history(history: List[Dict], save_dir: str, timestamp: str):
    """
    绘制进化历史图表

    Args:
        history: 进化历史
        save_dir: 保存目录
        timestamp: 时间戳
    """
    if not history:
        return

    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    worst_fitness = [h['worst_fitness'] for h in history]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 图1: 适应度曲线
    ax1.plot(generations, best_fitness, 'g-', label='Best', linewidth=2)
    ax1.plot(generations, avg_fitness, 'b-', label='Average', linewidth=1)
    ax1.plot(generations, worst_fitness, 'r-', label='Worst', linewidth=1)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Evolution Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 改进率
    improvements = []
    for i in range(1, len(best_fitness)):
        improvement = (best_fitness[i - 1] - best_fitness[i]) / best_fitness[i - 1] * 100
        improvements.append(improvement)

    if improvements:
        ax2.bar(generations[1:], improvements, color='blue', alpha=0.6)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Generation-wise Improvement')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    plot_file = os.path.join(save_dir, f"evolution_plot_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"图表保存到: {plot_file}")


def format_tree_expression(tree_str: str) -> str:
    """
    格式化树表达式为更易读的形式

    Args:
        tree_str: 树的字符串表示

    Returns:
        格式化后的表达式
    """
    # 简化表达式
    replacements = {
        '||': 'norm(',
        'σ': 'mms',
        'ζ': 'zsn',
        '√': 'sqrt',
        '²': '^2',
    }

    formatted = tree_str
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)

    return formatted


def calculate_sparsity(model) -> float:
    """
    计算模型的实际稀疏度

    Args:
        model: PyTorch模型

    Returns:
        稀疏度（0到1之间）
    """
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

    sparsity = zero_params / total_params if total_params > 0 else 0
    return sparsity


def get_model_size(model) -> Dict[str, float]:
    """
    获取模型大小信息

    Args:
        model: PyTorch模型

    Returns:
        模型大小信息字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算模型大小（MB）
    param_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_size_mb': param_size_mb
    }


def create_experiment_summary(ga_instance, evaluator, best_individual):
    """
    创建实验总结

    Args:
        ga_instance: 遗传算法实例
        evaluator: 评估器实例
        best_individual: 最佳个体

    Returns:
        实验总结字典
    """
    summary = {
        'experiment_info': {
            'model_path': evaluator.model_path,
            'sparsity_ratio': evaluator.sparsity_ratio,
            'calibration_samples': evaluator.calibration_samples,
            'population_size': ga_instance.config.population_size,
            'max_iterations': ga_instance.config.max_iterations,
            'mutation_prob': ga_instance.config.mutation_prob,
            'seed': ga_instance.config.seed
        },
        'best_solution': {
            'tree_expression': repr(best_individual.tree),
            'formatted_expression': format_tree_expression(repr(best_individual.tree)),
            'fitness': best_individual.fitness,
            'tree_size': best_individual.tree.size(),
            'tree_depth': calculate_tree_depth(best_individual.tree),
            'leaf_counts': best_individual.tree.aggregate_leaf(),
            'ops_counts': best_individual.tree.aggregate_ops()
        },
        'evolution_stats': {
            'total_generations': len(ga_instance.generation_history),
            'convergence_generation': find_convergence_point(ga_instance.generation_history),
            'initial_best': ga_instance.generation_history[0][
                'best_fitness'] if ga_instance.generation_history else None,
            'final_best': ga_instance.generation_history[-1][
                'best_fitness'] if ga_instance.generation_history else None,
            'improvement': calculate_improvement(ga_instance.generation_history)
        },
        'generation_history': ga_instance.generation_history
    }

    return summary


def calculate_tree_depth(tree) -> int:
    """
    计算树的深度

    Args:
        tree: GPTree实例

    Returns:
        树的深度
    """
    if tree is None or tree.data in ['W', 'G', 'X']:
        return 0

    left_depth = calculate_tree_depth(tree.left) if tree.left else 0
    right_depth = calculate_tree_depth(tree.right) if tree.right else 0

    return 1 + max(left_depth, right_depth)


def find_convergence_point(history: List[Dict], window: int = 20, threshold: float = 0.01) -> Optional[int]:
    """
    寻找收敛点

    Args:
        history: 进化历史
        window: 检查窗口大小
        threshold: 收敛阈值

    Returns:
        收敛的代数，如果未收敛返回None
    """
    if len(history) < window:
        return None

    for i in range(window, len(history)):
        recent = [h['best_fitness'] for h in history[i - window:i]]
        if max(recent) - min(recent) < threshold:
            return i

    return None


def calculate_improvement(history: List[Dict]) -> float:
    """
    计算总体改进率

    Args:
        history: 进化历史

    Returns:
        改进百分比
    """
    if not history or len(history) < 2:
        return 0.0

    initial = history[0]['best_fitness']
    final = history[-1]['best_fitness']

    if initial == 0:
        return 0.0

    improvement = (initial - final) / initial * 100
    return improvement