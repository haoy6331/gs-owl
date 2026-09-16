# lib/genetic/__init__.py

from .ga_search import PrunerZeroGA, GAConfig, Individual
from .pruning_evaluator import PruningEvaluator, create_fitness_function

# 辅助函数
import os
import json
import logging
from datetime import datetime


def setup_logging(log_dir: str, experiment_name: str) -> str:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")

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


def save_results(results: dict, output_dir: str) -> str:
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存JSON结果
    result_file = os.path.join(output_dir, "results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 保存最佳树
    if 'best_tree' in results:
        tree_file = os.path.join(output_dir, "best_tree.json")
        with open(tree_file, 'w') as f:
            json.dump(results['best_tree'], f, indent=2)

    logging.info(f"结果保存到: {result_file}")
    return result_file


def create_experiment_summary(ga: PrunerZeroGA, evaluator: PruningEvaluator,
                              best_individual) -> dict:
    """创建实验总结"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': ga.config.__dict__,
        'best_fitness': best_individual.fitness,
        'best_metric': repr(best_individual.tree),
        'best_tree': best_individual.tree._serialize_tree(),
        'generations': ga.generation,
        'history': ga.generation_history,
        'model_path': evaluator.model_path,
        'sparsity_ratio': evaluator.sparsity_ratio,
        'original_ppl': evaluator.original_ppl
    }

    # 添加树的分析
    if hasattr(best_individual.tree, 'aggregate_leaf'):
        summary['leaf_counts'] = best_individual.tree.aggregate_leaf()

    if hasattr(best_individual.tree, 'aggregate_ops'):
        summary['ops_counts'] = best_individual.tree.aggregate_ops()

    summary['tree_size'] = best_individual.tree.size()

    return summary


__all__ = [
    'PrunerZeroGA',
    'GAConfig',
    'Individual',
    'PruningEvaluator',
    'create_fitness_function',
    'setup_logging',
    'save_results',
    'create_experiment_summary'
]