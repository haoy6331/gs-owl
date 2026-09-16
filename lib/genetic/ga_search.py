import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random
import copy
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import time

from lib.gptree import GPTree, FUNCTIONS, TERMINALS, UNARY_FUNCTIONS, BINARY_FUNCTIONS
# 导入具体的函数
from lib.gptree import add, sub, mul, div, sqr, neg, abs, log, exp, sqrt, tanh, pow, skp, mms, zsn


@dataclass
class GAConfig:
    """遗传算法配置 - 集成OWL参数"""
    population_size: int = 100
    max_iterations: int = 100
    sample_ratio: float = 0.4
    top_k: int = 20
    selection_ratio: float = 0.2
    mutation_prob: float = 0.3
    min_depth: int = 2
    max_depth: int = 4
    calibration_samples: int = 128
    sparsity_ratio: float = 0.5
    model_path: str = "/home/yh114/workdir/DSnoT/models/decapoda-research-llama-7B-hf"
    device: str = "cuda"
    seed: int = 42
    checkpoint_freq: int = 10
    elite_size: int = 10
    # OWL参数
    use_owl: bool = False
    hyper_m: float = 5.0
    lamda: float = 0.08


class Individual:
    """种群个体"""

    def __init__(self, tree: GPTree, fitness: float = float('inf')):
        self.tree = tree
        self.fitness = fitness
        self.age = 0
        self.tree_str = self.get_tree_string()

    def get_tree_string(self):
        """安全地获取树的字符串表示"""
        try:
            return str(self.tree)
        except:
            return "unknown"

    def __lt__(self, other):
        return self.fitness < other.fitness

    def copy(self):
        return Individual(self.tree.build_subtree(), self.fitness)

    def to_dict(self):
        return {
            'tree': self.tree._serialize_tree(),
            'fitness': self.fitness,
            'age': self.age
        }

    @staticmethod
    def from_dict(data):
        tree = GPTree._deserialize_tree(data['tree'])
        ind = Individual(tree, data['fitness'])
        ind.age = data.get('age', 0)
        return ind


class PrunerZeroGA:
    """Pruner-Zero遗传算法 - 支持OWL"""

    def __init__(self, config: GAConfig, evaluator=None):
        self.config = config
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation_history: List[Dict] = []
        self.evaluator = evaluator
        self.generation = 0
        self.evaluated_trees = {}

        # 定义对立操作
        self.opposing_operations = {
            exp: log,
            log: exp,
            sqr: sqrt,
            sqrt: sqr,
            neg: neg,  # neg是自己的逆
            add: sub,
            sub: add,
            mul: div,
            div: mul,
        }

        # 设置随机种子
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        # 如果使用OWL，显示信息
        if config.use_owl:
            logging.info(f"启用OWL层级自适应稀疏 - Hyper_m: {config.hyper_m}, Lambda: {config.lamda}")

    def create_good_initial_trees(self) -> List[GPTree]:
        """创建已知效果好的初始树 - 包含X"""
        good_trees = []

        # 1. |W| * |W| - 权重幅度平方
        tree1 = GPTree(mul)
        tree1.left = GPTree(abs)
        tree1.left.left = GPTree('W')
        tree1.right = GPTree(abs)
        tree1.right.left = GPTree('W')
        good_trees.append(tree1)

        # 2. W^2 - 简单平方
        tree2 = GPTree(sqr)
        tree2.left = GPTree('W')
        good_trees.append(tree2)

        # 3. |W| * |G| - 权重梯度乘积
        tree3 = GPTree(mul)
        tree3.left = GPTree(abs)
        tree3.left.left = GPTree('W')
        tree3.right = GPTree(abs)
        tree3.right.left = GPTree('G')
        good_trees.append(tree3)

        # 4. |W*W| * mms(G) - 类似best_tree.json
        tree4 = GPTree(mul)
        tree4.left = GPTree(abs)
        tree4.left.left = GPTree(mul)
        tree4.left.left.left = GPTree('W')
        tree4.left.left.right = GPTree('W')
        tree4.right = GPTree(mms)
        tree4.right.left = GPTree('G')
        good_trees.append(tree4)

        # 5. 包含X的树: |W| * |X| - Wanda风格
        tree5 = GPTree(mul)
        tree5.left = GPTree(abs)
        tree5.left.left = GPTree('W')
        tree5.right = GPTree(abs)
        tree5.right.left = GPTree('X')
        good_trees.append(tree5)

        # 6. W^2 * X
        tree6 = GPTree(mul)
        tree6.left = GPTree(sqr)
        tree6.left.left = GPTree('W')
        tree6.right = GPTree('X')
        good_trees.append(tree6)

        # 7. |W| * sqrt(|G|) * X
        tree7 = GPTree(mul)
        tree7.left = GPTree(mul)
        tree7.left.left = GPTree(abs)
        tree7.left.left.left = GPTree('W')
        tree7.left.right = GPTree(sqrt)
        tree7.left.right.left = GPTree(abs)
        tree7.left.right.left.left = GPTree('G')
        tree7.right = GPTree('X')
        good_trees.append(tree7)

        # 8. |W| - 最简单的
        tree8 = GPTree(abs)
        tree8.left = GPTree('W')
        good_trees.append(tree8)

        # 9. |G| - 梯度幅度
        tree9 = GPTree(abs)
        tree9.left = GPTree('G')
        good_trees.append(tree9)

        # 10. |X| - 激活幅度
        tree10 = GPTree(abs)
        tree10.left = GPTree('X')
        good_trees.append(tree10)

        # 11. W^2 * sqrt(G) - 对梯度更敏感
        tree11 = GPTree(mul)
        tree11.left = GPTree(sqr)
        tree11.left.left = GPTree('W')
        tree11.right = GPTree(sqrt)
        tree11.right.left = GPTree(abs)
        tree11.right.left.left = GPTree('G')
        good_trees.append(tree11)

        # 12. |W| * zsn(G) - 梯度标准化
        tree12 = GPTree(mul)
        tree12.left = GPTree(abs)
        tree12.left.left = GPTree('W')
        tree12.right = GPTree(zsn)
        tree12.right.left = GPTree('G')
        good_trees.append(tree12)

        # 13. 包含X的复杂树: |W| * mms(G) * sqrt(X)
        tree13 = GPTree(mul)
        tree13.left = GPTree(mul)
        tree13.left.left = GPTree(abs)
        tree13.left.left.left = GPTree('W')
        tree13.left.right = GPTree(mms)
        tree13.left.right.left = GPTree('G')
        tree13.right = GPTree(sqrt)
        tree13.right.left = GPTree('X')
        good_trees.append(tree13)

        return good_trees

    def initialize_population(self) -> None:
        """初始化种群"""
        self.population = []

        # 添加已知好的树
        good_trees = self.create_good_initial_trees()
        for tree in good_trees:
            self.population.append(Individual(tree))

        logging.info(f"添加了 {len(good_trees)} 个种子树")

        # 添加一些简单的单函数树
        for func in [abs, sqr, log, sqrt, tanh, mms, zsn]:
            if func in UNARY_FUNCTIONS:
                # W版本
                tree_w = GPTree(func)
                tree_w.left = GPTree('W')
                self.population.append(Individual(tree_w))

                # G版本
                tree_g = GPTree(func)
                tree_g.left = GPTree('G')
                self.population.append(Individual(tree_g))

                # X版本
                tree_x = GPTree(func)
                tree_x.left = GPTree('X')
                self.population.append(Individual(tree_x))

        # 填充随机树
        while len(self.population) < self.config.population_size:
            depth = random.randint(self.config.min_depth, self.config.max_depth)
            tree = GPTree()
            grow = random.random() > 0.5
            tree.random_tree(grow=grow, max_depth=depth)

            # 不再避免X，让它自然出现
            self.population.append(Individual(tree))

        logging.info(f"初始化种群，大小: {len(self.population)}")

    def tournament_selection(self, k: int = None) -> List[Individual]:
        """锦标赛选择"""
        if k is None:
            k = self.config.top_k

        sample_size = max(k * 2, int(self.config.sample_ratio * len(self.population)))
        sample_size = min(sample_size, len(self.population))

        candidates = random.sample(self.population, sample_size)
        candidates.sort()
        return candidates[:k]

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """子树交叉"""
        offspring = parent1.copy()
        parent2_copy = parent2.copy()

        size1 = offspring.tree.size()
        size2 = parent2_copy.tree.size()

        if size1 > 1 and size2 > 1:
            point1 = random.randint(1, size1)
            point2 = random.randint(1, size2)

            subtree1 = offspring.tree.scan_tree([point1], None)
            subtree2 = parent2_copy.tree.scan_tree([point2], None)

            if subtree1 and subtree2:
                new_size = size1 - subtree1.size() + subtree2.size()
                if new_size <= 20:  # 限制树大小
                    offspring.tree.scan_tree([point1], subtree2)
                    offspring.tree_str = offspring.get_tree_string()

        return offspring

    def mutate(self, individual: Individual) -> Individual:
        """节点突变 - 不再避免X"""
        if random.random() < self.config.mutation_prob:
            mutated = individual.copy()
            tree_size = mutated.tree.size()

            if tree_size > 0:
                mutation_point = random.randint(1, tree_size)
                self._mutate_at_point(mutated.tree, [mutation_point])
                mutated.tree_str = mutated.get_tree_string()

            return mutated
        return individual

    def _mutate_at_point(self, tree: GPTree, count: List[int]) -> None:
        """在特定点突变 - 允许所有终端符号"""
        count[0] -= 1
        if count[0] == 0:
            # 突变当前节点
            if tree.data in UNARY_FUNCTIONS:
                other_unary = [f for f in UNARY_FUNCTIONS if f != tree.data]
                if other_unary:
                    tree.data = random.choice(other_unary)
            elif tree.data in BINARY_FUNCTIONS:
                other_binary = [f for f in BINARY_FUNCTIONS if f != tree.data]
                if other_binary:
                    tree.data = random.choice(other_binary)
            elif tree.data in TERMINALS:
                # 现在平等对待所有终端符号
                other_terminals = [t for t in TERMINALS if t != tree.data]
                if other_terminals:
                    tree.data = random.choice(other_terminals)
        else:
            if tree.left and count[0] > 0:
                self._mutate_at_point(tree.left, count)
            if tree.right and count[0] > 0:
                self._mutate_at_point(tree.right, count)

    def simplify_tree(self, tree: GPTree) -> GPTree:
        """实现OOS (Opposing Operation Simplification)"""
        # 创建树的副本
        simplified = tree.build_subtree()

        # 递归简化
        self._simplify_node(simplified)

        return simplified

    def _simplify_node(self, node: GPTree) -> bool:
        """递归简化节点，返回是否进行了简化"""
        if node is None or node.data in TERMINALS:
            return False

        simplified = False

        # 先递归简化子树
        if node.left:
            simplified |= self._simplify_node(node.left)
        if node.right:
            simplified |= self._simplify_node(node.right)

        # 检查当前节点是否可以简化
        if node.data in self.opposing_operations:
            # 检查子节点是否是对立操作
            if node.left and node.left.data in FUNCTIONS:
                if node.left.data == self.opposing_operations.get(node.data):
                    # 找到对立操作对，简化它们
                    # 例如: sqrt(sqr(X)) -> X
                    if node.left.left:
                        # 用子节点的子节点替换当前节点
                        new_data = node.left.left.data
                        new_left = None
                        new_right = None

                        # 安全地获取left和right
                        if hasattr(node.left.left, 'left'):
                            new_left = node.left.left.left
                        if hasattr(node.left.left, 'right'):
                            new_right = node.left.left.right

                        # 更新节点
                        node.data = new_data
                        node.left = new_left
                        node.right = new_right
                        simplified = True

        # 其他简化规则
        if node.data == mul:
            # 检查是否是 X * X -> sqr(X)
            if (node.left and node.right and
                    node.left.data == node.right.data and
                    node.left.data in TERMINALS):
                node.data = sqr
                node.right = None
                simplified = True

        elif node.data == add:
            # 检查是否是 X + X -> 2*X (但我们不能表示常数乘法，所以跳过)
            pass

        elif node.data == sub:
            # 检查是否是 X - X -> 0 (但我们不能表示0，所以跳过)
            pass

        elif node.data == div:
            # 检查是否是 X / X -> 1 (但我们不能表示1，所以跳过)
            pass

        # 简化嵌套的abs
        if node.data == abs and node.left and node.left.data == abs:
            # abs(abs(X)) -> abs(X)
            if node.left.left:  # 添加检查
                node.left = node.left.left
                simplified = True

        # 简化嵌套的neg
        if node.data == neg and node.left and node.left.data == neg:
            # neg(neg(X)) -> X
            if node.left.left:  # 添加检查
                new_data = node.left.left.data
                new_left = None
                new_right = None

                # 安全地获取left和right
                if hasattr(node.left.left, 'left'):
                    new_left = node.left.left.left
                if hasattr(node.left.left, 'right'):
                    new_right = node.left.left.right

                # 更新节点
                node.data = new_data
                node.left = new_left
                node.right = new_right
                simplified = True

        return simplified

    def evaluate_individual(self, individual: Individual) -> float:
        """评估个体 - 带早停"""
        simplified_tree = self.simplify_tree(individual.tree)
        simplified_str = str(simplified_tree)

        # 检查是否已评估
        if simplified_str in self.evaluated_trees:
            return self.evaluated_trees[simplified_str]

        # 快速检查是否包含危险操作
        if self.contains_dangerous_ops(simplified_tree):
            logging.warning(f"跳过危险指标: {simplified_str[:50]}")
            self.evaluated_trees[simplified_str] = 10000.0
            return 10000.0

        if self.evaluator:
            fitness = self.evaluator.evaluate_metric(simplified_tree)
        else:
            fitness = random.uniform(6.0, 10.0)

        self.evaluated_trees[simplified_str] = fitness
        return fitness

    def contains_dangerous_ops(self, tree: GPTree) -> bool:
        """检查是否包含数值不稳定的操作组合"""
        # 检查log(G)被用作除数
        if self.check_pattern(tree, pattern="div_log"):
            return True
        # 检查neg在根节点
        if tree.data == neg:
            return True
        # 其他危险模式...
        return False

    def check_pattern(self, tree: GPTree, pattern: str) -> bool:
        """检查树中是否包含特定的危险模式"""
        if pattern == "div_log":
            return self._has_div_log_pattern(tree)
        # 可以添加其他模式
        elif pattern == "log_zero":
            return self._has_log_zero_pattern(tree)
        elif pattern == "deep_nesting":
            return self._has_deep_nesting(tree)

        return False

    def _has_div_log_pattern(self, tree: GPTree) -> bool:
        """检查是否有log(G)作为除数的危险模式"""
        if tree is None:
            return False

        # 检查当前节点是否是除法
        if tree.data == div:
            # 检查右子树（除数）是否包含log(G)
            if tree.right and self._contains_log_g(tree.right):
                return True

        # 递归检查子树
        left_has_pattern = tree.left and self._has_div_log_pattern(tree.left)
        right_has_pattern = tree.right and self._has_div_log_pattern(tree.right)

        return left_has_pattern or right_has_pattern

    def _contains_log_g(self, tree: GPTree) -> bool:
        """检查树中是否包含log(G)"""
        if tree is None:
            return False

        # 检查当前节点是否是log(G)
        if tree.data == log and tree.left and tree.left.data == 'G':
            return True

        # 递归检查子树
        left_has_log_g = tree.left and self._contains_log_g(tree.left)
        right_has_log_g = tree.right and self._contains_log_g(tree.right)

        return left_has_log_g or right_has_log_g

    def _has_log_zero_pattern(self, tree: GPTree) -> bool:
        """检查是否有可能导致log(0)的模式"""
        # 这里可以实现检测可能导致log(0)的模式
        # 暂时返回False
        return False

    def _has_deep_nesting(self, tree: GPTree, max_depth: int = 8) -> bool:
        """检查是否有过深的嵌套"""

        def get_depth(node):
            if node is None or node.data in TERMINALS:
                return 0
            left_depth = get_depth(node.left) if node.left else 0
            right_depth = get_depth(node.right) if node.right else 0
            return 1 + max(left_depth, right_depth)

        return get_depth(tree) > max_depth

    def evolve(self) -> Individual:
        """主进化循环"""
        # 初始化种群
        if not self.population:
            self.initialize_population()

        # 评估初始种群
        logging.info("评估初始种群...")
        for individual in tqdm(self.population, desc="初始评估"):
            individual.fitness = self.evaluate_individual(individual)

        # 排序种群
        self.population.sort()
        self.best_individual = self.population[0].copy()

        logging.info(f"初始最佳适应度: {self.best_individual.fitness:.4f}")
        logging.info(f"初始最佳指标: {self.best_individual.tree_str}")

        # 如果使用OWL，显示层级稀疏率信息
        if self.config.use_owl and hasattr(self.evaluator, 'layer_sparsity_ratios'):
            if self.evaluator.layer_sparsity_ratios is not None:
                logging.info("OWL层级稀疏率分布（前10层）:")
                for i in range(min(10, len(self.evaluator.layer_sparsity_ratios))):
                    logging.info(f"  层 {i}: {self.evaluator.layer_sparsity_ratios[i]:.3f}")

        # 记录无改进代数
        no_improvement = 0

        # 进化循环
        for generation in range(self.generation, self.config.max_iterations):
            self.generation = generation
            start_time = time.time()

            logging.info(f"\n===== 第 {generation + 1}/{self.config.max_iterations} 代 =====")

            # 新种群
            new_population = []

            # 保留精英
            elite = self.population[:self.config.elite_size]
            new_population.extend([ind.copy() for ind in elite])

            # 生成后代
            while len(new_population) < self.config.population_size:
                # 锦标赛选择
                parents = self.tournament_selection(k=2)
                if len(parents) >= 2:
                    parent1, parent2 = parents[0], parents[1]
                else:
                    parent1 = parent2 = parents[0] if parents else self.population[0]

                # 交叉
                offspring = self.crossover(parent1, parent2)

                # 突变
                offspring = self.mutate(offspring)

                # 简化（OOS）
                offspring.tree = self.simplify_tree(offspring.tree)
                offspring.tree_str = offspring.get_tree_string()

                # 评估
                offspring.fitness = self.evaluate_individual(offspring)

                new_population.append(offspring)

            # 更新种群
            self.population = new_population
            self.population.sort()

            # 更新最佳个体
            if self.population[0].fitness < self.best_individual.fitness:
                self.best_individual = self.population[0].copy()
                # 简化最佳个体
                self.best_individual.tree = self.simplify_tree(self.best_individual.tree)
                self.best_individual.tree_str = self.best_individual.get_tree_string()

                logging.info(f"🎉 发现新的最佳适应度: {self.best_individual.fitness:.4f}")
                logging.info(f"最佳指标（简化后）: {self.best_individual.tree_str}")
                no_improvement = 0
            else:
                no_improvement += 1

            # 统计
            fitnesses = [ind.fitness for ind in self.population if ind.fitness < 10000]
            if fitnesses:
                stats = {
                    'generation': generation + 1,
                    'best_fitness': self.best_individual.fitness,
                    'avg_fitness': np.mean(fitnesses),
                    'std_fitness': np.std(fitnesses),
                    'worst_fitness': max(fitnesses),
                    'time': time.time() - start_time,
                    'use_owl': self.config.use_owl
                }
            else:
                stats = {
                    'generation': generation + 1,
                    'best_fitness': self.best_individual.fitness,
                    'avg_fitness': 10000.0,
                    'std_fitness': 0.0,
                    'worst_fitness': 10000.0,
                    'time': time.time() - start_time,
                    'use_owl': self.config.use_owl
                }

            self.generation_history.append(stats)

            logging.info(f"当前最佳: {stats['best_fitness']:.4f}, "
                         f"平均: {stats['avg_fitness']:.4f} ± {stats['std_fitness']:.4f}, "
                         f"最差: {stats['worst_fitness']:.4f}, "
                         f"用时: {stats['time']:.2f}s")

            # 统计包含X的个体数量
            x_count = sum(1 for ind in self.population if 'X' in ind.tree_str)
            logging.info(f"包含X的个体数: {x_count}/{len(self.population)}")

            # 保存检查点
            if (generation + 1) % self.config.checkpoint_freq == 0:
                os.makedirs("checkpoints", exist_ok=True)
                checkpoint_path = f"checkpoints/ga_gen_{generation + 1}_{'owl' if self.config.use_owl else 'std'}.json"
                self.save_checkpoint(checkpoint_path)

            # 早停
            if no_improvement >= 30:
                logging.info(f"连续 {no_improvement} 代无改进，提前停止")
                break

            # 找到好解
            if self.best_individual.fitness < 15.0 and no_improvement >= 10:
                logging.info(f"找到满意解 (PPL={self.best_individual.fitness:.4f})，停止搜索")
                break

        # 最终简化
        self.best_individual.tree = self.simplify_tree(self.best_individual.tree)
        self.best_individual.tree_str = self.best_individual.get_tree_string()
        logging.info(f"最终最佳指标（简化后）: {self.best_individual.tree_str}")

        return self.best_individual

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            'generation': self.generation,
            'population': [ind.to_dict() for ind in self.population[:50]],
            'best_individual': self.best_individual.to_dict() if self.best_individual else None,
            'history': self.generation_history[-100:],
            'config': self.config.__dict__
        }
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logging.info(f"保存检查点到 {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)

        self.generation = checkpoint['generation']
        self.population = [Individual.from_dict(ind) for ind in checkpoint['population']]
        if checkpoint['best_individual']:
            self.best_individual = Individual.from_dict(checkpoint['best_individual'])
        self.generation_history = checkpoint['history']
        logging.info(f"从 {filepath} 加载检查点, 当前代数: {self.generation}")