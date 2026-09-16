"""
Visualization Script: Layer-wise Representation Perturbation Analysis
=====================================================================
Fixed version with proper import handling.

Usage (from project root directory):
    python lib/visualize_perturbation.py \
        --model meta-llama/Llama-2-7b-hf \
        --sparsity_ratio 0.6 \
        --methods magnitude wanda pruner-zero-owl \
        --gradient_path /path/to/gradients.pt \
        --json_tree data/best_tree.json
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import argparse
import os
import sys
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple

# ============ 修复导入路径 ============
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (lib的父目录)
project_root = os.path.dirname(current_dir)
# 如果脚本在根目录，project_root就是当前目录
if os.path.basename(current_dir) != 'lib':
    project_root = current_dir

# 添加项目根目录到sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

# 现在导入lib模块
try:
    from lib.prune import (
        prune_wanda,
        prune_magnitude,
        prune_sparsegpt,
        prune_ria,
        prune_pruner_zero,
        prune_pruner_zero_owl,
        find_layers,
        prepare_calibration_input
    )
    from lib.data import get_loaders
    from lib.gptree import GPTree
    print("Successfully imported from lib.prune")
except ImportError as e:
    # 如果还是失败，尝试相对导入
    print(f"Import error: {e}")
    print("Trying relative imports...")
    from prune import (
        prune_wanda,
        prune_magnitude,
        prune_sparsegpt,
        prune_ria,
        prune_pruner_zero,
        prune_pruner_zero_owl,
        find_layers,
        prepare_calibration_input
    )
    from data import get_loaders
    from gptree import GPTree
    print("Successfully imported using relative imports")
# =====================================

# Publication-quality figure settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def get_llm(model_name, cache_dir="llm_weights"):
    """Load a pretrained LLM model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    return model


class LayerOutputCapture:
    """Hook-based class to capture layer outputs during forward pass."""

    def __init__(self, model):
        self.model = model
        self.layer_outputs = {}
        self.hooks = []

    def _hook_fn(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            self.layer_outputs[layer_idx] = out.detach().cpu()
        return hook

    def register_hooks(self):
        """Register forward hooks on all transformer layers."""
        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._hook_fn(i))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_outputs(self):
        """Clear captured outputs."""
        self.layer_outputs = {}

    def get_outputs(self) -> Dict[int, torch.Tensor]:
        """Return captured layer outputs."""
        return self.layer_outputs


def compute_representation_similarity(outputs_dense: Dict[int, torch.Tensor],
                                       outputs_pruned: Dict[int, torch.Tensor]) -> Dict[int, float]:
    """Compute cosine similarity between dense and pruned model layer outputs."""
    similarities = {}

    for layer_idx in outputs_dense.keys():
        if layer_idx not in outputs_pruned:
            continue

        dense_out = outputs_dense[layer_idx].float()
        pruned_out = outputs_pruned[layer_idx].float()

        dense_flat = dense_out.reshape(dense_out.shape[0], -1)
        pruned_flat = pruned_out.reshape(pruned_out.shape[0], -1)

        cos_sim = torch.nn.functional.cosine_similarity(dense_flat, pruned_flat, dim=1)
        similarities[layer_idx] = cos_sim.mean().item()

    return similarities


def compute_relative_error(outputs_dense: Dict[int, torch.Tensor],
                           outputs_pruned: Dict[int, torch.Tensor]) -> Dict[int, float]:
    """Compute relative L2 error between dense and pruned model layer outputs."""
    errors = {}

    for layer_idx in outputs_dense.keys():
        if layer_idx not in outputs_pruned:
            continue

        dense_out = outputs_dense[layer_idx].float()
        pruned_out = outputs_pruned[layer_idx].float()

        diff_norm = torch.norm(pruned_out - dense_out, p=2, dim=-1).mean()
        dense_norm = torch.norm(dense_out, p=2, dim=-1).mean()

        rel_error = (diff_norm / (dense_norm + 1e-8)).item()
        errors[layer_idx] = rel_error

    return errors


def run_evaluation(model, dataloader, device, nsamples=8):
    """Run forward pass and capture all layer outputs."""
    capture = LayerOutputCapture(model)
    capture.register_hooks()

    all_outputs = []

    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= nsamples:
                break
            capture.clear_outputs()
            inp = batch[0].to(device)
            _ = model(inp)
            all_outputs.append(copy.deepcopy(capture.get_outputs()))
            sample_count += 1

    capture.remove_hooks()

    # Average outputs across samples
    avg_outputs = {}
    if len(all_outputs) > 0:
        num_layers = len(all_outputs[0])
        for layer_idx in range(num_layers):
            layer_outs = [out[layer_idx] for out in all_outputs if layer_idx in out]
            if layer_outs:
                avg_outputs[layer_idx] = torch.stack(layer_outs).mean(dim=0)

    return avg_outputs


def create_args_for_pruning(base_args, method_name):
    """Create args object for specific pruning method."""
    args = argparse.Namespace(**vars(base_args))
    args.prune_method = method_name
    args.use_variant = False
    args.save = None
    args.save_model = None

    if method_name in ['ria', 'ria-owl']:
        args.a = getattr(base_args, 'a', 0.5)
        args.per_outneuron = getattr(base_args, 'per_outneuron', False)

    if 'owl' in method_name:
        args.Hyper_m = getattr(base_args, 'Hyper_m', 5)
        args.Lamda = getattr(base_args, 'Lamda', 0.08)

    return args


def plot_combined_figure(similarity_results: Dict[str, Dict[int, float]],
                         error_results: Dict[str, Dict[int, float]],
                         sparsity: float,
                         model_name: str,
                         output_path: str):
    """Create the main paper figure with two subplots."""

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    colors = {
        'Magnitude': '#E74C3C',
        'Wanda': '#3498DB',
        'RIA': '#9B59B6',
        'SparseGPT': '#F39C12',
        'Pruner-Zero': '#1ABC9C',
        'Ours': '#27AE60',
    }

    markers = {
        'Magnitude': 's',
        'Wanda': '^',
        'RIA': 'd',
        'SparseGPT': 'v',
        'Pruner-Zero': 'p',
        'Ours': 'o',
    }

    linestyles = {
        'Magnitude': ':',
        'Wanda': '-.',
        'RIA': '--',
        'SparseGPT': (0, (3, 1, 1, 1)),
        'Pruner-Zero': '--',
        'Ours': '-',
    }

    # Plot (a): Cosine Similarity
    ax1 = axes[0]
    for method_name, layer_metrics in similarity_results.items():
        layers = sorted(layer_metrics.keys())
        values = [layer_metrics[l] for l in layers]
        ax1.plot(layers, values,
                 label=method_name,
                 color=colors.get(method_name, '#333333'),
                 marker=markers.get(method_name, 'o'),
                 linestyle=linestyles.get(method_name, '-'),
                 linewidth=2.2, markersize=4,
                 markerfacecolor='white', markeredgewidth=1.5, markevery=3)

    ax1.set_xlabel('Layer Index', fontweight='bold')
    ax1.set_ylabel('Cosine Similarity', fontweight='bold')
    ax1.set_title('(a) Layer-wise Representation Similarity', fontweight='bold', pad=10)
    ax1.legend(loc='lower left', framealpha=0.95, edgecolor='gray')
    ax1.set_ylim([0.5, 1.02])
    ax1.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Plot (b): Relative Error
    ax2 = axes[1]
    for method_name, layer_metrics in error_results.items():
        layers = sorted(layer_metrics.keys())
        values = [layer_metrics[l] for l in layers]
        ax2.plot(layers, values,
                 label=method_name,
                 color=colors.get(method_name, '#333333'),
                 marker=markers.get(method_name, 'o'),
                 linestyle=linestyles.get(method_name, '-'),
                 linewidth=2.2, markersize=4,
                 markerfacecolor='white', markeredgewidth=1.5, markevery=3)

    ax2.set_xlabel('Layer Index', fontweight='bold')
    ax2.set_ylabel('Relative L2 Error', fontweight='bold')
    ax2.set_title('(b) Layer-wise Perturbation Accumulation', fontweight='bold', pad=10)
    ax2.legend(loc='upper left', framealpha=0.95, edgecolor='gray')

    fig.suptitle(f'Pruning-induced Perturbation Analysis ({model_name}, Sparsity={int(sparsity*100)}%)',
                 fontweight='bold', fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize pruning-induced perturbation')
    parser.add_argument('--model', type=str, default='/home/yh114/workdir/DSnoT/models/Llama-2-7b-hf',required=True, help='Model name or path')
    parser.add_argument('--sparsity_ratio', type=float, default=0.6, help='Sparsity level')
    parser.add_argument('--nsamples', type=int, default=128, help='Calibration samples')
    parser.add_argument('--eval_samples', type=int, default=8, help='Evaluation samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cache_dir', type=str, default='llm_weights', help='Cache directory')
    parser.add_argument('--output_dir', type=str, default='./visualization_results', help='Output dir')

    parser.add_argument('--gradient_path', type=str, default='/home/yh114/workdir/Pruner-Zero/gradients/llama2/gradients_aggregrate_norm_l2_model_Llama-2-7b-hf_128_0.pth', help='Gradient file path')
    parser.add_argument('--json_tree', type=str, default='data/best_tree.json', help='GPTree json')
    parser.add_argument('--Hyper_m', type=float, default=6, help='OWL Hyper_m')
    parser.add_argument('--Lamda', type=float, default=0.19, help='OWL Lambda')
    parser.add_argument('--a', type=float, default=0.5, help='RIA exponent')
    parser.add_argument('--per_outneuron', action='store_true')

    parser.add_argument('--methods', type=str, nargs='+',
                        default=['magnitude', 'wanda', 'pruner-zero-owl', 'sparsegpt', 'ria'],
                        choices=['magnitude', 'wanda', 'sparsegpt', 'ria', 'pruner-zero', 'pruner-zero-owl'],
                        help='Methods to compare')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    print(f"Loading calibration data (n={args.nsamples})...")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                 seqlen=2048, tokenizer=tokenizer)

    # Load GPTree if needed
    engine = None
    if any('pruner-zero' in m for m in args.methods):
        json_tree_path = os.path.join(project_root, args.json_tree) if not os.path.isabs(args.json_tree) else args.json_tree
        if os.path.exists(json_tree_path):
            engine = GPTree.load_tree(json_tree_path)
            print(f"Loaded GPTree from {json_tree_path}")
        elif os.path.exists(args.json_tree):
            engine = GPTree.load_tree(args.json_tree)
            print(f"Loaded GPTree from {args.json_tree}")
        else:
            print(f"Warning: GPTree not found at {args.json_tree}")

    # Step 1: Dense model outputs
    print("\n" + "="*60)
    print("Step 1: Capturing dense model outputs...")
    print("="*60)

    model_dense = get_llm(args.model, args.cache_dir)
    model_dense.eval()
    model_dense.seqlen = 2048

    outputs_dense = run_evaluation(model_dense, dataloader, device, args.eval_samples)
    print(f"Captured outputs from {len(outputs_dense)} layers")

    del model_dense
    torch.cuda.empty_cache()

    similarity_results = {}
    error_results = {}

    method_display_names = {
        'magnitude': 'Magnitude',
        'wanda': 'Wanda',
        'sparsegpt': 'SparseGPT',
        'ria': 'RIA',
        'pruner-zero': 'Pruner-Zero',
        'pruner-zero-owl': 'Ours',
    }

    # Step 2: Evaluate each method
    for method in args.methods:
        print("\n" + "="*60)
        print(f"Evaluating: {method}")
        print("="*60)

        model = get_llm(args.model, args.cache_dir)
        model.eval()
        model.seqlen = 2048

        prune_args = create_args_for_pruning(args, method)

        try:
            if method == 'magnitude':
                prune_magnitude(prune_args, model, tokenizer, device)
            elif method == 'wanda':
                prune_wanda(prune_args, model, tokenizer, device)
            elif method == 'sparsegpt':
                prune_sparsegpt(prune_args, model, tokenizer, device)
            elif method == 'ria':
                prune_ria(prune_args, model, tokenizer, device)
            elif method == 'pruner-zero':
                if engine is None:
                    print(f"Skipping {method}: GPTree not loaded")
                    del model
                    torch.cuda.empty_cache()
                    continue
                prune_pruner_zero(prune_args, model, tokenizer, device, engine=engine)
            elif method == 'pruner-zero-owl':
                if engine is None:
                    print(f"Skipping {method}: GPTree not loaded")
                    del model
                    torch.cuda.empty_cache()
                    continue
                prune_pruner_zero_owl(prune_args, model, tokenizer, device, engine=engine)

            outputs_pruned = run_evaluation(model, dataloader, device, args.eval_samples)

            display_name = method_display_names.get(method, method)
            similarity_results[display_name] = compute_representation_similarity(outputs_dense, outputs_pruned)
            error_results[display_name] = compute_relative_error(outputs_dense, outputs_pruned)

            sims = list(similarity_results[display_name].values())
            errs = list(error_results[display_name].values())
            print(f"\n{display_name} Summary:")
            print(f"  Avg Similarity: {np.mean(sims):.4f} (min: {np.min(sims):.4f})")
            print(f"  Avg Error: {np.mean(errs):.4f} (max: {np.max(errs):.4f})")

        except Exception as e:
            print(f"Error evaluating {method}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del model
            torch.cuda.empty_cache()

    # Step 3: Generate figures
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    if len(similarity_results) == 0:
        print("No results to visualize!")
        return

    model_short_name = args.model.split('/')[-1]

    plot_combined_figure(
        similarity_results, error_results,
        args.sparsity_ratio, model_short_name,
        os.path.join(args.output_dir, f'perturbation_analysis_s{int(args.sparsity_ratio*100)}.pdf')
    )
    plot_combined_figure(
        similarity_results, error_results,
        args.sparsity_ratio, model_short_name,
        os.path.join(args.output_dir, f'perturbation_analysis_s{int(args.sparsity_ratio*100)}.png')
    )

    # Save raw data
    import json
    results_data = {
        'model': args.model,
        'sparsity': args.sparsity_ratio,
        'similarity': {k: {str(kk): vv for kk, vv in v.items()} for k, v in similarity_results.items()},
        'error': {k: {str(kk): vv for kk, vv in v.items()} for k, v in error_results.items()},
    }
    with open(os.path.join(args.output_dir, f'results_s{int(args.sparsity_ratio*100)}.json'), 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nDone! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
