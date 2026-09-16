"""
Parameter Search Script for TSP
============================================
Search for optimal Hyper_m and Lamda parameters by evaluating
representation similarity and perturbation error.

Usage:
    python param_search.py \
        --model meta-llama/Llama-2-7b-hf \
        --sparsity_ratio 0.6 \
        --gradient_path /path/to/gradients.pt \
        --json_tree data/best_tree.json \
        --output_dir ./param_search_results
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
import json
import time
from datetime import datetime
from itertools import product

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple

# ============ 修复导入路径 ============
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if os.path.basename(current_dir) != 'lib':
    project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

try:
    from lib.prune import prune_pruner_zero_owl, find_layers
    from lib.data import get_loaders
    from lib.gptree import GPTree
    from lib.eval import eval_ppl

    print("Successfully imported from lib")
except ImportError:
    from prune import prune_pruner_zero_owl, find_layers
    from data import get_loaders
    from gptree import GPTree
    from eval import eval_ppl

    print("Successfully imported using relative imports")


# =====================================


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
        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._hook_fn(i))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_outputs(self):
        self.layer_outputs = {}

    def get_outputs(self) -> Dict[int, torch.Tensor]:
        return self.layer_outputs


def compute_metrics(outputs_dense: Dict[int, torch.Tensor],
                    outputs_pruned: Dict[int, torch.Tensor]) -> Dict:
    """
    Compute both similarity and error metrics.
    Returns summary statistics.
    """
    similarities = []
    errors = []

    for layer_idx in sorted(outputs_dense.keys()):
        if layer_idx not in outputs_pruned:
            continue

        dense_out = outputs_dense[layer_idx].float()
        pruned_out = outputs_pruned[layer_idx].float()

        # Cosine similarity
        dense_flat = dense_out.reshape(dense_out.shape[0], -1)
        pruned_flat = pruned_out.reshape(pruned_out.shape[0], -1)
        cos_sim = torch.nn.functional.cosine_similarity(dense_flat, pruned_flat, dim=1).mean().item()
        similarities.append(cos_sim)

        # Relative L2 error
        diff_norm = torch.norm(pruned_out - dense_out, p=2, dim=-1).mean()
        dense_norm = torch.norm(dense_out, p=2, dim=-1).mean()
        rel_error = (diff_norm / (dense_norm + 1e-8)).item()
        errors.append(rel_error)

    return {
        'avg_similarity': np.mean(similarities),
        'min_similarity': np.min(similarities),
        'final_similarity': similarities[-1] if similarities else 0,
        'avg_error': np.mean(errors),
        'max_error': np.max(errors),
        'final_error': errors[-1] if errors else 0,
        'layer_similarities': similarities,
        'layer_errors': errors,
    }


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


def evaluate_params(args, model_name, tokenizer, dataloader, outputs_dense,
                    engine, device, hyper_m, lamda):
    """
    Evaluate a specific parameter combination.
    Returns metrics dict.
    """
    print(f"\n  Testing Hyper_m={hyper_m}, Lamda={lamda}...")

    # Load fresh model
    model = get_llm(model_name, args.cache_dir)
    model.eval()
    model.seqlen = 2048

    # Create args for pruning
    prune_args = argparse.Namespace(
        model=model_name,
        sparsity_ratio=args.sparsity_ratio,
        nsamples=args.nsamples,
        seed=args.seed,
        gradient_path=args.gradient_path,
        Hyper_m=hyper_m,
        Lamda=lamda,
        use_variant=False,
        prune_method='TSP',
    )

    start_time = time.time()

    try:
        # Apply pruning
        prune_pruner_zero_owl(prune_args, model, tokenizer, device, engine=engine)
        prune_time = time.time() - start_time

        # Capture pruned model outputs
        outputs_pruned = run_evaluation(model, dataloader, device, args.eval_samples)

        # Compute metrics
        metrics = compute_metrics(outputs_dense, outputs_pruned)
        metrics['prune_time'] = prune_time
        metrics['status'] = 'success'

        # Optionally compute perplexity (slower but more accurate)
        if args.eval_ppl:
            ppl = eval_ppl(prune_args, model, tokenizer, device)
            metrics['perplexity'] = ppl
            print(f"    PPL: {ppl:.2f}")

        print(f"    Avg Sim: {metrics['avg_similarity']:.4f}, "
              f"Final Sim: {metrics['final_similarity']:.4f}, "
              f"Avg Err: {metrics['avg_error']:.4f}")

    except Exception as e:
        print(f"    Error: {e}")
        metrics = {
            'status': 'failed',
            'error': str(e),
            'avg_similarity': 0,
            'min_similarity': 0,
            'final_similarity': 0,
            'avg_error': float('inf'),
            'max_error': float('inf'),
            'final_error': float('inf'),
        }

    finally:
        del model
        torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Parameter search for TSP')
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--sparsity_ratio', type=float, default=0.6, help='Sparsity level')
    parser.add_argument('--nsamples', type=int, default=128, help='Calibration samples')
    parser.add_argument('--eval_samples', type=int, default=8, help='Evaluation samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cache_dir', type=str, default='llm_weights', help='Cache directory')
    parser.add_argument('--output_dir', type=str, default='./param_search_results', help='Output dir')

    parser.add_argument('--gradient_path', type=str, required=True, help='Gradient file path')
    parser.add_argument('--json_tree', type=str, default='data/best_tree.json', help='GPTree json')

    # Parameter search ranges
    parser.add_argument('--hyper_m_values', type=float, nargs='+',
                        default=[3, 4, 5, 6, 7, 8],
                        help='Hyper_m values to search')
    parser.add_argument('--lamda_values', type=float, nargs='+',
                        default=[0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.17, 0.19],
                        help='Lamda values to search')

    # Optional: also evaluate perplexity (slower)
    parser.add_argument('--eval_ppl', action='store_true', help='Also evaluate perplexity')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load tokenizer
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Load calibration data
    print(f"Loading calibration data (n={args.nsamples})...")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=2048, tokenizer=tokenizer)

    # Load GPTree
    json_tree_path = os.path.join(project_root, args.json_tree) if not os.path.isabs(args.json_tree) else args.json_tree
    if os.path.exists(json_tree_path):
        engine = GPTree.load_tree(json_tree_path)
    elif os.path.exists(args.json_tree):
        engine = GPTree.load_tree(args.json_tree)
    else:
        raise FileNotFoundError(f"GPTree not found: {args.json_tree}")
    print(f"Loaded GPTree")

    # ========== Step 1: Get Dense Model Outputs ==========
    print("\n" + "=" * 60)
    print("Step 1: Capturing dense model outputs (baseline)...")
    print("=" * 60)

    model_dense = get_llm(args.model, args.cache_dir)
    model_dense.eval()
    model_dense.seqlen = 2048

    outputs_dense = run_evaluation(model_dense, dataloader, device, args.eval_samples)
    print(f"Captured outputs from {len(outputs_dense)} layers")

    del model_dense
    torch.cuda.empty_cache()

    # ========== Step 2: Grid Search ==========
    print("\n" + "=" * 60)
    print("Step 2: Parameter Grid Search")
    print("=" * 60)
    print(f"Hyper_m values: {args.hyper_m_values}")
    print(f"Lamda values: {args.lamda_values}")
    print(f"Total combinations: {len(args.hyper_m_values) * len(args.lamda_values)}")

    results = []
    best_result = None
    best_score = -float('inf')

    total_combinations = len(args.hyper_m_values) * len(args.lamda_values)
    current = 0

    for hyper_m, lamda in product(args.hyper_m_values, args.lamda_values):
        current += 1
        print(f"\n[{current}/{total_combinations}] Hyper_m={hyper_m}, Lamda={lamda}")

        metrics = evaluate_params(
            args, args.model, tokenizer, dataloader, outputs_dense,
            engine, device, hyper_m, lamda
        )

        result = {
            'hyper_m': hyper_m,
            'lamda': lamda,
            **metrics
        }
        results.append(result)

        # Score: higher similarity + lower error
        # You can adjust this scoring function based on what matters most
        if metrics['status'] == 'success':
            # Option 1: Prioritize final layer performance
            # score = metrics['final_similarity'] - metrics['final_error']

            # Option 2: Balance average and final
            score = (metrics['avg_similarity'] + metrics['final_similarity']) / 2 - metrics['avg_error']

            # Option 3: If using perplexity (lower is better)
            if args.eval_ppl and 'perplexity' in metrics:
                score = -metrics['perplexity']  # Negative because lower PPL is better

            if score > best_score:
                best_score = score
                best_result = result
                print(f"    *** New best! Score: {score:.4f}")

        # Save intermediate results
        with open(os.path.join(args.output_dir, 'results_intermediate.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # ========== Step 3: Summary ==========
    print("\n" + "=" * 60)
    print("Step 3: Results Summary")
    print("=" * 60)

    # Sort by score
    successful_results = [r for r in results if r.get('status') == 'success']

    if args.eval_ppl:
        successful_results.sort(key=lambda x: x.get('perplexity', float('inf')))
        print("\nTop 5 by Perplexity (lower is better):")
        for i, r in enumerate(successful_results[:5]):
            print(f"  {i + 1}. Hyper_m={r['hyper_m']}, Lamda={r['lamda']}: "
                  f"PPL={r.get('perplexity', 'N/A'):.2f}, "
                  f"AvgSim={r['avg_similarity']:.4f}")
    else:
        successful_results.sort(key=lambda x: x['avg_similarity'], reverse=True)
        print("\nTop 5 by Average Similarity (higher is better):")
        for i, r in enumerate(successful_results[:5]):
            print(f"  {i + 1}. Hyper_m={r['hyper_m']}, Lamda={r['lamda']}: "
                  f"AvgSim={r['avg_similarity']:.4f}, "
                  f"FinalSim={r['final_similarity']:.4f}, "
                  f"AvgErr={r['avg_error']:.4f}")

    print("\n" + "-" * 40)
    if best_result:
        print(f"BEST PARAMETERS:")
        print(f"  Hyper_m = {best_result['hyper_m']}")
        print(f"  Lamda = {best_result['lamda']}")
        print(f"  Avg Similarity: {best_result['avg_similarity']:.4f}")
        print(f"  Final Similarity: {best_result['final_similarity']:.4f}")
        print(f"  Avg Error: {best_result['avg_error']:.4f}")
        if args.eval_ppl and 'perplexity' in best_result:
            print(f"  Perplexity: {best_result['perplexity']:.2f}")

    # Save final results
    final_output = {
        'model': args.model,
        'sparsity_ratio': args.sparsity_ratio,
        'search_space': {
            'hyper_m_values': args.hyper_m_values,
            'lamda_values': args.lamda_values,
        },
        'best_params': {
            'hyper_m': best_result['hyper_m'] if best_result else None,
            'lamda': best_result['lamda'] if best_result else None,
        },
        'best_metrics': best_result,
        'all_results': results,
        'timestamp': datetime.now().isoformat(),
    }

    output_file = os.path.join(args.output_dir, f'param_search_s{int(args.sparsity_ratio * 100)}.json')
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Also save a simple summary
    summary_file = os.path.join(args.output_dir, f'best_params_s{int(args.sparsity_ratio * 100)}.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Sparsity: {args.sparsity_ratio}\n")
        f.write(f"Best Hyper_m: {best_result['hyper_m'] if best_result else 'N/A'}\n")
        f.write(f"Best Lamda: {best_result['lamda'] if best_result else 'N/A'}\n")
        if best_result:
            f.write(f"Avg Similarity: {best_result['avg_similarity']:.4f}\n")
            f.write(f"Final Similarity: {best_result['final_similarity']:.4f}\n")
            f.write(f"Avg Error: {best_result['avg_error']:.4f}\n")
            if args.eval_ppl and 'perplexity' in best_result:
                f.write(f"Perplexity: {best_result['perplexity']:.2f}\n")

    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()