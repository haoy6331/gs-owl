import time
import heapq
import gc
import torch 
import torch.nn as nn
import numpy as np
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .prune import (
    V9_PRUNING_METRIC_LABELS,
    _get_v9_gradient,
    build_v9_pruning_metric,
    build_wanda_mask,
    owl_density_from_outlier_ratios,
    renormalize_density_with_bounds,
)

from .ablate import AblateGPT 


def _module_device(module, fallback=torch.device("cuda:0")):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def _opt_input_device(model, fallback=torch.device("cuda:0")):
    decoder = getattr(getattr(model, "model", None), "decoder", None)
    embed_tokens = getattr(decoder, "embed_tokens", None)
    if embed_tokens is not None:
        return embed_tokens.weight.device
    return torch.device(fallback)


def _move_calibration_buffers(inps, outs, attention_mask, device):
    inps = inps.to(device)
    outs = outs.to(device)
    attention_mask = attention_mask.to(device) if attention_mask is not None else None
    return inps, outs, attention_mask

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.decoder.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    device = _opt_input_device(model, device)

    dtype = next(iter(model.parameters())).dtype
    nsamples = len(dataloader) if hasattr(dataloader, "__len__") else 128
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.decoder.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten())[0][int(W.numel()*args.sparsity_ratio)]
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        dev = _module_device(layer, device)
        inps, outs, attention_mask = _move_calibration_buffers(
            inps, outs, attention_mask, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def _safe_to_device(tensor, device):
    return tensor.to(device) if tensor is not None else None


def prune_metric_owl_v9_opt(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                            metric_name=None):
    if metric_name is None:
        raise ValueError("metric_name must be provided for prune_metric_owl_v9_opt")
    return _prune_owl_v9_opt(
        args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m,
        pruning_metric=metric_name)


def _prune_owl_v9_opt(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0,
                      pruning_metric="wsqrtg"):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if pruning_metric not in V9_PRUNING_METRIC_LABELS:
        raise ValueError(f"Unknown OWL-V9 pruning metric: {pruning_metric}")
    if pruning_metric != "wanda" and args.gradient_path is None:
        raise ValueError(f"--gradient_path is required for OWL-V9 metric '{pruning_metric}'")

    gradients = None
    if pruning_metric != "wanda":
        print(f"loading gradients from {args.gradient_path}")
        gradients = torch.load(args.gradient_path, map_location=torch.device("cpu"))
        if not isinstance(gradients, dict):
            raise ValueError("The gradient file must contain a dictionary of layer tensors")
        print(f"loaded {len(gradients)} gradient tensors")

    Hyper_m = getattr(args, "Hyper_m", 5)
    Lamda = getattr(args, "Lamda", 0.08)
    anchor_alpha = float(np.clip(getattr(args, "Owl_alpha", 0.2), 0.0, 1.0))
    core_start = 4
    core_end = 10
    tail_start = 16
    tail_end = 30
    eps = 1e-12

    print("=" * 50)
    print("OPT OWL-V9 pruning starts (Core-Preserving Anchored OWL)")
    print(f"Stage-2 pruning metric: {pruning_metric} = {V9_PRUNING_METRIC_LABELS[pruning_metric]}")
    print(f"Hyper_m(outlier multiplier): {Hyper_m}, Lambda: {Lamda}, "
          f"anchor_alpha: {anchor_alpha}, core_layers: {core_start}-{core_end}, "
          f"tail_cap_layers: {tail_start}-{tail_end}")
    print("=" * 50)

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers
    layer_key_template = "model.decoder.layers.{}"
    count_ratios = []
    severity_ratios = []

    print("Stage 1: collecting OPT V9 core-preserving OWL ratios...")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)

        dev = _module_device(layer, device)
        inps, outs, attention_mask = _move_calibration_buffers(
            inps, outs, attention_mask, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_abs = torch.abs(subset[name].weight.data)
            X_l2 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            outlier_score = W_abs * X_l2
            layer_wmetric.append(outlier_score.cpu())

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric]).float()
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_mask = layer_wmetric > threshold
        outlier_count = outlier_mask.sum().item()
        count_ratio = outlier_count / layer_wmetric.numel() * 100
        tail_mass_ratio = layer_wmetric[outlier_mask].sum() / (layer_wmetric.sum() + eps) * 100

        if outlier_count > 0:
            relative_excess = (layer_wmetric[outlier_mask] - threshold) / (threshold + eps)
            severity = torch.log1p(relative_excess.mean()).item()
            severity_ratio = count_ratio * (1.0 + severity)
        else:
            severity = 0.0
            severity_ratio = 0.0

        print(f"layer {i} V9 raw ratios: count={count_ratio:.2f}%, "
              f"severity_ratio={severity_ratio:.2f}% "
              f"(mean={mean_val.item():.6e}, threshold={threshold.item():.6e}, "
              f"severity={severity:.4f}, tail_mass={tail_mass_ratio.item():.2f}%)")
        count_ratios.append(count_ratio)
        severity_ratios.append(severity_ratio)

    torch.cuda.empty_cache()

    print("\nRaw V9 true OWL count ratios:", [f"{x:.2f}" for x in count_ratios])
    print("Raw V9 severity-weighted ratios:", [f"{x:.2f}" for x in severity_ratios])

    count_density, count_sparsity = owl_density_from_outlier_ratios(
        count_ratios, args.sparsity_ratio, Lamda)
    severity_density, severity_sparsity = owl_density_from_outlier_ratios(
        severity_ratios, args.sparsity_ratio, Lamda)

    anchored_density = (1.0 - anchor_alpha) * count_density + anchor_alpha * severity_density
    constrained_density = anchored_density.copy()
    n_layers = constrained_density.size

    core_mask = np.zeros(n_layers, dtype=bool)
    lo = min(max(core_start, 0), n_layers)
    hi = min(max(core_end + 1, 0), n_layers)
    core_mask[lo:hi] = True
    constrained_density[core_mask] = np.maximum(constrained_density[core_mask], count_density[core_mask])

    tail_lo = min(max(tail_start, 0), n_layers)
    tail_hi = min(max(tail_end + 1, 0), n_layers)
    if tail_lo < tail_hi:
        constrained_density[tail_lo:tail_hi] = np.minimum(
            constrained_density[tail_lo:tail_hi], count_density[tail_lo:tail_hi])

    all_layer_ratio, layer_sparsity_ratios = renormalize_density_with_bounds(
        constrained_density, args.sparsity_ratio, Lamda, protected_mask=core_mask)

    print(f"Count-OWL sparsity - mean: {np.mean(count_sparsity):.3f}, "
          f"max: {np.max(count_sparsity):.3f}, min: {np.min(count_sparsity):.3f}")
    print(f"Severity-OWL sparsity - mean: {np.mean(severity_sparsity):.3f}, "
          f"max: {np.max(severity_sparsity):.3f}, min: {np.min(severity_sparsity):.3f}")
    print(f"Adjusted sparsity - mean: {np.mean(layer_sparsity_ratios):.3f}, "
          f"max: {np.max(layer_sparsity_ratios):.3f}, min: {np.min(layer_sparsity_ratios):.3f}")
    print("Count-OWL density ratios:", [f"{x:.3f}" for x in count_density])
    print("Severity-OWL density ratios:", [f"{x:.3f}" for x in severity_density])
    print("Anchored density ratios before constraints:", [f"{x:.3f}" for x in anchored_density])
    print("Adjusted sparsity ratios:", [f"{x:.3f}" for x in layer_sparsity_ratios])
    print("Adjusted density ratios:", [f"{x:.3f}" for x in all_layer_ratio])

    # Stage 2 rebuilds calibration activations. Release Stage 1 tensors first to
    # avoid doubling multi-GB activation buffers on larger OPT-style runs.
    del inps, outs, attention_mask
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nStage 2: applying {V9_PRUNING_METRIC_LABELS[pruning_metric]} "
          "with OPT V9 OWL layer sparsity...")
    print("=" * 50)

    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_key = layer_key_template.format(i)
        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = float(np.clip(layer_sparsity_ratio, 0.0, 0.99))
        print(f"\nlayer {i}: density={all_layer_ratio[i]:.3f}, sparsity={layer_sparsity_ratio:.3f}")

        dev = _module_device(layer, device)
        inps, outs, attention_mask = _move_calibration_buffers(
            inps, outs, attention_mask, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"  pruning {name}")
            gradient = None
            gradient_key = None
            if gradients is not None:
                gradient, gradient_key = _get_v9_gradient(gradients, i, name)

            W_metric = build_v9_pruning_metric(
                pruning_metric,
                subset[name].weight.data,
                gradient,
                wrapped_layers[name].scaler_row,
            )
            if gradient_key is not None:
                print(f"    gradient key: {gradient_key}")

            W_mask = build_wanda_mask(
                W_metric,
                layer_sparsity_ratio,
                use_variant=getattr(args, 'use_variant', False),
                prune_n=prune_n,
                prune_m=prune_m,
            )

            subset[name].weight.data[W_mask] = 0
            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    actual sparsity: {actual_sparsity:.3f}")

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print(f"OPT OWL-V9 pruning complete! metric={pruning_metric}")

@torch.no_grad()
@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    print('Starting ...')
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    dev = _opt_input_device(model, dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # 调试点 1
    print("Before first empty_cache")
    torch.cuda.synchronize()
    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"Error at first empty_cache: {e}")
        # 可以选择继续执行
        pass

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        print(f"\nProcessing layer {i}/{len(layers)}")
        layer = layers[i]

        dev = _module_device(layer, dev)
        print(f"layer {i} device {dev}")
        inps, outs, attention_mask = _move_calibration_buffers(
            inps, outs, attention_mask, dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            # 在剪枝前同步
            torch.cuda.synchronize()
            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer

        # 调试点 2
        print(f"Before empty_cache for layer {i}")
        torch.cuda.synchronize()
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Error at layer {i} empty_cache: {e}")
            pass

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    # 调试点 3
    print("Before final empty_cache")
    torch.cuda.synchronize()
    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"Error at final empty_cache: {e}")
        pass

@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    dev = _opt_input_device(model, dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        dev = _module_device(layer, dev)
        print(f"layer {i} device {dev}")
        inps, outs, attention_mask = _move_calibration_buffers(
            inps, outs, attention_mask, dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, 
                                        prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_pruner_zero(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, engine=None):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    with open(args.gradient_path, 'rb') as file:
        gradients = torch.load(
            args.gradient_path, map_location=torch.device('cpu'))

    layers = model.model.decoder.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        dev = _module_device(layer, device)
        inps, outs, attention_mask = _move_calibration_buffers(
            inps, outs, attention_mask, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            indexed_name = f'{name}_layer_{i}'
            print(f"pruning layer {i} name {name}")
            # W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            W = torch.abs(subset[name].weight.data)
            G = gradients[indexed_name]
            X = wrapped_layers[name].scaler_row.reshape((1, -1))  # 添加这行
            W_metric = engine.forward(
                W.to(dtype=torch.float32),
                G.to(device=W.device, dtype=torch.float32),
                X.to(device=W.device, dtype=torch.float32),  # 添加X参数
            )
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

def prune_TSP_opt(args, model, tokenizer, device=torch.device("cuda:0"),
                              prune_n=0, prune_m=0, engine=None):
    """
    OPT模型的Pruner-Zero with OWL剪枝
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # 检查必要参数
    if args.gradient_path is None:
        raise ValueError("gradient_path参数未指定")
    if engine is None:
        raise ValueError("需要GPTree engine，请指定--json_tree参数")

    # 加载梯度
    try:
        with open(args.gradient_path, 'rb') as file:
            gradients = torch.load(file, map_location=torch.device('cpu'))
    except Exception as e:
        raise RuntimeError(f"加载梯度文件错误: {e}")

    print("=" * 50)
    print("OPT模型 OWL-Pruner-Zero剪枝开始")
    print("=" * 50)

    # ========== 第一阶段：计算每层的异常值比例 ==========
    print("第一阶段：计算LOD（层级异常值分布）...")

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(
            model, dataloader, device)

    layers = model.model.decoder.layers  # OPT模型结构
    all_layer_ratio = []

    # 第一遍遍历：计算异常值
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # 处理多GPU - OPT的设备映射
        dev = _module_device(layer, device)
        inps, outs, attention_mask = _move_calibration_buffers(
            inps, outs, attention_mask, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                # OPT前向传播：不使用position_ids
                outs[j] = layer(inps[j].unsqueeze(0),
                                attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        # 收集当前层的W_metric
        layer_wmetric = []

        for name in subset:
            indexed_name = f'{name}_layer_{i}'

            W = torch.abs(subset[name].weight.data)
            X = wrapped_layers[name].scaler_row.reshape((1, -1))
            G = gradients.get(indexed_name, None)

            if G is None:
                print(f"  警告：未找到梯度 {indexed_name}，使用默认值")
                G = torch.ones_like(W)

            # 使用Pruner-Zero的metric计算
            W_metric = engine.forward(
                W.to(dtype=torch.float32),
                G.to(device=W.device, dtype=torch.float32),
                X.to(device=W.device, dtype=torch.float32),
            )

            layer_wmetric.append(W_metric.cpu())

        # 前向传播到下一层
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                                attention_mask=attention_mask)[0]
        inps, outs = outs, inps

        # 计算异常值比例
        layer_wmetric = torch.cat([torch.flatten(x) for x in layer_wmetric])

        Hyper_m = getattr(args, 'Hyper_m', 5)
        mean_val = torch.mean(layer_wmetric)
        threshold = mean_val * Hyper_m
        outlier_ratio = ((layer_wmetric > threshold).sum().item() /
                         layer_wmetric.numel() * 100)

        print(f"层 {i} 异常值比例: {outlier_ratio:.2f}%")
        all_layer_ratio.append(outlier_ratio)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # ========== 第二阶段：调整稀疏率 ==========
    print("\n调整前的异常值比例:", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)
    Lamda = getattr(args, 'Lamda', 0.08)

    print(f"Hyper_m: {Hyper_m}, Lambda: {Lamda}")

    # OWL调整算法
    if all_layer_ratio.max() > all_layer_ratio.min():
        all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) *
                           (1 / (all_layer_ratio.max() - all_layer_ratio.min()) * Lamda * 2))
    else:
        all_layer_ratio = np.ones_like(all_layer_ratio) * Lamda

    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    all_layer_ratio = np.clip(all_layer_ratio, 0.01, 0.99)

    print(f"调整后 - 均值: {np.mean(all_layer_ratio):.3f}")
    print("调整后的密度比例:", [f"{x:.3f}" for x in all_layer_ratio])

    # ========== 第三阶段：执行剪枝 ==========
    print("\n第三阶段：开始剪枝...")

    # 重新初始化
    model.config.use_cache = False
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    with torch.no_grad():
        inps, outs, attention_mask = prepare_calibration_input(
            model, dataloader, device)

    layers = model.model.decoder.layers

    # 第二遍遍历：执行剪枝
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        layer_sparsity_ratio = np.clip(layer_sparsity_ratio, 0.0, 0.99)

        print(f"\n层 {i}: 密度={all_layer_ratio[i]:.3f}, 稀疏率={layer_sparsity_ratio:.3f}")

        # 多GPU处理
        dev = _module_device(layer, device)
        inps, outs, attention_mask = _move_calibration_buffers(
            inps, outs, attention_mask, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                                attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        # 剪枝每个子模块
        for name in subset:
            indexed_name = f'{name}_layer_{i}'
            print(f"  正在修剪 {name}")

            W = torch.abs(subset[name].weight.data)
            X = wrapped_layers[name].scaler_row.reshape((1, -1))
            G = gradients.get(indexed_name, None)

            if G is None:
                G = torch.ones_like(W)

            W_metric = engine.forward(
                W.to(dtype=torch.float32),
                G.to(device=W.device, dtype=torch.float32),
                X.to(device=W.device, dtype=torch.float32),
            )

            W_mask = (torch.zeros_like(W_metric) == 1)

            if prune_n != 0:
                # 结构化剪枝
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n,
                                                           dim=1, largest=False)[1], True)
            else:
                # 非结构化剪枝
                if layer_sparsity_ratio > 0:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

            actual_sparsity = W_mask.sum().item() / W_mask.numel()
            print(f"    实际稀疏率: {actual_sparsity:.3f}")

        # 更新到下一层
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                                attention_mask=attention_mask)[0]

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("OPT模型 OWL-Pruner-Zero剪枝完成！")
    print("=" * 50)
