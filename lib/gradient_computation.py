import argparse
import os
import random
import gc
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional
import pickle

print('torch', torch.__version__)
print('# of gpus:', torch.cuda.device_count())


def maybe_empty_cuda_cache(context, force=False):
    """Avoid known multi-GPU driver failures caused by routine empty_cache calls."""
    if not torch.cuda.is_available():
        return
    if not force and os.environ.get('SKIP_CUDA_EMPTY_CACHE', '0') == '1':
        print(f'SKIP_CUDA_EMPTY_CACHE: skip torch.cuda.empty_cache() {context}')
        return
    torch.cuda.empty_cache()


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers,
                               name=name + '.' + name1 if name != '' else name1))
    return res


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """数据加载，带简单缓存机制"""
    cache_file = f'.cache/wikitext2_{nsamples}_{seed}_{seqlen}.pkl'

    # 尝试从缓存加载
    if os.path.exists(cache_file):
        print(f"从缓存加载数据: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # 原始加载逻辑
    local_path = "/home/yh114/workdir/DSnoT/wikitext/wikitext-2-raw-v1"

    try:
        traindata = load_dataset(local_path, split='train')
        testdata = load_dataset(local_path, split='test')
    except:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text'][:100]), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        trainloader.append((inp, tar))

    # 保存到缓存
    os.makedirs('.cache', exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump((trainloader, testenc), f)

    return trainloader, testenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)


def get_llm(model, cache_dir='llm_weights', enable_checkpointing=True):
    """内存优化的模型加载"""
    if not torch.cuda.is_available():
        raise RuntimeError('梯度计算需要CUDA GPU')

    device_map = os.environ.get('MODEL_DEVICE_MAP', 'auto').strip() or 'auto'
    reserve_gib = float(os.environ.get('GRADIENT_GPU_RESERVE_GIB', '4'))
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        usable_gib = max(int(total_gib - reserve_gib), 1)
        max_memory[i] = f'{usable_gib}GiB'
        print(
            f'GPU {i}: {torch.cuda.get_device_name(i)}, total={total_gib:.1f} GiB, '
            f'max_memory={max_memory[i]}'
        )

    print(f'device_map策略: {device_map}')

    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map=device_map,
        max_memory=max_memory
    )

    if enable_checkpointing:
        model.gradient_checkpointing_enable()
        print("已启用梯度检查点")

    print('GPU分配:')
    print(model.hf_device_map)

    model.seqlen = 1024
    return model


class GradientComputation:
    """优化的梯度计算器 - 仅计算L2梯度"""

    def __init__(self, model, scale, use_mixed_precision=False):
        self.model = model
        self.gradients_l2 = dict()
        self.nsample = 0
        self.scale = scale
        self.device = torch.device('cpu')
        self.use_mixed_precision = use_mixed_precision
        self.gradients_init()

    def _get_layers(self):
        if hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
            return self.model.model.decoder.layers
        raise AttributeError("未找到可识别的layers列表")

    def gradients_init(self):
        layers = self._get_layers()
        for i in tqdm(range(len(layers)), desc='初始化梯度存储'):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f'{name}_layer_{i}'
                # 只初始化L2梯度存储
                self.gradients_l2[indexed_name] = torch.zeros_like(
                    subset[name].weight, dtype=torch.float32,
                    device=self.device).pin_memory()

    def update_gradient_batch(self, model, batch_size=1):
        """批量更新梯度，减少循环开销 - 仅更新L2梯度"""
        layers = self._get_layers()

        # 批量收集梯度
        with torch.no_grad():
            for i in range(len(layers)):
                layer = layers[i]
                subset = find_layers(layer)
                for name in subset:
                    indexed_name = f'{name}_layer_{i}'
                    if subset[name].weight.grad is not None:
                        # 非阻塞传输
                        grad = subset[name].weight.grad.detach().to(
                            self.device, non_blocking=True, dtype=torch.float32
                        )
                        grad_scaled = grad * self.scale

                        # 只更新L2梯度
                        self.gradients_l2[indexed_name] += (grad_scaled ** 2)

        self.nsample += batch_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsamples', type=int, default=128, help='样本数量')
    parser.add_argument('--scale', type=int, default=100, help='梯度缩放')
    parser.add_argument('--seqlen', type=int, default=1024, help='序列长度')
    parser.add_argument('--llama_version', type=int, default=2)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--enable_checkpointing', action='store_true', default=True)
    parser.add_argument('--save_root', type=str, default='./gradients')
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Optional exact output .pth path. Overrides the default family subdirectory.',
    )
    parser.add_argument('--use_mixed_precision', action='store_true', default=False)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    args = parser.parse_args()

    print(f'计算梯度: nsamples={args.nsamples}, scale={args.scale}, seqlen={args.seqlen}')

    # 环境设置
    set_seed(args.seed)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # 启用cudnn benchmark加速
    torch.backends.cudnn.benchmark = True

    # 加载模型
    model = get_llm(args.model, enable_checkpointing=args.enable_checkpointing)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # 获取设备
    embed_tokens = getattr(getattr(model, 'model', None), 'embed_tokens', None)
    if embed_tokens is not None:
        device = embed_tokens.weight.device
    elif 'model.embed_tokens' in model.hf_device_map:
        mapped_device = model.hf_device_map['model.embed_tokens']
        device = torch.device(f'cuda:{mapped_device}') if isinstance(mapped_device, int) else mapped_device
    else:
        device = next(model.parameters()).device
    print(f'输入设备: {device}')

    # 加载数据
    print('加载校准数据...')
    dataloader, _ = get_loaders(
        'wikitext2',
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer
    )
    print('数据加载完成')

    # 优化器设置
    optimizer = AdamW(model.parameters(), lr=0.01, eps=0.01)
    optimizer.zero_grad()

    # 梯度计算器
    computer = GradientComputation(model, args.scale, args.use_mixed_precision)

    # 混合精度训练（可选）
    scaler = GradScaler() if args.use_mixed_precision else None

    # 训练模式
    model.train()

    # 主循环
    nsample = 0
    accumulation_steps = args.accumulation_steps

    for batch_idx, (input_ids, labels) in enumerate(tqdm(dataloader, desc='计算梯度')):
        nsample += 1

        # 数据传输（使用非阻塞）
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        try:
            # 前向传播
            if args.use_mixed_precision:
                with autocast():
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / accumulation_steps
                loss.backward()

            print(f'样本 {nsample}, loss: {loss.item() * accumulation_steps:.4f}')

            # 梯度累积
            if (batch_idx + 1) % accumulation_steps == 0:
                # 批量更新梯度
                computer.update_gradient_batch(model, accumulation_steps)

                if args.use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()

                optimizer.zero_grad()

                # 定期清理内存
                if (batch_idx + 1) % (accumulation_steps * 10) == 0:
                    maybe_empty_cuda_cache('during gradient accumulation')

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f'OOM at sample {nsample}, clearing cache...')
                maybe_empty_cuda_cache('after OOM', force=True)
                gc.collect()
                optimizer.zero_grad()
                continue
            else:
                raise e

    print('梯度计算完成，保存结果...')

    # 保存梯度
    model_name = os.path.basename(args.model)

    # 处理L2梯度
    gradients_l2 = computer.gradients_l2
    for name in gradients_l2:
        grad_sqrt = torch.sqrt(gradients_l2[name])
        gradients_l2[name] = grad_sqrt.to(dtype=torch.float16)

    # 保存 - 只保存L2梯度
    family = getattr(model.config, "model_type", "unknown").lower()
    if args.output_path:
        output_path = os.path.abspath(args.output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        save_subdir = f"llama{args.llama_version}" if family == "llama" else family
        save_dir = os.path.join(args.save_root, save_subdir)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(
            save_dir,
            f'gradients_l2_{model_name}_{args.nsamples}_{args.seed}.pth',
        )

    torch.save(computer.gradients_l2, output_path)
    print(f'L2梯度已保存到 {output_path}')


if __name__ == '__main__':
    main()
