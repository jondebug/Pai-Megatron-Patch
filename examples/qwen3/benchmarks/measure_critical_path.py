"""
measure_critical_path.py -- Measure critical path and routing metrics on a HuggingFace checkpoint.

Loads a HuggingFace MoE model, runs forward passes on a dataset, and collects
per-layer routing statistics including the critical path metric.

Usage:
    python measure_critical_path.py \
        --model-path /path/to/hf/checkpoint \
        --dataset-path /path/to/dataset  \
        --num-batches 100 \
        --batch-size 1 \
        --seq-length 128 \
        --output-file critical_path_results.json

The critical path is: sum over all MoE layers of max(tokens_per_expert[layer]).
This is the theoretical compute bottleneck for expert-parallel inference.
"""

import argparse
import json
import os
import time
import sys
from collections import defaultdict

import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Measure MoE critical path metrics")
    parser.add_argument("--model-path", required=True, help="Path to HuggingFace model checkpoint")
    parser.add_argument("--num-batches", type=int, default=100, help="Number of batches to evaluate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--output-file", default="critical_path_results.json", help="Output JSON file")
    parser.add_argument("--dataset", default=None, help="HuggingFace dataset name (default: random input)")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--label", default=None, help="Label for this checkpoint (e.g., 'baseline' or 'optimized')")
    return parser.parse_args()


class RoutingMetricsCollector:
    """Hooks into MoE router forward passes to collect per-layer routing statistics."""

    def __init__(self):
        self.hooks = []
        self.batch_metrics = []  # list of per-batch metric dicts
        self._current_batch = {}  # layer_idx -> tokens_per_expert

    def attach(self, model):
        """Find MoE layers and attach forward hooks to routers."""
        layer_idx = 0
        for name, module in model.named_modules():
            # Look for router/gate modules that produce routing decisions
            # Common patterns: TopKRouter, MoeGate, SparseMoeBlock
            if self._is_moe_block(name, module):
                hook = module.register_forward_hook(
                    self._make_hook(layer_idx, name)
                )
                self.hooks.append(hook)
                layer_idx += 1

        if layer_idx == 0:
            print("WARNING: No MoE layers found. Trying alternative detection...")
            layer_idx = self._attach_fallback(model)

        print(f"Attached routing hooks to {layer_idx} MoE layers")
        return layer_idx

    def _is_moe_block(self, name, module):
        """Detect MoE block modules by class name."""
        cls_name = type(module).__name__.lower()
        # Common HuggingFace MoE module names
        moe_patterns = ["sparsemoeblock", "qwen3moespraseblock", "moelayer",
                        "qwen2moesparsemoeblock", "mixtralsparsemoeblock"]
        return any(p in cls_name for p in moe_patterns)

    def _attach_fallback(self, model):
        """Fallback: look for modules with a 'gate' submodule."""
        layer_idx = 0
        for name, module in model.named_modules():
            if hasattr(module, 'gate') and hasattr(module, 'experts'):
                hook = module.register_forward_hook(
                    self._make_hook(layer_idx, name)
                )
                self.hooks.append(hook)
                layer_idx += 1
        return layer_idx

    def _make_hook(self, layer_idx, name):
        """Create a forward hook that captures routing decisions."""
        def hook_fn(module, input, output):
            # Try to extract routing information from the module
            # After forward, many MoE blocks store routing info
            try:
                hidden = input[0] if isinstance(input, tuple) else input
                if hasattr(module, 'gate'):
                    gate = module.gate
                    with torch.no_grad():
                        if hasattr(gate, 'weight'):
                            logits = torch.nn.functional.linear(
                                hidden.view(-1, hidden.shape[-1]).float(),
                                gate.weight.float()
                            )
                            # Top-k selection (use k from model config if available)
                            k = getattr(module, 'num_experts_per_tok',
                                       getattr(module, 'top_k', 8))
                            _, top_indices = logits.topk(k, dim=-1)

                            num_experts = logits.shape[-1]
                            # Count tokens per expert
                            expert_counts = torch.zeros(num_experts, device=logits.device)
                            for ki in range(k):
                                expert_counts.scatter_add_(
                                    0, top_indices[:, ki],
                                    torch.ones(top_indices.shape[0], device=logits.device)
                                )

                            self._current_batch[layer_idx] = {
                                'tokens_per_expert': expert_counts.cpu().numpy(),
                                'max_tokens': expert_counts.max().item(),
                                'min_tokens': expert_counts.min().item(),
                                'mean_tokens': expert_counts.float().mean().item(),
                                'std_tokens': expert_counts.float().std().item(),
                                'num_experts': num_experts,
                                'topk': k,
                                'num_tokens': logits.shape[0],
                            }
            except Exception as e:
                pass  # Don't crash the forward pass

        return hook_fn

    def start_batch(self):
        """Call before each forward pass."""
        self._current_batch = {}

    def end_batch(self):
        """Call after each forward pass. Computes aggregate metrics."""
        if not self._current_batch:
            return None

        num_layers = len(self._current_batch)
        critical_path = sum(d['max_tokens'] for d in self._current_batch.values())

        # Ideal critical path: if perfectly balanced
        if self._current_batch:
            sample = next(iter(self._current_batch.values()))
            ideal_per_layer = sample['num_tokens'] * sample['topk'] / sample['num_experts']
            ideal_critical_path = num_layers * ideal_per_layer
        else:
            ideal_critical_path = 0

        # Per-layer entropy
        entropies = []
        for layer_data in self._current_batch.values():
            counts = layer_data['tokens_per_expert']
            total = counts.sum()
            if total > 0:
                probs = counts / total
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log(probs))
                max_entropy = np.log(layer_data['num_experts'])
                entropies.append(entropy / max_entropy)  # normalized [0, 1]

        metrics = {
            'critical_path': critical_path,
            'ideal_critical_path': ideal_critical_path,
            'imbalance_ratio': critical_path / max(ideal_critical_path, 1),
            'num_moe_layers': num_layers,
            'per_layer_max_tokens': [d['max_tokens'] for d in sorted(self._current_batch.items())],
            'per_layer_mean_tokens': [d['mean_tokens'] for d in sorted(self._current_batch.items())],
            'per_layer_std_tokens': [d['std_tokens'] for d in sorted(self._current_batch.items())],
            'mean_normalized_entropy': float(np.mean(entropies)) if entropies else 0,
        }

        self._current_batch = {}
        self.batch_metrics.append(metrics)
        return metrics

    def summary(self):
        """Compute aggregate statistics over all batches."""
        if not self.batch_metrics:
            return {}

        crit_paths = [m['critical_path'] for m in self.batch_metrics]
        ideal_paths = [m['ideal_critical_path'] for m in self.batch_metrics]
        entropies = [m['mean_normalized_entropy'] for m in self.batch_metrics]

        # Per-layer averages
        num_layers = self.batch_metrics[0]['num_moe_layers']
        per_layer_avg_max = []
        for l in range(num_layers):
            layer_maxes = [m['per_layer_max_tokens'][l] for m in self.batch_metrics
                          if l < len(m['per_layer_max_tokens'])]
            per_layer_avg_max.append(float(np.mean(layer_maxes)))

        return {
            'num_batches': len(self.batch_metrics),
            'num_moe_layers': num_layers,
            'critical_path': {
                'mean': float(np.mean(crit_paths)),
                'std': float(np.std(crit_paths)),
                'min': float(np.min(crit_paths)),
                'max': float(np.max(crit_paths)),
                'p50': float(np.percentile(crit_paths, 50)),
                'p95': float(np.percentile(crit_paths, 95)),
            },
            'ideal_critical_path': float(np.mean(ideal_paths)),
            'imbalance_ratio': {
                'mean': float(np.mean(crit_paths)) / max(float(np.mean(ideal_paths)), 1),
            },
            'normalized_entropy': {
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
            },
            'per_layer_avg_max_tokens': per_layer_avg_max,
            # Top-5 most imbalanced layers
            'bottleneck_layers': sorted(
                range(num_layers), key=lambda l: per_layer_avg_max[l], reverse=True
            )[:5],
        }

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def load_model(model_path, device):
    """Load a HuggingFace model for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device if device == "auto" else {"": device},
    )
    model.eval()

    print(f"Model loaded: {type(model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def generate_inputs(tokenizer, batch_size, seq_length, device, dataset=None):
    """Generate input batches for evaluation."""
    if dataset:
        # Load real dataset
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset, split="test", streaming=True)
            for item in ds:
                text = item.get("text", item.get("content", str(item)))
                tokens = tokenizer(
                    text, return_tensors="pt", max_length=seq_length,
                    truncation=True, padding="max_length"
                )
                yield {k: v.to(device) for k, v in tokens.items()}
        except Exception as e:
            print(f"WARNING: Could not load dataset '{dataset}': {e}")
            print("Falling back to random input.")

    # Random input fallback
    vocab_size = tokenizer.vocab_size or 32000
    while True:
        input_ids = torch.randint(100, vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones_like(input_ids)
        yield {"input_ids": input_ids, "attention_mask": attention_mask}


def main():
    args = parse_args()

    model, tokenizer = load_model(args.model_path, args.device)
    collector = RoutingMetricsCollector()
    num_layers = collector.attach(model)

    if num_layers == 0:
        print("ERROR: No MoE layers detected. Is this an MoE model?")
        sys.exit(1)

    device = next(model.parameters()).device
    input_gen = generate_inputs(tokenizer, args.batch_size, args.seq_length, device, args.dataset)

    print(f"\nRunning {args.num_batches} forward passes "
          f"(batch_size={args.batch_size}, seq_length={args.seq_length})...")

    lm_losses = []
    for batch_idx in range(args.num_batches):
        inputs = next(input_gen)

        collector.start_batch()
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.get("input_ids"))

        metrics = collector.end_batch()

        if outputs.loss is not None:
            lm_losses.append(outputs.loss.item())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{args.num_batches}: "
                  f"crit_path={metrics['critical_path']:.0f}, "
                  f"lm_loss={lm_losses[-1]:.4f}" if lm_losses else "")

    # Compute summary
    summary = collector.summary()
    summary['lm_loss'] = {
        'mean': float(np.mean(lm_losses)) if lm_losses else None,
        'std': float(np.std(lm_losses)) if lm_losses else None,
        'perplexity': float(np.exp(np.mean(lm_losses))) if lm_losses else None,
    }
    summary['config'] = {
        'model_path': args.model_path,
        'num_batches': args.num_batches,
        'batch_size': args.batch_size,
        'seq_length': args.seq_length,
        'label': args.label or os.path.basename(args.model_path),
    }

    # Print results
    print("\n" + "=" * 70)
    print(f"CRITICAL PATH METRICS: {summary['config']['label']}")
    print("=" * 70)
    cp = summary['critical_path']
    print(f"  MoE Layers:            {summary['num_moe_layers']}")
    print(f"  Critical Path (mean):  {cp['mean']:.1f} tokens")
    print(f"  Critical Path (std):   {cp['std']:.1f}")
    print(f"  Critical Path (p95):   {cp['p95']:.1f}")
    print(f"  Ideal Critical Path:   {summary['ideal_critical_path']:.1f}")
    print(f"  Imbalance Ratio:       {summary['imbalance_ratio']['mean']:.3f}x")
    print(f"  Normalized Entropy:    {summary['normalized_entropy']['mean']:.4f}")
    if summary['lm_loss']['mean']:
        print(f"  LM Loss (mean):        {summary['lm_loss']['mean']:.4f}")
        print(f"  Perplexity:            {summary['lm_loss']['perplexity']:.2f}")
    print(f"\n  Top-5 Bottleneck Layers: {summary['bottleneck_layers']}")
    print("=" * 70)

    # Save results
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    collector.remove_hooks()
    return summary


if __name__ == "__main__":
    main()
