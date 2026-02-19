#!/usr/bin/env python3
"""
measure_critical_path.py -- Critical Path & Load Metrics

Loads a HuggingFace Qwen3-MoE checkpoint, runs forward passes on a
held-out dataset, and measures per-layer routing statistics.

Metrics collected per batch (then averaged):
  - num_tokens_on_critical_path (sum of max_tokens_per_expert across layers)
  - Per-layer max_tokens_per_expert (the bottleneck)
  - Per-layer load_balancing_entropy
  - Per-layer tokens_per_expert std (imbalance measure)
  - lm_loss (perplexity on held-out data)

Also computes theoretical speedup:
  theoretical_speedup = baseline_critical_path / optimized_critical_path

Usage:
    python measure_critical_path.py \
        --model-path /path/to/hf/checkpoint \
        --output-path ./critical_path_results.json \
        [--dataset-name wikitext --dataset-config wikitext-2-raw-v1] \
        [--num-batches 100] \
        [--batch-size 4] \
        [--seq-length 2048] \
        [--baseline-critical-path <float>]
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Measure MoE critical path and load metrics")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to HuggingFace model checkpoint")
    parser.add_argument("--output-path", type=str, default="./critical_path_results.json",
                        help="Path to save results JSON")
    parser.add_argument("--dataset-name", type=str, default="wikitext",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1",
                        help="HuggingFace dataset config")
    parser.add_argument("--dataset-split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--num-batches", type=int, default=100,
                        help="Number of batches to evaluate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--seq-length", type=int, default=2048,
                        help="Sequence length")
    parser.add_argument("--baseline-critical-path", type=float, default=None,
                        help="Baseline critical path value for speedup computation")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    return parser.parse_args()


class RoutingHook:
    """Captures routing decisions from MoE layers via forward hooks."""

    def __init__(self):
        self.layer_stats = defaultdict(list)
        self._hooks = []

    def _get_gate_hook(self, layer_idx):
        """Create a hook for a specific MoE gate layer.

        The HuggingFace Qwen3 MoE block's forward returns
        (final_hidden_states, router_logits). We hook the entire MoE block
        to capture router_logits, then compute statistics.
        """
        def hook_fn(module, input, output):
            # output is (hidden_states, router_logits)
            if isinstance(output, tuple) and len(output) >= 2:
                router_logits = output[1]
            else:
                return

            with torch.no_grad():
                # router_logits: [num_tokens, num_experts]
                num_tokens, num_experts = router_logits.shape

                # Compute routing probabilities
                routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

                # Get top-k selections (use model's top_k if available)
                topk = getattr(module, 'top_k', None) or getattr(module, 'num_experts_per_tok', 8)
                _, selected_experts = torch.topk(routing_weights, topk, dim=-1)

                # Build routing map: [num_tokens, num_experts] boolean
                routing_map = torch.zeros(num_tokens, num_experts,
                                          dtype=torch.bool, device=router_logits.device)
                routing_map.scatter_(1, selected_experts, True)

                # tokens_per_expert: [num_experts]
                tokens_per_expert = routing_map.sum(dim=0).float()

                # max_tokens_per_expert (the critical path bottleneck for this layer)
                max_tpe = tokens_per_expert.max().item()

                # Load balancing entropy
                token_dist = tokens_per_expert / (tokens_per_expert.sum() + 1e-10)
                epsilon = 1e-10
                token_dist = token_dist + epsilon
                lb_entropy = -torch.sum(token_dist * torch.log(token_dist)).item()

                # Maximum possible entropy (uniform distribution)
                max_entropy = math.log(num_experts)

                # Std of tokens per expert (imbalance measure)
                tpe_std = tokens_per_expert.std().item()
                tpe_mean = tokens_per_expert.mean().item()

                # Ideal load (perfectly balanced)
                ideal_load = num_tokens * topk / num_experts

                self.layer_stats[layer_idx].append({
                    "max_tokens_per_expert": max_tpe,
                    "load_balancing_entropy": lb_entropy,
                    "max_entropy": max_entropy,
                    "tokens_per_expert_std": tpe_std,
                    "tokens_per_expert_mean": tpe_mean,
                    "ideal_load_per_expert": ideal_load,
                    "num_tokens": num_tokens,
                    "num_experts": num_experts,
                    "topk": topk,
                })

        return hook_fn

    def register_hooks(self, model):
        """Register forward hooks on all MoE layers in the model."""
        layer_idx = 0
        for name, module in model.named_modules():
            # Match Qwen3 MoE block class names
            module_class = type(module).__name__
            if "SparseMoe" in module_class or "MoeBlock" in module_class:
                hook = module.register_forward_hook(self._get_gate_hook(layer_idx))
                self._hooks.append(hook)
                layer_idx += 1
                continue
            # Also check for modules that have 'gate' and 'experts' attributes (standard MoE pattern)
            if (hasattr(module, 'gate') and hasattr(module, 'experts')
                    and not any("SparseMoe" in type(p).__name__ or "MoeBlock" in type(p).__name__
                                for p in module.children())):
                hook = module.register_forward_hook(self._get_gate_hook(layer_idx))
                self._hooks.append(hook)
                layer_idx += 1

        print(f"Registered routing hooks on {layer_idx} MoE layers")
        return layer_idx

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear_stats(self):
        self.layer_stats.clear()

    def get_batch_stats(self):
        """Get statistics for the current batch (latest entry per layer)."""
        stats = {}
        for layer_idx, entries in self.layer_stats.items():
            if entries:
                stats[layer_idx] = entries[-1]
        return stats


def prepare_dataset(tokenizer, args):
    """Load and tokenize dataset, return a DataLoader."""
    from datasets import load_dataset

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)

    # Concatenate all text and chunk into seq_length segments
    all_text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(all_text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0]

    # Create chunks of seq_length
    seq_length = args.seq_length
    num_chunks = len(input_ids) // seq_length
    input_ids = input_ids[:num_chunks * seq_length].reshape(num_chunks, seq_length)

    # Create a simple dataset
    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids):
            self.input_ids = input_ids

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            ids = self.input_ids[idx]
            return {"input_ids": ids, "labels": ids.clone()}

    dataset = TokenDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    return dataloader


def main():
    args = parse_args()

    # Load model and tokenizer
    print(f"Loading model from: {args.model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Prepare dataset
    print("Preparing dataset...")
    dataloader = prepare_dataset(tokenizer, args)

    # Set up routing hooks
    routing_hook = RoutingHook()
    num_moe_layers = routing_hook.register_hooks(model)

    if num_moe_layers == 0:
        print("WARNING: No MoE layers found. Check model architecture.")

    # Run evaluation
    print(f"Running evaluation for {args.num_batches} batches...")
    all_batch_metrics = []
    total_loss = 0.0
    num_evaluated = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break

            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)

            routing_hook.clear_stats()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()
            total_loss += loss
            num_evaluated += 1

            # Collect routing statistics for this batch
            batch_stats = routing_hook.get_batch_stats()

            # Compute critical path for this batch
            critical_path = sum(
                batch_stats[l]["max_tokens_per_expert"]
                for l in sorted(batch_stats.keys())
            )

            batch_metrics = {
                "batch_idx": batch_idx,
                "lm_loss": loss,
                "perplexity": math.exp(min(loss, 20)),  # Clamp to avoid overflow
                "num_tokens_on_critical_path": critical_path,
                "per_layer": {},
            }

            for layer_idx in sorted(batch_stats.keys()):
                layer_data = batch_stats[layer_idx]
                batch_metrics["per_layer"][layer_idx] = {
                    "max_tokens_per_expert": layer_data["max_tokens_per_expert"],
                    "load_balancing_entropy": layer_data["load_balancing_entropy"],
                    "max_entropy": layer_data["max_entropy"],
                    "tokens_per_expert_std": layer_data["tokens_per_expert_std"],
                    "tokens_per_expert_mean": layer_data["tokens_per_expert_mean"],
                    "ideal_load_per_expert": layer_data["ideal_load_per_expert"],
                }

            all_batch_metrics.append(batch_metrics)

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_evaluated
                print(f"  Batch {batch_idx + 1}/{args.num_batches}: "
                      f"loss={avg_loss:.4f}, critical_path={critical_path:.0f}")

    routing_hook.remove_hooks()

    if num_evaluated == 0:
        print("ERROR: No batches evaluated.")
        return

    # Aggregate results
    avg_loss = total_loss / num_evaluated
    avg_ppl = math.exp(min(avg_loss, 20))
    avg_critical_path = sum(m["num_tokens_on_critical_path"] for m in all_batch_metrics) / num_evaluated

    # Per-layer averages
    per_layer_avg = {}
    for layer_idx in range(num_moe_layers):
        layer_entries = [m["per_layer"].get(layer_idx) for m in all_batch_metrics
                         if layer_idx in m.get("per_layer", {})]
        if layer_entries:
            per_layer_avg[layer_idx] = {
                "max_tokens_per_expert": sum(e["max_tokens_per_expert"] for e in layer_entries) / len(layer_entries),
                "load_balancing_entropy": sum(e["load_balancing_entropy"] for e in layer_entries) / len(layer_entries),
                "max_entropy": layer_entries[0]["max_entropy"],
                "tokens_per_expert_std": sum(e["tokens_per_expert_std"] for e in layer_entries) / len(layer_entries),
                "tokens_per_expert_mean": sum(e["tokens_per_expert_mean"] for e in layer_entries) / len(layer_entries),
                "ideal_load_per_expert": layer_entries[0]["ideal_load_per_expert"],
            }

    # Compute ideal critical path (perfectly balanced)
    if per_layer_avg:
        first_layer = per_layer_avg[min(per_layer_avg.keys())]
        ideal_critical_path = num_moe_layers * first_layer["ideal_load_per_expert"]
    else:
        ideal_critical_path = 0

    # Theoretical speedup
    speedup_vs_ideal = avg_critical_path / ideal_critical_path if ideal_critical_path > 0 else float("inf")
    speedup_vs_baseline = None
    if args.baseline_critical_path is not None and args.baseline_critical_path > 0:
        speedup_vs_baseline = args.baseline_critical_path / avg_critical_path

    # Build results
    results = {
        "summary": {
            "avg_lm_loss": avg_loss,
            "avg_perplexity": avg_ppl,
            "avg_num_tokens_on_critical_path": avg_critical_path,
            "ideal_critical_path": ideal_critical_path,
            "critical_path_ratio": speedup_vs_ideal,
            "theoretical_speedup_vs_baseline": speedup_vs_baseline,
            "num_moe_layers": num_moe_layers,
            "num_batches_evaluated": num_evaluated,
        },
        "per_layer_averages": {str(k): v for k, v in per_layer_avg.items()},
        "config": {
            "model_path": args.model_path,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "num_batches": args.num_batches,
            "dtype": args.dtype,
        },
    }

    # Print summary
    print()
    print("=" * 70)
    print("CRITICAL PATH & LOAD METRICS SUMMARY")
    print("=" * 70)
    print(f"  Model:                          {args.model_path}")
    print(f"  MoE layers:                     {num_moe_layers}")
    print(f"  Batches evaluated:              {num_evaluated}")
    print(f"  Avg LM loss:                    {avg_loss:.4f}")
    print(f"  Avg perplexity:                 {avg_ppl:.2f}")
    print(f"  Avg critical path (tokens):     {avg_critical_path:.1f}")
    print(f"  Ideal critical path (tokens):   {ideal_critical_path:.1f}")
    print(f"  Critical path ratio:            {speedup_vs_ideal:.4f}x ideal")
    if speedup_vs_baseline is not None:
        print(f"  Speedup vs baseline:            {speedup_vs_baseline:.4f}x")
    print()

    # Per-layer table
    print("Per-layer averages:")
    print(f"  {'Layer':>5s}  {'MaxTPE':>8s}  {'Entropy':>8s}  {'MaxEnt':>8s}  {'StdTPE':>8s}  {'MeanTPE':>8s}  {'Ideal':>8s}")
    print("  " + "-" * 62)
    for layer_idx in sorted(per_layer_avg.keys()):
        d = per_layer_avg[layer_idx]
        print(f"  {layer_idx:5d}  {d['max_tokens_per_expert']:8.1f}  "
              f"{d['load_balancing_entropy']:8.4f}  {d['max_entropy']:8.4f}  "
              f"{d['tokens_per_expert_std']:8.2f}  {d['tokens_per_expert_mean']:8.2f}  "
              f"{d['ideal_load_per_expert']:8.2f}")
    print("=" * 70)

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
