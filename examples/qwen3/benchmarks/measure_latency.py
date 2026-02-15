"""
measure_latency.py -- Measure actual inference latency for MoE models.

Compares wall-clock forward pass latency between checkpoints to validate
that critical path reduction translates to real speedup.

Usage:
    python measure_latency.py \
        --model-path /path/to/hf/checkpoint \
        --num-batches 100 \
        --warmup-batches 10 \
        --batch-size 1 \
        --seq-length 128 \
        --output-file latency_results.json

For comparing two checkpoints:
    python measure_latency.py \
        --model-path /path/to/baseline \
        --compare-path /path/to/optimized \
        --output-file latency_comparison.json
"""

import argparse
import json
import os
import time
import sys

import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Measure MoE inference latency")
    parser.add_argument("--model-path", required=True, help="Path to HuggingFace model")
    parser.add_argument("--compare-path", default=None, help="Second model to compare against")
    parser.add_argument("--num-batches", type=int, default=100, help="Number of batches to time")
    parser.add_argument("--warmup-batches", type=int, default=10, help="Warmup batches (discarded)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--output-file", default="latency_results.json", help="Output JSON file")
    parser.add_argument("--device", default="cuda:0", help="Device to use (single GPU)")
    parser.add_argument("--label", default=None, help="Label for this checkpoint")
    parser.add_argument("--compare-label", default=None, help="Label for comparison checkpoint")
    return parser.parse_args()


def measure_model_latency(model_path, device, num_batches, warmup_batches,
                          batch_size, seq_length, label=None):
    """Load a model and measure per-batch forward pass latency."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Measuring latency: {label or model_path}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()

    vocab_size = tokenizer.vocab_size or 32000

    # Warmup
    print(f"  Warming up ({warmup_batches} batches)...")
    for _ in range(warmup_batches):
        input_ids = torch.randint(100, vocab_size, (batch_size, seq_length), device=device)
        with torch.no_grad():
            _ = model(input_ids)
        torch.cuda.synchronize()

    # Timed runs
    print(f"  Timing {num_batches} batches...")
    latencies_ms = []
    for i in range(num_batches):
        input_ids = torch.randint(100, vocab_size, (batch_size, seq_length), device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(input_ids)

        torch.cuda.synchronize()
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies_ms.append(latency_ms)

    latencies = np.array(latencies_ms)

    results = {
        'label': label or os.path.basename(model_path),
        'model_path': model_path,
        'num_batches': num_batches,
        'warmup_batches': warmup_batches,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'device': device,
        'latency_ms': {
            'mean': float(latencies.mean()),
            'std': float(latencies.std()),
            'min': float(latencies.min()),
            'max': float(latencies.max()),
            'p50': float(np.percentile(latencies, 50)),
            'p90': float(np.percentile(latencies, 90)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
        },
        'throughput_tokens_per_sec': float(
            (batch_size * seq_length) / (latencies.mean() / 1000)
        ),
    }

    # Print summary
    lat = results['latency_ms']
    print(f"\n  Results:")
    print(f"    Mean:   {lat['mean']:.2f} ms")
    print(f"    Std:    {lat['std']:.2f} ms")
    print(f"    P50:    {lat['p50']:.2f} ms")
    print(f"    P95:    {lat['p95']:.2f} ms")
    print(f"    P99:    {lat['p99']:.2f} ms")
    print(f"    Throughput: {results['throughput_tokens_per_sec']:.0f} tokens/sec")

    # Free memory
    del model
    torch.cuda.empty_cache()

    return results


def main():
    args = parse_args()

    results = {}

    # Measure primary model
    primary = measure_model_latency(
        args.model_path, args.device, args.num_batches, args.warmup_batches,
        args.batch_size, args.seq_length,
        label=args.label or "primary"
    )
    results['primary'] = primary

    # Measure comparison model if provided
    if args.compare_path:
        comparison = measure_model_latency(
            args.compare_path, args.device, args.num_batches, args.warmup_batches,
            args.batch_size, args.seq_length,
            label=args.compare_label or "comparison"
        )
        results['comparison'] = comparison

        # Compute speedup
        speedup = primary['latency_ms']['mean'] / comparison['latency_ms']['mean']
        results['speedup'] = {
            'mean_latency_ratio': speedup,
            'throughput_ratio': comparison['throughput_tokens_per_sec'] / primary['throughput_tokens_per_sec'],
        }

        print(f"\n{'='*60}")
        print(f"COMPARISON: {primary['label']} vs {comparison['label']}")
        print(f"{'='*60}")
        print(f"  {primary['label']:>20s}: {primary['latency_ms']['mean']:.2f} ms (mean)")
        print(f"  {comparison['label']:>20s}: {comparison['latency_ms']['mean']:.2f} ms (mean)")
        print(f"  {'Speedup':>20s}: {speedup:.3f}x")
        print(f"{'='*60}")

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
