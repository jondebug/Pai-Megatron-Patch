#!/usr/bin/env python3
"""
measure_latency.py -- Actual Inference Timing

Measures wall-clock inference latency for a HuggingFace model checkpoint.
Runs N forward-pass batches, discards warmup iterations, and reports
latency statistics (mean, p50, p95, p99).

Can compare two checkpoints (baseline vs optimized) in a single run.

Usage:
    # Single model
    python measure_latency.py \
        --model-path /path/to/hf/checkpoint \
        --output-path ./latency_results.json \
        [--num-batches 100] [--warmup-batches 10] \
        [--batch-size 4] [--seq-length 2048]

    # Compare two models
    python measure_latency.py \
        --model-path /path/to/optimized \
        --baseline-model-path /path/to/baseline \
        --output-path ./latency_comparison.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Measure inference latency")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to HuggingFace model checkpoint")
    parser.add_argument("--baseline-model-path", type=str, default=None,
                        help="Path to baseline HuggingFace model for comparison")
    parser.add_argument("--output-path", type=str, default="./latency_results.json",
                        help="Path to save results JSON")
    parser.add_argument("--num-batches", type=int, default=100,
                        help="Number of batches for timing (excluding warmup)")
    parser.add_argument("--warmup-batches", type=int, default=10,
                        help="Number of warmup batches to discard")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--seq-length", type=int, default=2048,
                        help="Sequence length")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    return parser.parse_args()


def generate_synthetic_batch(batch_size, seq_length, vocab_size, device):
    """Generate a synthetic input batch for timing."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    return input_ids


def measure_model_latency(model_path, args, device):
    """Load a model and measure its forward-pass latency.

    Returns:
        dict with latency statistics and the loaded model's config info.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]

    print(f"Loading model: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    vocab_size = config.vocab_size

    total_batches = args.warmup_batches + args.num_batches
    latencies_ms = []

    print(f"Running {args.warmup_batches} warmup + {args.num_batches} timed batches "
          f"(bs={args.batch_size}, seq_len={args.seq_length})...")

    with torch.no_grad():
        for i in range(total_batches):
            input_ids = generate_synthetic_batch(
                args.batch_size, args.seq_length, vocab_size, device
            )

            # Synchronize before timing
            if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_ids=input_ids)

            if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
                torch.cuda.synchronize()

            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0

            if i >= args.warmup_batches:
                latencies_ms.append(elapsed_ms)

            if (i + 1) % 20 == 0:
                phase = "warmup" if i < args.warmup_batches else "timed"
                print(f"  Batch {i + 1}/{total_batches} ({phase}): {elapsed_ms:.2f} ms")

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute statistics
    latencies = np.array(latencies_ms)
    stats = {
        "model_path": model_path,
        "num_batches": len(latencies),
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "dtype": args.dtype,
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "throughput_samples_per_sec": float(args.batch_size * 1000.0 / np.mean(latencies)),
        "throughput_tokens_per_sec": float(
            args.batch_size * args.seq_length * 1000.0 / np.mean(latencies)
        ),
    }

    return stats


def print_stats(stats, label="Model"):
    """Print latency statistics in a formatted table."""
    print(f"\n  {label}:")
    print(f"    Path:                {stats['model_path']}")
    print(f"    Mean latency:        {stats['mean_ms']:8.2f} ms  (std: {stats['std_ms']:.2f})")
    print(f"    P50 latency:         {stats['p50_ms']:8.2f} ms")
    print(f"    P95 latency:         {stats['p95_ms']:8.2f} ms")
    print(f"    P99 latency:         {stats['p99_ms']:8.2f} ms")
    print(f"    Min / Max:           {stats['min_ms']:8.2f} / {stats['max_ms']:.2f} ms")
    print(f"    Throughput:          {stats['throughput_samples_per_sec']:8.2f} samples/sec")
    print(f"    Token throughput:    {stats['throughput_tokens_per_sec']:8.0f} tokens/sec")


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Measure primary model
    print("=" * 70)
    print("LATENCY MEASUREMENT")
    print("=" * 70)

    optimized_stats = measure_model_latency(args.model_path, args, device)

    # Optionally measure baseline
    baseline_stats = None
    if args.baseline_model_path is not None:
        print()
        baseline_stats = measure_model_latency(args.baseline_model_path, args, device)

    # Build results
    results = {
        "optimized": optimized_stats,
    }

    if baseline_stats is not None:
        results["baseline"] = baseline_stats
        speedup = baseline_stats["mean_ms"] / optimized_stats["mean_ms"]
        results["comparison"] = {
            "speedup": speedup,
            "baseline_mean_ms": baseline_stats["mean_ms"],
            "optimized_mean_ms": optimized_stats["mean_ms"],
            "latency_reduction_pct": (1.0 - optimized_stats["mean_ms"] / baseline_stats["mean_ms"]) * 100,
        }

    # Print summary
    print()
    print("=" * 70)
    print("LATENCY RESULTS SUMMARY")
    print("=" * 70)

    if baseline_stats is not None:
        print_stats(baseline_stats, "Baseline")
    print_stats(optimized_stats, "Optimized" if baseline_stats else "Model")

    if baseline_stats is not None:
        speedup = results["comparison"]["speedup"]
        reduction = results["comparison"]["latency_reduction_pct"]
        print(f"\n  Comparison:")
        print(f"    Speedup:             {speedup:.4f}x")
        print(f"    Latency reduction:   {reduction:.2f}%")

    print("=" * 70)

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
