#!/usr/bin/env python3
"""cp_microbench.py -- Critical-Path → Inference-Latency Microbenchmark.

Tests the hypothesis that reducing the routing critical path (max-tokens-per-
expert summed over MoE layers) actually reduces wall-clock inference latency
under expert-parallel deployment.

Method:
  1. For each candidate model, run forward passes on a held-out dataset
     (default: WikiText-2 test). Capture per-layer per-expert token counts via
     forward-hooks on the MoE blocks (no extra cost beyond the forward pass).
  2. Pre-time the per-expert FFN on this GPU at a grid of batch sizes
     [1,2,4,8,...,4096]. The FFN follows Qwen3-MoE's SwiGLU layout
     (gate_proj 2048→768, up_proj 2048→768, down_proj 768→2048).
  3. For each EP layout in {1,2,4,8,16,32,64,128}, simulate per-step expert-
     compute time with a block expert-to-GPU layout
     (expert e → gpu e // (E/EP)):
       tokens_on_gpu[g] = sum_{e on g} observed_tokens_for_expert[e]
       layer_time = max_g( ffn_time(tokens_on_gpu[g]) )
       step_time   = sum over the 48 MoE layers
  4. Compare baseline vs trained. Report per-model theoretical CP, simulated
     step time at each EP, and implied speedup (baseline / trained).

This isolates the CP→latency relationship cleanly: same kernels, same hardware,
same dataset, same EP policy — only the routing distribution differs.

Outputs:
  - JSON with per-model + per-EP metrics
  - Printed summary table

Usage:
    python cp_microbench.py \
        --baseline-model /lustre/.../Qwen3-30B-A3B-complete \
        --trained-model  /lustre/.../hf_converted_iter2000 \
        --output ./microbench_results.json \
        [--num-batches 64] [--batch-size 4] [--seq-length 2048]
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Qwen3-30B-A3B architectural constants (matches config.json).
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 8
NUM_MOE_LAYERS = 48
DTYPE = torch.bfloat16

# Batch sizes used to pre-characterise per-expert FFN latency. Anything beyond
# the largest is extrapolated via the slope of the last two points.
FFN_TIMING_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Expert-parallel layouts to simulate. EP=128 means every expert on its own GPU
# (the regime where CP === step latency).
EP_LAYOUTS = [1, 2, 4, 8, 16, 32, 64, 128]


# ---------------------------------------------------------------------------
# Routing-trace capture (a small simplification of measure_critical_path.py)
# ---------------------------------------------------------------------------


class _RoutingHook:
    """Captures token-per-expert vectors per (batch, MoE layer)."""

    def __init__(self):
        # traces[batch_idx][layer_idx] = np.ndarray[num_experts] of token counts
        self.traces: List[Dict[int, np.ndarray]] = []
        self._current: Dict[int, np.ndarray] = {}
        self._handles: List = []

    def _make_hook(self, layer_idx: int):
        def _hook(module, _inputs, output):
            if not (isinstance(output, tuple) and len(output) >= 2):
                return
            router_logits = output[1]
            if router_logits is None:
                return
            with torch.no_grad():
                num_tokens, num_experts = router_logits.shape
                topk = (
                    getattr(module, "top_k", None)
                    or getattr(module, "num_experts_per_tok", NUM_EXPERTS_PER_TOK)
                )
                routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                _, selected = torch.topk(routing_weights, topk, dim=-1)
                routing_map = torch.zeros(
                    num_tokens, num_experts, dtype=torch.bool, device=router_logits.device
                )
                routing_map.scatter_(1, selected, True)
                tpe = routing_map.sum(dim=0).to(torch.int64).cpu().numpy()
            self._current[layer_idx] = tpe
        return _hook

    def attach(self, model: nn.Module) -> int:
        idx = 0
        for _name, module in model.named_modules():
            cls = type(module).__name__
            if "SparseMoe" in cls or "MoeBlock" in cls:
                self._handles.append(module.register_forward_hook(self._make_hook(idx)))
                idx += 1
        return idx

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def begin_batch(self):
        self._current = {}

    def end_batch(self):
        self.traces.append(dict(self._current))


# ---------------------------------------------------------------------------
# Per-expert FFN kernel timing (the "what does B tokens cost on one GPU" curve)
# ---------------------------------------------------------------------------


class _ExpertFFN(nn.Module):
    """Single-expert SwiGLU FFN matching Qwen3-MoE's expert layout."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _time_expert_ffn(
    device: torch.device, repeats: int = 50, warmup: int = 10
) -> Dict[int, float]:
    """Time one Qwen3-MoE expert FFN at each batch size in FFN_TIMING_GRID.

    Returns {batch_size: median_ms_per_call}.
    """
    print(f"  Pre-timing per-expert FFN on {device} (repeats={repeats}, warmup={warmup})…")
    expert = _ExpertFFN(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE).to(device=device, dtype=DTYPE).eval()

    timings_ms: Dict[int, float] = {}
    for bs in FFN_TIMING_GRID:
        x = torch.randn(bs, HIDDEN_SIZE, device=device, dtype=DTYPE)
        with torch.no_grad():
            for _ in range(warmup):
                _ = expert(x)
            torch.cuda.synchronize(device)
            samples = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = expert(x)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
        med = float(np.median(samples))
        timings_ms[bs] = med
        print(f"    B={bs:>5d}  median={med:7.4f} ms")

    del expert
    torch.cuda.empty_cache()
    return timings_ms


def _ffn_time_at(load: int, grid: Dict[int, float]) -> float:
    """Linear-interpolate the pre-timed FFN curve at an arbitrary load `B`."""
    if load <= 0:
        return 0.0
    points = sorted(grid.items())
    sizes = [b for b, _ in points]
    times = [t for _, t in points]
    if load <= sizes[0]:
        return times[0] * (load / sizes[0])
    if load >= sizes[-1]:
        # Extrapolate via slope of the last two grid points.
        b1, b2 = sizes[-2], sizes[-1]
        t1, t2 = times[-2], times[-1]
        slope = (t2 - t1) / max(b2 - b1, 1)
        return t2 + slope * (load - b2)
    for i in range(len(sizes) - 1):
        if sizes[i] <= load <= sizes[i + 1]:
            frac = (load - sizes[i]) / (sizes[i + 1] - sizes[i])
            return times[i] + frac * (times[i + 1] - times[i])
    return times[-1]  # unreachable


# ---------------------------------------------------------------------------
# Forward-pass driver: capture routing traces from one model
# ---------------------------------------------------------------------------


def _build_dataloader(tokenizer, args):
    """Load text data and chunk into seq_length token sequences.

    `args.dataset_name` may be:
      - A Megatron mmap dataset path-prefix (no extension, sibling .idx + .bin
        files exist) — loaded via megatron.core.datasets.IndexedDataset. This
        path skips tokenization entirely (the data is already token IDs).
      - A path to a local .arrow / .parquet file (loaded directly).
      - A HuggingFace dataset id (used with `load_dataset`).
    """
    # --- Megatron mmap path -------------------------------------------------
    if os.path.isfile(args.dataset_name + ".idx") and os.path.isfile(args.dataset_name + ".bin"):
        # Add Megatron-LM to PYTHONPATH so the import works.
        repo = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch"
        meg = os.path.join(repo, "backends/megatron/Megatron-LM-250624")
        if meg not in sys.path:
            sys.path.insert(0, meg)
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        ds = IndexedDataset(args.dataset_name)
        print(f"  Loaded Megatron mmap dataset: {len(ds)} documents (skipping tokenization)")
        # Concatenate documents into a single token stream, then chunk.
        all_ids = []
        total_needed = (args.num_batches + 4) * args.batch_size * args.seq_length
        for i in range(len(ds)):
            doc = ds[i]
            # IndexedDataset returns numpy array; coerce to list for fast extend
            all_ids.extend(int(x) for x in doc)
            if len(all_ids) >= total_needed:
                break
        ids = torch.tensor(all_ids[:(len(all_ids) // args.seq_length) * args.seq_length],
                           dtype=torch.long)
        n = len(ids) // args.seq_length
        ids = ids.reshape(n, args.seq_length)
        print(f"  Built {n} chunks of {args.seq_length} tokens from Megatron mmap")
        return ids

    # --- Local arrow / parquet ----------------------------------------------
    if os.path.isfile(args.dataset_name):
        from datasets import Dataset
        if args.dataset_name.endswith(".arrow"):
            ds = Dataset.from_file(args.dataset_name)
        elif args.dataset_name.endswith(".parquet"):
            from datasets import load_dataset
            ds = load_dataset("parquet", data_files=args.dataset_name, split="train")
        else:
            raise ValueError(f"Unrecognized local dataset format: {args.dataset_name}")
        print(f"  Loaded local file {args.dataset_name}: {len(ds)} rows")
    else:
        from datasets import load_dataset
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)

    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    ids = enc.input_ids[0]
    n = len(ids) // args.seq_length
    ids = ids[: n * args.seq_length].reshape(n, args.seq_length)
    print(f"  Built {n} chunks of {args.seq_length} tokens from {args.dataset_name}")
    return ids


def capture_routing_traces(model_path: str, args) -> List[Dict[int, np.ndarray]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n→ Loading {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    chunks = _build_dataloader(tok, args)

    hook = _RoutingHook()
    n_moe = hook.attach(model)
    print(f"  Attached routing hooks on {n_moe} MoE layers (expected {NUM_MOE_LAYERS})")
    if n_moe != NUM_MOE_LAYERS:
        print(f"  WARNING: expected {NUM_MOE_LAYERS} MoE layers, got {n_moe}")

    n_batches = min(args.num_batches, chunks.shape[0] // args.batch_size)
    print(f"  Running {n_batches} batches (bs={args.batch_size}, seq={args.seq_length})")

    device = next(model.parameters()).device
    with torch.no_grad():
        for b in range(n_batches):
            start = b * args.batch_size
            x = chunks[start : start + args.batch_size].to(device)
            hook.begin_batch()
            _ = model(input_ids=x)
            hook.end_batch()
            if (b + 1) % 10 == 0:
                print(f"    batch {b + 1}/{n_batches}")

    hook.detach()
    traces = hook.traces
    del model
    torch.cuda.empty_cache()
    return traces


# ---------------------------------------------------------------------------
# Simulated EP-aware step latency from captured traces
# ---------------------------------------------------------------------------


def _simulate_step_time_ms(
    layer_traces: Dict[int, np.ndarray], ep: int, ffn_grid: Dict[int, float]
) -> float:
    """Compute simulated per-step expert-compute time for a single batch.

    Block expert-to-GPU layout: experts [g*K, (g+1)*K) live on GPU g, where
    K = NUM_EXPERTS / EP. The GPU's per-layer load is the sum of token counts
    over its experts. Layer time = max_g(ffn_time(load_g)). Step time = sum
    over MoE layers.
    """
    assert NUM_EXPERTS % ep == 0, f"EP={ep} must divide {NUM_EXPERTS}"
    experts_per_gpu = NUM_EXPERTS // ep
    total_ms = 0.0
    for layer_idx, tpe in layer_traces.items():
        # tpe is np.ndarray shape [num_experts]
        per_gpu = tpe.reshape(ep, experts_per_gpu).sum(axis=1)
        layer_ms = max(_ffn_time_at(int(load), ffn_grid) for load in per_gpu)
        total_ms += layer_ms
    return total_ms


def summarise_model(
    name: str,
    traces: List[Dict[int, np.ndarray]],
    ffn_grid: Dict[int, float],
) -> Dict:
    """Aggregate traces into per-EP step-time stats, plus theoretical CP."""
    cp_per_batch = []
    for trace in traces:
        cp = sum(int(tpe.max()) for tpe in trace.values())
        cp_per_batch.append(cp)

    by_ep = {}
    for ep in EP_LAYOUTS:
        ms = [_simulate_step_time_ms(t, ep, ffn_grid) for t in traces]
        by_ep[ep] = {
            "mean_ms": float(np.mean(ms)),
            "std_ms": float(np.std(ms)),
            "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)),
        }

    summary = {
        "name": name,
        "num_batches": len(traces),
        "critical_path_mean": float(np.mean(cp_per_batch)),
        "critical_path_std": float(np.std(cp_per_batch)),
        "critical_path_min": int(min(cp_per_batch)),
        "critical_path_max": int(max(cp_per_batch)),
        "step_time_by_ep_ms": by_ep,
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="CP→latency microbenchmark")
    p.add_argument("--baseline-model", required=True,
                   help="Path to high-CP HF model (e.g. Alibaba pretrained)")
    p.add_argument("--trained-model", required=True,
                   help="Path to low-CP HF model (router-trained checkpoint)")
    p.add_argument("--baseline-name", default="baseline")
    p.add_argument("--trained-name", default="trained")
    p.add_argument("--output", default="./microbench_results.json")
    # Default to the HF-Hub-hosted Salesforce/wikitext (parquet) — the legacy
    # `wikitext` config tries to fetch from s3.amazonaws.com which compute
    # nodes can't reach. Salesforce/wikitext is a drop-in replacement.
    p.add_argument("--dataset-name", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", default="test")
    p.add_argument("--num-batches", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--ffn-repeats", type=int, default=50)
    p.add_argument("--ffn-warmup", type=int, default=10)
    p.add_argument("--skip-trace", action="store_true",
                   help="Skip trace capture, load from a previous run's JSON")
    p.add_argument("--load-traces-from", default=None,
                   help="Reload traces from a previous JSON instead of re-capturing")
    return p.parse_args()


def _print_table(baseline: Dict, trained: Dict):
    print()
    print("=" * 92)
    print("MICROBENCHMARK RESULTS")
    print("=" * 92)
    print(f"  {baseline['name']:>12s}  CP_mean={baseline['critical_path_mean']:7.1f}  "
          f"CP_std={baseline['critical_path_std']:5.1f}  "
          f"(min={baseline['critical_path_min']}, max={baseline['critical_path_max']})")
    print(f"  {trained['name']:>12s}  CP_mean={trained['critical_path_mean']:7.1f}  "
          f"CP_std={trained['critical_path_std']:5.1f}  "
          f"(min={trained['critical_path_min']}, max={trained['critical_path_max']})")
    print()
    cp_ratio = baseline["critical_path_mean"] / max(trained["critical_path_mean"], 1e-9)
    print(f"  CP reduction:    "
          f"{(1 - trained['critical_path_mean'] / baseline['critical_path_mean']) * 100:5.2f}%   "
          f"(theoretical CP-speedup = {cp_ratio:.4f}x)")
    print()
    print(f"  Per-step expert-compute time (sum over {NUM_MOE_LAYERS} MoE layers, ms):")
    print(f"  {'EP':>4s}  {baseline['name'][:14]:>14s}  {trained['name'][:14]:>14s}  "
          f"{'speedup':>8s}  {'reduction':>10s}")
    print("  " + "-" * 60)
    for ep in EP_LAYOUTS:
        b = baseline["step_time_by_ep_ms"][ep]["mean_ms"]
        t = trained["step_time_by_ep_ms"][ep]["mean_ms"]
        sp = b / max(t, 1e-9)
        red = (1 - t / b) * 100 if b > 0 else 0
        print(f"  {ep:>4d}  {b:>14.4f}  {t:>14.4f}  {sp:>8.4f}x  {red:>9.2f}%")
    print("=" * 92)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA required.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda:0")
    print(f"Device for FFN timing: {device} ({torch.cuda.get_device_name(0)})")

    ffn_grid = _time_expert_ffn(device, repeats=args.ffn_repeats, warmup=args.ffn_warmup)

    if args.load_traces_from:
        with open(args.load_traces_from) as f:
            blob = json.load(f)
        baseline_traces = [
            {int(k): np.array(v, dtype=np.int64) for k, v in t.items()}
            for t in blob["baseline_traces"]
        ]
        trained_traces = [
            {int(k): np.array(v, dtype=np.int64) for k, v in t.items()}
            for t in blob["trained_traces"]
        ]
        print(f"Loaded {len(baseline_traces)} + {len(trained_traces)} traces from {args.load_traces_from}")
    else:
        print("\n=== BASELINE ===")
        t0 = time.time()
        baseline_traces = capture_routing_traces(args.baseline_model, args)
        print(f"  Done ({time.time() - t0:.1f}s)")

        print("\n=== TRAINED ===")
        t0 = time.time()
        trained_traces = capture_routing_traces(args.trained_model, args)
        print(f"  Done ({time.time() - t0:.1f}s)")

    baseline_summary = summarise_model(args.baseline_name, baseline_traces, ffn_grid)
    trained_summary = summarise_model(args.trained_name, trained_traces, ffn_grid)

    _print_table(baseline_summary, trained_summary)

    # Cross-EP speedup matrix.
    cross_ep_speedup = {
        ep: baseline_summary["step_time_by_ep_ms"][ep]["mean_ms"]
        / max(trained_summary["step_time_by_ep_ms"][ep]["mean_ms"], 1e-9)
        for ep in EP_LAYOUTS
    }

    out = {
        "config": {
            "baseline_model": args.baseline_model,
            "trained_model": args.trained_model,
            "dataset": f"{args.dataset_name}/{args.dataset_config}/{args.dataset_split}",
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "device_name": torch.cuda.get_device_name(0),
            "ep_layouts": EP_LAYOUTS,
        },
        "ffn_timing_ms_by_batch_size": ffn_grid,
        "baseline": baseline_summary,
        "trained": trained_summary,
        "speedup_by_ep": cross_ep_speedup,
        # Persist raw traces so a future run can re-do the simulation without
        # re-running both 30B-model forward passes.
        "baseline_traces": [
            {str(k): v.tolist() for k, v in t.items()} for t in baseline_traces
        ],
        "trained_traces": [
            {str(k): v.tolist() for k, v in t.items()} for t in trained_traces
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
