#!/usr/bin/env python3
"""
STRESS-TEST of the Phase A expert-FFN floor/knee finding (job 28803024).
Adversarial goal: REFUTE that the ~0.136ms floor is "weight-load-bound" and that
the knee is ~345-512. Uses METHODS INDEPENDENT of Phase A (which timed a single
eager nn.Linear ExpertFFN with per-call CUDA-event timing).

Independent legs here:
  L1  EAGER (reproduce Phase A path, sanity) -> floor_eager
  L2  CUDA-GRAPH captured single-expert FFN -> floor_graph
        If floor_graph << floor_eager: Phase A floor was LAUNCH-OVERHEAD, not weight-load.
        If floor_graph ~= weight_bytes/HBM_BW: floor is genuinely weight-load-bound.
  L3  GROUPED/BATCHED multi-expert FFN (vLLM-like fused path): E experts each with
        n_e tokens, total T tokens fixed; vary the BUSIEST expert load while others
        are light -> directly probes "does CP (max-tokens-on-one-expert) drive time?"
  L4  WEIGHT-LOAD MECHANISM probe: scale expert intermediate dim (=> weight bytes)
        at fixed small B; if floor scales ~linearly with weight bytes -> weight-load-bound.
  Each leg: fresh warmup, >=5 trials, report median/mean/std/p05/p95.

Roofline anchors printed for comparison.
"""
import argparse, json, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 4096
INTER = 1536           # Qwen3-235B-A22B moe_intermediate_size
DTYPE = torch.bfloat16
HBM_BW = 2039e9
PEAK_BF16 = 312e12

FFN_GRID = [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320,
            345, 384, 448, 512, 640, 768, 1024, 1536, 2048, 3072, 4096]


def stats(samples):
    a = np.array(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(a)),
        "mean_ms": float(np.mean(a)),
        "std_ms": float(np.std(a)),
        "p05_ms": float(np.percentile(a, 5)),
        "p95_ms": float(np.percentile(a, 95)),
        "n": int(a.size),
    }


class ExpertFFN(nn.Module):
    def __init__(self, inter=INTER):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, inter, bias=False)
        self.up = nn.Linear(HIDDEN, inter, bias=False)
        self.down = nn.Linear(inter, HIDDEN, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def time_callable(fn, device, repeats, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        samples.append(s.elapsed_time(e))
    return stats(samples)


# ---- L1 eager (reproduce Phase A) ----
def leg_eager(device, repeats, warmup):
    out = {}
    expert = ExpertFFN().to(device, DTYPE).eval()
    with torch.no_grad():
        for b in FFN_GRID:
            x = torch.randn(b, HIDDEN, device=device, dtype=DTYPE)
            out[b] = time_callable(lambda: expert(x), device, repeats, warmup)
            print(f"  [L1 eager] B={b:>5d} med={out[b]['median_ms']:.4f} std={out[b]['std_ms']:.4f}", flush=True)
    del expert; torch.cuda.empty_cache()
    return out


# ---- L2 CUDA graph captured ----
def leg_graph(device, repeats, warmup):
    out = {}
    expert = ExpertFFN().to(device, DTYPE).eval()
    with torch.no_grad():
        for b in FFN_GRID:
            x = torch.randn(b, HIDDEN, device=device, dtype=DTYPE)
            # warmup on side stream (required before capture)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(s):
                for _ in range(5):
                    y = expert(x)
            torch.cuda.current_stream(device).wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y = expert(x)
            out[b] = time_callable(lambda: g.replay(), device, repeats, warmup)
            print(f"  [L2 graph] B={b:>5d} med={out[b]['median_ms']:.4f} std={out[b]['std_ms']:.4f}", flush=True)
            del g
    del expert; torch.cuda.empty_cache()
    return out


# ---- L3 grouped multi-expert: vary busiest-expert load, hold others light ----
def leg_grouped(device, repeats, warmup, n_experts, light):
    """E experts resident; expert0 gets B_hot tokens, the rest get `light` each.
    Time the whole grouped FFN -> does total time track B_hot (CP-like) once hot>knee?
    Implemented as a loop over experts (mirrors a simple grouped-GEMM dispatch)."""
    out = {}
    experts = [ExpertFFN().to(device, DTYPE).eval() for _ in range(n_experts)]
    with torch.no_grad():
        for b_hot in FFN_GRID:
            xs = [torch.randn(b_hot if i == 0 else light, HIDDEN, device=device, dtype=DTYPE)
                  for i in range(n_experts)]
            def run():
                for i in range(n_experts):
                    _ = experts[i](xs[i])
            out[b_hot] = time_callable(run, device, repeats, warmup)
            tot = b_hot + light * (n_experts - 1)
            print(f"  [L3 grp E={n_experts} light={light}] B_hot={b_hot:>5d} (tot={tot}) "
                  f"med={out[b_hot]['median_ms']:.4f}", flush=True)
    del experts; torch.cuda.empty_cache()
    return out


# ---- L4 weight-load mechanism: scale weight bytes via intermediate dim ----
def leg_weightscale(device, repeats, warmup, b_fixed):
    out = {}
    for inter in [384, 768, 1536, 3072, 6144]:
        expert = ExpertFFN(inter=inter).to(device, DTYPE).eval()
        x = torch.randn(b_fixed, HIDDEN, device=device, dtype=DTYPE)
        with torch.no_grad():
            r = time_callable(lambda: expert(x), device, repeats, warmup)
        wbytes = (2 * HIDDEN * inter + inter * HIDDEN) * 2
        r["weight_bytes"] = wbytes
        r["wload_pred_ms"] = wbytes / HBM_BW * 1e3
        out[inter] = r
        print(f"  [L4 wscale b={b_fixed}] inter={inter:>5d} Wbytes={wbytes/1e6:.1f}MB "
              f"med={r['median_ms']:.4f} wload_pred={r['wload_pred_ms']:.4f}", flush=True)
        del expert; torch.cuda.empty_cache()
    return out


def roofline():
    W_bytes = (2 * HIDDEN * INTER + INTER * HIDDEN) * 2
    return {
        "hbm_bw_bytes_s": HBM_BW, "peak_bf16_flops": PEAK_BF16,
        "ridge_flop_per_byte": PEAK_BF16 / HBM_BW,
        "expert_weight_bytes": W_bytes,
        "weight_load_floor_ms": W_bytes / HBM_BW * 1e3,
        "knee_compute_crosses_wload": (W_bytes / HBM_BW) * PEAK_BF16 / (6 * HIDDEN * INTER),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--trials", type=int, default=5, help="independent re-measures of each leg for variance-of-median")
    ap.add_argument("--n-experts", type=int, default=16)
    ap.add_argument("--light", type=int, default=4)
    ap.add_argument("--wscale-b", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--legs", default="eager,graph,grouped,weightscale")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"device={torch.cuda.get_device_name(0)} legs={args.legs}", flush=True)
    legs = set(args.legs.split(","))
    res = {"meta": {"device": torch.cuda.get_device_name(0), "hidden": HIDDEN,
                    "inter": INTER, "dtype": "bf16", "repeats": args.repeats,
                    "warmup": args.warmup, "trials": args.trials,
                    "n_experts": args.n_experts, "light": args.light},
           "roofline": roofline()}
    # multi-trial wrapper for the two key single-expert legs (variance of the median)
    def multitrial(fn):
        per_b = {}
        for t in range(args.trials):
            r = fn()
            for b, s in r.items():
                per_b.setdefault(b, []).append(s["median_ms"])
        agg = {}
        for b, meds in per_b.items():
            a = np.array(meds)
            agg[b] = {"median_of_medians_ms": float(np.median(a)),
                      "mean_ms": float(np.mean(a)), "std_ms": float(np.std(a)),
                      "min_ms": float(a.min()), "max_ms": float(a.max()),
                      "trials": len(meds)}
        return agg
    if "eager" in legs:
        print("== L1 EAGER ==", flush=True)
        res["eager"] = multitrial(lambda: leg_eager(device, args.repeats, args.warmup))
    if "graph" in legs:
        print("== L2 CUDA-GRAPH ==", flush=True)
        res["graph"] = multitrial(lambda: leg_graph(device, args.repeats, args.warmup))
    if "grouped" in legs:
        print("== L3 GROUPED ==", flush=True)
        res["grouped"] = leg_grouped(device, args.repeats, args.warmup, args.n_experts, args.light)
    if "weightscale" in legs:
        print("== L4 WEIGHT-SCALE ==", flush=True)
        res["weightscale"] = leg_weightscale(device, args.repeats, args.warmup, args.wscale_b)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
