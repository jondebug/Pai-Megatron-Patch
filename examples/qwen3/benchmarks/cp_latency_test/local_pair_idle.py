"""
Q: Does pretrained's peaked routing leave one of each rank's 2 local experts idle more
   often than r15's flatter routing?

For each (cell, bs) and each layer, compute the (M_localA, M_localB) pair for each of 64
ranks at EP=64 linear placement (rank i owns experts {2i, 2i+1}). Classify each pair:

  - BOTH-IDLE:        both experts have M = 0
  - ONE-IDLE:         exactly one expert has M = 0 (the other has >0)
  - BOTH-ACTIVE:      both > 0, but skewed (MIN < 10% of MAX)
  - BOTH-BALANCED:    both > 0 and MIN/MAX >= 10%

Outputs aggregate counts per cell × bs across 94 layers × 64 ranks = 6016 pairs.
"""
import json, glob, os
import numpy as np

RES = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

def load_layers(path):
    d = json.load(open(path))
    pl = d["per_layer_per_expert_tokens"]
    if isinstance(pl, dict):
        return [np.array(pl[str(i)], dtype=np.float64) for i in range(d["n_layers_captured"])]
    return [np.array(L, dtype=np.float64) for L in pl]

def classify(local_arr, idle_threshold=0, skew_threshold=0.10):
    n = len(local_arr)
    n_idle = int(np.sum(local_arr == idle_threshold))
    if n_idle == n:
        return "ALL-IDLE"
    if n_idle > 0:
        return f"{n_idle}-OF-{n}-IDLE"
    mn, mx = float(local_arr.min()), float(local_arr.max())
    if mn < skew_threshold * mx:
        return "ACTIVE-SKEWED"
    return "ACTIVE-BALANCED"

def analyze(layers, ep_size=64, n_experts=128):
    from collections import Counter
    n_per_rank = n_experts // ep_size
    counts = Counter()
    skew_ratios = []
    rank_imbalance = []
    for L in layers:
        global_mean = L.mean()
        for rank in range(ep_size):
            pair = L[rank*n_per_rank:(rank+1)*n_per_rank]
            counts[classify(pair)] += 1
            if pair.min() > 0:
                skew_ratios.append(float(pair.min()) / float(pair.max()))
            rank_imbalance.append(float(pair.max()) / max(global_mean, 1))
    return counts, skew_ratios, rank_imbalance

print()
for ep_size in [64, 32, 16, 8]:
    n_per_rank = 128 // ep_size
    print()
    print("="*130)
    print(f"LOCAL-EXPERT GROUP ACTIVATION at EP={ep_size} ({n_per_rank} experts/rank, linear placement)")
    print("="*130)
    print(f"{'cell':<5} {'bs':<4} {'any-idle %':<12} {'avg n_idle/rank':<18} {'min/max p50':<14} {'min/mean p50':<14} {'rank_peak/global':<18}")
    for bs in [1, 2, 4]:
        for cell in ["pre", "r15"]:
            candidates = glob.glob(f"{RES}/routing_{cell}_bs{bs}_*.json")
            if not candidates: continue
            layers = load_layers(candidates[0])
            counts, skew, peak = analyze(layers, ep_size=ep_size)
            total = sum(counts.values())
            # any-idle = at least one of n_per_rank experts has M=0
            any_idle = sum(v for k, v in counts.items() if "IDLE" in k)
            # avg fraction-idle per rank
            n_idle_sum = 0
            for k, v in counts.items():
                if "OF" in k:
                    n_idle_sum += int(k.split("-")[0]) * v
                elif k == "ALL-IDLE":
                    n_idle_sum += n_per_rank * v
            avg_idle = n_idle_sum / total
            skew_p50 = float(np.median(skew)) if skew else 0
            peak_p50 = float(np.median(peak))
            print(f"{cell:<5} {bs:<4} {any_idle/total*100:>6.2f}%      {avg_idle:>8.3f}           {skew_p50:<14.3f} {'-':<14} {peak_p50:<.3f}")
print()
print("(any-idle = % of rank-layer slots where at least one local expert has M=0)")
print("(avg n_idle/rank = mean number of idle local experts per rank-layer slot)")
print("(skew min/max p50 = median of min(local M's) / max(local M's), for slots where all-active)")
print()
print("(skew = min(local M's) / max(local M's)) — lower = more imbalanced within rank's local experts")
