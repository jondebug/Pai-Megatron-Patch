"""
Routing-distribution analysis. Loads per-layer per-expert token counts from cp_routing_dump.py
output (routing_<cell>_bs<N>_<jobid>.json), computes per-rank load distribution at various EP
sizes, and quantifies what CP misses.

Q: Does per-rank load (M_max per rank) explain observed TTFT/TPOT deltas better than the
   expert-level CP metric does?

Inputs:
  /lustre/.../cp_latency_results/routing_{pre,r15}_bs{1,2,4}_*.json
Outputs:
  /lustre/.../cp_latency_results/routing_distribution_analysis.json
  stdout summary table
"""
import glob, json, os
import numpy as np
from collections import defaultdict

RES = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"
N_EXPERTS = 128
TOP_K = 8
KNEE_M = 1024  # rule-of-thumb; will be re-derived from gemm_knee job

# === Load all available routing dumps ===
dumps = {}  # (cell, bs) -> dict with per_layer_per_expert_tokens
for f in sorted(glob.glob(RES + "/routing_pre_bs*_*.json") + glob.glob(RES + "/routing_r15_bs*_*.json")):
    name = os.path.basename(f)
    parts = name.replace(".json","").split("_")
    # routing_pre_bs1_29334887  -> cell=pre bs=1
    cell = parts[1]
    bs = int(parts[2].replace("bs",""))
    d = json.load(open(f))
    dumps[(cell, bs)] = d
    print(f"loaded {f}: layers={d.get('n_layers_captured')} bs={bs} cell={cell}")

if not dumps:
    print("NO routing dumps found")
    raise SystemExit(1)

# === Analysis per (cell, bs) ===
def per_layer_dist(d):
    """Return list of np.ndarray(128,) — token counts per expert per layer."""
    pl = d["per_layer_per_expert_tokens"]
    if isinstance(pl, dict):
        return [np.array(pl[str(i)] if str(i) in pl else pl[i], dtype=np.float64) for i in range(d["n_layers_captured"])]
    return [np.array(L, dtype=np.float64) for L in pl]

def shannon_entropy(probs):
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0

def per_rank_load(layer_dist, ep_size, placement="linear"):
    """Sum expert tokens to rank tokens at given EP size + placement."""
    n_per_rank = N_EXPERTS // ep_size
    if placement == "linear":
        return layer_dist.reshape(ep_size, n_per_rank).sum(axis=1)
    elif placement == "round_robin":
        # round_robin: rank i owns experts [i, i+ep_size, i+2*ep_size, ...]
        return np.array([layer_dist[i::ep_size].sum() for i in range(ep_size)])
    raise ValueError(placement)

results = {"cells": list(set(c for c,_ in dumps.keys())), "batch_sizes": sorted(set(b for _,b in dumps.keys())), "ep_sizes": [8, 16, 32, 64], "per_cell_bs": []}

print()
print("="*120)
print("PER-LAYER EXPERT DISTRIBUTION SUMMARY (averaged across 94 layers)")
print("="*120)
print(f"{'cell':<6} {'bs':<4} {'plen':<6} {'total_tokens':<14} {'M_max':<10} {'M_p99':<10} {'M_mean':<10} {'M_p50':<8} {'N_active':<10} {'N>knee':<10} {'entropy':<10}")
for (cell, bs), d in sorted(dumps.items()):
    layers = per_layer_dist(d)
    seq_len = d["seq_length"]
    total = int(layers[0].sum())  # bs * plen * top_k
    m_max  = np.mean([L.max()       for L in layers])
    m_p99  = np.mean([np.percentile(L, 99)        for L in layers])
    m_mean = np.mean([L.mean()      for L in layers])
    m_p50  = np.mean([np.median(L)  for L in layers])
    n_act  = np.mean([(L > 0).sum() for L in layers])
    n_knee = np.mean([(L > KNEE_M).sum() for L in layers])
    entropy = np.mean([shannon_entropy(L / max(L.sum(), 1)) for L in layers])
    print(f"{cell:<6} {bs:<4} {seq_len:<6} {total:<14} {m_max:<10.1f} {m_p99:<10.1f} {m_mean:<10.1f} {m_p50:<8.1f} {n_act:<10.1f} {n_knee:<10.1f} {entropy:<10.3f}")
    results["per_cell_bs"].append({
        "cell": cell, "bs": bs, "seq_len": seq_len, "total_tokens": total,
        "m_max_avg": float(m_max), "m_p99_avg": float(m_p99), "m_mean_avg": float(m_mean), "m_p50_avg": float(m_p50),
        "n_active_avg": float(n_act), "n_above_knee_avg": float(n_knee), "entropy_avg": float(entropy),
    })

print()
print("="*120)
print("PER-RANK LOAD AT VARIOUS EP SIZES — busiest-rank token count per layer (linear placement)")
print("="*120)
print(f"{'cell':<6} {'bs':<4} {'EP':<5} {'rank_M_max':<14} {'rank_M_min':<14} {'rank_M_mean':<14} {'rank_imbalance':<16} {'rank_M_max/EP=8':<16}")
per_rank_data = []
for (cell, bs), d in sorted(dumps.items()):
    layers = per_layer_dist(d)
    for ep in [8, 16, 32, 64]:
        rank_max_per_layer = []
        rank_min_per_layer = []
        rank_mean_per_layer = []
        for L in layers:
            r = per_rank_load(L, ep)
            rank_max_per_layer.append(r.max())
            rank_min_per_layer.append(r.min())
            rank_mean_per_layer.append(r.mean())
        rmax = np.mean(rank_max_per_layer)
        rmin = np.mean(rank_min_per_layer)
        rmean = np.mean(rank_mean_per_layer)
        rimbal = rmax / max(rmean, 1)
        print(f"{cell:<6} {bs:<4} {ep:<5} {rmax:<14.1f} {rmin:<14.1f} {rmean:<14.1f} {rimbal:<16.3f}")
        per_rank_data.append({"cell": cell, "bs": bs, "ep": ep, "rank_M_max_avg": float(rmax), "rank_M_min_avg": float(rmin), "rank_M_mean_avg": float(rmean), "rank_imbalance": float(rimbal)})
results["per_rank_load"] = per_rank_data

# === Pretrained vs r15 deltas at each (bs, EP) ===
print()
print("="*120)
print("DELTA (r15 - pretrained) / pretrained, per (bs, EP) — busiest-rank load")
print("="*120)
print(f"{'bs':<4} {'EP':<5} {'pre rank_M_max':<16} {'r15 rank_M_max':<16} {'Δ %':<10} {'pre rank_imbalance':<20} {'r15 rank_imbalance':<20}")
deltas = []
for bs in sorted(set(b for _,b in dumps.keys())):
    if ("pre", bs) in dumps and ("r15", bs) in dumps:
        pre_l = per_layer_dist(dumps[("pre", bs)])
        r15_l = per_layer_dist(dumps[("r15", bs)])
        for ep in [8, 16, 32, 64]:
            pre_rm = np.mean([per_rank_load(L, ep).max() for L in pre_l])
            r15_rm = np.mean([per_rank_load(L, ep).max() for L in r15_l])
            pre_imb = pre_rm / np.mean([per_rank_load(L, ep).mean() for L in pre_l])
            r15_imb = r15_rm / np.mean([per_rank_load(L, ep).mean() for L in r15_l])
            d_pct = (r15_rm - pre_rm) / pre_rm * 100
            print(f"{bs:<4} {ep:<5} {pre_rm:<16.1f} {r15_rm:<16.1f} {d_pct:<+10.2f}% {pre_imb:<20.3f} {r15_imb:<20.3f}")
            deltas.append({"bs": bs, "ep": ep, "pre_rank_M_max": float(pre_rm), "r15_rank_M_max": float(r15_rm), "delta_pct": float(d_pct), "pre_rank_imbalance": float(pre_imb), "r15_rank_imbalance": float(r15_imb)})
results["pretrained_vs_r15_deltas"] = deltas

# === Save ===
out = RES + "/routing_distribution_analysis.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out}")
