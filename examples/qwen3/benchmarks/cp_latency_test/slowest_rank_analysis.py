"""
Q: At EP=64, the per-step wall time is bounded by the slowest rank's expert FFN. If
   the slowest rank in BOTH cells has all experts activated, what makes r15 slower?

Method: from the existing per-layer per-expert routing dumps, identify per layer which
rank is the busiest (highest total local tokens). Then look at the (M_local_A, M_local_B)
distribution for THAT specific rank — does pretrained's busiest rank have one fat expert
and one tiny, while r15's busiest rank has two roughly-equal experts?

Also compute the *predicted* per-rank FFN time using a simple cost model:
   T_rank = sum over local experts of: alpha + M_expert × beta(M_expert)
where alpha = per-expert launch/dispatch cost (~5-20 us per active expert)
      beta(M) = per-token compute cost, which is ~1/throughput(M), highest at low M.

This lets us check: even when pretrained's busiest rank has higher MAX-per-expert load,
does the per-expert dispatch cost paid by r15 (because both experts are active) flip the
total? Use the measured gemm_knee curve to derive beta(M).
"""
import json, glob, os
import numpy as np

RES = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

# Load measured GEMM-knee curve to get beta(M) = us per token at given M
knee = json.load(open(RES + "/gemm_knee_curve.json"))
KNEE_M = [r["M"] for r in knee["rows"]]
KNEE_us = [r["mean_us"] for r in knee["rows"]]
KNEE_per_M = [r["time_per_M_us"] for r in knee["rows"]]

def gemm_time_us(M):
    """Interpolate GEMM kernel time (us) for given M from the measured curve.
       Note: this is for one expert's GEMM (M tokens × K × N at K=4096 N=1536)."""
    if M <= 0: return 0.0
    # Find nearest M values
    arr = np.array(KNEE_M)
    if M <= arr.min(): return KNEE_us[0] * (M / arr.min())  # extrapolate down
    if M >= arr.max(): return KNEE_us[-1] * (M / arr.max())  # extrapolate up
    idx = np.searchsorted(arr, M)
    lo, hi = arr[idx-1], arr[idx]
    frac = (M - lo) / (hi - lo)
    return KNEE_us[idx-1] * (1-frac) + KNEE_us[idx] * frac

def load_layers(path):
    d = json.load(open(path))
    pl = d["per_layer_per_expert_tokens"]
    if isinstance(pl, dict):
        return [np.array(pl[str(i)], dtype=np.float64) for i in range(d["n_layers_captured"])]
    return [np.array(L, dtype=np.float64) for L in pl]

print("="*120)
print("PER-RANK FFN-TIME MODEL using MEASURED gemm_knee curve")
print("="*120)
print(f"GEMM-time(M) calibration (selected points):")
for M in [1, 8, 32, 128, 256, 512, 1024, 2048, 4096]:
    print(f"  M={M:5d}:  {gemm_time_us(M):7.2f} us  ({gemm_time_us(M)/M:.3f} us/token)")
print()

for bs in [1, 2, 4]:
    print(f"\n--- bs={bs} prefill (plen=8192, total tokens/fwd = {8192*bs}) ---")
    for cell in ["pre", "r15"]:
        path = sorted(glob.glob(f"{RES}/routing_{cell}_bs{bs}_*.json"))[0]
        layers = load_layers(path)
        # For each layer, compute per-rank (M_A, M_B) at EP=64 and find busiest rank
        ep = 64; npr = 128 // ep
        busiest_total = []   # total local tokens for the busiest rank, per layer
        busiest_min   = []   # smaller local expert's M for the busiest rank
        busiest_max   = []   # larger local expert's M for the busiest rank
        busiest_n_active = []  # n active local experts
        busiest_model_time = []   # predicted T_rank
        all_rank_times = []  # all 64 ranks' predicted T, per layer
        for L in layers:
            per_rank_times = []
            per_rank_pairs = []
            for r in range(ep):
                pair = L[r*npr:(r+1)*npr]
                # Time = sum over active local experts of GEMM(M_i)
                t = sum(gemm_time_us(m) for m in pair)
                per_rank_times.append(t)
                per_rank_pairs.append(pair)
            slow_rank = int(np.argmax(per_rank_times))
            busiest_total.append(float(per_rank_pairs[slow_rank].sum()))
            busiest_min.append(float(per_rank_pairs[slow_rank].min()))
            busiest_max.append(float(per_rank_pairs[slow_rank].max()))
            busiest_n_active.append(int((per_rank_pairs[slow_rank] > 0).sum()))
            busiest_model_time.append(per_rank_times[slow_rank])
            all_rank_times.append(per_rank_times)
        # Layer-aggregated stats
        bt = np.array(busiest_total); bx = np.array(busiest_max); bn = np.array(busiest_min); bf = np.array(busiest_n_active); bmt = np.array(busiest_model_time)
        print(f"  {cell:4s}: busiest-rank stats (avg over 94 layers)")
        print(f"     total tokens    : {bt.mean():7.1f} ± {bt.std():.1f}")
        print(f"     max-expert M    : {bx.mean():7.1f} ± {bx.std():.1f}")
        print(f"     min-expert M    : {bn.mean():7.1f} ± {bn.std():.1f}")
        print(f"     n_active local  : {bf.mean():.2f} / 2")
        print(f"     predicted T_rank: {bmt.mean():7.2f} us ± {bmt.std():.1f}   <-- the slowest GPU's predicted FFN time")
    # Also compute predicted T_rank for the WHOLE batch (max across ranks per layer, summed across layers)
    print(f"     ↑ predicted busiest-GPU FFN time per layer, comparison for slowest rank")

print()
print("="*120)
print("DIRECT COMPARISON: predicted busiest-GPU FFN time per layer, pretrained vs r15")
print("="*120)
print(f"{'bs':<4} {'pre_T (us/layer)':<18} {'r15_T (us/layer)':<18} {'Δ %':<10}")
for bs in [1, 2, 4]:
    pre_path = sorted(glob.glob(f"{RES}/routing_pre_bs{bs}_*.json"))[0]
    r15_path = sorted(glob.glob(f"{RES}/routing_r15_bs{bs}_*.json"))[0]
    pre_layers = load_layers(pre_path); r15_layers = load_layers(r15_path)
    ep=64; npr=2
    def slowest_t(layers):
        t_per_layer = []
        for L in layers:
            best = 0
            for r in range(ep):
                pair = L[r*npr:(r+1)*npr]
                t = sum(gemm_time_us(m) for m in pair)
                if t > best: best = t
            t_per_layer.append(best)
        return np.mean(t_per_layer)
    pt = slowest_t(pre_layers); rt = slowest_t(r15_layers)
    print(f"{bs:<4} {pt:<18.2f} {rt:<18.2f} {(rt-pt)/pt*100:+.2f}%")

print()
print("="*120)
print("SAMPLE: WHAT THE SLOWEST GPU LOOKS LIKE at bs=1 plen=8192 (layer 0)")
print("="*120)
for cell in ["pre", "r15"]:
    path = sorted(glob.glob(f"{RES}/routing_{cell}_bs{bs}_*.json"))[0]
    layers = load_layers(path)
    L0 = layers[0]
    ep=64; npr=2
    pairs = [L0[r*npr:(r+1)*npr] for r in range(ep)]
    times = [sum(gemm_time_us(m) for m in p) for p in pairs]
    slow_r = int(np.argmax(times))
    pair = pairs[slow_r]
    print(f"  {cell:5s} layer0: slowest rank={slow_r}, local experts=({2*slow_r}, {2*slow_r+1})")
    print(f"         M=({pair[0]:.0f}, {pair[1]:.0f})  total={pair.sum():.0f}  n_active={int((pair>0).sum())}/2  T_pred={times[slow_r]:.1f}us")
    # Also median rank
    med_r = int(np.argsort(times)[ep//2])
    pair = pairs[med_r]
    print(f"         median rank={med_r}: M=({pair[0]:.0f}, {pair[1]:.0f})  T={times[med_r]:.1f}us")
