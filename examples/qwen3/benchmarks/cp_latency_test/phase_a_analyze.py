"""Phase A replication analysis. Read both runs, pool trials, compute Welch t per bs."""
import json, glob, numpy as np
RES = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

files = sorted(glob.glob(f"{RES}/phase_A_A*.json"))
print(f"Reading {len(files)} replicate(s):")
for f in files: print("  ", f.split("/")[-1])

# Aggregate: per (cell, bs), collect all per-trial measurements across runs
agg = {}  # (cell, bs) -> {"ttft": [list of trial values], "e2e": [...]}
for f in files:
    d = json.load(open(f))
    for m in d.get("models", []):
        cell = "pre" if m["name"].startswith("pre") else "r15"
        for ck, cv in m["cells"].items():
            bs = cv["batch_size"]
            key = (cell, bs)
            if key not in agg: agg[key] = {"ttft_mean": [], "ttft_std": [], "n": [], "e2e_mean": [], "e2e_std": []}
            agg[key]["ttft_mean"].append(cv["ttft_ms_mean"])
            agg[key]["ttft_std"].append(cv["ttft_ms_std"])
            agg[key]["n"].append(cv["ttft_ms_n"])
            agg[key]["e2e_mean"].append(cv["end_to_end_ms_mean"])
            agg[key]["e2e_std"].append(cv["end_to_end_ms_std"])

# Pool runs into combined mean/std per (cell, bs)
def pool(means, stds, ns):
    means = np.array(means); stds = np.array(stds); ns = np.array(ns)
    N = ns.sum()
    grand = (means * ns).sum() / N
    # Pooled variance: sum((n-1) * s^2 + n * (mean-grand)^2) / (N - k)
    var = (((ns - 1) * stds**2).sum() + (ns * (means - grand)**2).sum()) / (N - len(ns))
    return grand, np.sqrt(var), N

print()
print(f"{'cell':<5} {'bs':<4} {'pre_TTFT (ms)':<22} {'r15_TTFT (ms)':<22} {'Δ %':<8} {'t-stat':<8} {'sig':<8} {'pre_e2e (ms)':<22} {'r15_e2e (ms)':<22} {'Δe2e%':<8}")
print("="*180)
bs_vals = sorted(set(b for (c,b) in agg.keys()))
for bs in bs_vals:
    if ("pre", bs) not in agg or ("r15", bs) not in agg: continue
    pre = agg[("pre", bs)]; r15 = agg[("r15", bs)]
    pre_m, pre_s, pre_N = pool(pre["ttft_mean"], pre["ttft_std"], pre["n"])
    r15_m, r15_s, r15_N = pool(r15["ttft_mean"], r15["ttft_std"], r15["n"])
    pre_e, pre_es, _ = pool(pre["e2e_mean"], pre["e2e_std"], pre["n"])
    r15_e, r15_es, _ = pool(r15["e2e_mean"], r15["e2e_std"], r15["n"])
    d_pct = (r15_m - pre_m) / pre_m * 100
    # Welch t
    se = np.sqrt(pre_s**2/pre_N + r15_s**2/r15_N)
    t = (r15_m - pre_m) / se if se > 0 else 0
    sig = "***" if abs(t) > 2.58 else ("**" if abs(t) > 1.96 else ("*" if abs(t) > 1.64 else "ns"))
    de2e = (r15_e - pre_e) / pre_e * 100
    print(f"{'8':<5} {bs:<4} {pre_m:7.2f} ± {pre_s:5.2f} (N={pre_N})     {r15_m:7.2f} ± {r15_s:5.2f} (N={r15_N})     {d_pct:+6.2f}%  {t:+6.2f}  {sig:<8} {pre_e:7.2f} ± {pre_es:5.2f}            {r15_e:7.2f} ± {r15_es:5.2f}            {de2e:+6.2f}%")
print()
print("sig: * p<0.10  ** p<0.05  *** p<0.01  ns = not significant")
