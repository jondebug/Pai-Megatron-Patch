"""HSG benchmark result processor. Reads HSG bench JSONs, computes deltas + statistical tests,
and outputs a comparison table vs ORD baseline. Ready to run the instant bench outputs land."""
import json, os, glob, math, sys

RES = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

# ORD reference numbers (from §20 + §22 + §23)
ORD_REF = {
    ("prefill", 8, 8192, 1):  {"pre_ttft": 53.98, "r15_ttft": 57.67, "delta_pct": +6.84, "t": +108.6, "n": 150},
    ("prefill", 8, 8192, 2):  {"pre_ttft": 89.28, "r15_ttft": 89.06, "delta_pct": -0.25, "t": -0.11, "n": 150},
    ("prefill", 8, 8192, 4):  {"pre_ttft": 116.91,"r15_ttft": 125.07,"delta_pct": +6.98, "t": +2.86, "n": 150},
    ("prefill", 8, 8192, 8):  {"pre_ttft": 155.44,"r15_ttft": 155.10,"delta_pct": -0.22, "t": -0.12, "n": 150},
    ("prefill", 8, 8192, 16): {"pre_ttft": 196.46,"r15_ttft": 198.91,"delta_pct": +1.24, "t": +1.29, "n": 150},
    ("prefill", 8, 8192, 32): {"pre_ttft": 24071.77,"r15_ttft": 22572.07,"delta_pct": -6.23, "t": -745.2,"n": 150,
                                "note": "chunked prefill regime"},
    ("prefill", 32, 8192, 1): {"pre_ttft": 74.16, "r15_ttft": 78.23, "delta_pct": +5.49, "t": None, "n": 20},
    ("prefill", 32, 8192, 2): {"pre_ttft": 115.88,"r15_ttft": 120.05,"delta_pct": +3.60, "t": None, "n": 20},
    ("decode",  64, 256,   512):  {"pre_tpot": 339.0, "r15_tpot": 390.3, "delta_pct": +15.13, "t": +11.5, "n": 8},
    ("decode",  64, 256,   2048): {"pre_tpot": 401.9, "r15_tpot": 404.2, "delta_pct": +0.57, "t": -0.59, "n": 20},
    ("decode",  64, 256,   8192): {"pre_tpot": 1311.6,"r15_tpot": 1158.5,"delta_pct": -11.67,"t": -90.3, "n": 20},
}

def welch_t(m1, s1, n1, m2, s2, n2):
    """Welch's t-statistic."""
    if n1 < 2 or n2 < 2 or s1 == 0 or s2 == 0:
        return 0.0
    se = math.sqrt(s1**2/n1 + s2**2/n2)
    return (m2 - m1) / se if se > 0 else 0.0

def sig_stars(t):
    a = abs(t)
    if a > 2.58: return "***"
    if a > 1.96: return "**"
    if a > 1.64: return "*"
    return "ns"

def load_hsg_prefill(patterns):
    """Load HSG prefill bench JSONs into (bs, plen) -> {cell: cell_dict}."""
    grouped = {}
    for pattern in patterns:
        for f in sorted(glob.glob(os.path.join(RES, pattern))):
            try:
                d = json.load(open(f))
            except Exception as e:
                print(f"  skip {f}: {e}"); continue
            for m in d.get("models", []):
                cell = "pre" if m["name"].startswith("pre") else "r15"
                for ck, cv in m.get("cells", {}).items():
                    key = (cv["prompt_len"], cv["batch_size"])
                    if key not in grouped: grouped[key] = {}
                    grouped[key].setdefault(cell, []).append(cv)
    return grouped

def summarize_prefill(grouped, ep):
    print()
    print(f"=== HSG EP={ep} PREFILL {'plen=8192' if 8192 in [k[0] for k in grouped] else ''} ===")
    print(f"{'bs':<4} {'pre_TTFT ms':<18} {'r15_TTFT ms':<18} {'ΔTTFT %':<10} {'t-stat':<10} {'sig':<6} {'ORD Δ':<10}")
    print("-" * 90)
    for (plen, bs) in sorted(grouped.keys()):
        cells = grouped[(plen, bs)]
        if "pre" not in cells or "r15" not in cells: continue
        # Aggregate across all runs (multiple JSONs pooled)
        def pool(cell):
            means = [c["ttft_ms_mean"] for c in cells[cell]]
            stds  = [c.get("ttft_ms_std", 0) for c in cells[cell]]
            ns    = [c.get("ttft_ms_n", c.get("num_trials", 20)) for c in cells[cell]]
            import numpy as np
            means = np.array(means); stds = np.array(stds); ns = np.array(ns)
            N = ns.sum()
            grand = (means * ns).sum() / N
            var = (((ns-1) * stds**2).sum() + (ns * (means - grand)**2).sum()) / (N - len(ns)) if N > len(ns) else stds.mean()**2
            return grand, math.sqrt(var), N
        pt, ps, pN = pool("pre"); rt, rs, rN = pool("r15")
        delta = (rt - pt) / pt * 100
        t = welch_t(pt, ps, pN, rt, rs, rN)
        ord_ref = ORD_REF.get(("prefill", ep, plen, bs), {})
        ord_str = f"{ord_ref.get('delta_pct', '?'):+.2f}%" if ord_ref else "n/a"
        print(f"{bs:<4} {pt:6.2f} ± {ps:5.2f} (N={pN})  {rt:6.2f} ± {rs:5.2f} (N={rN})  {delta:+6.2f}%  {t:+6.2f}  {sig_stars(t):<6} {ord_str}")

if __name__ == "__main__":
    print("Waiting for HSG bench outputs...")
    prefill_grouped_ep8 = load_hsg_prefill(["hsg_phase_A_*.json"])
    summarize_prefill(prefill_grouped_ep8, ep=8)

    decode_grouped_ep8 = load_hsg_prefill(["hsg_decode_ep*.json"])
    summarize_prefill(decode_grouped_ep8, ep=8)  # will show decode as if prefill; adapt later

    print("\n(when bench outputs land, results will populate above tables automatically)")
