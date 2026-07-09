"""
Deeper nsys analysis: per-call duration *distribution* for fused_moe_kernel and other MoE kernels.
Earlier analysis showed mean fused_moe_kernel time is essentially identical between cells
(56.0 us pre, 53.6 us r15). But MEANS hide tails. If r15's distribution has more outliers or a
fatter slow tail at the per-launch level, that would point to specific GEMM tile shapes being
problematic. This script extracts the full per-launch duration distribution.

Q: Within a single nsys trace, does fused_moe_kernel call-duration distribution differ between
   pretrained, r15, r05 in ways the mean hides? Specifically: does r15 have a fatter slow tail?

Outputs:
  /lustre/.../cp_latency_results/nsys_moe_kernel_distribution.json
  stdout summary tables
"""
import sqlite3, os, json
import numpy as np

RES = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"
TRACES = {
  "pretrained_EP8_prefill": "nsys_ep8_prefill_pretrained_235b_29201913.sqlite",
  "r15_EP8_prefill":        "nsys_ep8_prefill_r15_cp4682_29230389.sqlite",
  "r05_EP8_prefill":        "nsys_ep8_prefill_r05_iter2500_cp_29230390.sqlite",
}
KERNELS_OF_INTEREST = [
    "fused_moe_kernel",
    "ncclDevKernel_AllReduce_Sum_bf16_RING_LL",
    "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn",
    "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn",
    "flash_fwd_splitkv_kernel",
    "topkGating",
    "act_and_mul_kernel",
]

results = {}

for label, path in TRACES.items():
    full = os.path.join(RES, path)
    if not os.path.exists(full):
        print(f"SKIP missing {full}")
        continue
    db = sqlite3.connect(full)
    c = db.cursor()
    print(f"\n=== {label} ({path}) ===")
    results[label] = {}
    for kname in KERNELS_OF_INTEREST:
        c.execute("""
            SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName=s.id
            WHERE s.value LIKE ?
        """, (kname + "%",))
        durs_us = []
        for s, e in c.fetchall():
            durs_us.append((e - s) / 1000.0)
        if not durs_us:
            print(f"  {kname[:50]:50s}: NO MATCHES")
            results[label][kname] = None
            continue
        a = np.array(durs_us)
        stats = {
            "count": len(a), "mean_us": float(a.mean()), "std_us": float(a.std()),
            "p50_us": float(np.percentile(a, 50)),  "p90_us": float(np.percentile(a, 90)),
            "p99_us": float(np.percentile(a, 99)),  "p99_9_us": float(np.percentile(a, 99.9)),
            "max_us": float(a.max()), "min_us": float(a.min()),
            "total_ms": float(a.sum()) / 1000.0,
            "cv": float(a.std() / a.mean()) if a.mean() > 0 else 0,
        }
        results[label][kname] = stats
        print(f"  {kname[:50]:50s}: n={stats['count']:>6} mean={stats['mean_us']:7.1f}us p50={stats['p50_us']:7.1f} p90={stats['p90_us']:7.1f} p99={stats['p99_us']:8.1f} p99.9={stats['p99_9_us']:9.1f} max={stats['max_us']:9.1f} cv={stats['cv']:.3f}")
    db.close()

print("\n" + "="*120)
print("Cross-cell deltas (per-launch percentile r15 vs pretrained)")
print("="*120)
if "pretrained_EP8_prefill" in results and "r15_EP8_prefill" in results:
    p = results["pretrained_EP8_prefill"]
    r = results["r15_EP8_prefill"]
    print(f"{'kernel':52s} {'percentile':12s} {'pre us':>10} {'r15 us':>10} {'Δ%':>10}")
    for k in KERNELS_OF_INTEREST:
        if p.get(k) is None or r.get(k) is None: continue
        for pct in ["p50_us", "mean_us", "p90_us", "p99_us", "max_us"]:
            pv, rv = p[k][pct], r[k][pct]
            d = (rv - pv) / max(pv, 1e-9) * 100
            print(f"{k[:50]:52s} {pct:12s} {pv:>10.2f} {rv:>10.2f} {d:>+9.1f}%")
        print()

# Save
out = RES + "/nsys_moe_kernel_distribution.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out}")
