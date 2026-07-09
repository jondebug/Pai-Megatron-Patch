"""Single-GPU micro-benchmark: per-expert grouped-GEMM time vs M for Qwen3-235B-A22B FFN shapes.
K = 4096 (hidden), N = 1536 (intermediate per expert per FFN1/FFN2).
We don't have an easy isolated handle on vLLM's fused_moe_kernel from outside the layer, so we
benchmark Triton's standard matmul at the actual K,N to get the per-expert GEMM-time-vs-M curve.
This is a proxy for the relevant cost, not the exact fused_moe kernel — labelled accordingly.
"""
import json, time, sys
import torch
import torch.cuda as cu

# Qwen3-235B-A22B moe FFN shape
K = 4096
N = 1536
DTYPE = torch.bfloat16
DEV = "cuda"

# Sweep M values that bracket the expected knee
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
N_WARMUP = 5
N_TRIALS = 30

torch.cuda.set_device(0)

W = torch.randn(K, N, dtype=DTYPE, device=DEV)

results = {"K": K, "N": N, "dtype": "bfloat16", "n_trials": N_TRIALS, "rows": []}

print(f"{'M':>6} {'mean_us':>10} {'std_us':>8} {'tflops_per_s':>14} {'time_per_M_us':>14}")
for M in M_VALUES:
    A = torch.randn(M, K, dtype=DTYPE, device=DEV)
    # warmup
    for _ in range(N_WARMUP):
        Y = A @ W
    cu.synchronize()
    times_us = []
    for _ in range(N_TRIALS):
        cu.synchronize()
        t0 = time.perf_counter_ns()
        Y = A @ W
        cu.synchronize()
        t1 = time.perf_counter_ns()
        times_us.append((t1 - t0) / 1000.0)
    import statistics as st
    mu = st.mean(times_us)
    sd = st.stdev(times_us) if len(times_us) > 1 else 0.0
    flops = 2 * M * K * N
    tflops_per_s = flops / (mu * 1e-6) / 1e12 if mu > 0 else 0.0
    tpm = mu / M
    print(f"{M:>6} {mu:>10.2f} {sd:>8.2f} {tflops_per_s:>14.3f} {tpm:>14.4f}")
    results["rows"].append({"M": M, "mean_us": mu, "std_us": sd, "tflops_per_s": tflops_per_s, "time_per_M_us": tpm})

# Identify the knee: M where tflops_per_s reaches 90% of asymptote
asymptote = max(r["tflops_per_s"] for r in results["rows"])
knee = None
for r in results["rows"]:
    if r["tflops_per_s"] >= 0.9 * asymptote:
        knee = r["M"]
        break
print()
print(f"Asymptote: {asymptote:.3f} TFLOPS/s")
print(f"Knee (M at 90% of asymptote): {knee}")
results["asymptote_tflops_per_s"] = asymptote
results["knee_M_at_90pct"] = knee

out_path = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results/gemm_knee_curve.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out_path}")
