"""GEMM saturation knee microbench for Qwen3-235B-A22B FFN shape on GB200.
K=4096, N=1536, bf16 matmul, M-sweep 1..8192.
Direct comparison to ORD H100 knee at M=4096."""
import json, time
import torch
import torch.cuda as cu

K, N = 4096, 1536
DTYPE = torch.bfloat16
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
N_WARMUP, N_TRIALS = 5, 30

torch.cuda.set_device(0)
W = torch.randn(K, N, dtype=DTYPE, device="cuda")
results = {"K": K, "N": N, "dtype": "bfloat16", "device": torch.cuda.get_device_name(0),
           "compute_cap": torch.cuda.get_device_capability(0), "n_trials": N_TRIALS, "rows": []}

print(f"GEMM knee on {results['device']} (compute_cap={results['compute_cap']}), K={K}, N={N}, bf16")
print(f"{'M':>6} {'mean_us':>10} {'std_us':>8} {'tflops_per_s':>14} {'time_per_M_us':>14}")
for M in M_VALUES:
    A = torch.randn(M, K, dtype=DTYPE, device="cuda")
    for _ in range(N_WARMUP):
        _ = A @ W
    cu.synchronize()
    times_us = []
    for _ in range(N_TRIALS):
        cu.synchronize()
        t0 = time.perf_counter_ns()
        _ = A @ W
        cu.synchronize()
        times_us.append((time.perf_counter_ns() - t0) / 1000.0)
    import statistics as st
    mu, sd = st.mean(times_us), (st.stdev(times_us) if len(times_us) > 1 else 0.0)
    tflops = 2 * M * K * N / (mu * 1e-6) / 1e12 if mu > 0 else 0.0
    print(f"{M:>6} {mu:>10.2f} {sd:>8.2f} {tflops:>14.3f} {mu/M:>14.4f}")
    results["rows"].append({"M": M, "mean_us": mu, "std_us": sd, "tflops_per_s": tflops, "time_per_M_us": mu/M})

asymp = max(r["tflops_per_s"] for r in results["rows"])
knee_M = next((r["M"] for r in results["rows"] if r["tflops_per_s"] >= 0.9 * asymp), None)
results["asymptote_tflops_per_s"] = asymp
results["knee_M_at_90pct"] = knee_M
print()
print(f"Asymptote: {asymp:.3f} TFLOPS/s   Knee (M at 90%): {knee_M}")

with open("/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results/hsg_gemm_knee.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved.")
