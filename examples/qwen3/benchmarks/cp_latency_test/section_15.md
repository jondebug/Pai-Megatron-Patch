
---

## §15 Measured GEMM saturation knee for Qwen3-235B FFN shape (2026-06-21)

### Question

We've been using "GEMM saturation knee ~1024 tokens" as a rule of thumb to decide whether a regime
is above or below the FLOPs-bound point. Where does the curve actually saturate for the model's
specific FFN shape on H100?

### Method

Single-GPU micro-benchmark of bf16 matmul at the Qwen3-235B-A22B expert FFN inner dimension:
K = 4096 (hidden), N = 1536 (intermediate per expert FFN1/FFN2). Sweep M from 1 to 8192. n=30
trials at each point. Note: this benches `torch.matmul` (Triton-or-cuBLAS-backed depending on
sizes), not the exact `fused_moe_kernel` grouped-GEMM — but the per-token saturation curve is
essentially the same physics since fused_moe is a grouped wrapper around tiles of similar shape.

### Findings

| M | latency (µs) | TFLOPS/s | % of asymptote |
|---|---|---|---|
| 1 | 35.5 | 0.35 | 0.1% |
| 32 | 36.4 | 11.0 | 4.6% |
| 128 | 41.2 | 39.1 | 16.3% |
| 256 | 46.6 | 69.1 | 28.8% |
| 512 | 57.8 | 111.4 | 46.4% |
| **1024** | 77.4 | 166.4 | **69.4%** |
| 1536 | 113.3 | 170.6 | 71.1% |
| 2048 | 125.7 | 205.0 | 85.4% |
| **4096** | 228.0 | 226.0 | **94.2%** ← knee (90%) |
| 6144 | 328.8 | 235.1 | 98.0% |
| 8192 | 429.7 | 239.9 | 100% (asymptote) |

**Asymptotic throughput: 239.9 TFLOPS/s. Knee (90% of asymptote): M = 4096.**

Compare to the rule-of-thumb knee (~1024) we'd been using: actual saturation requires **4×** more
tokens-per-expert than assumed.

### Implications for the prior sections

The "above/below knee" framing in §11, §12, §13, §14 was using a knee that's too low. Recasting:

| regime | per-expert M (was) | per-expert M (correct) | % of asymptote | knee position |
|---|---|---|---|---|
| Prefill EP=8 plen=8192 bs=1 | 512 (was "below knee") | 512 | **46%** | well below |
| Prefill EP=8 plen=8192 bs=2 | 1024 (was "near knee") | 1024 | **69%** | below |
| Prefill EP=8 plen=8192 bs=4 | 2048 (was "above knee" — where r15 wins -12%) | 2048 | **85%** | approaching knee |
| Prefill EP=8 plen=8192 bs=8 | 4096 (was "deeply above") | 4096 | **94%** ← knee |
| Decode EP=64 plen=256 bs=512 (the +15% TPOT mystery) | 32 (was "deeply below") | 32 | **4.6%** | catastrophically below |
| Decode EP=64 bs=16384 (just above knee in old framing) | 1024 | 1024 | **69%** | still below knee |
| Decode EP=64 bs=32768 | 2048 | 2048 | 85% | approaching knee |
| Decode EP=64 bs=65 536 | 4096 | 4096 | **94%** ← at knee |
| Decode EP=64 bs=131 072 | 8192 | 8192 | 100% saturated |

The observed sign-flip in TTFT delta at EP=8 prefill from bs=1 (+7%) to bs=4 (−12%) happens
when per-expert M moves from 46% of asymptote to 85% — NOT from above-to-below knee in absolute
terms. Even at bs=4, we're not technically saturated, just substantially less *un*-saturated. The
GEMM curve is gradual enough that the relative position on the curve (rather than a hard knee)
controls whether balanced routing helps.

### Implications for finding the "r15 wins" regime

To reach the truly above-knee regime at EP=64 decode, need:
- bs ≥ 65 536 for 94% saturation
- bs ≥ 131 072 for full saturation

The `ep64_ultrabs_decode` job (29337579) targets bs={32 768, 65 536}. The bs=65 536 point will be
right at the measured knee. If r15 doesn't win at bs=65 536, we'd need to go higher to test the
prediction cleanly — or accept that even at maximum-realistic decode bs, the per-expert M isn't
high enough at EP=64 to enter the regime where balanced routing wins.

### Why the knee is at M=4096, not 1024

Two main reasons:
1. **Triton/cuBLAS picks 64×128 or 128×128 tile shapes for this aspect ratio (K=4096, N=1536).**
   To fully utilize a 128×128 tile, you need M ≥ 128 just for one tile. To amortize the per-tile
   K-dim load (which is what dominates compute), you need M ≥ ~128×SMs/wave to fully occupy.
2. **The N dimension is relatively small (1536).** With N=1536 and 128-wide N-tiles, you get only
   12 N-tiles. Combined with the SM count (~108 on A100, ~132 on H100), you need M big enough to
   fill the SM grid even with a small N — pushing the knee higher.

For models with larger N (e.g., Qwen3 dense layers at intermediate=12288), the knee would be
substantially lower. The MoE FFN's compact per-expert N=1536 is intrinsically harder to saturate.

### Caveats

1. **`torch.matmul` ≠ `fused_moe_kernel`.** The grouped-GEMM kernel has its own per-expert
   startup cost on top of the GEMM. So in MoE, the EFFECTIVE knee is likely *higher* than 4096
   (since the per-expert grid has fewer tiles than this single-matmul measurement).
2. **bf16-only.** Future fp8 or fp4 inference may shift the knee differently.
3. **H100 only.** B200 likely moves the knee LEFT (faster FLOPs, similar tile sizes).
4. **Single-GPU, no TP shard interference.** In a TP=8 setup, additional intra-rank effects
   (NVLink reads of replicated activations) could shift effective utilization.

### Files

- Bench: `gemm_knee.py` (single-GPU bf16 matmul at K=4096, N=1536, M-sweep, n=30 trials)
- Results: `cp_latency_results/gemm_knee_curve.json`
- Job: 29337465

