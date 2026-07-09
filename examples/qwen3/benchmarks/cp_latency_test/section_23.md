
---

## §23 EP=64 high-bs decode — r15 sign-flips from regression to win as bs grows (2026-06-28)

**MAJOR FINDING. The §16 +15% TPOT regression at EP=64 decode bs=512 was bs-specific. At
bs=8192 the sign FLIPS to r15 −12% TPOT FASTER, statistically rock-solid (t≈−90). The
"router-RL hurts decode" claim only holds at small-to-moderate bs at this EP scale.**

### Question

§16 showed r15 +15% TPOT slower at EP=64 decode bs=512 (statistically rock-solid). §12
serving-cost simulation used this as evidence that r15 is bad for production decode-heavy
workloads. But the prediction "below-knee r15 hurts, above-knee r15 wins" was falsified at
EP=8 prefill (§21). What does the bs sweep look like for EP=64 decode? Where (if anywhere)
does r15 start winning?

### Method

Submitted `highbs_decode_v2` (29564290) and `highbs_r15` (29566824). EP=64 decode, plen=256,
max_tokens=16, n=20 trials per cell. Bs ∈ {2048, 8192}. Used the env-var patch
(VLLM_MAX_NUM_BATCHED_TOKENS, VLLM_MAX_NUM_SEQS, VLLM_GPU_MEMORY_UTILIZATION) so the bench
respects the larger max_num_batched_tokens without choking on argparse.

The first run (29564290) timed out (3h walltime) after getting only pretrained bs=2048 and
bs=8192; the r15 cells were resubmitted standalone in 29566824 with 3.5h walltime.

### Results

| bs | cell | TTFT (ms) | e2e (ms) | TPOT (ms/tok) |
|---|---|---|---|---|
| 2048 | pretrained | 3776.1 | 9804.9 ± 473.8 | 401.92 |
| 2048 | r15 | 3660.5 | 9723.6 ± 401.7 | 404.21 |
| 2048 | **Δ** | **−3.06%** | **−0.83% (ns, t=−0.59)** | **+0.57%** |
| 8192 | pretrained | 146 598.1 | 166 272.4 ± 354.8 | 1311.62 |
| 8192 | r15 | 138 025.7 | 155 403.4 ± 404.8 | 1158.51 |
| 8192 | **Δ** | **−5.85%** | **−6.54% (t=−90.30, p<0.001)** | **−11.67%** |

### The bs-vs-TPOT picture at EP=64 decode

Combining with §16:

```
bs=512   (§16):              r15 +15.0% TPOT slower   (t ≈ 11.5)
bs=2048  (§23):              r15  +0.6% TPOT slower   (t ≈ 0.6, ns)
bs=8192  (§23):              r15 −11.7% TPOT FASTER   (t ≈ 90, highly sig)
```

**Monotonic sign flip as bs grows from 512 → 8192.** The +15% TPOT regression that anchored
the §12 "router-RL hurts decode" narrative is specific to small-to-moderate bs at high EP.
At deployment-realistic high-throughput serving bs (where you'd actually run a 235B model
at EP=64), r15 IS faster.

### Mechanism

Per-local-expert M at bs=8192 EP=64 = 8192·8/128 = **512**, far below the GEMM knee (M=4096
per §15). So the win is NOT GEMM-saturation. Candidate explanations:

1. **Comm-overhead amortization**: at small bs decode, per-step comm cost (TP AllReduce
   after MoE) is a fixed expense regardless of bs. r15's slightly slower per-rank compute
   loses relatively more time at small bs (small numerator) than at large bs (large
   numerator).
2. **Python/scheduler overhead amortization**: same logic — per-step scheduler cost is
   ~constant; at bs=8192 it amortizes over 8192 requests so its relative impact on TPOT
   is small.
3. **Routing-variance averaging within a step**: at bs=8192, the routing assignments
   within a single step average over 8192 tokens, smoothing out the variance. At bs=512
   the routing has more per-step shot noise. r15's "flatter mean routing" only pays off
   when there are enough tokens per step to realize the mean.
4. **KV cache and memory-bw effects**: at bs=8192 the workload is memory-bound on KV
   reads (not compute-bound). r15's routing produces different attention patterns that
   may be slightly more cache-friendly. Speculation.

None of these are testable from bench data alone. nsys at bs=8192 vs bs=512 would
discriminate. We don't have it.

### Implication for §12 serving-cost simulation

§12 used the bs=512 EP=64 +15% TPOT regression and concluded "r15 is +15% slower for any
output length at EP=64 decode bs=512." This was correctly stated — but extrapolating that
to "EP=64 is bad for r15 in production" was wrong. At bs=8192 EP=64 decode, r15 is
substantially faster (−11.67% TPOT, −6.5% e2e). Production-realistic high-throughput
serving at bs=8192+ is exactly the regime where r15 helps.

§12 break-even N* numbers at EP=64 bs=512 should not be extrapolated to higher bs.

### Implication for the project headline

The story is now:

- **EP=8 prefill, all M we can measure**: r15 +5-11% slower TTFT (§20, §21, §22 — solid)
- **EP=32 prefill bs=1-2**: r15 +5.5% slower TTFT (§22 — solid)
- **EP=64 decode bs=512**: r15 +15% slower TPOT (§16 — solid)
- **EP=64 decode bs=2048**: r15 within noise (§23)
- **EP=64 decode bs=8192**: r15 **−12% FASTER TPOT** (§23 — highly significant)

The original CP→inference-time question gets a regime-dependent answer:
- At low bs prefill (any EP): r15 hurts
- At low bs EP=64 decode (bs=512): r15 hurts
- At high bs EP=64 decode (bs=8192): r15 helps substantially
- The crossover for EP=64 decode is between bs=2048 and bs=8192

**For production high-throughput decode-heavy serving at EP=64 (the realistic deployment
target for 235B at scale), router-RL IS a win.** The narrow operating points where it
hurts (small bs at any regime, EP=8 prefill at any bs) are typically not the
revenue-relevant ones for large-model serving.

### Caveats

1. **bs=16384 was lost** in the timeout. Would tell us if the gap widens further with bs.
2. **n=20 trials per point.** The bs=8192 result is statistically firm regardless (CV<0.3%,
   t≈90) but bs=2048's smaller delta has wider CI.
3. **Only r15 vs pretrained** — r05/r60 not measured at high bs. The §6 surprise behavior
   was shared across all fine-tuned cells at low bs; we'd want to confirm the high-bs win
   shares that universality.
4. **The mechanism for the sign flip is unmeasured.** Hypothesizing comm/overhead
   amortization with bs, but nsys would be needed to confirm.
5. **Highbs_decode_v2 used chunked prefill behavior** during the prefill portion since
   bs×plen = 2M tokens exceeds max_num_batched_tokens. But the decode portion (steady
   state after TTFT) operates at step_tokens=bs (decode is 1 token/request per step,
   so 8192 tokens per step exactly fits in 8192 chunk limit — no chunking at decode).

### Files

- `cp_latency_results/ep64_highbs_v2_pretrained_235b_bs2048_29564290.json`
- `cp_latency_results/ep64_highbs_v2_pretrained_235b_bs8192_29564290.json`
- `cp_latency_results/ep64_highbs_v2_r15_cp4682_bs2048_29566824.json`
- `cp_latency_results/ep64_highbs_v2_r15_cp4682_bs8192_29566824.json`

