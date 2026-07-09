
---

## §22 EP=32 prefill multi-node — r15 regression replicates at higher EP (2026-06-28)

### What landed

After ~6 days in the polar4 priority queue, 3 of the 6 multi-node EP=32/EP=64 prefill jobs
finally dispatched and completed:

| Job | Config | State |
|---|---|---|
| 29329796 | EP=32 pretrained (4 nodes, plen=8192, bs=1,2, n=20) | COMPLETED |
| 29329804 | EP=32 r15 (4 nodes, same config) | COMPLETED |
| 29329814 | EP=64 r15 (8 nodes, same config) | COMPLETED |
| 29329798 | EP=64 pretrained (8 nodes) | **FAILED 7m42s** — resubmitted as 29564289 |
| 29329805 | EP=32 r05 (4 nodes) | FAILED |
| 29329815 | EP=64 r05 (8 nodes) | FAILED |

So we have a clean EP=32 pretrained-vs-r15 prefill comparison (the §9 closer this report
was originally written to provide).

### Results — EP=32 prefill, plen=8192, n=20 trials

| bs | pre TTFT (ms) | r15 TTFT (ms) | Δ % | pre e2e | r15 e2e |
|---|---|---|---|---|---|
| 1 | 74.16 (p95=74.65) | 78.23 (p95=79.34) | **+5.49%** | 272.96 ± 0.96 | 284.61 ± 1.88 |
| 2 | 115.88 (p95=145.19) | 120.05 (p95=149.49) | **+3.60%** | 339.50 ± 29.63 | 344.77 ± 34.93 |

### EP=64 prefill r15 only (no pretrained baseline)

| bs | r15 TTFT (ms) | r15 e2e |
|---|---|---|
| 1 | 86.52 | 459.59 ± 9.74 |
| 2 | 160.28 | 494.58 ± 40.26 |

Cannot compare cells until 29564289 (pretrained resubmit) lands.

### EP scaling of r15 TTFT at bs=1, plen=8192

```
EP=8  : 57.67 ms     (Phase A)
EP=32 : 78.23 ms     (this section)
EP=64 : 86.52 ms     (r15 only)
```

EP scaling of pretrained TTFT at bs=1, plen=8192:

```
EP=8  : 53.98 ms     (Phase A)
EP=32 : 74.16 ms     (this section)
EP=64 : ??           (pending 29564289 resubmit)
```

### Headline finding

**The r15 disadvantage is EP-invariant at small bs.** Across measurable single-forward EP=8
and EP=32 prefill at bs=1, r15 is +5-7% slower TTFT than pretrained:
- EP=8 bs=1: r15 +6.84% slower (Phase A, N=150)
- EP=32 bs=1: r15 +5.49% slower (this section, N=20)
- EP=32 bs=2: r15 +3.60% slower

This contradicts the "per-rank-load reduces with EP, so r15 should win MORE at higher EP"
prediction from §11. With EP scaling from 8 → 32 → 64, the per-rank-load reduction for r15
should grow (12% → 25% → 31% reductions in busiest-rank tokens per §11). If that translated
to time, r15 should hurt LESS as EP grows. **Instead the disadvantage is roughly flat at ~5-7%.**

The §21 finding (GEMM-tile-economics mechanism doesn't appear at the EP=8 knee) is reinforced:
r15's TTFT regression is **regime-stable** across both M (§21) and EP (§22). Whatever causes
the +5-7% is not GEMM saturation, not per-rank load, not EP scaling.

### Notes on absolute scaling with EP

Pretrained TTFT scales as EP grows: 54 → 74 → ? ms (8 → 32 → 64). This is the price of
more cross-rank comm at higher EP. r15 scales similarly: 58 → 78 → 87. **Both cells suffer
proportionally from higher EP — the relative gap stays near constant.**

This suggests:
- The CP→inference-time relationship continues to fail at higher EP prefill at small bs
- At small bs, prefill is dominated by something other than per-expert FFN — exactly what
  §13 (per-launch kernels equal across cells) and §21 (M-invariance) already showed

### Caveats

1. **n=20 trials** is moderate — Phase A's n=150 is more reliable but only EP=8.
2. **bs=1, bs=2 only.** Higher bs at EP=32 / EP=64 would need to fit memory; would need
   the env-var override patches.
3. **r05 cells failed at both EP=32 and EP=64** — only pretrained + r15 in this dataset.
4. **EP=64 pretrained still pending resubmit (29564289).** Without it, EP=64 cell comparison
   incomplete.
5. **§7 in the queue note**: the original 6 ord_rayEP_prefill jobs were submitted 2026-06-20
   and sat in polar4 priority queue for 5+ days before partial dispatch.

### Files

- `cp_latency_results/vllm_ep32_prefill_sweep_pretrained_235b_29329796.json`
- `cp_latency_results/vllm_ep32_prefill_sweep_r15_29329804.json`
- `cp_latency_results/vllm_ep64_prefill_sweep_r15_29329814.json`
- Resubmits: 29564289 (EP=64 pretrained prefill), 29564290 (high-bs decode v2 fixed for env vars)

### Status of the project

With §22, we now have replicated single-forward measurements across:

| Regime | r15 vs pretrained TTFT |
|---|---|
| EP=8 prefill bs=1 (M=512) | r15 +6.84% slower |
| EP=8 prefill bs=2-16 (chunked) | within noise |
| EP=8 prefill un-chunked M=2048 | r15 +11.42% slower |
| EP=8 prefill un-chunked M=4096 (knee) | r15 +7.18% slower |
| EP=8 prefill bs=32 chunked | r15 -6.23% (scheduler artifact) |
| **EP=32 prefill bs=1** | **r15 +5.49% slower** |
| **EP=32 prefill bs=2** | **r15 +3.60% slower** |

**Across all single-forward EP={8, 32} prefill regimes at small bs, r15 is consistently +3-11%
slower TTFT.** The mechanism is not identified but is EP-invariant and M-invariant. The
project's headline question — "does CP reduction → faster inference?" — answers NO at this
operating point on this hardware, across both EP scales and the M-range we can measure.

