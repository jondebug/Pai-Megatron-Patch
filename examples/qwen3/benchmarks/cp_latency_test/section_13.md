
---

## §13 Per-kernel-launch distribution from nsys — the +9% is NOT a kernel-time effect (2026-06-21)

### Question

Where does the +7-9% TTFT regression for r15 at EP=8 bs=1 prefill physically come from?
The bench-level measurement is firm (n=30, std<0.5 ms, delta ~5 ms). At the kernel level it has to manifest
somewhere. We pulled per-launch duration distributions from the EP=8 prefill nsys traces (pretrained,
r15, r05) and compared at p50 / mean / p90 / p99 / max for the dominant kernels.

### Method

Parsed `CUPTI_ACTIVITY_KIND_KERNEL` from each `.sqlite` nsys export. For each kernel class
(fused_moe_kernel, AllReduce, ampere GEMM, flash attention, topkGating, act_and_mul) computed
the full per-launch distribution rather than just the mean. (Means hide tails.)

### Findings

Distributions are essentially identical between cells at every percentile up to p99:

| kernel | percentile | pre (µs) | r15 (µs) | Δ% |
|---|---|---|---|---|
| fused_moe_kernel | p50 | 18.34 | 18.30 | -0.2% |
| fused_moe_kernel | mean | 56.0 | 53.6 | -4.3% |
| fused_moe_kernel | p90 | 134.0 | 133.1 | -0.7% |
| fused_moe_kernel | p99 | 577.4 | 576.9 | -0.1% |
| fused_moe_kernel | max | 4282 | 4677 | +9.2% |
| ampere s16816gemm 128x128 | p50/mean/p90/p99 | 18.9/19.3/23.8/68.7 | 18.9/19.2/23.8/67.5 | all <±2% |
| ampere s16816gemm 128x64 | p50/mean/p90/p99 | 14.2/18.0/34.7/35.7 | 14.3/18.0/34.4/35.8 | all <±1% |
| topkGating | p50/mean/p90/p99 | 7.3/7.5/7.7/18.4 | 7.3/7.5/7.8/18.0 | all <±2% |
| act_and_mul | p50/mean/p90/p99 | 11.4/16.8/23.2/354.9 | 11.2/16.0/22.9/349.5 | all <±5% |

**Conclusion at the per-kernel level: there is no slowdown in r15.** Median fused_moe call duration is
identical to within 0.2%, GEMM kernels identical to within 2%, attention identical (in fact r15 is faster
at p99 — likely artifact of different stall patterns).

### Launch counts differ slightly

The other axis is *how many* kernels fire:
- pretrained: 130848 `fused_moe_kernel` launches
- r15: 134002 launches (**+2.4% more**)
- r05: 134002 launches (+2.4% more)

This is r15's flatter routing forcing slightly more expert-active fused_moe invocations across the run.
With identical per-call cost, that accounts for +2.4% on the MoE kernel budget — but actual MoE time
*decreases* (-1.4%) because the slightly faster mean per launch (-4.3%) nearly cancels the count
increase. **Total fused_moe time is essentially equal between cells.**

### So where is the +9%?

Three possible homes for the missing time:

1. **AllReduce sync-wait artifact.** AllReduce per-call distribution is the *one* place with a huge
   difference — r15 mean 18166 µs, pretrained 2463 µs. But this is dominated by tail outliers:
   r15's max single AllReduce is **2.6 seconds**, pretrained's max is 351 ms. The p50 is 645 µs for
   *all three cells* — equal to within 1%. So the AR-kernel itself does the same work in the same
   time at the median; the difference is that ONE side hits a long sync-wait. This is **not real
   communication time — it's barrier wait time**, which gets billed to whatever kernel is on the
   GPU when the other ranks haven't arrived.

2. **Driver / scheduler / Python-overhead time outside CUDA kernels.** vLLM's per-step Python
   overhead (forward dispatch, request scheduling, sampling) doesn't show up in the CUDA kernel
   table at all. The +2.4% additional fused_moe launches plus their associated dispatch/Python work
   could account for several percent of step time.

3. **CUDA-graph capture/replay overhead.** With CUDA graphs ON, the captured graph is replayed each
   step. If r15's routing produces graphs that have to re-capture more often (e.g., dynamic expert
   selection paths that flip), that's invisible to per-kernel stats but visible to step latency.
   We have no direct measurement of replay vs re-capture rates.

### Conclusion

**The +7-9% TTFT regression is not a kernel-time effect.** Per-launch durations for every kernel
class are equal between cells. The bench-level measurement is real, so the time is being consumed
somewhere — most likely:
- Driver/Python overhead from the +2.4% additional kernel launches
- Or AllReduce sync-wait redistribution (r15's rank-time variance pattern lands the wait inside the
  AR kernel rather than between kernels)
- Or CUDA-graph re-capture/dispatch overhead

This is precisely the kind of effect that nsys at EP=8 *cannot* resolve — the time exists between
or above the kernels, not within them. The same analysis at the EP=64 decode trace (queued, job
29336222) will tell us whether the +15% TPOT lives inside kernels or between them — that will be
much more diagnostic because the AllReduce there spans 8 nodes and any wait-pattern delta would be
proportionally larger.

### Files

- Analysis: `nsys_kernel_dist.py`
- Results JSON: `cp_latency_results/nsys_moe_kernel_distribution.json`
- Source nsys: `nsys_ep8_prefill_{pretrained_235b,r15_cp4682,r05_iter2500_cp}_*.sqlite`

### Caveat

These traces are from **mixed-workload bench runs** (full sweep + profile cell), not isolated bs=1
prefill. Launch counts therefore include warmup + multiple bs configurations across the whole nsys
session. The per-launch *distribution* is still valid because each launch is its own measured event;
but it's not a fair "same workload" count comparison. To get the count number right we'd need an
isolated bs=1-only nsys run. The distribution-level finding (per-call costs are equal) does not
depend on count and stands.

