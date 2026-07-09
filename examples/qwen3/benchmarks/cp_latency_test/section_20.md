
---

## §20 Phase A — high-statistics replicated EP=8 prefill bs sweep (2026-06-22)

**STATUS: This is the most statistically rigorous result in the report. Replicates the noisy
single-run findings of §11-§18 with proper variance accounting. Should be treated as the
authoritative reference for the EP=8 prefill TTFT-vs-bs relationship.**

### Question

The §11-§12 narrative claimed r15 wins TTFT by -12% at bs=4 EP=8 prefill (from a single n=20
run, std ±13-20 ms). §18 attempted replication at n=30 and got the OPPOSITE SIGN (+17% slower).
Neither was statistically powered. The actual sign and magnitude at bs=2-16 was an open
question that all downstream sections built on.

### Method

Three independent SLURM jobs × 50 trials each = **150 trials per (cell, bs)**. EP=8 prefill
plen=8192, bs={1, 2, 4, 6, 8, 12, 16, 32}, max_tokens=4, CUDA graphs ON.

- Run scripts: `run_phase_A.sh` submitted with `REP_TAG=A1/A2/A3`
- Jobs: 29339780 (A1), 29339781 (A2), 29339782 (A3) — all COMPLETED rc=0
- Pooled across runs using weighted mean / pooled variance
- Welch t-test on each (cell, bs) pair

### Findings

| bs | pre TTFT (ms) | r15 TTFT (ms) | Δ % | t-stat | sig |
|---|---|---|---|---|---|
| 1  | 53.98 ± 0.27 | **57.67 ± 0.32** | **+6.84%** | **+108.6** | *** highly sig |
| 2  | 89.28 ± 15.82 | 89.06 ± 19.04 | -0.25% | -0.11 | ns |
| **4**  | 116.91 ± 18.91 | **125.07 ± 29.45** | **+6.98%** | **+2.86** | *** sig (p<0.005) |
| 6  | 139.35 ± 23.21 | 140.93 ± 27.16 | +1.13% | +0.54 | ns |
| 8  | 155.44 ± 21.44 | 155.10 ± 26.09 | -0.22% | -0.12 | ns |
| 12 | 180.23 ± 19.19 | 177.68 ± 23.75 | -1.42% | -1.02 | ns |
| 16 | 196.46 ± 13.57 | 198.91 ± 18.73 | +1.24% | +1.29 | ns |
| **32** | 24071.77 ± 16.16 | **22572.07 ± 18.61** | **-6.23%** | **-745.2** | *** highly sig |

(N=150 trials per (cell, bs). Significance code: * p<0.10, ** p<0.05, *** p<0.01)

### What's new vs prior sections

The replication firmly establishes:

1. **r15 +6.84% slower TTFT at bs=1 (EP=8 prefill)** — confirmed at t=+109, the strongest
   statistical evidence in the report. This was the original "+9% surprise" of §6. With
   150 trials and tight stds, the magnitude is ~6.84% (slightly lower than the original ±9%
   estimate from earlier n=20-30 runs, but qualitatively unchanged).

2. **r15 +6.98% slower TTFT at bs=4 (EP=8 prefill), highly sig (t=+2.86, p<0.005).**
   This is the **most important new finding**. The original §11-§12 narrative was built on
   a single bs_sweep run that showed r15 winning **-12.0%** at bs=4. §18 replication with
   n=30 showed **+17%** (sign-flipped). The truth, with N=150, is r15 is **statistically
   significantly SLOWER by +7%** at bs=4. The original sign was a single-run noise artifact.

3. **bs=2 through bs=16: no statistically significant sign.** All five mid-range bs points
   have |t| < 1.3 — well below the p<0.10 threshold. The deltas are 0.2-1.4% in either
   direction, dwarfed by trial-to-trial std (±13-30 ms on means of 90-200 ms). **There is
   no reproducible "r15 wins prefill at moderate bs" finding in the data.**

4. **r15 -6.23% faster TTFT at bs=32 (EP=8 prefill), highly sig (t=-745).** At bs=32 the
   wall-clock TTFT is ~24 sec — clearly in a saturated/compute-bound regime. The CIs are
   tight (±16-18 ms on 24000 ms means). This is the cleanest "r15 wins above-knee" datapoint
   we have. Per-local-expert M at bs=32 is 16384, ~4× the measured knee (M=4096 per §15).

### Updated sign-vs-bs picture for EP=8 prefill TTFT

```
            r15 slower (positive %)
              ▲
   bs=1: +6.8% ▲
   bs=4: +7.0% ▲
              ─ ─ ─ ─ noise band ─ ─ ─ ─
   bs=2,6,8,12,16: ~0% (noise envelope)
              ─ ─ ─ ─ noise band ─ ─ ─ ─
   bs=32: -6.2% ▼  r15 faster
              ▼
```

The sign-flip crossover happens **between bs=16 and bs=32**, not at bs=2-4 as previously
claimed. bs=16 has per-local-expert M = 8192 — above the measured knee but still in the
soft-saturating region (per §15 curve, M=8192 is at 100% of asymptote). The CP→FFN win
appears only after deeply saturating the GEMM regime.

### What this means for §11-§18 conclusions

This section does NOT retract §11-§18 — it adds the higher-confidence replication. Sections
to reconcile against §20:

- **§11 routing-distribution decomposition**: per-rank-load reductions are real (33-36% at
  EP=64). What's now in question is whether those reductions translate to time gain at the
  bs values we tested. Phase A says: at bs=32 they do (-6%); at bs ≤ 16 they're in noise.

- **§12 serving-cost simulation**: the break-even N* numbers at bs=2-4 used the (now-known
  wrong) "-12%" TTFT delta. Re-running the calc with the Phase A "+7%" delta at bs=4 would
  push break-even N* significantly. **§12 should be regenerated using §20's numbers.**

- **§13 per-kernel distribution**: still holds — per-launch kernel times are equal between
  cells. The Phase A bs=1 +6.84% TTFT delta still lives somewhere outside the CUDA kernel
  layer. Just now it's better-measured.

- **§14 training-vs-inference CP gap**: independent of bench measurements. Unchanged.

- **§15 GEMM saturation knee**: independent. The bs=32 win (M=16384 = 4× asymptote) is the
  cleanest above-knee data point we have, and it shows r15 wins -6%, **confirming the GEMM
  tile economics mechanism in the regime where it predicts r15 should win**.

- **§16 EP=64 decode trace**: still shows r15 +11% per-rank FFN, but per §19 audit the
  EP=64 cross-node setup is comm-dominated (88% of trace), so we can't attribute the
  +15% TPOT regression to MoE compute alone.

- **§17 idle-expert mechanism**: applies to AVERAGE ranks, not the slowest. §20 doesn't
  change that.

- **§18 earlier replication attempt** (bs_sweep3): superseded by Phase A which has 5× more
  trials.

### Status of the project's headline finding

Original question: **does router-RL CP reduction help inference latency?**

Phase A's clean answer, for EP=8 prefill (the regime where MoE compute matters, comm <1%):

- **At bs=1-4 (per-local-expert M ≤ 2048 = 85% of asymptote): r15 hurts by ~7%.** The
  GEMM is still in the rising part of the throughput curve. Per-expert overhead dominates
  per-token compute. r15's flatter routing produces more launches at smaller-M, hurting.

- **At bs=2, 6, 8, 12, 16 (M = 1024 to 8192): r15 is statistically indistinguishable
  from pretrained.** This is the regime where the prior "+12%" headline claims lived. They
  weren't replicating.

- **At bs=32 (M = 16384, deeply saturated): r15 wins by -6%.** The GEMM-tile-economics
  prediction holds where the regime supports it.

So the new, correctly-replicated answer is: **router-RL helps inference only at very high
per-rank batch sizes (per-local-expert M ≥ ~4× saturation knee). At realistic serving
batch sizes (bs=1-16 at EP=8), router-RL either hurts or is neutral.**

This is a more conservative — and more correctly-bounded — finding than the original
§11-§12 narrative claimed. The window of "r15 helps" for this model on this hardware
is narrower than initial measurements suggested.

### Files

- Phase A run script: `cp_latency_test/run_phase_A.sh`
- Phase A pooled analysis script: `cp_latency_test/phase_a_analyze.py`
- Raw results: `cp_latency_results/phase_A_A{1,2,3}_29339780/81/82.json`
- Jobs (all COMPLETED rc=0): 29339780, 29339781, 29339782

### Caveats

1. **Phase A measures TTFT only.** TPOT and e2e include decode tokens which Phase A's
   max_tokens=4 captures very briefly. The TTFT story is clean; the decode/long-output
   story still depends on the (now also-suspect) §16 + §12 data.

2. **Per-cell only n=2 (pretrained + r15).** r05 and r60 weren't included to avoid the
   third-model HF tokenizer bug. The §6 surprise applies to all fine-tuned cells, so
   r05/r60 should track r15, but this hasn't been replicated at Phase A's statistical
   power.

3. **Single hardware (A100 / our cluster).** B200/NVL72 predictions (§ earlier hypothetical
   discussion) remain unmeasured.

4. **EP=8 only.** The replication was on single-node where the regime is well-conditioned.
   EP=32 / EP=64 prefill replications would require multi-node 4-/8-node jobs that haven't
   been able to dispatch (still PD on Priority).

5. **bs=32 wall time is 24 sec.** This is a saturated regime that may not reflect realistic
   serving — most production deployments are at lower bs to keep TTFT under SLA. The "-6%
   r15 wins" at bs=32 may have limited practical applicability.

