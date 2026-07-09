
---

## §11 Routing-distribution decomposition — what CP misses (2026-06-21)

### Question

Does the canonical CP metric (busiest-expert token count) accurately predict per-rank wall-clock cost
at the EP sizes we actually deploy with? Specifically: at EP=64 the expert level CP→per-rank mapping
should compress (2 experts/rank), but if multiple effects beyond per-rank-FLOPs dominate, CP-based
optimization may overfit to the wrong objective.

### Method

Used `cp_routing_dump.py` output (`routing_{pre,r15}_bs{1,2,4}_*.json`) which captures
per-layer × per-expert token counts at plen=8192. For each (cell, bs):
- Computed expert-level distribution stats: M_max, M_p99, mean, p50, n_active, n_above_knee, Shannon entropy
- For each EP ∈ {8,16,32,64} with linear placement, aggregated experts → per-rank token counts
- Per-rank metrics: busiest-rank M, rank imbalance ratio (= rank_M_max / rank_M_mean)

### Findings

**Expert distribution (per layer, averaged over 94 layers):**

| cell | bs | M_max | M_mean | n_active | n_above_knee(1024) | entropy |
|---|---|---|---|---|---|---|
| pre | 1 | 2879 | 512 | 125.0 | **20.4** | 6.141 |
| r15 | 1 | 1845 | 512 | 127.5 | **9.8**  | 6.647 |
| pre | 2 | 5752 | 1024 | 127.9 | 46.9 | 6.207 |
| r15 | 2 | 3691 | 1024 | 127.9 | 56.7 | 6.668 |
| pre | 4 | 11056 | 2048 | 127.9 | **68.5** | 6.250 |
| r15 | 4 | 7243 | 2048 | 127.9 | **96.8** | 6.687 |

**Key inversion:** at bs=1, pretrained has 20.4 experts above the GEMM saturation knee while r15 has only 9.8 — r15's flatter routing puts MORE experts in the small-M tile-inefficient regime. At bs=4, the relationship flips: r15 has 96.8 experts above-knee vs pretrained's 68.5. **The sign of N(M > knee) flips between bs=1 and bs=4 — exactly tracking the observed sign-flip in TTFT delta (+7% at bs=1 → -12% at bs=4).**

**Per-rank busiest-rank load (linear placement):**

| bs | EP | pre rank_M_max | r15 rank_M_max | Δ % | pre imbalance | r15 imbalance |
|---|---|---|---|---|---|---|
| 1 | 8  | 11794 | 10337 | **-12.4%** | 1.44 | 1.26 |
| 1 | 16 |  7522 |  6083 | -19.1% | 1.84 | 1.49 |
| 1 | 32 |  5037 |  3756 | -25.4% | 2.46 | 1.83 |
| 1 | 64 |  3620 |  2509 | **-30.7%** | 3.54 | 2.45 |
| 4 | 8  | 46399 | 40925 | -11.8% | 1.42 | 1.25 |
| 4 | 64 | 13736 |  9689 | **-29.5%** | 3.35 | 2.37 |

**The per-rank-load reduction r15 delivers grows monotonically with EP** (12% at EP=8, 31% at EP=64).
If serving wall-clock was simply max-rank-FLOPs, r15 should help **more** at higher EP. But the measured
EP=64 decode bs=512 shows r15 *hurts* by +15% TPOT. So **per-rank token count is not the binding constraint
at EP=64 decode** — supports the earlier "below-knee" / dispatch-overhead-dominant story.

### Conclusion

CP-as-optimization-target is a *training* metric, not an *inference* metric.
- At inference scale, what matters is the **per-rank, per-expert work-distribution geometry**, gated by the GEMM saturation knee.
- For prefill above the knee at moderate bs, r15's reduction in per-rank load translates to TTFT win.
- For decode and small-bs prefill below the knee, the reduction in per-rank load is dwarfed by the cost of spreading work across more, less-efficiently-saturated expert GEMMs.

Provenance: `routing_{pre,r15}_bs{1,2,4}_29334887/29335689.json`, analysis at `routing_distribution_analysis.py`,
results JSON at `cp_latency_results/routing_distribution_analysis.json`.

Caveat: per-rank linear placement assumed. vLLM also supports round-robin placement; the relative
per-rank-load ratio between cells is invariant to placement strategy, so the qualitative conclusion holds.

---

## §12 Serving-cost simulation — break-even output-length per regime (2026-06-21)

### Question

Even when r15 wins TTFT (e.g. EP=8 bs=4 prefill: −12% TTFT) it loses TPOT (e.g. EP=8 bs=4: +2.35 ms/tok).
For a serving workload that produces N output tokens per request:
```
total_latency(N) = TTFT + (N - 1) * TPOT
```
Where does each regime stand at realistic N values?

### Method

For every (regime, EP, plen, bs) where we have measured TTFT and TPOT for both cells, computed:
- ΔTTFT (r15 − pre), ΔTPOT (r15 − pre)
- Break-even output length N*: total_latency_r15(N*) = total_latency_pre(N*) → N* = 1 − ΔTTFT / ΔTPOT
- Total-latency percent delta at N ∈ {1, 10, 50, 100, 500, 1000}

### Findings

**Total-latency r15 vs pretrained, percent difference at fixed N output tokens:**

| regime | EP | plen | bs | N=1 | N=10 | N=50 | N=100 | N=500 | N=1000 |
|---|---|---|---|---|---|---|---|---|---|
| **prefill** | 8 | 8192 | **4** | **-12.0%** | +1.3% | +9.0% | +10.4% | +11.7% | +11.9% |
| prefill | 8 | 8192 | 2 | -4.5% | +7.8% | +12.9% | +13.7% | +14.5% | +14.6% |
| prefill | 8 | 8192 | 8 | 0.0% | -1.3% | -2.1% | -2.3% | -2.4% | -2.4% |
| prefill | 8 | 8192 | 1 | +7.0% | +2.1% | +0.7% | +0.5% | +0.3% | +0.3% |
| decode | 8 | 256 | 1024 | -6.3% | -5.1% | -2.2% | -0.8% | +1.3% | +1.7% |
| decode | 8 | 256 | 512 | -3.1% | +0.5% | +1.7% | +1.9% | +2.0% | +2.0% |
| decode | 32 | 256 | 64 | +0.7% | +0.7% | +0.7% | +0.7% | +0.7% | +0.7% |
| decode | 32 | 256 | 512 | -1.3% | -0.1% | +0.2% | +0.3% | +0.3% | +0.3% |
| decode | 64 | 256 | 64 | -0.1% | +3.7% | +4.4% | +4.5% | +4.6% | +4.6% |
| **decode** | **64** | 256 | **512** | **+4.2%** | **+13.0%** | **+14.7%** | **+15.0%** | **+15.2%** | **+15.2%** |

**Key observations:**

1. **r15's TTFT wins are mostly for N≤10**, then erased and inverted by TPOT regressions.
2. **At EP=8 prefill bs=4 (where r15 wins TTFT by 12%), the break-even output length is N≈8 tokens.** A typical chat completion of 50 tokens already pays a +9% penalty for using r15.
3. **At EP=64 decode bs=512 (the standout deployment regime), r15 loses across all N.** From +4% at N=1 to +15% at N=1000. There is no output length where r15 is competitive in this regime.
4. **Only configuration where r15 is consistently better:** EP=8 prefill bs=8, by ~2% across all N. Marginal and within noise envelope.

**Break-even N* by regime (where r15 catches up to pretrained):**

| regime | EP | bs | N* | interpretation |
|---|---|---|---|---|
| prefill | 8 | 1 | N/A (r15 always slower) | small-bs prefill: r15 strictly worse |
| prefill | 8 | 2 | 3 | r15 wins only for ≤2 output tokens |
| prefill | 8 | 4 | 8 | r15 wins only for ≤7 output tokens |
| prefill | 8 | 8 | 1 | r15 wins for any output |
| decode | 8 | 1024 | 157 | r15 wins for output ≤157 tokens |
| decode | 32 | 512 | 13 | r15 wins for output ≤12 tokens |
| decode | 64 | 64 | 1 | r15 wins only for N<1 (i.e. never) |
| decode | 64 | 512 | N/A (r15 always slower) | strict loss |

### Conclusion

**For realistic serving workloads (chat: N~50-200, code: N~500-2000), router-RL fine-tuning *hurts* inference latency in nearly every regime we measured.** The single regime where r15 is a clear win for short-output serving is EP=8 prefill at bs=2-4 with N≤7 output tokens — essentially a "first-token-only" or classification serving pattern, not generative inference.

At the scaled deployment regimes (EP=32, EP=64) where router-RL's training benefits would in principle pay off most, the inference penalty is largest:
- EP=32 decode: r15 ≈ +0.3-0.7% across all N (essentially neutral)
- EP=64 decode bs=512: r15 +15.2% at N=1000 (substantial regression)

The +15% finding is the practical headline: **if you deploy this model at 64-GPU EP for high-throughput serving, router-RL costs you 15% TPOT** — equivalent to 15% lower tokens/sec/replica, 15% lower revenue per GPU-hour at that operating point.

### Caveats

1. **Statistical strength varies.** EP=64 bs=512 delta is t≈11.5 (rock-solid). EP=8 decode bs=1024 delta is single-trial-data and n=4-8 trials at older runs — wider CI than stated percent suggests.
2. **TPOT is computed as (e2e - TTFT)/(max_tok - 1).** This conflates ramp-up effects in the first few decode steps with steady-state TPOT.
3. **vLLM continuous batching** may make "bs=N steady-state" different from "bs=N first-step." Bench harness is one-shot, not continuous-batch.
4. **Only one fine-tuned cell (r15) extended through all regimes.** Other cells (r05, r60) confirmed in prefill EP=8; remaining EP/regimes not measured for them.
5. **The EP=8 decode bs=1024 case has very low statistical confidence** — only one run; the N*=157 should be re-confirmed.

Provenance: `serving_cost_sim.py`, all source `vllm_ep*_sweep_*.json` files referenced.

---
