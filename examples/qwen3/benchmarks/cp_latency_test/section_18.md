
---

## §18 bs_sweep replication — earlier "r15 wins bs=2-4 by 12%" does NOT replicate (2026-06-21)

### What happened

The original bs_sweep (29332445, n=20 trials) showed r15 winning -12% TTFT at bs=4 EP=8 prefill.
This was the headline result that motivated the whole "above-knee r15 wins" narrative in §11-§12.

Replication via bs_sweep3 (29338297, n=30 trials, same workload):

| bs | bs_sweep (n=20) | bs_sweep3 (n=30) | std on each | reproducible? |
|---|---|---|---|---|
| 1  | +7.0% | (n/a, didn't include bs=1) | ~0.5 ms | yes — multiple confirmations elsewhere |
| 2  | -4.5% | **+12.7%** | ~14-20 ms | **NO — sign flipped** |
| 4  | -12.0% | **+17.2%** | ~13-20 ms | **NO — sign flipped** |
| 8  | +0.05% | +1.7% | ~14-21 ms | inconsistent — both near zero |
| 16 | (n/a) | +7.6% | ~8-13 ms | new data |
| 32 | (n/a) | **-6.2%** | **±2-4 ms (CV<0.02%)** | strong, clean win |

### Why it doesn't replicate at bs=2-4

Standard deviation on the bench measurement at bs=2-4 is ~13-20 ms, while the deltas being claimed
are 5-15 ms. The signal-to-noise ratio is ~1:1. Individual runs can flip sign by chance. To resolve
the actual sign at bs=2-4 would need n≥100 trials and probably multiple independent SLURM jobs to
average out run-to-run system-state variance.

The bs=4 "-12% r15 wins" claim that I built §11-§12 narrative on top of is **not statistically defensible**
from one n=20 measurement when the std is ±13-20 ms.

### What IS robust

- **bs=1 EP=8 prefill**: r15 +7-9% slower. Confirmed across multiple runs (bs_sweep, bs_sweep3,
  earlier campaign at 29156714, the §6 surprise-reproducer). Std is tight (~0.3-0.6 ms). Real signal.
- **bs=32 EP=8 prefill**: r15 -6.2% faster. Std ±2-4 ms on 24-second TTFT (CV<0.02%). Very strong signal.
  Per-local-expert M = 16384 = 4× asymptote — deeply FLOPs-bound regime. The user's CP→TTFT prediction
  cleanly holds here.
- The qualitative trend (r15 wins more as bs grows, after some bs threshold) is preserved.
- The crossover bs is **somewhere between 8 and 32**, but our data can't pin it down to ±4 because
  of the noise at bs=2-16.

### Implications

§11/§12 narrative "r15 sign-flips by bs=2, wins -12% by bs=4" is **overclaim from one noisy run**.
The conservative restatement is:
- At small bs (≤1, perhaps ≤4), r15 loses by 7-17% TTFT
- At extreme bs (≥32), r15 wins by ~6% TTFT
- In between (bs=2-16), r15's relative position is within the measurement noise envelope; cannot
  declare a sign with current data

The serving-cost simulation (§12) used the bs_sweep numbers for break-even N* calculation. Those
N* values for bs=2-4 should be considered HIGH-VARIANCE; the headline finding "for short outputs at
bs=2-4 r15 wins, switching to losing at N>10" depends on the unreliable bs=2-4 deltas.

What survives §12 cleanly:
- EP=64 decode bs=512: r15 +15% TPOT — solid (t≈11.5)
- bs=1 EP=8 prefill: r15 +7-9% TTFT — solid
- bs=32 EP=8 prefill: r15 -6% TTFT — solid (very tight CI)

The "r15 wins prefill at moderate bs, loses decode at all bs" 2D map is partially true (decode side
robust) and partially not (prefill mid-bs side is in noise).

### What to do

1. **Retract** the §11/§12 claim that r15 wins -12% at bs=4 prefill. Replace with "r15 may win at
   moderate bs but the data has too high a variance to declare sign at bs=2-16."
2. **Re-run** bs={2,4,8} with n≥100 trials across multiple SLURM submissions to nail down the sign.
3. **Note** that the GEMM-knee at M=4096 (§15) means bs=2 (M=1024, 69% of asymptote) and bs=4
   (M=2048, 85%) are NOT above-knee — they're in the rising portion of the throughput curve.
   bs=8 (M=4096, 94%, the knee itself) and bs=32 (deeply above) are clearer regimes.

### Files

- bs_sweep (older): `cp_latency_results/bs_sweep_29332445.json` (20 trials)
- bs_sweep3 (newer): `cp_latency_results/bs_sweep3_29338297.json` (30 trials)
- bs_sweep2: failed at r05 cell HF tokenizer error before producing output

### Caveats

- bs=32 TTFT of 24 seconds suggests memory pressure / KV-cache thrash / dispatch overhead at extreme
  bs. The clean r15 -6% win at that point may not generalize to better-conditioned regimes (e.g.
  smaller plen with similar M).
- The replication is single-run vs single-run. Multiple replications would establish the true
  underlying variance distribution.

