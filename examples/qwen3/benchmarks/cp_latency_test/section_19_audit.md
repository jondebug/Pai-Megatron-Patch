
---

## §19 STATUS AUDIT — what's solid, what's retracted, what's misframed (2026-06-21)

This audit supersedes the §1 TL;DR (written 2026-06-16, before the trace-decomposition and
replication work in §13-§18). Read this section before relying on any earlier numeric claims.

### Solid findings (replicated or directly measured)

1. **bs=1 EP=8 prefill, r15 +7-9% slower TTFT.** Replicated across ≥4 runs (28837949,
   29156714, 29186556, bs_sweep_29332445, bs_sweep3_29338297). Std ~0.3-0.6 ms vs ~5 ms delta.
   The mechanism is NOT compile-cache (§10 falsified), NOT per-launch kernel time (§13 showed
   identical distributions), but is real and intrinsic to r15's router weights.

2. **bs=32 EP=8 prefill, r15 -6.2% faster TTFT.** Std ±2-4 ms on ~24 sec TTFT (CV<0.02%). bs=32
   gives per-local-expert M = 16384 = 4× the measured asymptote (knee = M=4096, §15). Deeply
   FLOPs-bound regime — exactly where CP-reduction *should* help, and does.

3. **GEMM saturation knee for Qwen3-235B FFN shape (K=4096, N=1536) is at M=4096.** Direct
   single-GPU microbench (§15). 90%-asymptote crossover at M=4096; 100%-asymptote at M=8192.
   This is 4× higher than the "M~1024 rule of thumb" we'd been using; multiple §11-§14 claims
   need this correction (now noted in §15).

4. **Training-vs-inference CP gap (§14).** r15's CP reduction is 43% at training shape but only
   36% at inference shape (identical total tokens). 7 percentage points of the trained metric
   don't transfer to inference.

5. **vLLM 0.20.2 uses NoDPEP at TP=N+dp=1 with --enable-expert-parallel.** Each rank fully
   owns n_experts/EP experts. **No token dispatch** between ranks; each rank sees full hidden
   state via TP replication, computes only its local experts, single AllReduce after MoE.
   Confirmed both in source (config.py:1184-1198) and in nsys traces (a2a_ms=0 for every rank).

6. **EP=64 decode bs=512: r15 +15% TPOT, t≈11.5.** Statistically rock-solid as a *measurement*.
   But see retractions below for what it means.

### Retracted / qualified

R1. **"r15 wins -12% TTFT at bs=4 EP=8 prefill" — RETRACTED.** Single n=20 run (bs_sweep,
    29332445) said -12%. n=30 replication (bs_sweep3, 29338297) said +17%. Std is ±13-20 ms
    on ~10 ms delta, so individual runs flip sign. Sign at bs=2-16 is in the noise envelope
    and we cannot declare it with current data. Reproducibility requires n≥100 trials over
    multiple SLURM jobs. (§18)

R2. **"§17 idle-expert mechanism explains the +15% EP=64 decode regression" — QUALIFIED.**
    §17 showed 5-7× more idle local experts in pretrained at the AVERAGE rank during prefill
    bs=1. But the *slowest rank* (which controls wall-clock) at prefill has both experts
    active in BOTH cells (§17 correction). The mechanism applies to light/middle ranks, not
    the wall-clock-bounding rank. The §16 +11% busiest-rank FFN delta requires another
    explanation, likely related to launch count or per-step routing patterns we can't see
    with the dump methodology.

R3. **"EP=64 decode bs=512 +15% TPOT IS evidence of CP→inference relationship" — RETRACTED.**
    Per-rank trace decomposition (§16): expert_FFN is only **5%** of per-rank time at this
    regime; comm_other is **88%**. The +15% TPOT can't physically come from MoE compute
    delta (which is only ±0.5% of total time on a +11% basis). It must come from how
    routing affects comm patterns — sync wait, NCCL algorithm transitions, or something in
    the un-instrumented Python/scheduler layer. **The EP=64 cross-node IB regime is comm-
    dominated and is the WRONG REGIME to test the MoE-compute-vs-CP hypothesis.**

R4. **"Routing dumps at seq_length=1 reflect real decode-time routing" — RETRACTED.** The
    cp_routing_dump captures gate output for tokens **without preceding context** (each
    "sequence" is 1 token, no KV history). Real decode-time routing is determined by gate
    input that has 256+ tokens of generation context flowing through it. The captured
    pattern at seq_length=1 doesn't match what vLLM actually sees during decode.

R5. **"Per-rank-load reduction at EP=64 of 31% means r15 should be 31% faster at EP=64" —
    QUALIFIED.** The 31% reduction is real (§11) but applies only in a FLOPs-bound regime
    on the busiest rank. We're not in that regime at EP=64 decode bs=512 (per-expert M=32,
    far below knee; comm dominates). Per-rank-load reduction would translate to time
    reduction at sufficiently high bs / sufficiently fast comm.

R6. **"Serving-cost break-even N* (§12)" — DEPENDS ON RETRACTED INPUTS.** The N* calculations
    for prefill bs=2-4 used the unreliable -12% measurement. The decode N* values use the
    +15% TPOT which is real but mechanism-misattributed. The qualitative shape (TTFT can win
    short outputs, TPOT loses long outputs) survives but specific N* numbers should not be
    cited.

### What's misframed in the project so far

M1. **The 8-node EP=64 setup is comm-bound, not MoE-FLOPs-bound.** Anything we measure on it
    is testing "how does router-RL affect comm patterns" not "how does router-RL affect MoE
    compute efficiency." For testing the CP→time relationship, single-node EP=8 (where comm
    is <1%) is the right regime.

M2. **The bs we've been sampling at prefill (bs=1-8) is below or near the GEMM knee.** At
    M=4096 knee, bs needs to be 8 (M=4096) at EP=8 to be at the knee, 16 (M=8192) to be
    above. bs=1-4 = below knee, where the "CP helps" effect is muted/inverted.

M3. **"Per-rank token count" was our optimization target for the analysis.** But the actual
    cost is `sum over active local experts of GEMM(M_expert)`, and at small bs this is
    dominated by the per-active-expert fixed cost (~35 µs floor below M=128). So "per-rank
    tokens" is the wrong feature; "per-rank active-expert count × per-expert M distribution"
    is closer.

M4. **The "+11% busiest-rank expert FFN at EP=64 decode" (§16) is real but explains only
    ~0.5% of the +15% TPOT regression.** The rest is in the un-instrumented Python /
    CUDA-graph / scheduler / comm-wait layers. We've been writing as if §16 + §17 closes
    the +15% mystery; they don't.

### Where to go next

Given the audit, the clean experiment plan is:

**Phase A — Tight EP=8 prefill statistics (single-node, comm <1%)**

Goal: pin down the sign and magnitude of r15-vs-pretrained TTFT delta at each bs, with
proper statistical power, in the regime where MoE compute matters and comm doesn't.

- 3 independent SLURM jobs × n=50 trials each (= 150 trials per cell-bs combo)
- bs = {1, 2, 4, 6, 8, 12, 16, 32}, plen=8192
- pretrained + r15 + r05 + r60 (4 cells; if HF tokenizer bug recurs, split per cell)
- Compute Welch t-test on each (cell, bs) pair
- Report only sign-significant results at α=0.01

Expected outcome: cleanly identifies the bs threshold for the sign flip, with proper CIs.

**Phase B — Per-rank FFN time vs measured CP at high bs (single-node EP=8 with traces)**

Goal: directly test "busiest-rank FFN time tracks busiest-rank token count" — the FLOPs-bound
prediction.

- For each (cell, bs) at EP=8 with per-rank torch.profiler traces
- Plot per-rank FFN time vs per-rank token count (measured from routing dump)
- Compute regression: T_rank = α + β × M_max_local + γ × n_active_local
- Test whether β (FLOPs-bound term) dominates above bs=8 and γ (overhead term) dominates below

**Phase C — Single-node decode at large bs to reach above-knee at decode regime**

Goal: see if CP-reduction helps decode latency in a regime where MoE compute matters.

- bs={2048, 4096} EP=8 (max single-node memory allows)
- Per-local-expert M = bs/16, so bs=4096 → M=256 (still below knee)
- This won't reach above-knee at decode with single-node EP=8
- Multi-node EP=64 at bs=16384+ would reach M=1024+ but multi-node IS the comm-dominated regime
- **NO clean way to test CP→decode-time with our hardware. Acknowledge this.**

**Phase D — Direct vLLM routing instrumentation (if mechanism verification matters)**

Goal: capture real per-step routing patterns during a vLLM decode bench. Modify
fused_moe layer to dump M_per_local_expert per step. Then correlate with per-step latency.

- 1-2 days of vLLM source modification + testing
- Would resolve the §17 idleness question definitively for decode

**Phase E — Skip the multi-node EP=64 prefill/decode jobs as evidence for CP→time**

The 6 ord_rayEP_prefill jobs currently in queue and ep64_highbs_decode + ep64_ultrabs_decode
ARE useful for documenting what happens at production-relevant deployment topologies, but
they should NOT be cited as evidence for or against the CP→inference-time hypothesis,
because the regime is comm-dominated.

### Bottom line for the original question

**Does CP reduction → faster inference?**

Honest answer at the audit-line:

- For above-knee per-expert M (bs ≥ knee-threshold per EP): **likely yes** — bs=32 EP=8
  prefill confirms r15 is -6% faster, the only fully-clean datapoint we have above-knee.
- For below-knee per-expert M (bs=1 EP=8 prefill; any EP decode in our setups): **the
  mechanism flips and r15 is slower** by 7-9% (small-bs prefill), but this isn't a robust
  serving-relevant regime.
- For comm-dominated regimes (EP=64 cross-node): **CP reduction's effect on MoE compute is
  drowned out by comm dynamics**; whatever delta we observe is not attributable to the
  MoE mechanism we set out to test.

The "headline finding" that originally motivated this project — does router-RL training make
inference faster — is **not yet cleanly answered for production-relevant serving**, because
the production-relevant serving regimes are either (a) below the GEMM knee where CP doesn't
help, or (b) comm-dominated where the question isn't well-posed in our setup.

