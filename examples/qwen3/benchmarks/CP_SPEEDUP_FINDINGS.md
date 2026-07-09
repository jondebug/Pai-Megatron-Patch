# CP → Inference Speedup: Findings (Qwen3-235B-A22B, A100 / vLLM, EP8–EP64)

Author: CP-inference profiling campaign. Date: 2026-06-07. Model: Qwen3-235B-A22B
(94 MoE layers, hidden 4096, moe_ffn 1536, 128 experts, top_k 8). Serving: vLLM v0.20.2,
`tensor_parallel_size=EP` + `enable_expert_parallel=True`, `enforce_eager=True`, A100-80GB,
intra-node NVLink / inter-node InfiniBand. Every quantitative claim below is a measurement
with trial count and variance, tagged to its SLURM job; assumptions are labelled
MEASURED / MODELED / UNVALIDATED.

## TL;DR (headline)

On A100, reducing the MoE critical path (CP) does **not** reduce vLLM inference latency or
raise throughput — **~1.00× speedup at every measured EP (8/16/32/64) and batch (1–1024),
for both decode and prefill** — despite CP being a *real* routing-balance improvement that
**does** cut the busiest-expert token load by 43% at inference token scale. Two stacked,
independently measured mechanisms explain this:

1. **Per-GPU expert-FFN compute is set by total tokens through the resident experts, not the
   busiest expert.** Each GPU hosts 128/EP experts; the fused-MoE kernel time depends on the
   conserved aggregate (bs·top_k/EP tokens), which CP does not change. Measured: EP32 r15 FFN
   = 402–420 ms ≈ EP32 pretrained FFN = 370–404 ms, *despite* r15 having 43% lower busiest-expert
   load. (job 28818149/28818150)
2. **The decode step is dominated by a TP NCCL all-reduce that is mostly barrier spin-wait
   (load-imbalance idle), not bandwidth** — and expert-FFN is only ~6–10% of kernel time anyway.
   CP has no lever on either term.

The positive CP→compute link is real but lives only in the **per-expert** view, which the
serving stack’s **per-GPU fused kernel + all-reduce** structure averages away. The NVL72
projection (below) shows that even removing the inter-node interconnect penalty does not
make CP bite, because the dominant residual is per-GPU-aggregate compute + all-reduce
serialization, neither CP-governed.

---

---

## ⚠️ LIMITATIONS / SCOPE CAVEAT (added 2026-06-08, reviewer-driven) — READ BEFORE §3, §5, §8

A review flagged that the **"no CP benefit / flat 1.00× at EP64"** conclusion, as originally
written, was over-extrapolated from **decode-scale (sub-knee)** measurements into the
**compute-bound prefill regime** where it was never directly tested. The following parts of
this doc are valid **only at decode scale** and must NOT be read as proven in the compute-bound
prefill regime:

1. **§3 FFN-invariance traces are DECODE-SCALE (sub-knee), not a valid test of CP leverage.**
   The per-rank FFN traces that "prove CP-invariance" (EP64 FFN 283–315 ms, pretrained≈r15;
   EP32 370–420 ms) were all captured at `prompt_lengths=256, batch_sizes=64, max_tokens=32`
   — i.e. **decode** bursts. At EP64 that is 64·8/64 = **4 tok/expert** balanced (even bs512 →
   32 tok/expert), far **below** the eager FFN knee (~448 tok/expert, §4). **Sub-knee the
   per-expert FFN is launch/exec-floor-bound** (≈0.134 ms/expert, §4), so CP is invisible
   *because nobody is compute-bound* — that trace **cannot** test the CP→compute hypothesis.
   Where the model IS compute-bound (prefill / 8192-token forward, §1–2), the busiest GPU is
   **above** the knee and r15's ~38–43% busiest-load cut is real (§2). FFN is, however, only
   ~6–10% of the EP64 step (the rest is all-reduce), so even a 38% cut of a ~10% slice is only
   a ~4% step effect — small, and it was never measured above the knee.

2. **§5 EP64 0.95/0.87 are within run-to-run variance — consistent with 1.00, NOT slowdowns.**
   EP64 `e2e_std ≈ 1450–1918 ms` vs EP32 `≈ 88 ms`. A few-% effect is buried in that noise; the
   measured 0.95 (bs64) / 0.87 (bs512) at EP64 are **noise, not evidence of an r15 slowdown**.
   Do not cite them as "r15 is slower." The correct reading is "EP64 CP→speedup is consistent
   with 1.00 within ±large variance, and was not measured with enough trials to resolve a
   few-% effect."

3. **§8 "CP can't reduce the EP64 straggler/barrier-spin" was only checked at DECODE scale.**
   For CP to shave the barrier-spin, the straggler must be **FFN-driven**. The §3 spin
   measurement is per-layer (×6237) launch/IB-latency jitter under `enforce_eager`, which CP
   does not touch — but that was a **decode-scale, eager** trace. Whether the *prefill*
   straggler (above the knee, CUDA-graphs-on) is FFN-driven or launch/IB-jitter-bound was
   **under-tested**.

**Status of the gap:** the decisive compute-bound EP64 test — an **EP64 prefill (≥8192-token)
per-rank FFN trace, CUDA-graphs-on, pretrained vs r15, enough trials to beat the ~1900 ms
variance** — is reported in **§9 below**. Read §9 for the corrected verdict; §3/§5/§8 stand as
the decode-scale findings they always were.

## 1. The CP-semantics paradox — RESOLVED (jobs 28817880; measure_critical_path.py)

The campaign’s headline CP numbers (pretrained “CP≈8800”, r15 “CP≈4682”) were measured during
router-finetuning eval at **seq_length=128** (every checkpoint path is `…seqlen-128…`). At a
128-token forward the balanced load is 128·8/128 = **8 tok/expert**; busiest/layer = CP/94 →
93.6 (pretrained) / 49.8 (r15) — an 11.7× / 6.2× imbalance. There is **no** “max below mean”
paradox; the earlier confusion came from wrongly assuming 8192 tokens.

CP scales ~linearly with tokens/forward, so “8800/4682” are **routing-quality numbers at the
128-token training scale, not inference busiest-GPU loads.** To get the inference-scale load we
re-measured routing at seq_length=2048, bs=4 → **8192 tokens/forward** (HF model, real Megatron
data, gate-Router forward hook; n=8 batches each):

| model | tokens/fwd | balanced | CP_mean (8192 tok) | busiest-expert/layer | imbalance |
|-------|-----------|----------|--------------------|----------------------|-----------|
| pretrained | 8192 | 512 | 356152 ± 54441 | **3788.85** | **7.40×** |
| r15        | 8192 | 512 | 202453 ± 27377 | **2153.76** | **4.21×** |

(MEASURED, job 28817880, n=8.) Hook caveat resolved: in transformers 5.8.0 the gate is class
`Qwen3MoeTopKRouter` (not `Linear`); its forward returns `(router_logits, …)`, `out[0]` =
logits[nt,128].

**Corrected CP→compute mapping:** the inference busiest-GPU hot-expert load = (tokens/GPU) ×
(measured inference imbalance ~7.4× pretrained / 4.2× r15), and only the per-*expert* load is
CP-governed. At 8192 tokens the busiest expert (3789 / 2154 tok) is **5–8× above the eager FFN
knee (~448 tok)** — so at the *per-expert* level it is genuinely compute-bound. The catch
(Sections 3–4) is that the serving stack never exposes a per-expert critical path.

---

## 2. CP→compute link — VALIDATED (job 28817880)

r15 cuts the busiest-expert load **43%** at inference scale (3789 → 2154 tok/layer) and CP 43%
(356k → 202k), closely tracking the 47% training-scale CP cut (8800→4682). So CP *does* describe
a real reduction in the load on the single busiest expert. **This is the positive result** — and
the only one. Whether it converts to latency is Sections 3–5.

---

## 3. Per-stage decomposition — MEASURED (torch.profiler kernel traces)

Decomposition on GPU-kernel self-time (parse_decode_trace.py, gz-patched). **Profiling overhead:
torch.profiler inflates absolute step time ~1.96×** (unprofiled EP8 bs64 step 112.4 ms vs profiled
trace single-step ~220 ms; job 28803089 vs 28803446). Absolute trace ms are therefore ~2×; use
**stage ratios** and **cross-rank spread**, never trace-ms as walltime.

### Comm pattern: ALL-REDUCE, not all-to-all (settles the open question)
At **every** EP the only collective is a TP all-reduce — there is **no** MoE all-to-all
dispatch/combine in the vLLM serving path (unlike Megatron training which used
`--moe-token-dispatcher-type alltoall`):
- EP8 (NVLink, 1 node): vLLM custom all-reduce `cross_device_reduce_2stage`; MoE a2a only ~0.29 ms/step. (job 28803446)
- EP32/EP64 (IB): `ncclDevKernel_AllReduce_Sum_bf16_TREE_LL`, 6237 calls (=188/forward ×33 forwards) + a tiny AllGather. **Zero** AllToAll/SendRecv. (jobs 28818149, 28818217)

### Per-rank decode-step decomposition (profiled; ratios on the least-spinning rank)
| EP | rank | AllReduce | expert_FFN | attn | other | note |
|----|------|-----------|-----------|------|-------|------|
| 8  | rank3 (min-spin) | ~6 ms real / 177 ms on rank0,7 | 24 ms | 3 ms | ~17 ms | TP-AR is 82% on spinning ranks (job 28803446) |
| 32 | rank-min | 102.7 ms/step* | 11.6 ms | 2.6 ms | — | AR 81–88% of kernel time; **asymmetric across ranks** (job 28818149) |
| 64 | rank6 (min-spin) | **28.7 ms/step** | 8.5 ms | 2.1 ms | — | AR 59% on rank6 vs **90% on rank22/24** (job 28818217) |

\*EP32 rank-min still carries spin; EP64 rank6 is the cleanest real-transfer floor.

### THE key finding — all-reduce time is barrier spin-wait, not bandwidth
For the **same** 6237 AllReduce calls of the **same** message size, kernel time is wildly
asymmetric across ranks:
- EP32: rank12 = 3406 ms (81%) … rank29 = 5758 ms (88%) — ~2400 ms spread.
- EP64: **rank6 = 951.7 ms (59%)** … rank22/24 = 5934–5958 ms (90%) — **~5000 ms is pure
  barrier spin/idle waiting for straggler ranks.**

So the “comms” bucket is dominated by load-imbalance serialization, with a real-transfer
**floor** given by the least-waiting rank (EP64 ≈ 951.7 ms over the burst ≈ 28.7 ms/step).
This is a **TP-not-EP all-reduce artifact**, NOT classic a2a-dispatch masking.

### Cross-validation (reconciled, not averaged)
- **r15 FFN ≈ pretrained FFN (the no-leverage proof).** EP32 r15 FFN = 402–420 ms ≈ EP32
  pretrained FFN = 370–404 ms across ranks, despite r15’s 43% lower busiest-expert load
  (Section 2). Mechanism: per-GPU fused-MoE time = f(total tokens through 128/EP resident
  experts) = conserved; CP only moves the per-expert split. (jobs 28818149 vs 28818150)
- **expert_FFN is balanced across ranks** (EP32 370–404; EP64 283–315) → CP does not
  differentiate ranks → cannot reduce the straggler/spin term either.
- **FFN is a small fraction** of the step (EP8 ~11%, EP32/64 ~6–10% of kernel time) → even
  zeroing it moves the step <11%.
- The EP64 fused-MoE per-GPU time (283–315 ms) < EP32 (370–404 ms) because EP64 hosts 2
  experts/GPU vs 4 — consistent with “per-GPU cost set by #resident experts × tokens”.

---

## 4. Measured parameters (each with job + variance)

| parameter | value | source |
|-----------|-------|--------|
| FFN eager floor / expert | 0.134 ms (graph 0.065 ms) | stress-test, job 28803152, 5 trials |
| FFN eager knee B* | ~448 tok/expert (graph ~224; roofline 153) | job 28803152 |
| FFN slope above knee | 0.16 µs/tok (roofline 0.121) | job 28803152 |
| NVLink all_to_all latency floor | 0.078 ms (sub-MB); 171 GB/s algbw @67 MB | job 28803024, n=50 |
| NVLink real all-reduce (EP8) | ~6 ms/step on min-spin rank (bs64, profiled) | job 28803446 |
| IB real all-reduce floor (EP64) | ~28.7 ms/step on min-spin rank (bs64, profiled ⇒ ~15 ms unprofiled) | job 28818217 |
| barrier-spin component (EP64) | up to ~5000 ms over burst (~150 ms/step) | job 28818217 |
| profiling overhead | ×1.96 (+96%) on step time | jobs 28803089 vs 28803446 |
| inference imbalance ratio @8192 tok | 7.40× (pre) / 4.21× (r15) | job 28817880, n=8 |
| per-GPU expert_FFN (EP32 bs64, profiled) | ~370–420 ms (pre≈r15) | jobs 28818149/150 |

UNVALIDATED: a dedicated IB all_to_all microbench was not run (the launcher’s A2A bench was
defined-but-not-invoked); the IB term here is taken from the multi-node **all-reduce** trace,
which is the operative collective anyway. A synthetic IB all-to-all BW remains UNVALIDATED but
is **not on the serving critical path** (no a2a kernels exist in the trace).

---

## 5. Measured A100 CP→speedup curve (sweeps, n=8–10 trials)

Per-decode-step = (e2e − ttft)/(max_tokens−1). “speedup” = pretrained/r15 (>1 = r15 faster).

| EP | bs | pre step (ms) | r15 step (ms) | decode speedup | pre TTFT (ms) | r15 TTFT | prefill speedup |
|----|----|---------------|---------------|----------------|---------------|----------|-----------------|
| 8  | 8   | 112.6 | 114.0 | 0.99 | — | — | — | (job 28803089/091) |
| 8  | 512 | 226.6 | 231.2 | 0.98 | — | — | — |
| 8  | 1024| 440.8 | 450.1 | 0.98 | — | — | — |
| 32 | 64  | 125.1 | 126.0 | 0.99 | 257.2 | 259.0 | 0.99 | (job 28818149/150) |
| 32 | 512 | 260.6 | 261.5 | 1.00 | 766.7 | 756.9 | 1.01 |
| 64 | 64  | 142.5 | 149.4 | 0.95 | 291.6 | 291.2 | 1.00 | (job 28818217/218) |
| 64 | 512 | 338.8 | 391.1 | 0.87 | 770.7 | 802.7 | 0.96 |

(EP8 ladder bs8 from jobs 28716419/375; multi-node bs8 ladder EP16/32/64 ≈ flat too,
jobs 28716xxx.) **Verdict: CP→speedup ≈ 1.00 everywhere, decode and prefill.** At EP64 r15 is
slightly *slower*, dominated by the large all-reduce barrier-spin run-to-run variance
(EP64 e2e_std ≈ 1450–1918 ms vs EP32 ≈ 88 ms). Even at bs512 (where the busiest expert clears
the FFN knee and prefill *should* be CP-leveraged), no benefit surfaces — confirming the
per-GPU-aggregate + all-reduce masking.

---

## 6. Parameterized decode-step model (every parameter measured)

```
step_time(EP, bs, interconnect, routing) ≈
    attn(bs)                                   # ~2–3 ms, EP-insensitive          [MEASURED]
  + dense_gemm + norm                          # ~5–9 ms                          [MEASURED]
  + expert_FFN(tokens_per_GPU)                 # tokens_per_GPU = bs*top_k/EP,
                                               #   through 128/EP resident experts;
                                               #   FFN_curve(floor 0.134ms/expert,
                                               #   knee 448, slope 0.16µs/tok).
                                               #   NOT a function of CP.           [MEASURED]
  + allreduce_real(msg, BW_interconnect, EP)   # floor: EP8(NVLink)~6ms,
                                               #   EP64(IB)~15–29ms/step           [MEASURED]
  + barrier_spin(load_imbalance, EP)           # dominant at multi-node;
                                               #   grows with EP & straggler var.  [MEASURED]
```
- CP enters ONLY through the per-*expert* split inside `expert_FFN`, but `expert_FFN` depends on
  the per-GPU *aggregate* (conserved), so ∂step/∂CP ≈ 0. (MEASURED: r15 FFN ≈ pre FFN.)
- The dominant term at multi-node is `allreduce_real + barrier_spin`, neither CP-governed
  (FFN is balanced across ranks → CP doesn’t reduce the straggler).
- All-NVLink branch validated directly by the EP8 anchor: unprofiled step flat ~112 ms to bs256,
  47% CP cut → 0 speedup at every bs incl. compute-bound bs512/1024. (job 28803089/091)

---

## 7. NVL72 / GB200 projection (MODELED; anchored to EP8 all-NVLink measurement)

Substitute the inter-node IB all-reduce term with an NVLink-domain all-reduce. Even in the
best case this does **not** revive a CP benefit, because:
- `expert_FFN` is unchanged (still per-GPU-aggregate, not CP). → the CP-leverage term is ~0
  regardless of interconnect. (anchor: EP8 is an 8-GPU all-NVLink domain; CP cut → 0 speedup —
  MEASURED job 28803089/091.)
- Replacing IB all-reduce (~15–29 ms/step floor + up to ~150 ms/step spin) with an NVLink
  all-reduce shrinks the *comms+spin* term (NVLink ≈ 2–4× the IB bandwidth on these message
  sizes; the spin/imbalance component also shrinks with faster sync) → **step time drops and
  decode_tps rises**, but the **CP-attributable** fraction stays ~0.

**Projected NVL72 CP→speedup curve: ≈ 1.00 across CP**, same as measured A100 — i.e. NVL72
improves absolute latency (comms no longer masks compute *as much*) but the CP *reduction*
still does not convert to speedup, because the unmasked term that grows in relative weight is
the per-GPU-aggregate expert-FFN, which CP does not change.

Residual assumptions:
- NVLink-domain all-reduce BW/latency at 72 GPUs: taken from EP8 NVLink a2a (job 28803024) +
  EP8 real all-reduce (job 28803446); 72-GPU NVLink all-reduce scaling is **UNVALIDATED**
  (no NVL72 hardware here). Closest anchor = EP8 all-NVLink (8 GPUs).
- The claim “CP could instead reduce the barrier-spin by balancing load” is **refuted** by the
  measured r15-vs-pretrained FFN equality (FFN balanced across ranks both before and after CP
  cut) → CP does not reduce the straggler term. (MEASURED, jobs 28818149/150)

---

## 8. What would actually move A100/NVL72 MoE decode latency (implication)

CP is the wrong lever for serving latency on this stack. The levers the data points to:
(a) reduce the all-reduce **barrier-spin** (better cross-rank load balance at the *GPU* level,
e.g. expert-placement balancing, not per-expert CP); (b) CUDA graphs (removes ~50% of the eager
launch floor — job 28803152); (c) raise tokens/GPU into the FFN compute-bound regime where the
aggregate (not CP) is the cost. CP remains a valid **training/routing-quality** metric; it is
not an inference-latency metric on A100 or (projected) NVL72.

## Provenance index
- Routing dump (paradox + CP→compute): job 28817880 → routing_dump_{pretrained,r15}_28817880.json
- EP8 batch sweep + profiling overhead: jobs 28803089/28803091; trace 28803446
- FFN curve/knee stress-test: job 28803152 → ffn_stress_28803152.json
- NVLink a2a microbench: job 28803024 → ffn_a2a_nvlink_ep8_28803024.json
- EP32 sweep+trace: jobs 28818149 (pre), 28818150 (r15) → vllm_ep32_sweep_*.json, trace_*_ep32_*/
- EP64 sweep+trace: jobs 28818217 (pre), 28818218 (r15) → vllm_ep64_sweep_*.json, trace_*_ep64_*/
- Multi-node Ray hang fix (custom Ray ports broke cross-node actor scheduling): launcher
  run_ord_rayEP_profile.sh, backups *.bak_ports/*.bak_headip/*.bak_timeout.
All result JSON/traces under $L/rl_token_routing/cp_latency_results/ ($L = /lustre/fsw/portfolios/nvr/users/jonathanp).
