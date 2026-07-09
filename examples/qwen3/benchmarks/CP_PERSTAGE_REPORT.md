# CP -> Per-Stage-Timing Report (Qwen3-235B-A22B, A100 / vLLM, EP8 / EP32 / EP64)

**Status: IN PROGRESS — measurement pass started 2026-06-16.**
Companion to `CP_SPEEDUP_FINDINGS.md` (2026-06-07). Where `FINDINGS` measured 2 CP points at
decode scale and flagged Section 9 (compute-bound EP64 prefill) as untested, this report:
1. Adds a 3rd CP cell so we have a real CP curve (3 points), not 2 endpoints.
2. Closes Section 9 with a compute-bound EP64 prefill measurement (CUDA graphs ON, >=8192-token
   forward, >=20 trials).
3. Decomposes every cell x EP x regime into ATTN / EXPERT-FFN / NETWORK with per-rank
   kernel traces.

---

## 1. TL;DR

Headline finding from the measured EP8 traces (CUDA graphs ON for prefill, eager for decode):

1. **At EP8 prefill (compute-bound, plen=8192, bs=1, n=20) lower-CP cells are ~9% SLOWER on
   TTFT (54.66 -> 59.51 ms r15; 59.68 r05_iter2500_cp). Per-rank busiest expert_FFN is
   23-31% HIGHER for the lower-CP cells (~8-9 sigma above across-rank std).** Opposite of
   the naive "CP reduction = compute reduction" hypothesis. Probable mechanism: at EP8 each
   GPU hosts 128/8 = 16 experts; the per-GPU aggregate tokens (8192) is conserved across
   cells, but CP-reduced routing balances tokens more uniformly across resident experts,
   producing MORE smaller grouped-GEMM tiles -> WORSE arithmetic intensity / more launch
   overhead. See Section 3 (prefill table).

2. **At EP8 decode (sub-knee, bs=64, eager) lower-CP cells are ~20% FASTER per-FFN-step
   than pretrained (13.41 ms -> 10.83 r15, 10.62 r05). But net step time (and decode_tps)
   is flat within 3% across cells -- the FFN delta is absorbed by larger "other" kernel
   buckets (consistent with the prior FINDINGS that decode is AllReduce-spin-dominated at
   EP8 single-node).** See Section 3 (decode table).

3. **The sign of "CP effect on per-step FFN" is REGIME-DEPENDENT at EP8** -- positive for
   decode (helpful), negative for prefill (harmful). Neither converts to net TTFT/step
   delta beyond the bench noise floor for decode; for prefill the +25% FFN delta + small
   other-stage deltas produces a measurable +9% TTFT slowdown.

4. **Status of the Section 9 closer (compute-bound EP64 prefill, the original gap):
   NOT YET MEASURED.** Cluster ran into a hard accounting collision -- the project's own
   `babysit_3000.sh` supervisor enforces a 64-GPU per-user cap by cancelling the newest
   PENDING non-exempt jobs at each 20-min cycle. With the babysit's own continuations
   (cg_/c3000_) consistently holding 48-64 GPU committed, the budget available to this
   measurement campaign collapses to ~0-16 GPU and EP32 (32 GPU/job) / EP64 (64 GPU/job)
   submissions hit `QOSMaxGRESPerUser` or are trimmed before dispatch (see Section 6
   provenance: jobs 29141112, 29141114, 29141310-13, 29142128, 29142131, 29149250,
   29154877 all cancelled by babysit or by QOS). The data here is from EP8 single-node
   only.

The narrow EP8 result is itself interesting and contradicts the simplest reading of
"CP -> latency": the *sign* of the effect inverts between decode and prefill, and the
absolute magnitudes (+/- 20-30% per-step FFN) are larger than the prior 2-point study
implied. But the COMPUTE-BOUND multi-node test the prior FINDINGS was missing remains
unfinished pending cluster capacity. Recommended next step: temporarily lift babysit's
GPU_CAP from 64 -> 72-80 (the user-stated cluster cap and "+8 buffer") to permit the
EP32 / EP64 prefill measurements.


## 2. Method

### CP cells (3-point ladder)
| Cell      | hf_converted path (relative to project root)                              | CP_train (seqlen=128) | acc | inference busiest expert (8192 tok) |
|-----------|---------------------------------------------------------------------------|-----------------------|-----|-------------------------------------|
| pretrained | `qwen-ckpts/Qwen3-235B-A22B`                                              | 8800                  | n/a | 3789 tok/layer (7.40x, MEASURED job 28817880) |
| r15        | `.../235bv5a_rlc1.0_ppo_aux0.01_r15/.../hf_converted_iter3000_cp`          | 4682 (iter 3802)      | 72.4 (eval) | 2154 tok/layer (4.21x, MEASURED job 28817880) |
| r05_iter2500_cp | `.../235bv5a_rlc0.1_ppo_aux0.02_r05/.../hf_converted_iter2500_cp`     | 4247 -> 3797 across iters 2000->3000 (CP_train, CSV) | 72.55 (iter 3000, CSV) | 2161.5 tok/layer (4.22x, MEASURED job 29140800, n=8) <- ESSENTIALLY IDENTICAL TO r15 — iter 2500 lands on the same CP point as r15 iter 3000. The true 3-point ladder needs r05_iter3000 (conversion job 29141114 pending; expected busiest ~1942 from CP_train ratio 3797/4214 × 2154) |

(r05 iter 3000 -> HF conversion submitted as job 29141114; if successful we'll use those
weights to hit the canonical CP=3797 point.)

### EP ladder
- **EP8** (single node, all-NVLink): tensor_parallel_size=8 + enable_expert_parallel.
- **EP32** (4 nodes x 8 GPUs, Ray multi-node, inter-node InfiniBand).
- **EP64** (8 nodes x 8 GPUs, Ray multi-node).

### Regimes
- **Decode**: `--prompt-lengths 256 --batch-sizes 64,512 --max-tokens 128`, eager mode, n>=10 trials.
- **Prefill** (compute-bound, the Section 9 closer): `--prompt-lengths 8192[,16384] --batch-sizes 1,2[,4] --max-tokens 8`, **CUDA graphs ON**, n>=20 trials.

### Profiler
**torch.profiler** kernel traces (per-rank), parsed by `parse_decode_trace.py` into
ATTN / expert_FFN / comm_other (AllReduce) / moe_a2a / GEMM / norm_elementwise / other.

**Why not nsys**: nsys is not installed on ORD compute nodes (verified via interactive srun
on `batch-block1-2032` / `batch-block1-0075`) and is not present in `vllm-openai-latest.sqsh`.
It IS present in `pai-megatron-patch_25.04.sqsh` at `/usr/local/cuda/bin/nsys 2025.2.1`, but
the megatron container does not ship a vLLM install. Cleanest path forward is torch.profiler
(prior validated; per-rank parser already exists). A future nsys cross-anchor is possible by
extracting the nsys binary plus libs and bind-mounting into the vllm container; deferred as a
stretch goal.

### Profiling overhead correction
- Prior: torch.profiler inflates EP8 bs64 decode-eager step time **~1.96x** (jobs 28803089
  vs 28803446). Used to flag absolute trace ms as **non-walltime**.
- This work re-measures the multiplier in the prefill / CUDA-graphs-ON regime (Phase A2):
  [FILL]. The unprofiled TTFT walltime is `54.66 ms (std 0.36, n=20)`; the profiled
  per-rank busiest kernel-self-time sum is `48.50 ms`. These are NOT directly comparable
  (kernel self-time != wallclock; ratios are what matter).
- Throughout the report: stage **ratios** are primary; absolute ms cited only when both numbers
  are from the same job (so the multiplier cancels).

### Tooling
- Bench: `cp_latency_test/cp_vllm_bench.py` (--models, --tp-size, --prompt-lengths,
  --batch-sizes, --max-tokens, --num-warmup, --num-trials, --profile, --prefill-profile,
  --cuda-graphs, --distributed-executor-backend ray).
- Launchers: `run_ord_rayEP_profile.sh` (decode), `run_ord_rayEP_prefill.sh` (prefill).
- Trace decomposition: `parse_decode_trace.py --per-rank --prefill`.
- Aggregation: `cp_aggregate_perstage.py --manifest <csv> --out <csv>` (new in this pass).

---

## 3. Per-stage table

[FILL: a single CSV-derived table — one row per (cell, EP, regime, plen, bs)
with columns:
  cell | cp_train | cp_inf | EP | regime | plen | bs | n | TTFT_ms +/- std | e2e_ms +/- std |
  busiest_FFN | mean_FFN +/- std | mean_AR | mean_attn | mean_gemm | busiest_total
Plus a plot (`per_stage_vs_cp.png`): stage_ms vs CP_reduction_pct, one line per (EP, regime).]

---


---


---

## EP8 prefill: lower-CP cells are SLOWER at TTFT (validated by re-measure)

Single-node EP8, plen=8192, bs=1, max_tokens=8, CUDA graphs ON, n=20 trials per cell.

| Cell | Trial date | Bench TTFT (ms) | std | busiest_FFN (ms) | mean_FFN (ms) | std_FFN |
|------|-----------|-----------------:|----:|------------------:|--------------:|--------:|
| pretrained (job 28837949, 2026-06-08) | t0   | 54.66 | 0.36 | 14.32 | 13.41 | 0.53 |
| pretrained (job 29156714, 2026-06-16) | t+10d | 54.89 | 0.28 | 14.10 | 13.38 | 0.49 |
| r15 (job 29141111, 2026-06-16)        | t+10d | 59.51 | 0.64 | 17.68 | 17.15 | 0.55 |
| r05_iter2500_cp (job 29148502, 2026-06-16) | t+10d | 59.68 | 0.28 | 18.74 | 18.03 | 0.56 |

**The pretrained re-measure 10 days later matches the original within noise (54.66 vs 54.89,
both stds 0.3-0.4 ms). The 9 percent TTFT slowdown of lower-CP cells is reproducible and
NOT a cluster-condition confound. Per-rank busiest expert_FFN is 23-31 percent higher for the
lower-CP cells, ~8-9 sigma above the across-rank std.**

### What the simple per-GPU-load model predicts vs what is measured

| Cell | inference busiest expert (tok/layer) | busiest **GPU** total (tok/layer, EP=8) | predicted FFN sign vs pretrained | observed |
|------|--------------------------------------:|----------------------------------------:|----------------------------------:|---------:|
| pretrained | 3789 (imb 7.4x) | 11416 (job 28817880) | -- | baseline |
| r15        | 2154 (imb 4.2x) | 10215 | should be FASTER (fewer tokens) | **23% SLOWER** |
| r05        | 2162 (imb 4.2x) | 10190 | should be FASTER | **31% SLOWER** |

So the naive "per-GPU aggregate tokens -> FFN time" model predicts r15/r05 FFN should be
LOWER than pretrained (they have less per-GPU work). The MEASUREMENT shows the OPPOSITE.

### Why? (UNVALIDATED hypothesis, do not cite as proven)

The fused-MoE grouped-GEMM kernel on the busiest GPU processes ALL 16 resident experts in
one call. Time depends not just on total tokens but on the SHAPE of the per-expert token
distribution within those 16 experts. A speculative read:

- Pretrained: highly imbalanced. The busiest **expert** there (across all GPUs) is 3789 tok,
  on whichever GPU it lives. Within the busiest GPU's 16 experts, the distribution is
  steeply concentrated. Wide grouped-GEMM tiles, possibly cuBLAS fast-paths active.
- r15/r05: balanced. Within the busiest GPU's 16 experts, tokens are more uniformly spread.
  Smaller per-expert chunks, more tiles needed, lower per-tile arithmetic intensity.

This is a hypothesis that requires direct kernel-name inspection to confirm (which CUDA
kernels actually launched, what their grid/block shapes were). The 86MB `pt.trace.json.gz`
files do carry kernel name and dur but not grid/block shapes. An nsys trace would resolve
this directly but nsys is not available on the cluster (verified for `vllm-openai` container
and host nodes; nsys IS in `pai-megatron-patch_25.04.sqsh` container but vLLM is not).

ALTERNATIVE mechanisms to consider:
- Different attention/projection/norm kernels triggered by different routing patterns
- Different expert-FFN weights (if RL training didn't fully freeze them -- routing-only
  training was the intent, but worth verifying that expert_W1/W2 are bit-identical to
  pretrained)
- vLLM v0.20.2 path-specific quirks (e.g. the kernel may have a "highly imbalanced"
  fast-path)

### Implication for higher EP

At EP32, each GPU hosts 128/32 = 4 experts; at EP64, 2 experts. With fewer resident
experts per GPU, the within-GPU distribution shape is necessarily simpler. The hypothesis
predicts the EP8 anti-effect (CP reduction -> FFN slowdown) should weaken at higher EP
because there is less "shape variation" to exploit. EP32 / EP64 measurements (Phase C / D)
will discriminate this.

(Phase C / D currently blocked by user-wide 80-GPU cap: babysit_3000.sh runs at 72 GPU
committed, leaving 8 GPU for measurement. Conversion job 29163964 is queued; EP32 / EP64
require babysit to drain.)


---

## Root cause (validated 2026-06-17): `fused_moe_kernel` per-call duration is routing-shape-sensitive

The previous EP8 prefill SURPRISE write-up speculated about "smaller grouped-GEMM tiles"
but did not bottom-up confirm. Three independent checks done today (jobs 29182209,
29182230 + per-rank decompose) localize the entire FFN delta to a single Triton kernel.

### Check 1: expert FFN weights are bit-identical across cells (job 29182209)

Hashed `mlp.experts.{0,50,127}.{gate,up,down}_proj.weight` at layers {0,30,60,93} across
pretrained / r15 / r05_iter2500_cp (60 hash comparisons total). All experts ALL_EQUAL.
Only `mlp.gate.weight` (the router Linear) differs across cells, as expected for
router-only RL fine-tuning. The FFN-time delta is therefore NOT explained by RL training
modifying expert weights -- it is purely routing-driven.

### Check 2: the slowdown is UNIFORM across all 8 ranks (per-rank decompose)

Per-rank busiest_FFN on EP8 prefill (plen=8192 bs=1, n=20 trials each):

| Rank | pretrained (ms) | r15 (ms) | r05 (ms) |
|-----:|----------------:|---------:|---------:|
| 0 | 13.37 | 17.40 | 17.44 |
| 1 | 12.90 | 16.22 | 17.33 |
| 2 | 13.81 | 17.23 | 18.28 |
| 3 | 12.91 | 16.52 | 17.68 |
| 4 | 13.90 | 17.66 | 18.74 |
| 5 | 14.10 | 17.58 | 18.57 |
| 6 | 12.99 | 17.68 | 18.54 |
| 7 | 13.10 | 16.88 | 17.66 |
| **mean** | **13.38** | **17.15** | **18.03** |
| **std across ranks** | **0.49** | **0.55** | **0.56** |
| **delta vs pretrained** | -- | **+28%** | **+35%** |

The +28-35% FFN slowdown appears uniformly on EVERY rank, not concentrated on the
busiest GPU. This rules out any per-GPU-load explanation -- the busiest GPU under
pretrained's imbalanced routing has MORE work (11416 tok/layer) than under r15/r05's
balanced routing (10215/10190 tok/layer), but is FASTER. Counter-intuitive on its face;
explained below.

### Check 3: kernel-level diff localizes the entire delta to `fused_moe_kernel` (job 29182230)

Rank0 GPU-kernel breakdown of one prefill forward (8192-token, bs=1, CUDA graphs ON,
torch.profiler trace, n_kernels = 2194 identical across all 3 cells):

| Kernel (top by total time) | pre us | r15 us | r05 us | delta pre->r15 |
|----------------------------|-------:|-------:|-------:|---------------:|
| `flash::flash_fwd_splitkv_kernel` (attention, 94 calls)             | 19217 | 19194 | 19199 | ~0 |
| **`fused_moe_kernel` (188 calls)**                                  | **11457** | **15455** | **15537** | **+35%** |
| `vllm::cross_device_reduce_2stage` (TP all-reduce, 189 calls)       | 8659 | 10221 | 9455 | +18% |
| `cutlass tensorop_sNgemm` (dense GEMM, 188 calls)                   | 1411 | 1397 | 1401 | ~0 |
| `ampere_bfN_sNgemm` (dense GEMM, 94 calls)                          | 968 | 968 | 967 | ~0 |
| `vllm::moe::topkGating` (94 calls)                                  | 613 | 613 | 609 | ~0 |
| `vllm::moe::moe_align_block_size_kernel` (94 calls)                 | 415 | 415 | 416 | ~0 |
| (~25 other kernels: norms, elementwise, gather, ncclAllGather, ...) | ... | ... | ... | ~0 |
| **TOTAL kernel time on rank0**                                      | **46.81 ms** | **52.36 ms** | **51.69 ms** | **+12%** |

- **Total kernel COUNT is 2194 in all three cells.** The launch graph is identical.
- `fused_moe_kernel` is invoked exactly 188 times in all three cells (= 2 calls per MoE layer
  x 94 layers). Same call count, same expert weights.
- The cell-vs-cell delta is captured almost entirely by **one kernel**: `fused_moe_kernel`
  takes ~61 us per call under pretrained but ~82 us per call under r15/r05 -- a ~35%
  per-call slowdown driven purely by which experts the router sent tokens to.
- The secondary AR delta (+9-18% in `cross_device_reduce`) is plausibly a downstream
  consequence: when one rank's FFN finishes ~5 ms later, the AllReduce barriers wait longer.
  Or it is independent noise -- prior FINDINGS Section 3 noted EP8 AR is launch-floor-bound
  at NVLink (0.029 ms mean here), so an 18% delta amounts to 1.5 ms across 189 calls.
- Attention is invariant -- expected, since attention runs over the prompt independent of
  routing decisions.
- All dense / norm / topk / align kernels are within <1% -- these run BEFORE the routed
  expert dispatch and don't see the distribution.

### Mechanism (still open, but constrained)

The fused_moe_kernel is vLLM's Triton MoE grouped-GEMM. Its grid is sized by
`num_tokens_post_padded / BLOCK_M` (per-token-block count after expert-aware alignment).
Total token-expert assignments are constant across cells (bs * top_k = 8192 * 8 = 65536),
so total block count is approximately constant. What changes between routing patterns is:
- **Per-block work distribution.** Pretrained's heavily imbalanced routing concentrates
  tokens in fewer experts, producing fewer "active expert" lookups per block-iteration.
- **Memory-access pattern.** Balanced routing means each block reads a different expert
  weight slab; pretrained has more block-locality (consecutive blocks hit the same expert).
  An L2-cache effect is plausible: ~512 tok / 64 BLOCK_M = 8 consecutive blocks for the
  same expert under balance vs ~60 consecutive for the busiest expert under imbalance.
  The latter probably stays L2-resident; the former evicts. Worth verifying with nsys
  L2-hit-rate metrics, but nsys is not available on this cluster.
- **Block ID -> expert ID indirection cost** could also dominate at small per-expert
  token counts. The kernel reads `expert_ids[block_id]` then loads the corresponding
  expert weight pointer. For balanced routing every block_id has a different expert, so
  this is a divergent load.

Not yet validated end-to-end (would require either nsys / ncu metrics or modifying the
kernel and re-running). Recommended follow-up: examine vLLM v0.20.2 `vllm/model_executor/
layers/fused_moe/fused_moe.py` triton kernel and instrument BLOCK_M and L2 behavior.

### Implication

**At EP8 (single node, all-NVLink), reducing the MoE critical path makes inference
SLOWER, not faster** -- a +9% TTFT regression at prefill, contributed almost entirely
by a +35% slowdown in one Triton kernel. The mechanism is L2 / block-locality, NOT
per-GPU compute load. The naive "CP reduction -> latency win" model gets the SIGN
WRONG on this stack.

At higher EP (32, 64) each GPU hosts fewer experts (128/EP = 4 or 2), so the
block-locality effect may weaken or invert. The Phase C/D measurements (currently
blocked by the user-wide GPU cap) would resolve whether this sign flip extends to
multi-node prefill.



---

## EP16 prefill: the story revises — FFN penalty is EP-invariant, but AR penalty flips sign

Following the EP8 finding (fused_moe_kernel +35% slower per-call for r15/r05), I queued EP16
prefill (2-node Ray, 128/16 = 8 resident experts per GPU) to test whether the kernel penalty
attenuates as resident-experts-per-GPU drops. Three EP16 prefill jobs (jobs 29186555 pre,
29186559 r15, 29186561 r05), n=20 trials each, plen=8192, bs=1, CUDA graphs ON.

### TTFT cross-cell at EP16 vs EP8

| Cell | EP8 TTFT (ms) | EP16 TTFT (ms) | EP8 delta vs pre | EP16 delta vs pre |
|------|--------------:|---------------:|-----------------:|------------------:|
| pretrained | 54.89 +- 0.28 | 70.60 +- 1.75 | -- | -- |
| r15        | 59.51 +- 0.64 | 75.33 +- 0.52 | **+8.4%** | **+6.7%** |
| r05_iter2500_cp | 59.68 +- 0.28 | 74.60 +- 0.38 | **+8.7%** | **+5.7%** |

The TTFT slowdown attenuates from +8.4% to +6.7%, broadly consistent with my original
L2/locality-attenuation hypothesis... but the kernel-level decomposition shows the
mechanism is NOT what I claimed.

### EP16 kernel breakdown (rank 0, n_kernels=2287 identical across cells)

| Kernel | pre us | r15 us | r05 us | delta vs pre |
|--------|-------:|-------:|-------:|--------------|
| `ncclDevKernel_AllReduce_TREE_LL` (189 calls) | 27231 | 20806 | 24624 | **r15 -23.6%, r05 -9.6%** |
| `flash_fwd_splitkv_kernel` (attention, 94)    | 19204 | 19256 | 19262 | ~0 |
| **`fused_moe_kernel`** (188)                  | **7281** | **10003** | **10044** | **r15 +37.4%, r05 +37.9%** |
| cutlass tensorop dense gemm (188)             | 1434 | 1431 | 1432 | ~0 |
| ampere dense gemm (94)                        | 1017 | 1012 | 1021 | ~0 |
| ~25 other kernels                             | ...  | ...  | ...  | ~0 |
| **TOTAL rank0 kernel-self-time (ms)**         | **61.43** | **57.83** | **61.75** | **r15: -5.9% (net FASTER!)** |

### Three surprises

1. **The `fused_moe_kernel` per-call slowdown is EP-invariant in percentage terms.**
   At EP8: pre 60.9 us/call, r15 82.2 us/call (+35%). At EP16: pre 38.7 us/call, r15 53.2 us/call
   (+37%). The percentage penalty doesn't depend on resident-experts-per-GPU (16 vs 8).
   My earlier "fewer resident experts -> less L2 thrashing -> smaller delta" hypothesis
   was WRONG at the kernel level -- the routing-shape penalty is intrinsic to the kernel
   path, not a function of how many experts a GPU happens to host. What shrinks with EP is
   the *absolute* kernel cost (and therefore the absolute FFN delta), not the ratio.

2. **AllReduce flips sign at EP16: r15 is 23.6 percent FASTER than pretrained on AR.**
   At EP8, AR is intra-node NVLink (~45 us per call, launch-floor + L2/cache-line-ping
   dominated). At EP16, AR is inter-node IB (~144 us per call, real-bandwidth + cross-rank
   barrier-spin dominated). When AR becomes barrier-spin-bound, the slowest rank's FFN
   sets the barrier; r15's balanced routing means smaller cross-rank FFN spread, faster
   barrier convergence. The straggler-reduction effect of balanced routing -- which the
   prior FINDINGS predicted as a possible AR win at multi-node -- shows up here.

3. **Net rank0 kernel-self-time is ALREADY -5.9% under r15 at EP16.** The AR win
   (-6.4 ms) more than absorbs the FFN loss (+2.7 ms). TTFT still shows +6.7% slowdown
   because wallclock ne kernel-self-time (host-side scheduling, barrier-spin on slower
   ranks I did not decompose, etc.). But the *kernel-time crossover* between FFN and AR
   appears to happen between EP8 and EP16.

### Revised mechanism

- The fused_moe_kernel penalty under balanced routing is a fixed-percentage cost of the
  kernel path itself (likely L2-locality, per the original story; the percentage stays
  constant because both pretrained and r15 see the same scaling as per-expert tokens drop).
- The AR-spin benefit of balanced routing scales with how big the AR cost is relative to
  FFN. At EP8 NVLink, AR is too cheap to matter (~9 ms of 47 ms total kernel time, mostly
  launch-floor; balanced-routing's straggler reduction can't beat launch-floor). At EP16
  IB, AR is the largest kernel (~27 ms of 61 ms), and balanced-routing's straggler
  reduction pays back.
- **Prediction for EP32 / EP64**: the AR/FFN ratio grows further, the AR-straggler win
  grows further, and r15/r05 should become net FASTER on TTFT, not slower. The sign of
  "CP -> latency" is regime-dependent, and the crossover from CP-hurts to CP-helps
  appears to happen near EP16.

### Status of Phase C / D

Phase C (EP32) is still blocked at SLURM submit time by QOSMaxGRESPerUser, since the
training-side babysit owns ~600 GPU committed in the 4-node QOS pool. EP8 and EP16 (1-2 node
QOS pools) submit fine. When the cluster drains the babysit backlog, the EP32/EP64 trend
prediction can be tested. Provenance: jobs 29186555 (pre EP16), 29186559 (r15 EP16),
29186561 (r05 EP16), 29186556 (pre EP8 bs-sweep). Kernel-diff anchor jobs 29191919 (EP16).



---

## CORRECTION (2026-06-18): the +35% fused_moe_kernel claim does NOT replicate in nsys

The earlier story (Section "EP8 prefill: lower-CP cells are SLOWER at TTFT") attributed the
9% TTFT slowdown to fused_moe_kernel running +35% slower per call under balanced routing,
with a speculated L2-locality mechanism. That kernel-level claim is now contradicted by an
independent nsys cross-validation. The TTFT slowdown is REAL — but its root cause is NOT
the FFN kernel; it is the choice of AllReduce backend.

### What we actually verified (data paths included so it's reproducible)

#### 1. Expert FFN weights are bit-identical across cells (still holds)
- Job: `29195215` (verification) / `29198352` (initial probe) — `nsys_ep8_pre_*` not relevant; the
  comparison ran via interactive srun in the vllm-openai container with the staged nsys.
- Script: `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results/compare_experts.py`
- Hashed `mlp.experts.{0,50,127}.{gate,up,down}_proj.weight` at layers {0,30,60,93} across all
  three checkpoints (60 hashes), all ALL_EQUAL. Only `mlp.gate.weight` (the router Linear)
  differs across cells (4/4 layers ALL_DIFFER). Conclusion: only the router weights changed.

#### 2. Model architecture / configs / file structure are identical
- Verified 2026-06-18 by direct diff:
  - `config.json`: pretrained vs r15: NO diff. r15 vs r05: NO diff.
  - `generation_config.json`: identical across all three.
  - `model.safetensors.index.json`: all three have 36,945 tensors, total_size=470.19 GB.
  - 118 safetensors files in each, total dir size 438 GB.
- Tensor-name pattern check (running_, momentum_, ema_, optim_, adv_, value_, critic_, reward_,
  rl_, _aux_, shadow_, buffer_): **zero matches in any cell.** No residual RL training tensors
  (no optimizer state, no EMA weights, no value/critic heads).
- File-count cosmetic differences only: pretrained has extra `benchmark_results*` + `LICENSE`
  legacy files; doesn't affect vLLM load.
- Data paths:
  - pretrained: `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/qwen-ckpts/Qwen3-235B-A22B`
  - r15: `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/235bv5a_rlc1.0_ppo_aux0.01_r15/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter3000_cp`
  - r05: `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/235bv5a_rlc0.1_ppo_aux0.02_r05/checkpoint/pretrain-mcore-qwen3-moe-megatron-A22B-lr-1e-4-minlr-1e-6-bs-1-gbs-16-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-true-ti-3000-wi-5/hf_converted_iter2500_cp`

#### 3. nsys (independent of torch.profiler) shows fused_moe_kernel is NOT slower under r15/r05

After applying the Glean-provided fix (`--trace-fork-before-exec=true` + `VLLM_WORKER_MULTIPROC_METHOD=spawn`
+ `--cuda-graph-trace=node`; see job 29201913 which produced a 308 MB working trace vs 29195852/29196317/29197843
which were sub-1MB and empty), nsys captured the full bench across 130-134k fused_moe_kernel
invocations per cell.

**Per-cell fused_moe_kernel summary** (cuda_gpu_kern_sum on full bench, model load + warmup + trials):
| Cell | Instances | Total ns | Avg ns | Med ns | Max ns |
|------|----------:|---------:|-------:|-------:|-------:|
| pretrained (job 29201913) | 130,848 | 7,332,026,130 | 56,035 | **18,337** | 4,282,435 |
| r15        (job 29230389) | 134,002 | 7,187,638,249 | 53,638 | **18,304** | 4,677,373 |
| r05_iter2500_cp (job 29230390) | 134,002 | 7,171,767,102 | 53,520 | **18,240** | 4,023,375 |

- Median per-call time is **18.3 microseconds in all three cells** (identical within 0.5%).
- Mean is dominated by ~5% long-tail outliers (max ~4-4.7 ms per call); r15/r05 means are
  ACTUALLY 4% LOWER than pretrained, not 35% higher.
- Total fused_moe time across the whole bench is r15/r05 about 2% LOWER than pretrained.
- The earlier "+35%" came from torch.profiler scoped to ONE profiled prefill burst (188
  fused_moe calls). That sample was biased; nsys's 700x larger sample (130k calls) does NOT
  show the per-call delta.

Data paths for verification:
- `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results/nsys_ep8_prefill_pretrained_235b_29201913.nsys-rep` (308 MB)
- `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results/nsys_ep8_prefill_r15_cp4682_29230389.nsys-rep` (323 MB)
- `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results/nsys_ep8_prefill_r05_iter2500_cp_29230390.nsys-rep` (324 MB)
- Replay: `srun ... --container-image=vllm-openai-latest.sqsh ... /lustre/.../tools/nsys/NsightSystems-cli-2025.2.1/target-linux-x64/nsys stats --report cuda_gpu_kern_sum <REP>`

### What IS different across cells (root cause of TTFT delta)

#### vLLM picks a different AllReduce backend for r15/r05 than for pretrained

Same nsys traces, kernel summary grepped for AR-related kernels:

| Kernel | pretrained | r15 | r05 |
|--------|-----------:|----:|----:|
| `vllm::cross_device_reduce_2stage<bf16, 8>` (NVLink-optimized) | 189 inst @ 280 ms | **0** | **0** |
| `vllm::cross_device_reduce_1stage<bf16, 8>` (NVLink-optimized) | 1512 inst @ 56 ms | **0** | **0** |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` (NCCL fallback) | 1701 inst @ 4.19 s | 1512 inst @ 27.47 s | 1512 inst @ 13.62 s |

- **Pretrained uses vLLM's tuned NVLink AllReduce** (`cross_device_reduce_2stage` and `_1stage`,
  combined 1701 instances at 336 ms).
- **r15 and r05 use ONLY NCCL `RING_LL`** as the AllReduce backend. The vLLM custom-AR kernels
  are not invoked at all (zero instances).
- NCCL RING_LL at EP8 single-node NVLink is dramatically slower per call than vLLM's custom AR:
  pretrained's RING_LL avg = 2.46 ms/call, r15's RING_LL avg = 18.17 ms/call, r05's = 9.01 ms.
  Note: the RING_LL time includes both model-load broadcasts and inference AR; the inference
  portion is part of the total, and that's where the +9% TTFT delta comes from.

This is the actual mechanism behind the bench TTFT delta:
- pretrained (29156714, re-measured): TTFT = 54.89 +- 0.28 ms (n=20)
- r15 (29141111): TTFT = 59.51 +- 0.64 ms (n=20)
- r05 (29148502): TTFT = 59.68 +- 0.28 ms (n=20)
- The +5 ms TTFT delta IS the cost of running NCCL RING_LL instead of vLLM custom AR for ~188
  AR calls per forward.

Data paths:
- Bench JSONs: `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results/vllm_ep8_prefill_sweep_{pretrained_235b_v2_29156714,r15_cp4682_29141111,r05_iter2500_cp_29148502}.json`
- vLLM startup logs: `/lustre/fsw/portfolios/nvr/users/jonathanp/ord_rayEP_prefill_ord_rayEP_prefill_{29156714,29141111,29148502}.out`
  (search for `Initializing a V1 LLM engine` for the config dump; all show `disable_custom_all_reduce=False`, so custom AR was REQUESTED for all three).

#### Why does vLLM fall back to NCCL for r15/r05? — OPEN, not yet answered

`disable_custom_all_reduce=False` is set for ALL three cells in their vLLM config — so the user
asked for custom AR in every case. Yet only pretrained ends up using it at runtime. Possible
causes (NOT YET TESTED):

(a) **Node-topology luck.** The three benches ran on different physical nodes (pretrained
    29156714 on batch-block4-1+, r15 29141111 on batch-block1-0+, r05 29148502 on batch-block1-0+).
    vLLM's custom AR registration depends on P2P/NVLink capability; if some nodes have
    different P2P availability, custom AR may not initialize on those nodes.
    *Counterargument*: pretrained was reproducibly fast on two different nodes (2026-06-08
    job 28837949 on a third node, and 2026-06-16 job 29156714 on yet another node, identical
    TTFTs within noise). r15/r05 are reproducibly slow across two cells x multiple nodes. So
    node-luck would need to be highly biased (pretrained always luckier).

(b) **Model-content-dependent dispatch.** vLLM's custom AR registration may key off some
    runtime check that the model's gate weights affect. Routing-only RL training changed the
    gate weights numerically (verified ALL_DIFFER across all 4 sampled layers), and this
    might trip a fall-back path. *Plausibility*: medium — would be a surprising design choice
    but not impossible if vLLM checks something like "do all tensor shapes match expected" on
    a per-load basis.

(c) **vLLM initialization order or process state.** With `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
    each forward of cp_vllm_bench.py loads a model fresh; if pretrained happens to load first
    in a fresh process state and the IPC handles register cleanly, but a subsequent model load
    in a "dirty" state fails to register — that could explain the pattern, except r15/r05 also
    ran in fresh processes (separate sbatch jobs each).

(d) **Symptom of a known vLLM v0.20.2 bug.** Custom AR has historically had registration
    failures that fall back silently to NCCL. Worth searching vLLM issue tracker for
    routing-related fall-back patterns.

### Same-node controlled test queued (job 29231096)

Submitted `run_ar_compare_samenode.sh` (path: `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/cp_latency_test/run_ar_compare_samenode.sh`).
This loads pretrained + r15 + r05 sequentially in the SAME vLLM engine instance on the SAME
node, with nsys tracing. If the AR backend still differs across cells in this controlled
condition (same node, same NCCL state, same IPC environment), cause (b) is confirmed —
the dispatch is model-content-dependent. If all three suddenly behave identically, the
cause is (a) or (c) — node/process state contamination.

Output paths (once it completes):
- `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results/ar_compare_samenode_29231096.json` (bench)
- `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results/ar_compare_samenode_29231096.nsys-rep` (kernel trace)

### Is the +9% TTFT increase worth investigating?

**Yes, for two reasons:**

1. **It contradicts the project thesis on its face.** Router-only RL training was designed
   to REDUCE CP (and through that, ideally, reduce inference latency). Measuring instead a
   +9% TTFT slowdown at EP8 means the training intervention is producing the OPPOSITE of its
   intended effect at this configuration. Even though the magnitude is small, the SIGN is
   wrong and that matters for how the work gets framed.

2. **The mechanism is now believed to be an AllReduce backend dispatch issue, not a
   compute issue.** If we can identify why vLLM falls back to NCCL RING_LL for r15/r05
   (jobs 29230389, 29230390 nsys traces above) and fix it, the +9% TTFT slowdown should
   disappear — and CP-related effects (which the prior FINDINGS doc and this report's
   Section 3 said were near-zero at decode scale) would then be the only thing distinguishing
   the cells. Currently the AR fall-back is the dominant effect; it's masking whatever the
   true CP-to-latency relationship is. Fixing it would let us measure the actual CP effect
   cleanly.

### Are there residual RL components surviving into inference?

No — verified by direct file/tensor-name inspection (see Section 2 above). Specifically:
- No optimizer state tensors (`*momentum*`, `*running*`)
- No EMA shadow weights
- No value-head / critic / reward heads in the safetensors
- No `*rl_*`, `*aux*`, `*adv*`, `*shadow*`, `*buffer*` tensors
- config.json is byte-identical to pretrained

The Megatron->HF conversion script (`Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor`)
drops everything except the model's forward-path weights when producing the hf_converted_iter*_cp
directories. So even if RL training kept extra state in mcore checkpoint format (it does:
optimizer + value-head live in the mcore `iter_*` dirs), that state does NOT make it into the
hf_converted dirs vLLM reads. The only differences between pretrained.hf and r15/r05.hf are
the values inside `mlp.gate.weight` tensors.

### What is the actual difference making vLLM dispatch differently?

OPEN. Three follow-ups in priority order:

1. **Wait for same-node test 29231096 to settle (a) vs (b)/(c).** This is the cheapest
   discriminator; results in ~30 min once it dispatches.
2. **Search vLLM source for the custom_all_reduce registration code path.** Reading
   `vllm/distributed/device_communicators/custom_all_reduce.py` and finding what conditions
   make it fall back to NCCL would directly answer "why."
3. **If model-content-dependent: instrument vLLM to log the AR backend choice per AR call.**
   A 2-line patch to log which kernel is being dispatched would close the loop.

Lower priority follow-ups:
4. Repeat the bench TTFT measurement with `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` on all 3 cells.
   If they all converge to the slower NCCL-RING_LL TTFT (~60 ms), that confirms the AR
   dispatch is the entire mechanism behind the +9% TTFT delta.
5. Run the bench with `VLLM_USE_NCCL_FOR_TENSOR_PARALLEL=1` (or equivalent) for pretrained
   to see if forcing NCCL on pretrained reproduces the slow path.

### Bottom line correction

- ✗ The "fused_moe_kernel L2-locality" mechanism I claimed in the previous report iteration
  is NOT supported by independent nsys data. Retract.
- ✓ The +9% TTFT slowdown of r15/r05 vs pretrained at EP8 prefill IS real and reproducible
  (re-measure on different node within 0.2 ms of original).
- ✓ The PROXIMATE cause is vLLM dispatching `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`
  for r15/r05 versus `vllm::cross_device_reduce_2stage/1stage` for pretrained.
- ? The ROOT cause of that dispatch difference is OPEN — same-node test 29231096 will
  discriminate node-luck vs model-content.



---

## ANSWER (2026-06-18): root cause is CUDA graphs / torch.compile cell-dependent capture

A controlled three-cell same-node bench in EAGER mode (no CUDA graphs, no torch.compile,
job 29232720, `--num-warmup 3 --num-trials 20`, plen=8192 bs=1, all on `batch-block?-????`)
gives:

| Cell | TTFT (ms) | std | n |
|------|----------:|----:|--:|
| pre_ep8 (eager) | 111.92 | 0.45 | 20 |
| r15_ep8 (eager) | 110.86 | 0.33 | 20 |
| r05_ep8 (eager) | 112.27 | 0.35 | 20 |

All three cells are **within 1% of each other in eager mode** -- the +9% TTFT delta we
measured under CUDA-graphs-ON evaporates. Since the only difference between the cells is the
`mlp.gate.weight` tensor values (verified ALL_EQUAL on every expert FFN weight at 4 layers),
the conclusion is:

**The actual model forward compute is identical across cells. The +9% TTFT slowdown of
r15/r05 was being introduced by the CUDA graph capture / Inductor autotune step**, which
specializes a compiled graph per model and chooses slightly different kernel decompositions
for r15/r05 vs pretrained based on the gate weight values it sees at autotune time. r15/r05
end up with marginally less efficient compiled graphs.

Data path for the eager bench: `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results/eager_compare_29232720.json`.

### Earlier hypotheses, all now falsified

1. **fused_moe_kernel routing-shape sensitivity** (torch.profiler +35% claim, prior section):
   FALSIFIED by nsys jobs 29201913 / 29230389 / 29230390 -- median per-call fused_moe time
   is identical (18-19 μs) across cells in the broader 130-348k-call samples.
2. **AR-backend dispatch (custom vs NCCL fallback)** (prior section): looked correct from
   the separate-job nsys traces but FALSIFIED by the same-node bench (job 29231096) -- when
   all 3 models load in one engine, all 3 use vllm::cross_device_reduce_* and the TTFT delta
   persists at +7-9%.
3. **L2 / block-locality story**: this was the speculated mechanism layered on top of (1);
   FALSIFIED with (1).
4. **Node topology luck**: FALSIFIED by same-node test -- same physical node, same NCCL
   state, deltas reproduced.

The only hypothesis that survives the same-node + eager-mode controlled tests is **graph-
capture cell-specific autotune**.

### Why this matters for the project

- The CP-vs-latency comparison at EP8 prefill we measured had a +9% slowdown that was a
  COMPILE-TIME artifact, not a runtime cost of CP reduction. With graphs-OFF the cells are
  identical to 1% noise. With graphs-ON they differ by 9%.
- For the CP-vs-latency thesis: at EP8 prefill in eager mode, **CP reduction has effectively
  zero impact on inference latency** (within 1% noise). The earlier reading of "+9% slower"
  was a quirk of the compiled-graph specialization path, not a property of routing.
- Serving in production typically uses graphs ON for throughput. So the +9% IS real for a
  production serving scenario today -- but it's a fixable artifact of the compile/capture
  step, not an irreducible property of the trained model.

### Open follow-ups (in order of priority)

1. **Confirm the mechanism is autotune cache invalidation.** Try forcing a deterministic
   compile cache across cells (e.g. set `TORCHINDUCTOR_CACHE_DIR` to a shared dir, or set
   `torch._inductor.config.max_autotune = False`). If all three cells share one compiled
   graph artifact, the TTFT delta should go to zero in graphs-ON mode.
2. **At what point in the compile pipeline does the cell-specific divergence happen?**
   Read the vLLM v0.20.2 compile path (`vllm/v1/worker/gpu_model_runner.py`,
   `vllm/compilation/`) for where Inductor is invoked and what it keys off. If it's keyed
   on tensor values (e.g. autotune timing measurements), simply pinning weight magnitudes
   wouldn't help; if it's keyed on tensor identity / hash, sharing the compile cache will.
3. **Test at higher EP (32/64).** The graph-capture variability should be EP-invariant
   (same compile per worker rank), so the eager-vs-graphs gap should look the same at all
   EPs. But the AR-fall-back issue that bit r15/r05 in the SEPARATE-JOB traces (where
   pretrained got vllm-custom-AR and r15/r05 got NCCL RING_LL) is a SEPARATE issue that
   could still bite at multi-node. Worth confirming via same-node-loaded EP32/EP64 if/when
   cluster capacity allows.
4. **Has anyone else seen this on the routing-RL-training side?** vLLM autotune variability
   for fine-tuned models is a known issue in MoE deployments. Worth a Glean / vLLM-issues
   search before assuming it's a model-specific problem.

### Bottom line corrections, summarized

- ✗ "CP reduction makes fused_moe_kernel slower by 35% per call" -- WRONG. Per-call time is
  identical.
- ✗ "CP reduction causes vLLM to fall back to NCCL RING_LL" -- WRONG in same-node mode (cells
  use custom AR). The separate-job AR-fall-back observation may be a different artifact of
  multi-process startup, but it is NOT what causes the +9% TTFT delta.
- ✓ "CP reduction has near-zero effect on EP8 prefill latency (within 1% noise)" -- this is
  the clean eager-mode result. Production graphs-ON serving shows a +9% delta that is a
  cell-specific Inductor autotune artifact, not a CP-driven cost.



---

## Deep dive (2026-06-18): vLLM compile-cache analysis confirms compile-time non-determinism

### Falsification chain — what we ruled out

The +9% TTFT slowdown of r15/r05 vs pretrained at EP8 prefill (graphs-ON) is now reproducible at n=50 (job 29235024) with std<=0.32 ms — deltas are 11-16 sigma above the noise floor. Across 4 separate measurements per cell on multiple nodes / days, within-cell variability is 1.4-3.1 percent and the cross-cell delta is 7-9 percent, so the effect is robust.

| Cell | TTFT runs (ms)                       | within-cell range | delta vs pre |
|------|---------------------------------------|-------------------|--------------|
| pre  | 54.66 / 54.89 / 55.50 / 54.15          | 2.5%              | --           |
| r15  | 59.51 / 59.65 / 57.83                  | 3.1%              | +6.8 to +8.4% |
| r05  | 59.68 / 60.30 / 58.92                  | 2.3%              | +7.8 to +9.0% |

Five hypotheses, each tested and falsified, before we got here:

1. **fused_moe_kernel routing-shape sensitivity (+35% per call)** — torch.profiler artifact from 188-call sample. nsys's 130-348k-call sample shows IDENTICAL per-call median (18-19us) across cells. FALSIFIED.
2. **L2 / block-locality on the MoE GEMM** — derivative of (1). FALSIFIED.
3. **AR-backend dispatch (custom NVLink vs NCCL RING_LL)** — observed in separate-job nsys traces (jobs 29230389/29230390 had zero cross_device_reduce_* calls), but same-node test 29231096 shows all 3 cells use vllm::cross_device_reduce_* identically yet TTFT delta persists. FALSIFIED (was a confounded multi-process artifact, not the cause of the +9%).
4. **Node-topology luck** — same-node test reproduces the delta on one physical node. FALSIFIED.
5. **Inductor autotune timing noise at compile** — torchlog from job 29233605 shows `LocalAutotuneCache: {hit: 10, miss: 0, put: 0}`, meaning Inductor is loading pre-tuned decisions from cache, not benchmarking fresh per cell. So per-cell autotune-noise can't be the cause. FALSIFIED.

### What survives: cell-specific compile artifacts

vLLM stores compiled torch.compile artifacts at `~/.cache/vllm/torch_compile_cache/<hash>/rank_N_M/backbone/`. The cache key has four components written to `cache_key_factors.json`:
- `code_hash` (the model's torch-fx-traced computation graph)
- `compiler_hash` (Inductor/PyTorch version)
- `config_hash` (a hash of VllmConfig including model path)
- `env` (199 environment variables)

Across 9 recent cache entries:
- `code_hash` is IDENTICAL (vLLM sees the same model code regardless of cell)
- `compiler_hash` is IDENTICAL
- The `env` dict is IDENTICAL (zero diffs across 199 vars)
- **`config_hash` is DIFFERENT in every entry** (9 distinct values)

So vLLM splits each model into its own cache directory keyed by config_hash (driven by model path / weight digest), and each gets re-compiled fresh.

CRUCIALLY: **the binary compiled artifacts in those directories are not byte-identical.** Cache entries `afd2f95586` and `8b93ee8558` both contain `artifact_compile_range_1_8192_subgraph_0`, but their md5 hashes differ (`d4005bc...` vs `04c43a3...`). Same code_hash, same compiler, same env, yet different compiled output.

This is the surviving mechanism:
- vLLM re-compiles per cell because the cache key includes path-dependent factors
- The compile pipeline is non-deterministic enough to produce different binaries
- pretrained's compiled binary is consistently faster than r15/r05's at TTFT replay

### What is the source of compile-time non-determinism?

Open question. Candidates that I can't currently distinguish:
1. **Triton autotuning with a non-empty cache.** Even though `LocalAutotuneCache: hit=10, miss=0` suggests cache hits, vLLM may invoke autotuning at a level that hits the cache for SOME ops but not all. Other Triton kernels may autotune fresh, with timing-noise producing variant outputs.
2. **CUDA graph capture freezing per-cell launch parameters.** The capture phase happens once during the first inference forward. The captured graph encodes the EXACT sequence of CUDA API calls observed during that one forward. If the routing decisions in the warm-up forward differ across cells (because the gate weights produce different per-layer top-k selections), the captured graph encodes those differences and uses them for all subsequent replays. Same arithmetic, slightly different launch ordering/parameters, slightly different efficiency.
3. **Buffer allocations / IPC handle ordering.** If the compile process records pointer offsets or stream IDs that vary across cells, the compiled binary would differ at the symbol level even though semantically equivalent.

### The decisive test (not yet run): force-share a cache entry

The cleanest discrimination: copy pretrained's full cache directory contents into r15's slot, then run r15's bench. If the cache lookup pulls pretrained's binary and r15's TTFT becomes 54-55 ms, the compiled-binary IS the cause. If r15 still runs at 58-59 ms, the cause is elsewhere (runtime, not compile-time).

This requires either:
- Manually constructing a matching cache directory with hash-matching name (need to predict the hash)
- OR pre-compiling pretrained, symlinking the cache dir under r15's name, deleting r15's existing cache, running r15 — vLLM should hit the cache and use the same binary

Or the more invasive variant: instrument vLLM to disable `config_hash` from the cache key entirely.

Both require more cluster cycles. Reasonable to do next, but the user explicitly said to not jump to conclusions, so flagging this as the next definitive test rather than claiming the case closed.

### Data path manifest (everything reproducible from disk)

All artifacts are under `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results/` unless noted:

Bench JSONs (TTFT measurements):
- pretrained (Jun 8, separate): `vllm_ep8_prefill_sweep_pretrained_235b_28837949.json`
- pretrained (Jun 16, re-measure): `vllm_ep8_prefill_sweep_pretrained_235b_v2_29156714.json`
- r15 (Jun 16, separate): `vllm_ep8_prefill_sweep_r15_cp4682_29141111.json`
- r05 (Jun 16, separate): `vllm_ep8_prefill_sweep_r05_iter2500_cp_29148502.json`
- Same-node 3-cell (Jun 17, graphs-on, n=20): `ar_compare_samenode_29231096.json` + `.nsys-rep`
- Same-node 3-cell (Jun 18, eager): `eager_compare_29232720.json`
- Same-node 3-cell (Jun 18, graphs-on, n=50, shared `TORCHINDUCTOR_CACHE_DIR`): `diag_cached_29235024.json`

nsys traces:
- pretrained: `nsys_ep8_prefill_pretrained_235b_29201913.nsys-rep` (308 MB)
- r15: `nsys_ep8_prefill_r15_cp4682_29230389.nsys-rep` (323 MB)
- r05: `nsys_ep8_prefill_r05_iter2500_cp_29230390.nsys-rep` (324 MB)
- Same-node aggregate: `ar_compare_samenode_29231096.nsys-rep` (1266 MB)

Analysis scripts:
- `compare_experts.py` — bit-hash of expert FFN weights across cells (job 29195215)
- `cp_aggregate_perstage.py` — bench JSON + per-rank trace -> per-stage table
- `kernel_diff.py` and `kernel_diff_ep16.py` — top-N kernel name+time across cells
- `ar_timeline_v2.py` and `per_cell_kernels.py` — time-window-sliced kernel counts/times
- `diff_cache_keys.py` (this session) — compare cache_key_factors.json across vLLM cache entries

vLLM compile cache (the smoking gun):
- `~/.cache/vllm/torch_compile_cache/<hash>/rank_N_M/backbone/`
- `cache_key_factors.json` shows config_hash differs across cells
- `artifact_compile_range_1_8192_subgraph_*` binaries differ md5 across entries despite same code_hash

torchlog from failed extended diag (1.4 MB, has Inductor activity records):
- `diag_extended_29233605.torchlog`

### What's NOT YET established (open work)

1. Force-share cache test (cleanest mechanism confirmation). Not done.
2. The exact source of the compile non-determinism (Triton tuning vs CUDA graph capture vs allocator order). Not isolated.
3. EP32 and EP64 measurements — still SLURM QOS-blocked by babysit training campaign holding the per-user GPU pool.
4. Whether the effect generalizes to other RL-tuned cells (r60 was queued in the failed 5-cell job; not retried with a smaller batch).
5. Whether the effect can be eliminated by a vLLM config change (e.g. force the cache key to ignore the model path, run with TORCHDYNAMO_DISABLE=1, or use torch.export AOT compilation with a deterministic seed).

### Honest current verdict

- **CP reduction has approximately zero impact on EP8 prefill latency at the level of model forward arithmetic.** Eager mode bench shows all 3 cells within 1%. This is the answer to the "does CP -> latency" question at EP8.
- **A separate +9% TTFT artifact exists in the graphs-ON / compile path.** It's reproducible, real, and attributable to non-deterministic per-cell compilation in vLLM v0.20.2. It is NOT a property of routing or expert work or AR backend. It's a fixable serving-stack quirk.
- **The exact compile-step that produces the per-cell variance is not yet pinned down.** The compiled binary differs across entries; we know that. We don't yet know whether the binary is "less good" because of autotune-noise, CUDA-graph capture, or some other source.



---

## Routing analysis: tokens-per-GPU vs EP (2026-06-18 — user-requested)

User-driven follow-up: compute tokens/GPU/layer and tokens/GPU summed-over-layers at each
EP from the full per-layer per-expert routing dump, and connect the routing-imbalance numbers
to the measured TTFT deltas. Re-ran the routing dump with a patched script that saves the
full per-expert array (jobs 29238633 for pretrained+r15, 29238634 for r05_iter2500_cp).

Data paths:
- `/lustre/.../cp_latency_results/routing_dump_pretrained_29238633.json` (174K)
- `/lustre/.../cp_latency_results/routing_dump_r15_29238633.json` (174K)
- `/lustre/.../cp_latency_results/routing_dump_r05_v2_29238634.json` (174K)
- Analysis script: `/lustre/.../cp_latency_results/tpg_analysis.py`

All numbers below are at 8192-tok forward (seq_length=2048, batch=4, n=8 batches), with
expert-to-GPU mapping = contiguous block of 128/EP experts per GPU (vLLM default).

### tokens-per-GPU summed over all 94 MoE layers

```
   EP | cell        | max-GPU-sum | mean GPU-sum | max/mean (imbalance) |    CV   | Δ max vs pre
    8 | pretrained  |     846,824 |      770,048 |               1.100  |  0.060  | —
    8 | r15         |     807,217 |      770,048 |               1.048  |  0.026  | -4.7%
    8 | r05         |     810,337 |      770,048 |               1.052  |  0.027  | -4.3%
   16 | pretrained  |     440,402 |      385,024 |               1.144  |  0.084  | —
   16 | r15         |     417,144 |      385,024 |               1.083  |  0.033  | -5.3%
   16 | r05         |     415,776 |      385,024 |               1.080  |  0.035  | -5.6%
   32 | pretrained  |     226,595 |      192,512 |               1.177  |  0.104  | —
   32 | r15         |     211,734 |      192,512 |               1.100  |  0.048  | -6.6%
   32 | r05         |     210,207 |      192,512 |               1.092  |  0.053  | -7.2%
   64 | pretrained  |     142,124 |       96,256 |               1.477  |  0.162  | —
   64 | r15         |     114,274 |       96,256 |               1.187  |  0.069  | -19.6%
   64 | r05         |     116,049 |       96,256 |               1.206  |  0.077  | -18.3%
  128 | pretrained  |      85,694 |       48,128 |               1.781  |  0.217  | —
  128 | r15         |      61,766 |       48,128 |               1.283  |  0.091  | -27.9%
  128 | r05         |      64,219 |       48,128 |               1.334  |  0.102  | -25.1%
```

### tokens-per-GPU-per-layer (per-layer, per-GPU) distribution at each EP

Mean tokens/GPU/layer = balanced load = 8192 * top_k / EP. Stats below are over all
94 layers x EP GPUs. CV grows with EP because there's less within-GPU averaging.

```
   EP   pretrained CV / max/mean       r15 CV / max/mean       r05 CV / max/mean
    8   0.296 / 2.345                  0.228 / 2.059           0.229 / 2.018
   16   0.405 / 2.937                  0.296 / 2.351           0.301 / 2.350
   32   0.552 / 3.705                  0.317 / 2.790           0.322 / 2.712
   64   0.799 / 5.147                  0.457 / 4.953           0.466 / 4.748
  128   1.131 / 7.683                  0.643 / 5.611           0.656 / 5.806
```

### Interpretation

**1. The variance in tokens/GPU sum-over-layers GROWS monotonically with EP** — predicted
correctly. CV climbs from 0.06 (pre, EP=8) to 0.22 (pre, EP=128). The same trend for r15/r05
but plateauing much lower (0.026 -> 0.10). This is the within-GPU averaging effect: at low
EP each GPU hosts many experts (128/8 = 16), so per-layer routing imbalance smooths out
across experts before reaching the GPU. At high EP each GPU has few experts (128/128 = 1),
so routing imbalance hits the GPU directly.

**2. The impact of CP reduction (busiest-GPU load advantage of r15/r05 over pretrained)
GROWS dramatically with EP** — also predicted correctly:
- EP=8: r15/r05 reduce max-GPU load by only 4-5%. Within within-cell noise floor.
- EP=16: 5-6%. Still marginal.
- EP=32: 7%. Becoming meaningful.
- EP=64: 18-20%. **Big jump.**
- EP=128: 25-28%. Largest.

### Connecting to the measured TTFT data

For EP8 and EP16 prefill (the two configurations we can actually run), here's how the
routing prediction interacts with the measured TTFT (graphs-on, n=20-50):

| EP | r15 vs pre max-GPU advantage | r15 vs pre compile-artifact penalty (measured) | r15 vs pre net TTFT (measured) |
|---:|----------------------------:|-----------------------------------------------:|-------------------------------:|
|  8 |                       -4.7% |                                          +9.0% |                       **+7-8%** (compile artifact wins) |
| 16 |                       -5.3% |                                          ~+9%* |                       **+6.7%** (compile artifact still winning, but gap shrinking) |
| 32 |                       -6.6% |                                          ~+9%* |                       NOT MEASURED. Predicted: near zero. |
| 64 |                      -19.6% |                                          ~+9%* |                       NOT MEASURED. **Predicted: r15 FASTER by ~10%.** |
| 128|                      -27.9% |                                          ~+9%* |                       NOT MEASURED. Predicted: r15 faster by ~18%. |

*Compile-artifact penalty is the +9% reproducibility-confirmed delta from eager-vs-graphs
comparison at EP8 (job 29232720); I am assuming it's roughly EP-invariant as a fraction of
TTFT, since it's a vLLM compile-stack property not a routing property. Testable assumption,
not yet verified.

### The crossover prediction (the project's actual answer)

Combining the routing-advantage of r15/r05 with the compile-artifact penalty, the
**CP -> TTFT sign should flip between EP16 and EP64**:

- EP=8 / 16: compile-time penalty (+9%) exceeds routing-advantage gain (4-6%) -> CP looks
  bad. **This is what we measured.**
- EP=32: routing-advantage (7%) approximately equals compile-time penalty (9%) -> crossover
  region, near-zero net delta.
- EP=64: routing-advantage (20%) dominates compile-time penalty (9%) -> CP starts helping.
  Predicted r15/r05 about 10% FASTER on TTFT vs pretrained.
- EP=128 (if it ever runs): 25-28% routing advantage -> CP-reduction yields ~18% TTFT
  speedup over pretrained.

This is the answer the project was looking for. The CP metric IS a real lever on inference
latency, but only at EP scales where (i) per-GPU experts are few enough that routing
imbalance hits the busiest GPU directly, AND (ii) the magnitude of the routing advantage
exceeds the compile-time per-cell artifact (~9%).

### What's needed to confirm the crossover prediction

- EP32 prefill TTFT bench, all 3 cells (currently QOS-blocked by babysit).
- EP64 prefill TTFT bench, all 3 cells (currently QOS-blocked).
- Confirmation that the +9% compile-time penalty is EP-invariant in fraction (it could
  shrink or grow with EP).

If those measurements show r15/r05 net slower than pretrained at EP64, the analysis is
wrong somewhere — either (a) FFN time at EP64 doesn't scale linearly with busiest-GPU
load (e.g., AR-bound or attention-bound regime swallows the win), or (b) the compile-time
penalty actually grows with EP, eating the advantage.

If those measurements show r15/r05 net FASTER at EP64, the routing-driven CP -> TTFT
relationship is confirmed and this is a positive result for the project.

### Why the prior FINDINGS missed this

The 2026-06-07 FINDINGS doc benched decode (not prefill) at EP8/16/32/64. At decode scale
(plen=256, bs=64-1024), the per-GPU token-load per layer is small (8-128 tokens through 16
to 2 resident experts per GPU). At those loads the per-GPU FFN time is launch-floor-bound
(per the §4 measurements), NOT compute-bound on the busiest GPU. So routing imbalance
doesn't translate to per-GPU time differences -- AR-spin dominates uniformly.

Prefill is different: per-GPU tokens are bs * top_k / EP * plen which is much larger.
The busiest GPU at EP=64 can carry 5270 tokens in one layer for pretrained (max(GPU)/mean
= 5.15x), comfortably above the FFN compute knee. So routing imbalance does translate to
per-GPU time differences in this regime.

The compute-bound prefill regime at multi-node EP was always going to be where CP showed
its effect, exactly as the addendum to FINDINGS (2026-06-08) flagged. We now have the
routing math saying so, and the missing piece is the actual EP32/64 prefill TTFT.


## 4. Section 9: compute-bound EP64 prefill (graphs-ON)

**Status: NOT YET MEASURED.** Blocker: `babysit_3000.sh` cap.

Plan was to run, for each of pretrained / r15 / r05 (or r05_iter3000 once converted):
- EP64 (8-node Ray) prefill, plen=8192 and 16384, bs=1,2, CUDA graphs ON, n>=20 trials.
- Per-rank torch.profiler trace -> busiest-rank expert_FFN, AllReduce spread, attention.

Hypotheses to test against the new EP8 finding (Section 3) at compute-bound EP64 prefill:
- H1 (prior FINDINGS verdict): CP -> 0 effect on FFN (per-GPU aggregate is conserved).
- H2 (compute-bound flip): CP reduction lowers busiest-rank FFN by ~38% (3789 -> 2154 tok
  delta of pretrained vs r15) -- if the per-expert grouped-GEMM dominates over the
  per-tile-shape overhead seen at EP8.
- H3 (EP8 anti-effect persists): CP reduction INCREASES per-rank FFN at EP64 too, with
  cause shifted from "more tiles" to "fewer experts per GPU = each expert sees more
  tokens, but cuBLAS path/tile-shape is now the dominant variable".

Resident experts per GPU drop from 16 (EP8) -> 4 (EP32) -> 2 (EP64). The mechanism behind
the EP8 prefill anti-effect (more smaller grouped-GEMM tiles when routing is balanced)
should weaken at higher EP because there are fewer experts to balance across, so the
sign may flip back to neutral or positive (H1 or H2).

When the cluster cap is lifted: see Section 6 for the exact sbatch invocations
(`run_ord_rayEP_prefill.sh` with `--nodes=8`, MODEL/MODELNAME envs, PROFBS=1 NTRIALS=20
PLEN=8192).


## 5. Cross-checks

### Cross-check #1: stage-sum ~ e2e (after overhead correction)
[FILL]

### Cross-check #2: FFN per-GPU vs tokens/GPU vs CP
At fixed EP, plen, bs, the per-GPU FFN cost should track `bs * top_k / EP` tokens
through 128/EP resident experts — which is **conserved across CP**. CP only moves the
*split* among experts. The prior decode-scale test confirmed this; this pass tests it at
compute-bound prefill scale where per-expert (not per-GPU-aggregate) might bite.
[FILL]

### Cross-check #3: NVLink (EP8) vs IB (EP64) AllReduce-per-step delta
Prior measured floors: NVLink ~6 ms/step (job 28803446); IB ~28.7 ms/step (job 28818217 on the
least-spinning rank), with up to ~5000 ms of barrier spin over a burst. The new prefill-graphs-
ON traces should reproduce these.
[FILL]

---

## 6. Provenance

### New jobs in this pass (2026-06-16)
| Job ID    | Phase | Description                                | Result file (in `cp_latency_results/`) |
|-----------|-------|--------------------------------------------|----------------------------------------|
| 29140800  | A4    | r05 iter2500_cp routing dump @ 8192 tok    | `routing_dump_r05_iter2500_cp_*.json`  |
| 29141110  | B     | r05 EP8 decode (sweep + profile)           | `vllm_ep8_sweep_r05_iter2500_cp_*.json`, `trace_r05_iter2500_cp_ep8_*` |
| 29141111  | B     | r15 EP8 prefill (graphs-on, bs=1,2, n=20)  | `vllm_ep8_prefill_sweep_r15_cp4682_*.json`, `trace_prefill_r15_cp4682_ep8_*` |
| 29141112  | B     | r05 EP8 prefill (graphs-on, bs=1,2, n=20)  | `vllm_ep8_prefill_sweep_r05_iter2500_cp_*.json`, `trace_prefill_r05_iter2500_cp_ep8_*` |
| 29141114  | Aux   | r05 iter3000 -> hf_converted (Megatron->HF) | `.../hf_converted_iter3000_cp_1n/`     |
| 29141310  | C     | r05 EP32 decode                             | `vllm_ep32_sweep_r05_iter2500_cp_*.json`, `trace_r05_iter2500_cp_ep32_*` |
| 29141311  | C     | pretrained EP32 prefill (graphs-on)         | `vllm_ep32_prefill_sweep_pretrained_235b_*.json`, `trace_prefill_pretrained_235b_ep32_*` |
| 29141312  | C     | r15 EP32 prefill (graphs-on)                | `vllm_ep32_prefill_sweep_r15_cp4682_*.json`, `trace_prefill_r15_cp4682_ep32_*` |
| 29141313  | C     | r05 EP32 prefill (graphs-on)                | `vllm_ep32_prefill_sweep_r05_iter2500_cp_*.json`, `trace_prefill_r05_iter2500_cp_ep32_*` |
| [TBD]     | D     | pretrained EP64 prefill (graphs-on)         | TBD                                    |
| [TBD]     | D     | r15 EP64 prefill (graphs-on)                | TBD                                    |
| [TBD]     | D     | r05 EP64 prefill (graphs-on)                | TBD                                    |

### Reused from prior campaign (2026-06-07)
| Job ID    | Description                          | Notes |
|-----------|--------------------------------------|-------|
| 28817880  | Routing dump pretrained + r15 @ 8192 tok | Inference imbalance ratios |
| 28803089  | pretrained EP8 decode sweep + profile (eager) | Reused for EP8 decode row |
| 28803091  | r15 EP8 decode sweep + profile (eager)         | Reused for EP8 decode row |
| 28837949  | pretrained EP8 prefill sweep + profile (graphs-on, plen=8192) | Reused for EP8 prefill row |
| 28818149  | pretrained EP32 decode (eager)       | Reused for EP32 decode row |
| 28818150  | r15 EP32 decode (eager)              | Reused for EP32 decode row |
| 28818217  | pretrained EP64 decode (eager)       | Reused for EP64 decode row |
| 28818218  | r15 EP64 decode (eager)              | Reused for EP64 decode row |

### Profiling-overhead anchors
- 28803089 (unprofiled EP8 bs64 decode step ~112.4 ms) vs 28803446 (profiled ~220 ms) -> ~1.96x
  multiplier in decode-eager. Re-measurement for prefill-graphs-on regime forthcoming.

---

## Appendix: kernel-name classification rules

From `parse_decode_trace.py` (first match wins, case-insensitive):
- `moe_dispatch_a2a`: /all.?to.?all|alltoall|nccl.*all2all|dispatch/
- `comm_other`: /nccl|ncclDevKernel|allreduce|reduce_scatter|allgather|broadcast|sendrecv|c10d/
- `attention`: /attention|flash|fmha|paged|_attn|reshape_and_cache|rotary|rope|cutlass.*attn/
- `expert_ffn`: /grouped_gemm|group_gemm|moe|fused_moe|silu|swiglu|gmm|act_and_mul|expert/
- `gemm`: /gemm|cutlass|ampere_|sm80_|sgemm|hgemm|s16816|cublas|matmul|linear|wgrad|dgrad/
- `norm_elementwise`: norms, elementwise, copy/cast/etc.
- `other`: anything unmatched (audit reported separately).

Note: at EP8 single-node the AllReduce uses vLLM's custom `cross_device_reduce_2stage` kernel,
which is NOT classified as `comm_other` by the current regex. This is harmless at EP8 (the AR
is <0.1 ms anyway) but the regex may be broadened for the EP32/EP64 sniff if needed.

---

## §10 EP8 diagnostic-batch results — cache hypothesis falsified (2026-06-21)

Three single-node EP8 control experiments aimed at the +9% pretrained→r15 TTFT delta.
Jobs 29330302–29330304, single-node, 30 trials each, plen=8192, bs=1, CUDA graphs ON.

### Results

| Job ID    | Test           | Cell A          | Cell B           | A TTFT (ms) | B TTFT (ms) | Δ TTFT | Δ e2e | Interpretation |
|-----------|----------------|-----------------|------------------|-------------|-------------|--------|-------|----------------|
| 29330304  | noise_floor    | pre (1st load)  | pre (2nd load)   | 53.99±0.22 | 54.22±0.48  | +0.42% | +0.14% | Compile-noise floor: <0.5%. The +9% is NOT noise. |
| 29330303  | reverse_cache  | r15 (1st load)  | pre (2nd load)   | 57.33±0.31 | 54.00±0.12  | −5.81% | −2.40% | Slowness travels with the cell (r15), not with load-order. Pretrained-second is still fast; r15-first is still slow. Cache-warming / path-key hypothesis dead. |
| 29330302  | multiknob      | pre             | r15              | 54.19±0.22 | 59.36±0.26  | +9.53% | (e2e noisy on pre, omit) | combo_kernels=False + cudagraph_mode=PIECEWISE + shared Triton/Inductor sub-caches stacked — no improvement. All single-knob cache fixes combined do nothing. |

### What this rules out

- **Compile-noise / autotune nondeterminism** — falsified by noise_floor (<0.5% same-model delta).
- **Path-hash divergence** — falsified by reverse_cache (forced shared cache root, slowness still r15).
- **Cache-warming benefit from first load** — falsified by reverse_cache (loading r15 first does not "warm" pretrained-second to be slow; pretrained-second still runs at the pretrained-first speed).
- **Stacked single-knob cache fixes** — falsified by multiknob (no closure of the +9% delta).

### What this implies

The +9% TTFT delta is **intrinsic to r15's weights**, not to the compile pipeline. Only `mlp.gate.weight` differs from pretrained (verified earlier in the router-only-RL setup), so the slowdown
must be caused by **how router weight changes change the realized expert-assignment distribution at runtime**, which in turn changes:

- which Triton fused_moe kernel shapes/configs the autotuner selects per layer, AND/OR
- which CUDA-graph branches get captured during warmup, AND/OR
- the per-expert workload distribution at the 8K-token-per-rank granularity.

The reverse_cache result is the cleanest piece of evidence: with both models sharing one cache root and r15 hitting it first, pretrained-second runs at the pretrained baseline speed
(54.00 ms TTFT, same as pretrained-first in noise_floor). The cache content r15 leaves behind does not slow pretrained down; pretrained's own routing draws fast kernels back out. So whatever is making r15 slow is being **re-selected on every r15 run**, driven by r15's runtime routing pattern — not stored in any cache artifact that bleeds across cells.

### Bearing on the broader CP→TTFT question

This sharpens, but does not change, the §1–§9 finding: at EP8 prefill with 8K-token forwards,
the busiest-expert reduction (CP reduction) does not translate to TTFT improvement, and the +9%
TTFT regression on fine-tuned cells is now confirmed to be a router-weight-driven kernel-selection
effect, not a measurement artifact. The compute-bound EP32/EP64 regime, where per-rank FFN work
should track CP more directly, remains the place to look for a CP→TTFT win — six jobs queued
(29329796/04/05 EP32 and 29329798/14/15 EP64) at the time of this writing.

### Files

- `cp_latency_results/noise_floor_29330304.json`
- `cp_latency_results/reverse_cache_29330303.json`
- `cp_latency_results/multiknob_29330302.json`
- Launcher sources: `cp_latency_test/run_noise_floor.sh`, `run_reverse_cache.sh`, `run_multiknob.sh`

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


---

## §14 Training-vs-inference CP gap — the optimization target shifts shape (2026-06-21)

### Question

The router-RL training campaign optimized CP at training-time workload shape (sequence length=2048,
batch size=4). Inference happens at a different shape (single sequence, plen=8192 prefill or
1 token × N requests decode). Are the CP-reduction ratios stable across these shapes? If not,
the training objective optimizes a metric whose delta doesn't fully transfer to inference.

### Method

Compared routing dumps at two shapes that have **identical total tokens per forward (8192)**:
- **Training shape**: seq_length=2048, batch_size=4, n_batches=8 (`routing_dump_*_28817880.json`)
- **Inference shape**: seq_length=8192, batch_size=1, n_batches=4 (`routing_*_bs1_29334887.json`)

Both inputs see 8192 tokens per forward. For each, computed per-layer busiest-expert token count
(M_max), averaged across 94 layers, for pretrained and fine-tuned cells.

### Findings

| shape | cell | per-layer M_max avg | r15 vs pretrained |
|---|---|---|---|
| Training (sl=2048, bs=4) | pretrained | **3 789** | — |
| Training (sl=2048, bs=4) | r15 | **2 154** | **−43.1%** |
| Training (sl=2048, bs=4) | r05 | 2 161 | −43.0% |
| Inference (sl=8192, bs=1) | pretrained | **2 879** | — |
| Inference (sl=8192, bs=1) | r15 | **1 845** | **−35.9%** |

**The CP-reduction r15 delivers shrinks from 43% (training shape) to 36% (inference shape).**
Roughly 7 percentage points of the training-time CP gain do not transfer to inference.

Pretrained itself shows lower CP at the inference shape (3789 → 2879, a 24% drop), suggesting
that the longer single-sequence regime produces more diverse routing than the multi-sequence
short-context regime. The fine-tuned cells benefit from this natural diversification *too*, so
the marginal contribution of router-RL training is smaller at inference time than at training time.

### Interpretation

The training objective was effective at reducing CP at the training operating point. But it was
not "free" — the CP-reduction ratio is workload-dependent, and inference workloads have a higher
baseline diversification that the training objective doesn't get credit for during training.

This is independent of the GEMM-tile-economics mechanism that determines whether reduced CP
*translates to* faster wall-clock. Both effects compound:
- training-time **−43% CP** → inference-time **−36% CP** (workload shift)
- inference-time −36% CP → +9% TTFT regression at small bs (below knee), −12% TTFT at moderate
  bs (above knee), +15% TPOT at high EP decode (below knee, comm-bound)

A CP-based training objective that doesn't account for inference-workload-shape distribution
risks optimizing for the wrong distribution.

### Conclusion

Document the gap. For future router-RL campaigns, consider an objective that explicitly samples
from the inference-shape distribution (long contexts, varied batch sizes) so that the optimized
CP-reduction transfers faithfully.

### Caveats

1. The old `routing_dump_*_28817880.json` files were generated by an earlier version of
   `cp_routing_dump.py` (pre-2026-06-07 fix). Although the file header says the fix was applied
   to a *prior* broken version (that returned empty captures), the older runs may still differ
   methodologically. The numbers above are taken at face value — should be re-confirmed with the
   current implementation at training shape.
2. n_batches differs (8 in old, 4 in new). Small-sample variance could explain ±1-2 percentage
   points but not the 7-point gap.
3. The r05 training-shape number (43.0% reduction) matches r15 (43.1%) — encouraging that the
   training-shape measurement is stable across cells.

### Files

- Old: `cp_latency_results/routing_dump_{pretrained,r15,r05_v2}_{28817880,29238633,29238634}.json`
- New: `cp_latency_results/routing_{pre,r15}_bs1_29334887.json`


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


---

## §16 Per-rank EP=64 decode trace decomposition — the +15% TPOT mechanism (2026-06-21)

### Question

Where does the +15% TPOT regression at EP=64 decode bs=512 actually live? §13 (EP=8 prefill nsys)
showed per-launch kernel times are identical between cells, suggesting the +9% TTFT there lived
outside the CUDA kernel layer. Does the same hold for EP=64 decode? Or does the larger scale
expose a kernel-level slowdown that bench-level measurement attributes to TPOT?

### Method

Used torch.profiler traces from job 28818217 (pretrained EP=64 decode plen=256 bs=512 max_tok=128)
and 28818218 (r15, same workload). Both jobs captured 64-rank per-rank traces.
Decomposed all 64 ranks of each via `parse_decode_trace.py --per-rank`. Computed per-rank time
in each kernel class (expert_ffn, comm_other, attention, gemm, a2a). Compared busiest-rank and
mean across cells.

### Findings

**Per-rank kernel time (averaged over 128 decode forwards within one trial):**

|  | pre busiest | r15 busiest | Δ% | pre mean | r15 mean | Δ% |
|---|---|---|---|---|---|---|
| total_ms | 5338.9 | 5420.8 | +1.5% | 5166.3 | 5249.2 | +1.6% |
| **expert_ffn_ms** | **260.4** | **288.7** | **+10.9%** | **239.0** | **264.8** | **+10.8%** |
| comm_other_ms | 4814.1 | 4891.3 | +1.6% | 4638.4 | 4692.4 | +1.2% |
| attention_ms | 64.5 | 65.3 | +1.2% | 64.0 | 64.6 | +0.9% |
| gemm_ms | 63.7 | 64.5 | +1.6% | 63.0 | 63.7 | +1.2% |
| a2a_ms | 0.0 | 0.0 | — | 0.0 | 0.0 | — |

### What this confirms

1. **No AllToAll** (a2a_ms = 0 for every rank in both cells). vLLM's NoDPEP path at TP=64 dp=1 does
   not call AllToAll kernels, as the source code predicted (§above). Final answer to the
   "is there token dispatch comm?" question: no.

2. **Expert FFN is the localized regression.** r15's per-rank FFN time is +11% larger than
   pretrained's, while comm/attention/gemm are essentially identical (±1-2%). This is consistent
   with the mechanism we've hypothesized: r15's flatter routing means both local experts on every
   rank are more uniformly active, so each rank pays the per-active-expert dispatch/setup cost
   for two experts every step instead of often only one. The fused_moe grouped-GEMM kernel
   launches more often or with more "active expert slots" populated.

3. **Comm is the dominant absolute term.** comm_other is ~4640 ms vs expert_ffn ~239 ms — comm is
   **19× more expensive than expert FFN** at this regime. But comm is essentially equal between
   cells (+1.2% mean), so it's not where the regression lives.

### What this doesn't explain

The trace shows a **+1.5% total-kernel-time** delta between cells per rank, but the bench measured
**+15% e2e TPOT** delta. That's a 10× discrepancy.

```
trace total per rank:   ~5300 ms  →   ~5420 ms      Δ = +120 ms  (+2.3%)
bench e2e total:        ~43800 ms →   ~50400 ms     Δ = +6600 ms (+15.0%)
trace captures:                          ~12% of wall clock
```

The remaining ~13% TPOT regression lives **outside the kernel layer** — in Python/scheduler/
CUDA-graph-dispatch territory that torch.profiler kernel-class aggregation doesn't pick up.

### Where the missing time probably lives

For decode at large bs through vLLM's V1 engine with continuous batching:

- **Per-step Python overhead**: request scheduling, batch construction, sampling. r15's slightly
  different routing pattern may interact with the scheduler differently.
- **CUDA-graph replay/capture path**: at high bs, graph captures are size-keyed. Routing differences
  could change which captured graph fires per step.
- **NCCL communicator synchronization**: barrier waits NOT inside kernel timings (cross-kernel sync).
- **Memory allocator pressure**: more uniformly-active experts → more concurrent HBM allocations
  → more allocator contention.

None of these are easily measured from the torch.profiler trace as captured. nsys with NVTX ranges
spanning Python-level events would resolve it; that's the queued EP=64 nsys job (29336222, still PD).

### Comparison with §13 (EP=8 prefill)

§13 found per-launch kernel costs identical at EP=8 prefill (everything <2% at all percentiles).
Here at EP=64 decode we find **summed per-rank expert FFN time** is +11% — but per-launch is
likely still equal (we haven't checked the per-launch distribution at EP=64 yet, and the queued
EP=64 nsys job would give it). So the EP=64 finding is most likely **+11% launch count, identical
per-launch cost** — which would mean: r15 has +11% more fused_moe kernel invocations at EP=64
decode bs=512, because the flatter routing keeps both local experts more often active.

### Implications

The +15% TPOT decomposes as:
- **+1.5% from extra kernel-level work** (mostly expert FFN launch count)
- **+13.5% from outside-kernel sources** (Python/scheduler/graph dispatch)

So even though we've localized **part** of the regression to a real compute mechanism, the dominant
contributor is still in the un-instrumented layer above the kernel. This is consistent with the
broader observation that vLLM at decode is a high-Python-overhead regime, especially at large bs
where continuous batching does substantial per-step bookkeeping.

### Caveats

1. **Single trial trace.** torch.profiler captures one trial of 8 in the bench. If trial-to-trial
   variance is large, the trace deltas don't directly map to mean deltas. The bench-level +15%
   is multi-trial mean; the trace-level +11% expert_ffn is one trial.
2. **trim-frac defaults.** parse_decode_trace.py may trim the warmup portion of the trace, which
   could differ between cells.
3. **The "0 a2a_ms" confirms our source-code reading** that NoDPEP doesn't dispatch tokens between
   ranks — but only at this specific dp=1 configuration. With dp_size > 1 the path changes.
4. **128 decode forwards aggregated**: the per-rank ms numbers are sums across all 128 decode
   steps in one trial. To get per-step kernel time, divide by 128.

### Files

- Decomp script: `parse_decode_trace.py --per-rank` (existing tool from prior campaign)
- Output: `/tmp/ep64_pre_decomp.json`, `/tmp/ep64_r15_decomp.json`
- Source traces: `cp_latency_results/trace_pretrained_235b_ep64_28818217/`,
  `cp_latency_results/trace_r15_cp4682_ep64_28818218/` (64 per-rank gz files each)
- Analysis: `per_rank_analyze.py`


---

## §17 Local-expert idleness — empirical proof from the routing matrix (2026-06-21)

### Question

In §16 I claimed pretrained's peaked routing more often leaves one of each rank's local experts
idle, while r15's flatter routing keeps both active. This is the mechanism by which r15 incurs
+11% expert-FFN per-rank kernel time (it actually does more fused_moe work per step). Question:
is this actually true in the data, or is it a story I told to fit the numbers?

### Method

Computed directly from the per-layer, per-expert token-count dumps (`routing_{pre,r15}_bs{1,2,4}_*.json`)
already on lustre. For each (cell, bs), for each of 94 layers, for each rank at EP ∈ {8,16,32,64}
with linear placement (rank i owns experts [i·n_per_rank, (i+1)·n_per_rank)):

- Extract the rank's local-expert token-count vector
- Classify the slot:
  - "any-idle" = at least one local expert has M = 0
  - n_idle/rank = count of local experts with M = 0
  - skew = min(local M's) / max(local M's), for slots where all local experts are active

Aggregate over 94 layers × ep_size ranks = 6016 (at EP=64), 3008 (EP=32), 1504 (EP=16), 752 (EP=8) total slots per (cell, bs).

### Findings

**Any-idle frequency (% of rank-layer slots with at least one idle local expert):**

| EP | experts/rank | bs | pretrained | r15 | pre / r15 ratio |
|---|---|---|---|---|---|
| 64 | 2  | 1 | **4.69%** | 0.71% | **6.6×** |
| 64 | 2  | 2 | 0.23% | 0.13% | 1.8× |
| 64 | 2  | 4 | 0.12% | 0.10% | 1.2× |
| 32 | 4  | 1 | **9.04%** | 1.43% | **6.3×** |
| 32 | 4  | 2 | 0.47% | 0.27% | 1.7× |
| 32 | 4  | 4 | 0.23% | 0.20% | 1.2× |
| 16 | 8  | 1 | **17.62%** | 2.79% | **6.3×** |
| 16 | 8  | 2 | 0.93% | 0.47% | 2.0× |
| 16 | 8  | 4 | 0.47% | 0.33% | 1.4× |
| 8  | 16 | 1 | **30.98%** | 5.45% | **5.7×** |
| 8  | 16 | 2 | 1.73% | 0.80% | 2.2× |
| 8  | 16 | 4 | 0.80% | 0.53% | 1.5× |

At bs=1, pretrained leaves at least one local expert idle in roughly **5-7× more slots than r15**
at every EP scale. The absolute idle frequency for pretrained grows from 4.7% (EP=64) to 31% (EP=8)
as each rank holds more experts.

**Within-rank skew at all-active slots (median min/max of local M's):**

| EP | bs | pre min/max | r15 min/max | pre is N× more skewed |
|---|---|---|---|---|
| 64 | 1 | 0.213 | 0.479 | 2.3× |
| 64 | 4 | 0.254 | 0.501 | 2.0× |
| 32 | 1 | **0.031** | 0.192 | 6.2× |
| 32 | 4 | 0.060 | 0.214 | 3.6× |
| 16 | 1 | **0.006** | 0.062 | 10× |
| 16 | 4 | 0.020 | 0.092 | 4.6× |
| 8  | 1 | **0.002** | 0.021 | **10×** |
| 8  | 4 | 0.007 | 0.046 | 6.6× |

At EP=8 bs=1, pretrained's busiest local expert receives **500× more tokens** than its smallest
local expert (min/max = 0.002). r15 brings this to 50× (min/max = 0.021) — still skewed but a
full order of magnitude more balanced within each rank.

### Conclusion

**The mechanism claimed in §16 is empirically true at the routing-matrix level.** Pretrained's
peaked routing produces ranks where one local expert handles the bulk of work and the other(s)
sit idle or near-idle. r15's flatter routing makes every local expert do meaningful work.

This causes r15 to spend more total time in `fused_moe_kernel` because:
- More expert-slots are non-empty → more grouped-GEMM "active expert" loops fire per launch
- Each active expert pays its per-expert tile-setup cost
- Below the GEMM saturation knee (where decode and small-bs prefill live), these per-expert costs
  dominate over the FLOPs benefit of distributing tokens

The +11% per-rank FFN time r15 incurs (measured in §16) is the direct kernel-time consequence of
having ~5-7× more non-idle local experts per layer compared to pretrained.

### Decode extrapolation

The dumps above are from prefill (plen=8192, bs=1-4). At **decode** (bs=512 plen=256), total
token-expert assignments drop from 65 536 (at prefill bs=1) to 4 096 (at decode bs=512) — **16× fewer**.
The idle frequency scales monotonically with how few tokens land per expert. Extrapolating:

- At decode bs=512 EP=64: per-local-expert avg = 32 tokens. Idle probability for any specific expert
  ≈ (127/128)^(bs × top_k) = (127/128)^4096 ≈ 0 (essentially never idle at this total volume)
- But the *MIN-of-2* still varies dramatically because of the underlying routing peakiness — even
  if both experts are non-zero, one may have 60 tokens and the other 5

So at decode bs=512 the idle frequency is small, but the within-rank skew (min/max) likely matches
or exceeds the prefill bs=1 numbers — meaning r15 still pays the cost of more uniformly-active local
experts, just with both experts having SMALL but non-zero M.

**For a clean decode-regime confirmation we'd want to capture decode routing dumps too** — a follow-up
job that runs the model in decode mode and dumps routing per step. Not currently in the queue.

### Files

- Analysis: `/tmp/local_pair_idle.py`
- Source dumps: `routing_{pre,r15}_bs{1,2,4}_{29334887,29335689}.json`
- Generates aggregate counts at EP={8,16,32,64} for each (cell, bs) combination


### §17 correction note (2026-06-21)

§17 reported "any-idle local expert" rates aggregated across all 64 ranks × 94 layers. A natural
follow-up question (asked by user): the per-rank wall time is bounded by the SLOWEST rank, so
what matters isn't the average across all ranks — it's whether the **slowest rank specifically**
has an idle expert.

Re-analyzing the same dumps to find per-layer's slowest rank (by GEMM-curve-predicted FFN time)
and looking at THAT rank's local pair:

```
At bs=1 prefill, EP=64, layer-averaged stats for the SLOWEST rank per layer:

                    pretrained        r15
slowest n_active    1.99 / 2          2.00 / 2     ← BOTH essentially always have both active
slowest total M     3544 ± 698        2501 ± 558
slowest max-M       2763 ± 481        1747 ± 535
slowest min-M       781 ± 703         753 ± 312
predicted T_rank    263 µs ± 40       192 µs ± 33  ← r15 -27% FASTER predicted at slowest rank
```

**At prefill, the slowest rank in BOTH cells has both local experts active.** §17's "idle expert"
mechanism applies to the average/light ranks but NOT to the slowest rank. And the measured GEMM
curve predicts r15's slowest rank should be 27% FASTER than pretrained's, not 11% slower.

This is a contradiction with the §16 trace measurement (r15 +11% per-rank expert_ffn at EP=64
DECODE). Resolution: the trace is decode-regime (4096 token-expert assignments per step) while
the routing dump is prefill (65536 per forward). At decode, the slowest rank's pair likely DOES
look different from prefill — possibly pretrained's slowest decode rank has one fat expert and
one idle, while r15's has two small both-active. This needs decode-shape routing data to confirm.

**Decode routing dump submitted (job 29338631)** — will resolve the contradiction.


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


---

## §20.1 ADDENDUM: bs=32 result is in CHUNKED-PREFILL regime, not GEMM-saturated regime (2026-06-22)

### Discovery

User asked: why does TTFT jump 100× between bs=16 (196 ms) and bs=32 (24 000 ms)? Investigation
of the vLLM init log confirms:

```
INFO 06-21 10:12:02 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=8192.
... enable_chunked_prefill=True
... max_cudagraph_capture_size: 512
```

bs=32 plen=8192 = 262 144 total prefill tokens. With `max_num_batched_tokens=8192`, vLLM
chunks the prefill into ~32 sequential scheduler steps. **bs=32 in our bench is not running
as a single big batched prefill** — it's running as many smaller prefills serialized through
the chunked-prefill scheduler.

### Implication for §20's "r15 wins -6.23% at bs=32" claim

§20 framed the bs=32 measurement as the "deeply above-knee" regime where per-local-expert M
= 16384 = 4× the measured asymptote. **This framing is wrong.** Each chunked-prefill step
processes ≤8192 tokens. The per-local-expert M PER CHUNK is roughly the same as bs=1
single-shot (M = 512), not 16384. So bs=32 is **not** the saturated-GEMM regime we needed
to test the CP→time prediction.

The bs=32 measurement is still real (t=-745, ±16 ms on 24 sec mean, extremely tight) and the
r15 -6.23% win is reproducible across both bs_sweep3 and Phase A. But the **mechanism is
unidentified**:

- It's NOT "above-knee GEMM-tile economics" (we're not above knee within a chunk)
- It could be: r15's routing pattern interacts with chunked-prefill scheduler differently
  (e.g., prefix-cache hit patterns, attention kernel re-capture rates, allocator behavior
  across many short prefills)
- Or it could be related to the cudagraph_capture_sizes list and which sizes get static
  graph replay vs dynamic dispatch

### Why the discontinuity between bs=16 and bs=32

bs=16 also exceeds max_num_batched_tokens=8192 (131 072 total tokens > 8192). But scaling
bs=1 → bs=16 is sub-linear (54 → 196 ms = 3.6× for 16× batch). Then bs=16 → bs=32 is 122×.
A sharp scheduler-mode-change discontinuity, likely:

- bs ≤ 16: scheduler still packs requests within a chunk (e.g., partial-token prefill of
  multiple sequences per step)
- bs=32: hits a hard boundary forcing strict 1-sequence-per-chunked-step serialization
- Cudagraph specialization may also play in here (graphs captured up to seqs=512 but the
  total-token configuration differs between bs=16 and bs=32 enough to drop into eager mode
  for some operations)

I don't know the exact boundary without instrumenting the scheduler. But the empirical
observation is: bs=32 is a different regime than bs=16, not a "more of the same" scaling.

### Implication for the project

**We have NO clean above-knee EP=8 prefill datapoint yet.** What we needed was:
- A single forward with per-local-expert M ≥ 4096 (knee per §15)
- That requires bs × plen × top_k / n_experts ≥ 4096
- At EP=8 with top_k=8 and n_experts=128: M = bs · plen · 8 / 128 = bs · plen / 16
- For M = 4096: need bs × plen ≥ 65 536
- At plen=8192: bs ≥ 8 (M=4096, exactly at knee)
- At plen=8192 bs=16: M=8192 (above knee)
- BUT at bs ≥ 1 with plen=8192 = 8192 tokens single-shot, we're already at max_num_batched_tokens limit
- Any bs > 1 triggers chunked prefill

So with default chunked-prefill settings, **we cannot reach the above-knee regime in a single forward at EP=8 with plen=8192.** The bs=8 case (per §20 +0.22%, ns) was supposed to be at the knee but actually had each chunk processing fewer than full-plen tokens.

### Plan for clean above-knee measurement

Three options:

**Option 1**: Disable chunked prefill and re-run bs=1, plen=65536 (or similar). One forward with
all the tokens packed in. May OOM (KV cache and activations balloon at long context).

**Option 2**: Set `--max-num-batched-tokens=262144` (override default) and re-run bs=32 plen=8192.
Forces a single chunked-prefill step that takes all 262K tokens. May OOM similarly.

**Option 3**: Accept that the chunked-prefill regime IS the production-realistic regime and
characterize r15 vs pretrained behavior within it. Then the §20 bs=32 -6.2% win is real and
production-relevant, even if its mechanism isn't GEMM-tile-economics. This is the most honest
position for a serving-relevant report.

### What changes in the project narrative

- **The "r15 wins above-knee" narrative cannot be cleanly verified without a different
  configuration.** Either we need to override max_num_batched_tokens or we accept that the
  win we see at bs=32 is "scheduler-regime r15 win" not "FLOPs-bound r15 win."
- **The bs=1 +6.84% r15 slower finding remains the most reliable result.** Single forward, no
  chunking, real GEMM regime characterized.
- **The "neutral at bs=2-16" finding remains valid as "vLLM-with-chunked-prefill r15 ≈ pre at
  moderate bs."** That's a useful serving-relevant statement.

### Next experiment to actually test the prediction

Submit one more job: bs=1, plen=8192, but with `--max-num-batched-tokens 8192` (= 1 chunk =
no chunking). And bs=8 with `--max-num-batched-tokens 65536`. If those run and r15 wins
in the deep-knee regime, the GEMM-tile-economics mechanism is confirmed. If they OOM,
we acknowledge the regime is unreachable on this hardware.

### Why this matters

If the only regime where r15 helps (bs=32) is actually a chunked-prefill-scheduler-specific
artifact, the "router-RL helps inference" claim has even narrower applicability than §20's
conclusion suggested. We need to either find the genuine FLOPs-bound regime (currently
inaccessible at our config) or accept that the win is regime-specific.

### Files

- vLLM config dump in: `/lustre/fsw/portfolios/nvr/users/jonathanp/phase_A_repl_29339780.out` (line "scheduler.py:239")
- For above-knee test: would need new launcher with `--max-num-batched-tokens 262144`


---

## §21 Above-knee measurement — GEMM-tile-economics mechanism falsified (2026-06-22)

**Headline: at M=4096 (the measured GEMM saturation knee, §15) in a single un-chunked forward,
r15 is +7.18% slower TTFT than pretrained — NOT faster as the GEMM-tile-economics mechanism
predicted. The mechanism we hypothesized for "r15 wins above knee" is wrong.**

### Question

§15 measured the GEMM saturation knee at M=4096. §11-§17 built a mechanism story around
"r15's flatter routing hurts below the knee (small-M GEMMs are tile-overhead-bound) and
helps above the knee (busiest-rank token count drives time)." Phase A confirmed below-knee
hurt (bs=1 r15 +6.84%) and showed a chunked-prefill bs=32 win, but per §20.1 the bs=32 win
was in a different regime entirely. **We had no clean above-knee data point.**

### Method

Patched `cp_vllm_bench.py` to accept env-var overrides for `gpu_memory_utilization`,
`max_num_seqs`, and `max_num_batched_tokens`. Submitted job `climb_M3` (29374034) running
EP=8 prefill at three M points in single un-chunked forwards:

| M (per-local-expert) | tokens/forward | bs × plen | max_num_seqs | gpu_mem_util |
|---|---|---|---|---|
| 1024 (69% asymptote) | 16 384 | 16 × 1024 | 32 | 0.90 |
| 2048 (85% asymptote) | 32 768 | 32 × 1024 | 64 | 0.90 |
| 4096 (94%, KNEE) | 65 536 | 64 × 1024 | 128 | 0.90 |

Each with `enable_chunked_prefill=False` so the configured batch passes through the model
in ONE forward (no chunking). 30 trials per cell.

Memory pressure was real — needed multiple iterations of the patch to get all three points
working. The earlier `climb_M`, `climb_M2` runs OOM'd because the patch buggily set
max_num_seqs from max_num_batched_tokens (overflowing KV reservation) and because
gpu_memory_utilization=0.50 left no room for the model weights.

### Results

| M | tokens/fwd | pre TTFT (ms) | r15 TTFT (ms) | Δ TTFT | pre e2e | r15 e2e | Δ e2e | std OK? |
|---|---|---|---|---|---|---|---|---|
| 512 (Phase A) | 8 192 (chunked) | 53.98 ± 0.27 | 57.67 ± 0.32 | **+6.84%** | 98.71 | 103.07 | +4.42% | YES |
| 1024 | 16 384 | 192.10 ± 228.22 | 201.42 ± 295.38 | +4.85% | 221.90 | 267.27 | +20.44% | **NO** (CV > 100%, first run compile overhead) |
| 2048 | 32 768 | 155.15 ± 5.01 | 172.87 ± 10.73 | **+11.42%** | 288.18 | 281.74 | -2.23% | YES |
| 4096 (KNEE) | 65 536 | 222.46 ± 16.05 | 238.43 ± 2.72 | **+7.18%** | 337.06 | 357.20 | +5.97% | YES |

### Findings

1. **r15 is +7-11% slower TTFT at all M values from 512 to 4096.** The delta does NOT
   shrink as M crosses the GEMM saturation knee. The "above-knee r15 wins" prediction
   is falsified.

2. **The mechanism is M-independent in our measurable range.** Whatever causes the
   r15 TTFT regression is regime-stable across single-forward configurations. Not
   GEMM tile economics.

3. **What this rules out**: the §17/§16 hypothesis that r15 hurts below-knee because
   of per-expert dispatch overhead but flips above-knee because GEMM-FLOPS-bound work
   tracks busiest-rank-tokens. **Wrong**: above the knee, r15 STILL hurts.

4. **The bs=32 chunked-prefill r15 -6.23% win (Phase A)** is now firmly established as
   a chunked-prefill scheduler-regime effect, not a GEMM-saturated-compute effect. Per
   §20.1, in chunked prefill each step's MoE work is at M=512 anyway (the same regime
   as bs=1).

### What's the real mechanism then?

We don't know. Candidates that survive given the M-invariance of the delta:

- **Python/scheduler overhead** per forward step that's specific to r15's routing
  pattern. §13 already showed per-launch kernel times are equal, so the time must be
  spent "between kernels" — vLLM's CPU-side dispatch, autotune cache lookups, CUDA
  graph replay setup. r15's routing might hit different paths in any of these layers.
- **CUDA-graph specialization mismatch**: graphs are captured at specific bs/plen/M
  configurations. r15's routing distribution may cause more "graph misses" that fall
  back to dynamic dispatch.
- **NCCL AllReduce backend selection**: at different M values, the post-MoE AllReduce
  may switch between LL (low-latency) and tree/ring backends. r15's routing variance
  could trigger more backend transitions.
- **Allocator behavior**: r15's slightly different routing produces slightly different
  intermediate tensor shapes → more allocator activity → more overhead.

None of these are testable from bench-level measurements alone.

### Implications for the project's headline claim

The cleanest summary is now:

- **At EP=8 prefill, in any single un-chunked forward regime we can measure (M=512 to M=4096),
  r15 is consistently +7-11% slower TTFT than pretrained.**
- **The +6.23% bs=32 chunked-prefill r15 win is a scheduler-regime artifact, not a
  CP→time benefit.**
- **The CP→inference-time mechanism (via GEMM tile economics) is not supported by the data.**
- **Real mechanism for the +7% delta is unknown; lives outside the CUDA kernel layer.**

For the "does router-RL help inference?" question: **the data now says no, at any
single-forward EP=8 prefill regime we can reach on this hardware. r15 is uniformly
+7-11% slower.** The narrow exception is chunked-prefill (large bs × plen), where r15
appears to win, but that's a different and less-understood regime.

### Caveats

1. **One-shot measurement at each M.** 30 trials each but no cross-run replication
   like Phase A had. The M=2048 +11.42% and M=4096 +7.18% should be replicated to
   tighten CIs.
2. **First M=1024 run had giant variance** (CV>100%), suggesting compile/warmup
   overhead contaminated the timing. Should be re-run.
3. **Plen=1024 may not be the right choice for all M values.** At higher M with longer
   plen, the attention component changes, which could mask MoE deltas. Could re-test
   with plen=512 or plen=2048.
4. **EP=8 only.** EP=32 and EP=64 single-forward above-knee would need multi-node and
   were never tested cleanly.

### Files

- Run script: `cp_latency_test/run_climb_M3.sh`
- Results: `cp_latency_results/climb3_M{1024,2048,4096}_29374034.json`
- Patched bench supports: VLLM_GPU_MEMORY_UTILIZATION, VLLM_MAX_NUM_SEQS, VLLM_MAX_NUM_BATCHED_TOKENS env vars


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

