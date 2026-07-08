# Benchmarking Methodology Lessons — multi-node LLM inference A/B on shared clusters

Distilled from ~150 A/B jobs on HSG GB200 (2026-06/07). Every rule below was paid for
with a retracted claim; see CAMPAIGN_SUMMARY.md for the ledger.

## 1. Controls that killed false positives (run them ALWAYS)

- **A/A null test**: bench the same weights as both sides of the A/B harness. Ours
  showed −2.82% for the "second" model (warm torchinductor/lustre caches) — the same
  magnitude as the effect we were about to claim. One 8-minute job.
- **Duplicate pairs**: measure the same nominal weights via two artifacts (router-swap
  vs full checkpoint). Disagreement bounds methodology noise: ours agreed to ~1pp at
  EP=32 mid-range but diverged up to 28pp at batch extremes.
- **Dose-response ladder + control cell**: any real effect must scale with the treatment
  (CP cut %) and vanish on the control. Our "winner" ladders repeatedly ranked the
  control first — instant refutation.
- **Cross-condition replication**: effects that sign-flip between EPs / allocations are
  noise. Sorted per-cell extremes of a 28-cell campaign WILL show ±7-9% from pure noise
  at ±4%/cell.

## 2. Noise structure (GB200, vLLM, SLURM shared cluster)

- Order bias: the model benched second in an engine session differs systematically
  (+1.3% slower in one harness, −2.8% faster in another — direction depends on cache
  effects). Randomize or interleave A/B/A/B; never fixed order.
- Batch-size extremes (bs=1, bs=32 prefill; bs=512 decode) carry one-sided straggler
  spikes up to +92% on 15-trial means. Medians + more trials, or drop the extremes.
- Node-allocation variance across jobs: ±2-4% on identical workloads. Same-job A/B only
  controls within-pair; cross-cell comparisons still carry allocation luck.
- bs=4-16 prefill was the cleanest regime (±1-2pp repeatability).

## 3. Trace-analysis traps

- **NCCL kernel names lie**: all non-TREE AllReduce algo/proto combos are named
  `*_RING_LL`. Attribute algorithm by launch signature: blockX=640 → NVLS+SIMPLE,
  ≤544 → RING+SIMPLE.
- **Kernel-time ≠ wall time**: NCCL kernel duration includes late-arrival spin; kernels
  span capture/warmup phases; per-node traces sum 4 GPUs. Compute exposure via interval
  unions within the steady bench burst only (see hsg_scripts/analyze_exposure.py).
- **Window attribution**: a session trace holds BOTH models. Split at the largest AR gap
  fails (lands in load phase); segment at >60s gaps and take the two densest segments.
  Validate windows with a routing-independent internal control (attention kernel time
  must match A vs B; if it doesn't, the window is contaminated).
- **Per-GPU, not per-node**: busiest-GPU effects (the straggler that gates the barrier)
  vanish under node-level sums. Split by deviceId. Report max and mean separately —
  totals are conserved, so the mean cannot move; only the max can.
- **Regime matters for FFN sensitivity**: below the GEMM knee (≲450 tok/expert) kernel
  time is fixed-cost bound and insensitive to token counts — routing changes show up as
  MORE kernel time (more active tiles), not less. Only above the knee does per-expert
  token count convert to time.
- **DP padding trap (vLLM)**: without CUDA graphs, DP padding is off and MoE selects
  Broadcast+Reduce instead of AG+RS — eager traces are not comparable to CG traces
  unless padding is forced.

## 4. Cross-rank collective analysis (adopted from the Nemotron Ultra study)

Core duration = min kernel duration across participating ranks per collective instance;
late-arrival = max − min. CG replay emits an identical kernel sequence per rank
(verifiable by sequence hash), enabling instance matching without NVTX. End-time
alignment works across nodes without PTP (~15-28µs skew). Intra-node matching (4 of N
ranks) badly underestimates global straggler effects — do global matching for
imbalance studies.

## 5. Process rules

- Search internal knowledge (Glean) for infra failures on new platforms BEFORE
  iterating experimentally: our DeepEP crash and the RING_LL misread each had existing
  internal answers that saved/would-have-saved days.
- Validation must exercise the full path: HybridEP's CG bug passed capture-based
  validation and failed on the first real request. Always send one request.
- Log EVERYTHING with job IDs; keep a run tracker; append findings incrementally with
  dated sections and explicit retractions rather than silent edits.
