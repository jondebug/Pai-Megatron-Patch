# CP→Speedup Existence Proof on eager hybrid_ep — Design

2026-07-10. Phase following the §A-§K campaign (no reproducible CP-RL e2e effect,
EP=8-64, four backends). Goal chosen by user: **existence proof** — engineer the regime
where CP reduction must show if the theory holds, rather than search production configs.

## 1. Hypothesis (pre-registered)

In a prefill-heavy, above-GEMM-knee workload on a load-proportional comms backend
(hybrid_ep, eager), reducing busiest-expert load (CP) increases end-to-end prefill
throughput. Predictions:
- r05 (~57% train-scale CP cut) beats pretrained by more than the noise envelope;
- ordering r05 ≥ r15 (~47%) > r46 (control) ≈ 0;
- the win co-occurs with measured busiest-rank MoE-FFN + dispatch kernel-time reduction.

Falsification: r05 ≤ noise in this regime — where busiest-GPU FFN −50% was directly
measured (§I) — means CP→e2e-speedup is wrong at the serving level, not merely unmeasured.

## 2. Workload (regime engineering)

- Serve-mode prefill saturation: plen=4096, gen=1, concurrency 64, 256 prompts/measurement.
- Chunked prefill mnbt=2048/rank → at EP=32 (DP=32, 8 nodes): ~65k tokens/step
  ≈ 4k tokens/expert = ~9× above the knee (~450 tok/expert), where per-expert token
  count converts to kernel time.
- Backend: hybrid_ep (eager). Eager launch overhead amortizes at large prefill kernels;
  the 2× penalty measured earlier was a small-batch decode artifact. Baseline caveat
  (not the fastest backend) is accepted and documented.
- Metric: batch wall time + TTFT distribution. Decode out of scope (sub-knee, settled).

## 3. Cells & controls

Jobs at EP=32 (8 nodes, --segment=8), launcher `hsg_serve_hybridep_ab.sh`, bench
`http_bench.py --plen 4096 --gen 1 --concurrency 64 --prompts 256`:
1. A/A null (pre vs pre)
2. r05 vs pre
3. r15 vs pre
4. r46 vs pre (control)
5. A/A replica
6. r05 replica

Decision rule (pre-registered): claim requires ALL of
(a) r05 delta > noise envelope defined by both A/As,
(b) ordering r05 ≥ r15 > r46 ≈ 0,
(c) r05 replicas agree within 2pp.
Anything less → null result, documented in the retraction-aware style of §J/§K.

## 4. Mechanism linkage

One nsys-traced pair (pre, r05), `--cuda-graph-trace=node`, worker-node reps (head-node
vllm hangs on SIGINT; worker traces are the reliable artifact). Extract in the prefill
window, per rank: MoE-FFN kernel time, dispatch/combine kernel time, busiest-rank vs
mean. A real e2e delta must be consistent with the busiest-rank reduction × CP-gated
fraction of the step.

## 5. Effort & risks

~10 jobs × 8 nodes × ~25 min; all tooling exists. Risks:
- vLLM DP-level load balancing may smooth per-rank token counts, shrinking the
  busiest-RANK gap even when expert-level imbalance persists — the trace pair detects
  this (per-rank FFN spread).
- Eager host overhead could still dominate prefill steps — the exposure split measures
  it; if idle > 50% of step, report "regime unreachable in eager mode" and the
  follow-up is fixing hybrid_ep CUDA-graph integration (declined option B).
- Companion option C (kernel-level microbench replaying recorded routing through
  deep_ep buffers) is deferred, not rejected.

## 6. Measurement sanity gates (auto-reject, never explain away)

Motivated by the retracted "NCCL = 90% of e2e" artifact:
1. **Accounting identity**: per-device compute + exposed-comms + idle = wall, from
   interval unions only. Summed kernel durations are NEVER reported as fractions of
   wall (kernel sums include late-arrival spin, span warmup, and add across GPUs).
2. **Bench↔trace cross-check**: trace-derived step time must match bench-implied step
   time within ~10%; mismatch = attribution error, discard the analysis not the bench.
3. **Plausibility floor**: kernel-table busy >50% of wall in a saturated burst;
   near-zero busy = graph-level-trace artifact (--cuda-graph-trace=node missing).
4. **Internal control**: attention kernel time (routing-independent) must match
   pre-vs-cell within noise, else window contamination.
5. **Spin separation**: comms time reported as core (min across ranks) + late-arrival
   (max−min), never one "network time" number.
6. Any measurement failing a gate is flagged NONSENSICAL in the results table and
   excluded from conclusions until root-caused.
