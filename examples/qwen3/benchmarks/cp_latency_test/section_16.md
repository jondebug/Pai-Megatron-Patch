
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

