
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

