
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

