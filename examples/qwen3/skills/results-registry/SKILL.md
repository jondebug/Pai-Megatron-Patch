---
name: results-registry
description: Keep the benchmark_results.csv a COMPLETE registry of every saved checkpoint (lm_loss, accuracy, critical path, checkpoint path, vLLM walltime) and manage the HF-conversion disk lifecycle — benchmark every saved checkpoint for both walltime and accuracy, but don't hoard HF copies of non-crucial checkpoints. Use when checkpoints finish training, when auditing coverage, or when reclaiming disk.
---

# Results registry & HF-checkpoint lifecycle

## The CSV is the single source of truth — every saved checkpoint must be in it
`benchmark_results.csv` (built by `collect_results_unified.py`) should have, for **every saved
distcp checkpoint** (each cell × each saved iter), a row with:
`run_name, bench_iteration, category, lm_loss, per-task acc, benchmark_avg, eval_crit_path (CP),
checkpoint_path, limit, wt_decode_tps/wt_ttft_ms/wt_e2e_ms (walltime)`.
- The collector `scan_all_distcp()` registers every distcp dir so nothing is silently untracked.
- If a checkpoint is missing accuracy or walltime, it's an **incomplete row → schedule the
  missing benchmark**. Don't consider a checkpoint "done" until it has both.

## Benchmark policy: BOTH walltime AND accuracy, for every saved checkpoint
1. **Accuracy**: lm-eval at **both** `limit=1000` (fast) and `limit=inf` (canonical). See
   [[accuracy-benchmark]].
2. **Walltime**: vLLM decode_tps/TTFT/e2e (TP=8 single-node minimum; multi-node EP for the
   EP-scaling study). See [[walltime-benchmark]].
3. **CP** (`eval_crit_path`) comes from *training* wandb (`num_tokens_on_critical_path`), joined
   in by the collector — not from lm-eval.
A Pareto point is only plottable with accuracy **and** CP; walltime is the latency axis.

## HF lifecycle: convert → benchmark → delete, keep only crucial copies
HF copies are large (235B ≈ 438 GB). Convert distcp→HF **only to benchmark**, then **delete the
`hf_converted_iter<N>` dir** — UNLESS the checkpoint is crucial.

**Crucial = keep the HF** if any of:
- it's on / near the Pareto frontier (frontier-cell exemption),
- it still needs a pending walltime or EP-scaling/microbench run,
- it's a designated paper candidate.

**Cleanup guard (idempotent):** delete `hf_converted_iter<N>` ONLY when the CSV has **all three**
for that (cell, iter): `limit=1000` row **and** `limit=inf` row **and** a walltime JSON. Deleting
before walltime exists **strands the pending walltime job** (a real bug we hit — 17 walltime jobs
failed because their HF was deleted early). Helper:
`examples/qwen3/benchmarks/cleanup_completed_hf.sh` (encodes the 3-way guard + frontier exemption).

## Coverage audit (do periodically)
- List cells with a saved iter that lack `benchmark_avg` or walltime → queue them.
- Prune **dominated** intermediate iters (keep frontier + latest), never frontier cells.
- Cross-cluster: NRT-run lm-eval writes locally and is **not** auto-ingested by the ORD
  collector — copy results to ORD (or extend the collector) so they reach the registry/frontier.
