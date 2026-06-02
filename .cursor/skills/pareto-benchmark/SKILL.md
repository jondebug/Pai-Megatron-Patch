---
name: pareto-benchmark
description: Find Pareto-optimal (CP vs LM-loss) checkpoints from a finished W&B sweep and submit benchmark jobs for them in batches. Use when the user wants to benchmark only the best runs from a sweep, identify the Pareto frontier of CP/accuracy trade-offs, follow up a finished sweep with accuracy measurements, or benchmark a specific training step (e.g. step 2000 vs step 3000) across many runs.
---

# Pareto benchmark a finished sweep

## When

A sweep has finished and the user wants accuracy numbers for its
Pareto-optimal points. Often the second step after `launch-sweep`.

## Two budget regimes — pick differently

- **Plenty of GPU budget** (e.g. 30B, 4-GPU jobs, ≥10 candidates): benchmark
  the whole Pareto front + a few near-frontier points across multiple iters.
  Use `pareto_benchmark.py --all` or per-iter loops.
- **Tight budget** (typical for 235B: 1.5 h × 8-GPU per run, user picks 2–3):
  see "Picking a small comparable pair when benchmark budget is tight" below
  before submitting anything. **The choice of which pair to benchmark is at
  least as important as the choice of iter.**

## Script

`examples/qwen3/benchmarks/pareto_benchmark.py` — pulls all runs from a
sweep, fetches `(eval_crit_path, eval_lm_loss)` at a target training
step, computes the Pareto front, and (with `--benchmark`) submits
`submit_batch_benchmark.sh` jobs for the selected checkpoints.

## Run from the bare host (not in container)

`wandb` must be installed in the user's local env (it usually is via
`pip install --user wandb`). Submission happens outside the container.

```bash
cd examples/qwen3/benchmarks

# Inspect only — prints sweep summary + Pareto front + writes a CSV
python3 pareto_benchmark.py --sweep-id <SWEEP_ID> --step 3000

# Submit batch benchmarks for the Pareto points (limit=1000 = quick screening)
python3 pareto_benchmark.py --sweep-id <SWEEP_ID> --step 3000 --benchmark --limit 1000 --per-job 3

# Benchmark ALL runs (not just Pareto) at the final step, full dataset
python3 pareto_benchmark.py --sweep-id <SWEEP_ID> --step final --benchmark --all --per-job 5
```

## Arguments

| Flag | Meaning |
|---|---|
| `--sweep-id` | W&B sweep ID (8-char). |
| `--step <N or final>` | Training iter to read CP/LM at. `final` uses summary. |
| `--benchmark` | Actually submit SLURM jobs (otherwise just print). |
| `--all` | Benchmark every run, not just Pareto-optimal ones. |
| `--limit 1000` | lm-eval limit per task. Omit for full. |
| `--per-job N` | Pack N checkpoints into one SLURM job — critical because QOS allows only 3 concurrent jobs. Use 3-5. |
| `--output-base` | Defaults to `/lustre/.../output_router_finetuning`. |

## Side effects

- Writes `<SWEEP_ID>_pareto_step<step>.csv` next to the script.
- Submits `bench_batch` SLURM jobs (1 node × 4 GPU each), with manifests written under the same directory.
- Each batch job converts (if needed) + runs lm_eval + logs to W&B + updates `benchmark_results.csv`.

## After submission

Monitor with the standard SLURM pattern, then:

```bash
python3 collect_benchmark_results.py    # refresh master CSV
python3 generate_pareto.py --limit-filter 1000 --min-accuracy 61 --clean --max-train-iters 5000
```

Open `pareto_accuracy_vs_cp.html` for the chart.

## Important conventions

- **CP source**: this script reads CP from W&B's `critical_eval/critical_path` (preferred) or falls back to `eval/num_tokens_on_critical_path`. The `lm_eval`-side accuracy benchmark does **not** measure CP — that comes from the training-time eval.
- **Step 2000 often beats step 3000/5000** for aggressive configs — accuracy degrades with more training when load balancing is too aggressive. Benchmark multiple steps when reporting.
- The W&B sweep must already be done (`finished`/`crashed` states). Running runs are skipped.
- `--per-job 1` will hit QOS limits fast on a 23-point Pareto front; always use ≥3.
- **Filter out over-trained outliers when making method-comparison plots.**
  One aux-only run trained past 8K iters (`norl_aux0.01_r08`) made the
  non-RL frontier look stronger than the comparable ≤8K regime. For 30B
  method plots, use `train_iters <= 8000` unless the point of the plot is
  explicitly "long training horizon".
- **Separate RL and non-RL frontiers when explaining method value.** A single
  unified Pareto front can hide the baseline frontier. For method writeups,
  plot CP reduction (%) on x (higher is better), accuracy on y, and draw one
  frontier for `rl+aux`/`rl_only` and one for `aux_only`.

## Per-job sizing: full vs limit (CRITICAL — wallclock budget)

The SLURM `interactive` partition caps jobs at **4 h**. Per-checkpoint cost on
the 30B A3B model:

| Workload | Conversion (mcore→HF) | lm_eval (3 tasks) | Total per ckpt |
|---|---|---|---|
| `--limit 1000` | ~100 s | ~20–30 min | ~30 min |
| Full dataset (no `--limit`) | ~100 s | ~2.5–3 h | ~3 h |

Practical `--per-job` ceilings (so the job actually finishes within 4 h):

| Mode | Safe `--per-job` |
|---|---|
| `--limit 1000` | 5 (default 3 also fine) |
| Full dataset | **1–2** (≥3 will time out mid-eval) |

If you submit `--per-job 5` with full dataset, only the first 1–2 checkpoints
complete; the rest are silently dropped on SLURM TIME LIMIT cancellation.
The end-of-job auto `collect_benchmark_results.py` **never runs** in that
case — you must call it manually (see below).

## Recovering after a TIME LIMIT kill

When a batch job hits the 4 h wall:

1. `accuracy_summary.json` is written **per checkpoint** as each completes,
   not at end-of-job. So partially-completed batches still contributed data
   — just not aggregated.
2. Run `python3 collect_benchmark_results.py` manually to ingest the
   completed checkpoints into `benchmark_results.csv`.
3. The skipped checkpoints will re-submit cleanly: the script's
   "already-benchmarked" check uses `accuracy_summary.json` existence, so
   completed ones are auto-skipped and only the unfinished ones run again.

## CRITICAL: never `Path(...).resolve()` manifest paths

The training/benchmark area lives at
`/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing`, which is a
**symlink** to
`/lustre/fs12/portfolios/nvr/projects/nvr_israel_scne/users/jonathanp/rl_token_routing`.

The container only mounts the `/lustre/fsw/...` path. If your submission
code calls `Path(manifest).resolve()` (which follows symlinks), the
resulting absolute path will be `/lustre/fs12/...` and inside the
container the manifest is `FileNotFoundError`.

**Always write manifests with the `/lustre/fsw/...` string and pass that
verbatim to `sbatch`. Do not `.resolve()`, `os.path.realpath`, or
`Path.absolute()` on container-bound paths.**

Symptom: job allocates, runs for ~1 s, then `traceback ... FileNotFoundError:
'/lustre/fs12/portfolios/nvr/projects/nvr_israel_scne/.../manifest.json'`.

When in doubt, translate explicitly:

```python
def to_fsw(p: str) -> str:
    return p.replace(
        '/lustre/fs12/portfolios/nvr/projects/nvr_israel_scne/users/jonathanp/rl_token_routing',
        '/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing',
    )
```

This same trap also bites `checkpoint_dir` paths read from older CSVs that
were written from the resolved side.

## Picking a small comparable pair when benchmark budget is tight

When the user gives you a single-digit benchmark budget (often 2 for 235B, where
each run is ~1.5 h on 8 GPU), don't auto-pick the top-N-by-CP. The two best
configs often share the same mechanism and the comparison is uninformative.
Instead, pick a **matched-CP pair across mechanisms** so the benchmark
isolates *what's actually responsible* for the downstream accuracy.

Algorithm:

1. From the finished runs in the sweep, find the Pareto-optimal points in
   (CP, LM) space.
2. Group them by mechanism: `rl_only` vs `aux_only` vs `rl+aux` vs `cpb_only`
   vs `rl+aux+cpb` etc. (parse from run name).
3. Pick a pair `(A, B)` where:
   - `A.mechanism ≠ B.mechanism`
   - `|A.cp − B.cp| / max(A.cp, B.cp) < 0.02` (within 2% CP)
   - `|A.lm − B.lm| < 0.02` (within 0.02 LM)
   - Both **`finished`**, not `crashed` (different training horizons confound
     downstream accuracy).
4. If no pair meets the CP+LM tolerance, prefer the CP match (LM differences
   <0.05 are within run-to-run noise; CP differences usually aren't).

**Why this works**: matched (CP, LM) controls for the load-balance and quality
state of the model. Any benchmark gap between A and B is then attributable to
*how each one got there*, which is the interesting question for the paper.

Worked example (q55w7678, 235B, picking 2 of 8 finished runs):

| Run | Mechanism | CP | LM | iter | Why chosen |
|---|---|---|---|---|---|
| `235b-pareto_rlc0.5_c256_ppo_lm1.0_aux0.01_r04` | RL+aux+LM_reward | 6240 | 2.8449 | 1194 | Best CP overall |
| `235b-pareto_norl_aux0.01_cpb_n1_a0.01_r14` | aux+CPB (no RL) | 6290 | 2.8455 | 1100 | Best non-RL baseline at matched CP/LM |

The 50 CP gap and 0.0006 LM gap are within noise; the **mechanism** is the
only meaningful difference. Whatever the benchmark gap is, that's the signal.

Don't pick `rlc0.5_aux0.001_r02` (CP=7859) just because it shows the largest
RL-vs-no-RL gap at *low aux* — you'll measure both "mechanism" and "training
horizon hit different points" and not be able to attribute. Save the
low-aux/wide-spread comparison for a separate benchmark pair specifically
designed to test it.

### Avoid `crashed` runs as benchmark candidates unless absolutely necessary

In a 235B sweep, "crashed" usually means **SLURM wall-time eviction**, not
training-collapse. The run is fine on disk, but if `iter` < `train_iters`,
its CP/LM is measured at a different point than `finished` runs and the
comparison is confounded by training horizon. If you must include a crashed
run, **resume it first** (see `resume-training-run`) so all runs are at
the same `train_iters` before benchmarking.

## Snapping eval steps to on-disk checkpoints

`pareto_benchmark.py --step final` reports the W&B `iteration` from the
run summary, but Megatron only saves at `save_interval` boundaries. For
crashed/time-killed runs, the last on-disk iter is often slightly **less**
than the W&B-reported final iteration (e.g. W&B final=2803, on-disk=2949
because the run resumed and saved past the summary checkpoint; or W&B
final=3000, on-disk=3000 cleanly).

Before submitting, build the manifest from the **on-disk** max iter:

```python
import re, os
def find_latest_iter(ckpt_dir):
    iters = []
    for d in os.listdir(ckpt_dir):
        m = re.match(r'iter_(\d+)$', d)
        if m and os.path.isdir(os.path.join(ckpt_dir, d)):
            iters.append(int(m.group(1)))
    return max(iters) if iters else None
```

Using this avoids the `pareto_benchmark.py` `--step <N>` mode silently
skipping runs whose `iter_NNNNNNN/` doesn't exist on disk.
