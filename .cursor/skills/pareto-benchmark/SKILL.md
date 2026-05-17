---
name: pareto-benchmark
description: Find Pareto-optimal (CP vs LM-loss) checkpoints from a finished W&B sweep and submit benchmark jobs for them in batches. Use when the user wants to benchmark only the best runs from a sweep, identify the Pareto frontier of CP/accuracy trade-offs, follow up a finished sweep with accuracy measurements, or benchmark a specific training step (e.g. step 2000 vs step 3000) across many runs.
---

# Pareto benchmark a finished sweep

## When

A sweep has finished and the user wants accuracy numbers for its
Pareto-optimal points. Often the second step after `launch-sweep`.

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
