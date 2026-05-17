---
name: collect-benchmark-results
description: Aggregate all per-checkpoint lm-eval accuracy_summary.json files under output_router_finetuning/ into the master benchmark_results.csv, deriving training config metadata (rl_enabled, aux_enabled, coefficients, sweep_id, iteration) directly from run names and paths. Use after any new benchmark job finishes, before generating Pareto charts, or when the CSV looks stale.
---

# Collect lm-eval results into the master CSV

## When

After any `run-lm-eval-benchmark` or `pareto-benchmark` job finishes (the
benchmark submit scripts call this automatically, but run it manually if
they didn't, or before generating a Pareto chart).

## How

```bash
cd examples/qwen3/benchmarks
python3 collect_benchmark_results.py
# Optional: attach a free-text note to a specific run name
python3 collect_benchmark_results.py --comment "my_run_v2=re-ran after KL bug fix"
```

No SLURM job needed — runs on the login node in seconds (just file
globbing + JSON parsing).

## What it does

Globs `/lustre/.../output_router_finetuning/**/benchmark_*/accuracy_summary.json`,
parses each, derives metadata from the parent run-name path, and writes
`examples/qwen3/benchmarks/benchmark_results.csv`.

## CSV schema (unique key = `run_name, bench_iteration, limit`)

| Column | Meaning |
|---|---|
| `run_name` | Sweep-generated name (`<sweep>_<config>_r<idx>`). |
| `bench_iteration` | Training iter the checkpoint was at. |
| `limit` | `1000` (quick) or `""` (full). |
| `sweep_id` | Parsed from run name or "manual". |
| `category` | `rl+aux`, `rl_only`, `aux_only`, `pretrained`. |
| `rl_enabled`, `aux_enabled` | Bool from name. |
| `train_iters`, `rl_reward_type`, `rl_loss_coeff`, `aux_loss_coeff`, `kl_loss_coeff` | From path/name. |
| `hellaswag`, `arc_challenge`, `winogrande` | Per-task accuracy %. |
| `benchmark_avg` | Mean of the 3. |
| `eval_crit_path`, `eval_lm_loss` | From training-time eval (pulled from a sibling W&B summary cache, if present). |

## Gotchas

- The script **does not** measure CP itself — `eval_crit_path` comes from training eval. If a row has `eval_crit_path=""`, the W&B summary cache was missing — re-fetch with `pareto_benchmark.py --step <N>`, which writes that data.
- Re-running is idempotent and overwrites the CSV in-place.
- New rows appear for every `(run, iter, limit)` combo — make sure benchmark output dirs follow the convention `benchmark_iter<N>_limit<L>/` or `benchmark_iter<N>_full/`.
