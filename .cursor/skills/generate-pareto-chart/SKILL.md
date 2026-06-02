---
name: generate-pareto-chart
description: Generate the interactive HTML Pareto-frontier chart (accuracy vs critical path) from benchmark_results.csv, with category coloring and filtering for limit, accuracy floor, and training iteration. Use after collect-benchmark-results, when reporting Pareto trade-offs to stakeholders, or to visualize CP/accuracy results across categories (rl+aux, rl_only, aux_only, pretrained).
---

# Generate the Pareto frontier chart

## When

Final reporting step — after benchmarks have run and the master CSV is
fresh.

## Standard invocations

```bash
cd examples/qwen3/benchmarks

# Quick benchmark (limit=1000) chart
python3 generate_pareto.py --limit-filter 1000 --min-accuracy 61 --clean \
    --max-train-iters 5000 --y-min 61 --y-max 66

# Full benchmark (no limit) chart
python3 generate_pareto.py --limit-filter "" --min-accuracy 61 --clean
```

Output: `pareto_accuracy_vs_cp.html` at the repo root (relative path
`../../../pareto_accuracy_vs_cp.html` from the script).

## Flags

| Flag | Purpose |
|---|---|
| `--csv <path>` | Override default `benchmark_results.csv`. |
| `--output <path>` | Override default output HTML. |
| `--limit-filter <1000 or "">` | Pick quick vs full benchmark rows. **Required** to avoid mixing. |
| `--min-accuracy <pct>` | Drop runs below this accuracy. |
| `--max-train-iters <N>` | Drop runs trained past N iters. |
| `--y-min`, `--y-max` | Y-axis clamp (accuracy %). |
| `--clean` | Hide non-Pareto / dominated points. |

## Output convention

- X-axis: `eval_crit_path` (lower is better). Baseline Qwen3-30B-A3B = **4780**.
- Y-axis: `benchmark_avg` (HellaSwag ∪ ARC ∪ WinoGrande mean).
- Points colored by `category`: `rl+aux` `rl_only` `aux_only` `pretrained`.
- Pareto front drawn as a connected line, dominated points dimmed (or hidden with `--clean`).

## Gotchas

- Always run `collect-benchmark-results` first — `generate_pareto.py` is just a renderer over the CSV.
- Mixing limit=1000 and full-dataset rows in one chart is misleading (different sample-size noise). The `--limit-filter` is mandatory in practice.
- Rows with `eval_crit_path=0` or `benchmark_avg=0` are dropped silently — verify the CSV has values for both before chart generation.
- **`--limit-filter ""` means "full-dataset rows only"** (CSV `limit` column is empty for full runs), NOT "no filter". An older version of `generate_pareto.py` had a bug where passing `--limit-filter ""` excluded every row with a non-empty `limit` *value* — i.e. all the limit=1000 rows — which is the intended behavior, *but* a string-vs-int mismatch silently dropped full-dataset rows too when their `limit` column happened to be cast to a non-empty representation. The current script treats empty-string filter as "no filter" (kept for backwards compat); if you intend "full only", pass `--limit-filter ""` AND confirm the row count in the resulting HTML matches `awk -F, '$<limit_col>==""' benchmark_results.csv | wc -l`. If those don't match, you have stale data or a re-introduced filter bug — re-run `collect_benchmark_results.py` and verify.
- **The "best iter is the final iter" assumption is wrong** for aggressive load-balancing configs. Pareto charts that only plot a run's final iter systematically miss the early-iter peaks. Pre-stage benchmarks at multiple iters (see `iter-selection-for-pareto`) before charting.
- **30B and 235B should not share a chart.** The CP scale, baseline (4780 vs 6500ish), and meaning of "good" all differ. Use `--csv` to point at separate filtered subsets, or filter by `run_name` prefix in the renderer.
