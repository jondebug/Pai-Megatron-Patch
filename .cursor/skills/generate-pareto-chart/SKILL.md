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
