---
name: run-lm-eval-benchmark
description: Run lm-evaluation-harness (HellaSwag, ARC-Challenge, WinoGrande) on a trained checkpoint and log results back to its W&B run. Use when the user wants to benchmark accuracy on a specific checkpoint, get HellaSwag/ARC/WinoGrande numbers, validate a single trained run, or run the same evaluation we use across all sweep results.
---

# Run lm-eval on a checkpoint

## When

User wants accuracy numbers for one checkpoint (or a small batch). This
skill does conversion + evaluation + W&B logging + CSV collection in one
SLURM job.

## Two scripts

| Use | Script | Behavior |
|---|---|---|
| 1 checkpoint | `examples/qwen3/benchmarks/submit_benchmark.sh` | One ckpt per job. |
| Many checkpoints in one allocation | `examples/qwen3/benchmarks/submit_batch_benchmark.sh` | Reads a JSON manifest, runs sequentially. |

Both: 1 node × 4 GPU × 4 h, on `nvr_israel_rlop`.

## Single-checkpoint submission

```bash
sbatch examples/qwen3/benchmarks/submit_benchmark.sh \
  --checkpoint-dir /lustre/.../output_router_finetuning/<run_name>/checkpoint/<pretrain-...>/ \
  --wandb-run-id <wandb_run_id>          # optional, logs back to that run
  --wandb-project qwen3-router-training \
  --run-name <run_name>                  \
  --model-size A3B                       \
  --tasks hellaswag,arc_challenge,winogrande \
  --batch-size 8                         \
  --limit 1000                           # OPTIONAL — quick benchmark, omit for full
```

Outputs (under the checkpoint dir):

- `hf_converted/*.safetensors` — converted HF model (skipped if already present).
- `benchmark_results/` (single) or `benchmark_iter<N>_limit<L>/` (batch) — lm-eval raw output.
- `<results_dir>/accuracy_summary.json` — parsed scores (also pushed to W&B summary as `benchmark/{task}` and `benchmark/average`).
- Appends a row to `examples/qwen3/benchmarks/benchmark_results.csv` (via `collect-benchmark-results`).

## Batch over a manifest

Manifest JSON format:

```json
[
  {"checkpoint_dir": "/lustre/.../ckpt", "run_name": "run1", "wandb_run_id": "abc123", "iteration": 3000},
  {"checkpoint_dir": "/lustre/.../ckpt2", "run_name": "run2", "wandb_run_id": "def456", "iteration": "final"}
]
```

```bash
sbatch examples/qwen3/benchmarks/submit_batch_benchmark.sh \
  --manifest /path/to/manifest.json \
  --limit 1000   # optional
```

Already-benchmarked entries (those with `accuracy_summary.json`) are
skipped automatically.

## Conventions

- **`--limit 1000`** = quick benchmark (~30 min), use for screening. Omit for full dataset benchmark (~2-3 h), use for Pareto-frontier reporting.
- Run names containing `rl` / `aux` / `cpb` / `kl` are auto-categorized by `collect_benchmark_results.py` into `rl+aux`, `rl_only`, `aux_only`, `pretrained`.
- `lm_eval` runs on `cuda:0` (single process) on purpose — multi-rank hit HF Hub rate limits (429) on shared datasets.

## What the agent should do after submission

1. Capture `JOB_ID` from `sbatch` output.
2. Poll `squeue -j $JOB_ID` until allocated.
3. Tail the log under `/lustre/.../benchmark_logs/benchmark_<JOB_ID>.out` and watch for `Step 1: Converting...`, `Step 2: Running lm-evaluation-harness`, `Step 3: Parsing results`, `Step 4: Logging results to WandB`.
4. When done, run `collect-benchmark-results` to update the master CSV.
5. Report the W&B URL and the parsed scores from `accuracy_summary.json`.

## Gotchas

- `HF_HOME` is pinned to `/lustre/.../.hf_cache` to avoid re-downloading datasets per job.
- Conversion (Step 1) is skipped only if `*.safetensors` exist — partial/failed conversions are detected by the absence of weight files, not config files.
- `--limit ""` (empty) is treated as "no limit" (full dataset).
- If `lm_eval` fails with `429`, retry — it's rate-limit, not a bug.
- **Don't pass `--log_samples`** to `lm_eval` — on Lustre with long checkpoint names it produces `OSError: File name too long` because the sample-log filenames include the full pretrained path.
- **Manifest paths must be `/lustre/fsw/...`, never `/lustre/fs12/...`.** The container only mounts `/lustre/fsw`. `Path(...).resolve()` silently rewrites to the `/lustre/fs12` real path and the job dies at start with `FileNotFoundError` on the manifest. See `pareto-benchmark` SKILL for details.

## Wallclock budget per checkpoint (30B A3B, 4 GPUs)

| Phase | `--limit 1000` | Full dataset |
|---|---|---|
| Megatron→HF conversion | ~100 s | ~100 s |
| `lm_eval` (hellaswag+arc+winogrande) | 20–30 min | 2.5–3 h |
| Total | ~30 min | ~3 h |

**Implication for `submit_batch_benchmark.sh`:** with the 4 h SLURM cap,
full-dataset batches can fit **at most 1, occasionally 2 checkpoints per
job**. Asking for `--per-job 3` or more with no `--limit` will silently
drop checkpoints 2+ on TIME LIMIT. Use `--per-job 1` for full runs.

## Output directory naming convention

The batch script writes per-checkpoint result dirs **inside the source
checkpoint dir** (not in a separate `benchmark_logs/` tree):

| Mode | Path |
|---|---|
| `--limit 1000` | `<ckpt_dir>/benchmark_iter<N>_limit1000/` |
| Full | `<ckpt_dir>/benchmark_iter<N>_full/` |
| No iter (legacy) | `<ckpt_dir>/benchmark_latest_<limit\|full>/` |

`collect_benchmark_results.py` keys on `(run_name, bench_iteration, limit)`
so the **same checkpoint can have both** a limit-1000 row and a full row in
the master CSV — they are not deduplicated against each other. **Never**
mix them in a single Pareto chart without the `--limit-filter` flag (see
`generate-pareto-chart`).

## If a batch job hit SLURM TIME LIMIT

The end-of-script auto-call to `collect_benchmark_results.py` does **not**
run on a TIME LIMIT kill (signal arrives mid-`lm_eval`). Per-checkpoint
`accuracy_summary.json` files were still written by completed checkpoints.

Recovery:

```bash
# 1. Aggregate the partial results
python3 examples/qwen3/benchmarks/collect_benchmark_results.py

# 2. Resubmit — the script auto-skips already-benchmarked checkpoints
sbatch examples/qwen3/benchmarks/submit_batch_benchmark.sh --manifest <same_manifest>
```
