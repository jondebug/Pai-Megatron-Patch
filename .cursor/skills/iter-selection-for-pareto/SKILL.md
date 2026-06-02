---
name: iter-selection-for-pareto
description: Pick which training iterations to benchmark for each Pareto-candidate run, using the i_early / i_mid / i_late methodology that accounts for accuracy regression with over-training. Use when about to submit benchmarks for a finished sweep, when extending an existing Pareto chart with more iter samples, or whenever the user mentions "benchmark across multiple iterations" or "pick iters per run".
disable-model-invocation: true
---

# Iteration Selection for Pareto Benchmarks

Empirical fact from the 30B sweeps: **the optimal benchmark iter varies
by config**. Aggressive load-balancing configs often peak around iter 2000
and degrade by 3000–5000; conservative configs keep improving to iter 5000.
Hardcoding `iter=3000` (or any single number) systematically misses points
that would otherwise be on the Pareto frontier.

This skill operationalises picking 3 iters per Pareto-candidate run.

## When to invoke

After a sweep's runs have all reached their target `train_iters` (use
`resume-training-run` to ensure this) and you're about to submit the
benchmark batch.

## The three iters

For each finished run, pick:

| Symbol | Definition | Purpose |
|---|---|---|
| `i_early` | First saved checkpoint where `eval_crit_path < baseline_CP * 0.95` | Catches configs that overshoot quickly then regress |
| `i_mid` | Saved iter with the **lowest CP** among iters whose LM-loss is within 0.1 of the run's final LM-loss | Best-known stable trade-off point |
| `i_late` | The run's last saved iteration | The "did it keep improving?" data point |

If two of these collapse to the same iter, drop the duplicate. Typical
result: 2–3 unique iters per run, usually 3.

`baseline_CP` is the **pretrained model baseline CP**:

- 30B (Qwen3-30B-A3B): **≈ 4780**
- 235B (Qwen3-235B-A22B): **≈ 6500** (varies slightly; pull from any
  iter-1 row in the sweep with `eval_crit_path` to confirm before using).

The 30B and 235B baselines are very different; don't accidentally use the
30B 4780 number when computing thresholds for a 235B sweep — `i_early`
will fire for almost every run and the filtering becomes useless.

## Step-by-step

### 1. Pull each run's CP and LM-loss history from W&B

```python
import wandb
api = wandb.Api()
sweep = api.sweep("nvr-israel/qwen3-router-training/<SWEEP_ID>")
runs = [r for r in sweep.runs if r.state == "finished"]

baseline_CP = 4780   # pretrained baseline; replace once 235B baseline benchmarked

picks = {}
for r in runs:
    h = r.history(keys=["iteration", "eval_crit_path", "eval_lm_loss"],
                  pandas=True, samples=2000)
    h = h.dropna()
    if h.empty:
        continue
    final_lm = h["eval_lm_loss"].iloc[-1]

    # i_early: first iter where CP < 0.95 * baseline
    early = h[h["eval_crit_path"] < 0.95 * baseline_CP]
    i_early = int(early["iteration"].iloc[0]) if not early.empty else None

    # i_mid: lowest CP among iters with LM-loss within 0.1 of final
    near = h[abs(h["eval_lm_loss"] - final_lm) < 0.1]
    i_mid = int(near.loc[near["eval_crit_path"].idxmin(), "iteration"]) \
            if not near.empty else None

    # i_late: last saved iter
    i_late = int(h["iteration"].iloc[-1])

    iters = sorted(set(x for x in (i_early, i_mid, i_late) if x is not None))
    picks[r.name] = iters

for name, iters in picks.items():
    print(f"{name:60s} iters={iters}")
```

### 2. Snap each picked iter to the nearest **on-disk** checkpoint

W&B logs `iteration` per training step; on-disk checkpoints exist only at
`save_interval` boundaries. Snap before submitting any benchmark:

```bash
RUN_DIR=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/<run_name>
CKPT=$(ls -d $RUN_DIR/checkpoint/pretrain-mcore-* | head -1)

# All saved iters as integers
saved_iters=$(ls -d $CKPT/iter_* 2>/dev/null | sed 's|.*/iter_0*||' | sort -n)
# Pick nearest <= target_iter (Megatron only saves multiples of save_interval)
nearest=$(echo "$saved_iters" | awk -v t=$TARGET_ITER 'BEGIN{best=0} $1<=t{best=$1} END{print best}')
echo "target=$TARGET_ITER nearest_on_disk=$nearest"

# Verify the dir is non-empty before submitting
ls "$CKPT/iter_$(printf %07d $nearest)/" >/dev/null || { echo "EMPTY checkpoint, skip"; }
```

If `nearest=0` (no checkpoint at-or-before target), drop this iter from
the benchmark plan.

### 3. Submit benchmarks

Two ways:

**Per-iter via pareto_benchmark.py:**

```bash
cd examples/qwen3/benchmarks
for ITER in $i_early $i_mid $i_late; do
  python3 pareto_benchmark.py --sweep-id <SWEEP_ID> --step $ITER \
    --benchmark --limit 1000 --per-job 3
done
```

**Custom manifest via submit_batch_benchmark.sh** (recommended when iters
differ across runs — `pareto_benchmark.py --step <i>` benchmarks the same
iter across all runs in the sweep, which is wasteful here):

```bash
# Build a manifest: one line per (run_name, iteration, ckpt_dir, hf_dir)
MANIFEST=/tmp/235b_pareto_manifest.txt
> $MANIFEST
for run_name in "${!picks[@]}"; do
  for iter in ${picks[$run_name]}; do
    # ... validate iter on disk per Step 2 ...
    echo "$run_name $iter $ckpt_dir $hf_dir" >> $MANIFEST
  done
done
sbatch --array=1-$(wc -l < $MANIFEST)%3 \
  examples/qwen3/benchmarks/submit_batch_benchmark.sh $MANIFEST
```

`%3` caps array concurrency at 3 to respect QOS.

### 4. 235B-only — convert each (run, iter) BEFORE benchmarking

`pareto_benchmark.py` and the default batch script invoke the 30B 4-GPU
converter, which OOMs and mismatches `TP*EP/world_size` on 235B. For each
selected iter:

```bash
TRAINED_MEGATRON_CKPT=<ckpt_dir> ITER_NUM=$ITER \
  sbatch examples/qwen3/benchmarks/cp_latency_test/submit_convert_235b.sh
```

Wait until `<run_dir>/hf_converted_iter${ITER}_cp/` exists before kicking
off the lm-eval. The batch benchmark will then skip conversion and just
run lm-eval on the existing HF dir.

## Output format for the user

After picking iters across the whole sweep, present the plan as a table
**before** submitting any benchmark, so the user can sanity-check it:

```
run_name                                     train_iters  picked_iters    notes
235b-rl0.5-aux0.01-r02                       1500         500, 1000, 1500
235b-rl0.5-aux0.001-r05                      1500         1000, 1500      (i_early == i_mid; collapsed)
235b-aux0.01-norl-r03                        1500         500, 1500       (no on-disk iter at target i_mid=1100; snapped to 1000, then collapsed with i_early)
```

Submitting is a one-shot QOS commit (3 GPUs × 30 min × 30+ jobs); presenting
the plan first avoids re-doing the whole batch if a heuristic is wrong.

## Anti-patterns

- **`--step 3000` blindly**: assumes 30B's modal optimum applies to 235B.
  It doesn't.
- **Benchmarking only the last iter**: misses configs that peaked early
  and regressed.
- **Skipping the on-disk validation**: `pareto_benchmark.py --step 1234`
  silently produces no results when iter_0001234 doesn't exist.
- **Benchmarking iters before runs hit `train_iters`**: produces a
  truncated comparison. Use `resume-training-run` to finish runs first.
