---
name: resume-training-run
description: Continue a partially-trained Megatron run so it reaches its target train_iters, using this repo's auto-resume mechanism (wandb_run_id.txt, latest_checkpointed_iteration.txt, cross-sweep config matching). Use when a sweep has under-target runs (crashed, OOM, walltime-evicted, QOS-evicted), when the user asks to "continue" / "extend" / "resume" a run, or before launching new sweeps that would re-do work already 60% done.
disable-model-invocation: true
---

# Resume / Continue a Training Run

## When to read this

Before launching any new sweep, audit existing run dirs for under-target
runs and continue them first. Throwing away a 60%-trained checkpoint to
restart from iter 0 wastes 4–8 GPU-hours per run on 235B.

## Mental model: how resume actually works in this repo

There are **two independent resume mechanisms**, both fully automatic, both
keyed off files on disk:

### 1. Megatron auto-resumes training

`<run_dir>/checkpoint/pretrain-mcore-qwen3-moe-megatron-.../` has:

```
latest_checkpointed_iteration.txt   # contains e.g. "1000"
iter_0000500/
iter_0001000/
iter_0001500/
```

If the checkpoint dir exists at training-start time, Megatron **reads
`latest_checkpointed_iteration.txt`, loads that iter, and continues from
there toward `train_iters`**. No flag is required.

### 2. wandb_agent_runner.py auto-resumes the W&B run

`<run_dir>/wandb_run_id.txt` contains the W&B run id created on the run's
first SLURM allocation. The agent runner reads it on every subsequent
launch and reuses the same W&B run, so charts, configs, and history are
continuous.

**Cross-sweep resume**: if the exact `<run_name>` directory does not exist
under `<output_basepath>/`, the agent runner falls back to matching by the
**config portion** of the run name (it strips the `r##` index suffix and
known prefixes). This means a config relaunched under a *new* sweep ID
still resumes the original training and the original W&B run, as long as
the config is identical.

To force from-scratch retraining, set `"fresh_start": true` in the sweep
config JSON. **The default is resume.**

## Pre-flight check: which runs are under-target?

```bash
# Replace the glob pattern to match the run-name prefix you care about
# (e.g. "235b-*" for the 235B sweeps, "pareto_*" for 30B Pareto sweeps).
OUT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning
for run in $OUT/235b-*; do
  ckpt=$(ls -d $run/checkpoint/pretrain-mcore-* 2>/dev/null | head -1)
  [ -z "$ckpt" ] && { echo "$(basename $run): NO CHECKPOINT (skip)"; continue; }
  last=$(cat $ckpt/latest_checkpointed_iteration.txt 2>/dev/null)
  saved_dirs=$(ls -d $ckpt/iter_* 2>/dev/null | wc -l)
  wbid=$(cat $run/wandb_run_id.txt 2>/dev/null || echo "MISSING")
  echo "$(basename $run): last_iter=$last  saved_dirs=$saved_dirs  wbid=$wbid"
done
```

Then compare `last_iter` to each run's configured `train_iters` (from the
sweep JSON or the W&B config) and flag anything below target.

## Decision matrix

| Run state | Action |
|---|---|
| `last_iter ≥ train_iters` | Done; benchmark, don't relaunch. |
| `last_iter < train_iters` AND `wandb_run_id.txt` exists AND checkpoint dir present | **Relaunch the sweep** (Path A). |
| `last_iter < train_iters` AND `wandb_run_id.txt` MISSING but checkpoint dir present | **Relaunch the sweep** (Path A). Cross-sweep resume by config will find it; verify in the agent log. |
| Checkpoint dir missing entirely | Nothing to resume. Either let the sweep agent run it from scratch or skip as failed. |
| Sweep is **still actively training** for this run (chains in `squeue`) | **Do nothing.** It's still progressing — interfering will create duplicates. |

## Path A — relaunch the sweep (the only normal case)

The same command that originally launched the sweep continues it:

```bash
cd examples/qwen3
./launch_sweep.sh <SWEEP_ID> --parallel 2 --agents 1 --8gpu   # 235B
./launch_sweep.sh <SWEEP_ID> --parallel 2 --agents 2          # 30B
```

`launch_sweep.sh` submits new SLURM chains; each chain spins up `wandb agent`
which picks up an unfinished run from the sweep, then `wandb_agent_runner.py`
+ Megatron resume from the existing checkpoint and W&B run.

**Confirm resume actually engaged within 5 minutes** by tailing the chain log:

```bash
tail -f /lustre/.../sweep_logs/sweep_<SWEEP_ID>_chain*_<JOB_ID>.out
```

Look for one of these lines:

```
RESUME: Found previous wandb run ID: ...               # same-sweep resume
CROSS-SWEEP RESUME: matched <prev_run_name> -> ...     # cross-sweep resume
loading checkpoint from .../iter_NNNNNNN               # Megatron resume
```

If you see `Initialized new wandb run` and Megatron starting from `iter 0`,
**stop the chain** (`scancel <JOB_ID>`) and investigate — something is
forcing fresh-start (likely `fresh_start: true` in the sweep config or a
`.txt` file not where the runner expects it).

## Path B — config doesn't exist as a sweep anymore

If the sweep was deleted on W&B but the run dir is still on disk: re-create
the sweep from the same config JSON, then launch it. Cross-sweep resume by
config name will match the existing checkpoint:

```bash
cd examples/qwen3
python3 wandb_sweep_config.py --config run_config_jsons/<original>.json
./launch_sweep.sh <NEW_SWEEP_ID> --parallel 2 --agents 1 --8gpu
```

Verify in the agent log: `CROSS-SWEEP RESUME: matched <old_run_name> -> ...`.

## Common gotchas

- **`fresh_start: true` silently overrides resume.** Always grep the sweep
  JSON for `fresh_start` before relaunching. If you see `true`, that's why
  it's restarting from iter 0.
- **`r##` index drift.** Cross-sweep resume strips the `r##` suffix when
  matching, so a config relaunched as `r07` still resumes a `r03`-suffixed
  run dir. The agent log shows which it matched.
- **Multiple chains may all grab the same run.** W&B only assigns one agent
  per run, so this is fine, but if you see `wandb agent` exit immediately
  with "no runs available", that's why — not a failure.
- **Don't relaunch a sweep that's still actively training.** It just adds
  more chains to the same active chains. Confirm `squeue` is empty for that
  sweep first.

## Verifying continuation worked

After a few minutes:

```bash
# 1. Iter line in run log shows iteration > previous last_iter
grep -oE "iteration\s+[0-9]+/" <run_log> | tail -3

# 2. On-disk checkpoint advances
cat <run_dir>/checkpoint/.../latest_checkpointed_iteration.txt

# 3. Same W&B run id, history continues
cat <run_dir>/wandb_run_id.txt
# In W&B UI, the run page shows continuous metrics, no duplicate run
```

If `wandb_run_id.txt` changed, Path A failed silently and a new run was
created. The original W&B run is now orphaned. To consolidate, you can
either keep both visible (annotate one as "abandoned") or, if the prior run
had no benchmarks downstream, simply ignore it.
