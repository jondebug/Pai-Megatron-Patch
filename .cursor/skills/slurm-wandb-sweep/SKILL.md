---
name: slurm-wandb-sweep
description: Design, register, and launch SLURM-backed W&B sweeps for Megatron training runs. Use when the user asks to launch a sweep, queue training runs, run a new experiment grid, design a sweep config JSON, or set up a hyperparameter scan on a SLURM cluster.
disable-model-invocation: true
---

# SLURM + W&B Sweep Workflow

For training experiments on the Pai-Megatron-Patch fork at `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch`. Adapt paths for other repos.

## Quick start

```
Workflow:
- [ ] 1. Author sweep_config.json under examples/qwen3/run_config_jsons/
- [ ] 2. Dry-run combo count (don't register W&B yet)
- [ ] 3. Register W&B sweep
- [ ] 4. Pick parallel/agents knobs based on EP size
- [ ] 5. Launch with launch_sweep.sh
- [ ] 6. Verify the first run actually starts training (check log within 15 min)
```

## Step 1: Author sweep config

Sweep configs live in `examples/qwen3/run_config_jsons/<name>_sweep.json`. Required structure: a flat dict of fixed params, list-valued sweep params, and a `filters` array that prunes the grid.

**Critical settings that must be right:**

- **`save_interval`**: must be **less than `train_iters`** so partial checkpoints save before SLURM 4-hour wall-time. Use `save_interval: 300` for 1500-iter 235B runs (5 saves per run). Setting `save_interval == train_iters` means **no resume across allocations** and a killed run loses all work.
- **`fresh_start: false`** if you want auto-resume across SLURM continuation chains. Set `true` only if you genuinely want every config to start from the pretrained checkpoint, never resuming.
- **`exit_duration_in_mins`**: graceful shutdown trigger. Set to ~10 min less than the SLURM wall-time (e.g., `230` for a 4h SLURM allocation). Megatron will save a checkpoint and exit cleanly when this is hit. Without this, runs get hard-killed by SLURM with no save.
- **`empty_unused_memory_level: 2`**, **`ckpt_assume_constant_structure: true`**, **`ckpt_fully_parallel_save: true`** for 235B (avoids OOM on save).
- **`wandb_run_name_base`** and **`wandb_run_tags`**: include a unique tag per sweep so you can filter in W&B later (e.g., `"kl-fix-v2"`).

**Filter rules** prune combinatorial blowups:
- `{"if": {"key": value}, "skip": true}` — drops combos matching the condition.
- `{"if": {"key": value}, "collapse": ["other_key"]}` — when condition holds, dedup across `other_key` (treat as if other_key isn't varying).
- Always collapse RL-specific knobs when `use_rl_loss: false`.
- Always collapse KL when `use_rl_loss: false` (KL is computed inside `loss_func_with_rl`).

## Step 2: Dry-run combo count

Always verify combo count before registering. The dry-run is free.

```bash
cd examples/qwen3 && python3 -c "
import sys; sys.path.insert(0, '.')
from wandb_sweep_config import generate_filtered_sweep_config, load_config
cfg = load_config('run_config_jsons/<name>_sweep.json')
_, valid, fixed = generate_filtered_sweep_config(cfg)
print(f'Total runs: {len(valid)}')
for i, c in enumerate(valid):
    print(f'  [{i:2d}] {c}')
"
```

If the count is wildly more than expected, fix the filters before paying the W&B-registration round trip.

## Step 3: Register the W&B sweep

```bash
cd examples/qwen3 && python3 wandb_sweep_config.py --config run_config_jsons/<name>_sweep.json
```

This is a **foreground** call (8–10s typical). Let it complete; do not background it. Capture the sweep ID from `SWEEP CREATED: <id>` and persist it:

```bash
echo "<sweep_id>" > examples/qwen3/sweep_logs/<name>_<timestamp>/sweep_id.txt
```

## Step 4: Pick parallel/agents knobs

The two settings on `launch_sweep.sh` are **`--parallel` (chains)** and **`--agents` (agents per chain)**. Each chain is its own SLURM allocation. **Agents within a chain share the GPUs in that allocation.**

| Model | EP | GPUs/run | Right knobs |
|---|---|---|---|
| 30B (A3B) | 4 | 4 | `--parallel N --agents M` (N×M = concurrent runs); script grabs 4 GPUs |
| 235B (A22B) | 8 | 8 | **`--parallel N --agents 1 --8gpu`** (only 1 agent per 8-GPU box) |

**Common mistake:** `--parallel 2 --agents 2 --8gpu` does **not** give you 4 concurrent 235B runs — both agents in a chain fight for the same 8 GPUs and either OOM or NCCL-bind-fail. **For 235B, always `--agents 1`.**

For QOS-limited clusters (3 concurrent jobs, 24 GPU max), the 235B ceiling is `--parallel 3 --agents 1 --8gpu`.

## Step 5: Launch

```bash
bash examples/qwen3/launch_sweep.sh <sweep_id> [--8gpu] --parallel <N> --agents <M>
```

Background-friendly. The script submits `<N>` independent chain jobs that auto-resubmit on completion (up to 20 hops). Returns `JOBID` per chain.

## Step 6: Verify it actually started

Within 15 minutes of allocation start, the run log should exist and have iter-1 metrics. Check:

```bash
# 1. SLURM allocations made it past PD into R
squeue -u $USER

# 2. The driver log for the SLURM job exists
ls -lat /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs/sweep_<sweep_id>_chain*

# 3. The actual training log has iter lines (not just the megatron startup banner)
cat /lustre/.../sweep_logs/<sweep_dir>/logs/<run_name>.log | grep -oE 'iteration\s+[0-9]+/' | tail -3
```

If the training log exists but has no iter lines after 15 min, training is hung — scancel that chain and inspect the error.

## Common gotchas (do not repeat)

- **`save_interval == train_iters`** silently breaks resume. Always pick a save interval < train_iters.
- **`fresh_start: true` + `exit_duration_in_mins`** means every continuation starts from the pretrained checkpoint, throwing away the partial checkpoint on disk. If you want resume, `fresh_start: false`.
- **`--agents 2 --8gpu`** does not double 235B throughput; it stalls both runs.
- **Cursor injects `--trailer`** into `git commit`, breaking on git 2.25.1. Commit from a real terminal with `git commit --no-verify -F /tmp/msg.txt`.
- **W&B sweep registration hangs sometimes.** If a foreground `wandb.sweep()` call hasn't returned in ~30s, the auth/network is wedged — kill it, retry. Don't background it.
- **`exit_duration_in_mins` triggers a save**, so a killed run with `save_interval=N >> exit_duration` is actually OK if `train_iters * iter_time > exit_duration`. The save happens just before the time limit hits.

## Run-time per iteration on 235B

For sanity-checking that training is healthy:

| Config | sec/iter | 1500-iter wallclock |
|---|---|---|
| RL+aux+CPB, kl=0 (no reference forward) | ~10s | ~4.5h (1 alloc) |
| RL+aux+CPB, kl>0 (reference forward) | ~22s | ~9.5h (2-3 allocs) |

If iter time is 2× higher than expected, suspect either contention with another job on the same node or that `--agents 2 --8gpu` is in effect.

## KL on 235B — defer until hook bug is verified-fixed

The KL gradient on this repo was broken from the start of the work in
`megatron_patch/template/helper.py`: the hook to capture `current_logits`
was registered against `model.output_layer` but Megatron training wraps the
model as `DistributedDataParallel(Float16Module(GPTModel))`, so the attribute
lookup returned `None` and the hook attached to nothing. The KL loss read
`current_logits = None` and short-circuited to 0 in the backward graph.

The user-visible effect ("KL improves accuracy +1.7 pp, +500 CP") was
**entirely a wallclock-budget compute confound**: the reference-model
forward pass roughly doubled iter time, so KL>0 runs completed ~half as
many iters within the 4-h SLURM budget. Less training → less CP reduction
*and* less LM degradation; both axes moved because the run was under-
trained, not because KL regularized anything.

Implications for sweep design:

- **Until the hook fix is verified end-to-end** (param-grad nonzero on
  router params, coefficient sweep changes outcome at matched-iter), do not
  vary KL as a sweep axis on 235B. You pay 2× wallclock per run for an inert
  regularizer.
- A non-zero KL on 235B currently behaves as a **wallclock throttle**, not
  a regularization signal. If that's actually what you want, say so
  explicitly and disable the rest of the KL machinery.
- When comparing pre-fix and post-fix runs in the same chart, **always
  label which side of the fix each run is on**. Numerical comparisons
  between them are not apples-to-apples.

See `training-bug-investigation` SKILL for the recursive-hook-registration
fix and Layer-4.5 iter-count parity check.

## Run-state taxonomy (235B in particular)

W&B `state` values on 235B sweeps don't mean what they suggest:

| W&B state | Meaning (235B) |
|---|---|
| `finished` | Hit `train_iters`. Compare these freely. |
| `crashed` | Almost always SLURM 4h wall-time eviction at iter < `train_iters`. **Resume before benchmarking**, otherwise comparisons across runs are horizon-confounded. See `resume-training-run`. |
| `failed` | Real failure — OOM, NCCL, code error. Inspect the .out log. |
| `killed` | User `scancel`, usually intentional. |

A "1500-iter sweep" with 8 `finished` and 7 `crashed` is normal — the seven
crashed runs are time-evicted, not broken. Don't draw conclusions from a
mixed `finished` + `crashed` table until the crashed ones are resumed to
their target.

## Reference

For sweep-config-syntax details (filter semantics, parameter inheritance, etc.) see the `wandb_sweep_config.py` source. The `wandb_agent_runner.py` source documents how the runner builds the CLI command and handles cross-sweep resume.
