---
name: training-log-forensics
description: Parse Megatron training logs into per-run comparison tables of LM loss, critical path, KL, entropy, expert load, and other metrics. Use when comparing the final state of multiple training runs, building a Pareto frontier table, or diagnosing why two runs diverged.
disable-model-invocation: true
---

# Megatron Training Log Forensics

## Quick start

For each run log file, extract per-iter metrics and present a side-by-side comparison of the last-window averages plus an iter-1 baseline.

## Where logs live

```
sweep_logs/<sweep_name>_<timestamp>/logs/<run_name>.log
```

Each log contains pipe-delimited iter lines, e.g.:

```
[2026-05-13 10:12:50] iteration       1/      80 | consumed samples:    8 | elapsed time per iteration (ms): 29546.7 | lm loss: 2.293943E+00 | num_tokens_on_critical_path: 5.319500E+03 | max_tokens_per_expert: 1.108229E+02 | output_entropy: 1.949811E+00 | kl_loss: 3.366815E-03 | ...
```

## Parser script

Use this template Python script — never hand-write the regex from scratch. Customize the `KEYS` list for the metrics you care about.

```python
import re, glob, os

LOG_DIR = "<path to logs dir>"
KEYS = [
    'lm loss',                       # LM cross-entropy
    'kl_loss',                       # KL constraint value
    'num_tokens_on_critical_path',   # CP — main load-balancing metric
    'max_tokens_per_expert',         # max load over all experts
    'output_entropy',                # logit distribution entropy
    'router_weight_drift',           # drift from KL snapshot
    'aux_loss',                      # Megatron load-balance loss
    'load_balancing_entropy',        # router decision entropy
    'rl_policy_loss',                # PPO loss
    'rl_entropy_bonus',              # PPO entropy bonus
    'grad norm',                     # optimizer health
]

def parse(log):
    iters = []
    for line in open(log):
        m = re.search(r"iteration\s+(\d+)/\s*\d+", line)
        if not m: continue
        row = {'iter': int(m.group(1))}
        for k in KEYS:
            mv = re.search(rf"{re.escape(k)}:\s*([0-9.eE+\-]+)", line)
            if mv: row[k] = float(mv.group(1))
        iters.append(row)
    return iters

def fmt(v, w, p=4):
    if v is None: return f"{'-':>{w}}"
    return f"{v:>{w}.{p}g}"

logs = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
print(f"{'run':<35} {'final':>5} {'lm':>7} {'CP':>6} {'mte':>5} {'ent':>5} {'kl':>9}")
print("-" * 80)
for log in logs:
    name = os.path.basename(log).replace('.log', '')
    iters = parse(log)
    if not iters: continue
    win = max(10, len(iters)//10)   # tail window: last 10% or 10 iters
    last = iters[-win:]
    avg = lambda k: (sum(r[k] for r in last if k in r) /
                     max(1, sum(1 for r in last if k in r))) if any(k in r for r in last) else None
    print(f"{name:<35} {iters[-1]['iter']:>5} "
          f"{fmt(avg('lm loss'),7,4)} {fmt(avg('num_tokens_on_critical_path'),6,0)} "
          f"{fmt(avg('max_tokens_per_expert'),5,3)} {fmt(avg('output_entropy'),5,3)} "
          f"{fmt(avg('kl_loss'),9,4)}")

# Always include an iter-1 baseline for context
b = parse(logs[0])[0] if logs else {}
print(f"\nUntrained baseline (iter 1): CP={b.get('num_tokens_on_critical_path',0):.0f}  "
      f"max_tok={b.get('max_tokens_per_expert',0):.1f}  lm={b.get('lm loss',0):.4f}")
```

## Common pitfalls

- **Python f-strings**: `f"{a if cond else 'b'}"` is fine but `f"{x.format() if x else '-'}"` cannot contain a backslash inside the expression. Format conditionally **outside** the f-string and inject the result, or use a `fmt()` helper like above.
- **Integer fallbacks for `None`**: `f"{v:.4g}"` crashes when `v is None`. Always guard.
- **Last-window choice**: use `last 10%` not `last N` — short runs (e.g. KL>0 runs that only got to 700/1500 iters before SLURM kill) need a smaller window than full runs.
- **Compare like-for-like**: if some runs ran 1500 iters and others 700, **flag this in the report**. Do not silently average over different training horizons.

## Output format

After producing the table, **always** include:

1. **An iter-1 baseline row** (untrained model state) so the magnitude of training effects is visible.
2. **The "final iter" column** so the reader knows which runs completed and which were truncated.
3. **A short interpretation paragraph** that:
   - Calls out the metric with the largest spread across runs (this is usually the most informative).
   - Notes any runs that didn't reach full training length (and warns that their numbers may not be comparable).
   - Distinguishes "X is mechanistically real" from "X correlates with Y" — see `training-bug-investigation` skill for separating gradient-flow from outcome-effect.

## Reference

Megatron's stdout iter line format is defined in `backends/megatron/Megatron-LM-250624/megatron/training/training.py` near `training_log` / `_print_iter_metrics`. Custom RL/KL metrics are added in `megatron_patch/template/helper.py:loss_func_with_rl`.
