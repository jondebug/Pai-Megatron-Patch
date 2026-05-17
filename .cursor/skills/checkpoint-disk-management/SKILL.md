---
name: checkpoint-disk-management
description: Manage Lustre disk usage during long-running sweeps. Use when df shows the project filesystem above ~80% full, when an sbatch fails with disk-quota or no-space errors, when a Megatron save fails mid-training, or proactively before launching a new 235B sweep that will produce many large checkpoints.
disable-model-invocation: true
---

# Checkpoint Disk Management

235B Megatron checkpoints are large (~470 GB per `iter_NNNNNNN/` dir for
A22B). A single 1500-iter run with `save_interval: 300` produces 5
checkpoints ≈ 2.3 TB. A full sweep can easily push the project Lustre
quota over the line and silently break new saves.

This skill: monitor usage, identify safe-to-delete checkpoints, perform
the deletion, document what was removed.

## When to read this

- `df` / quota warnings on the project filesystem.
- Megatron save failure mid-training (OSError, ENOSPC, EDQUOT).
- Before launching any sweep that will produce >5 TB of checkpoints
  (any 235B sweep with >3 runs at `save_interval: 300`).
- After every successful Pareto chart refresh — opportunity to clear
  checkpoints proven non-frontier.

## Quick check

```bash
df -h /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/
# project quota:
lfs quota -h -p $(stat -c %g /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/) /lustre/fsw 2>/dev/null || \
  du -sh /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/
```

If used > 80% of quota, act. If > 90%, **act before the next sweep
launch** — saves silently truncate when the FS fills mid-write and
those checkpoints become unreadable.

## What is safe to delete

In **decreasing** safety order (most safe at the top):

| Tier | Item | Safe to delete? |
|---|---|---|
| 1 | `hf_converted_iter*_cp/` for runs already benchmarked and recorded in `benchmark_results.csv` | Yes — regeneratable from the mcore checkpoint in ~25 min if needed again. |
| 2 | Megatron checkpoints from runs that **completed** their `train_iters` AND are NOT on the current Pareto frontier AND are NOT within ε of it (see "Distance to frontier" below) | Yes, **after** chart refresh confirms they're not frontier. |
| 3 | Mid-run checkpoints (not `i_early/i_mid/i_late` per `iter-selection-for-pareto`) for non-frontier runs | Yes. |
| 4 | The pretrained checkpoint dirs `qwen-ckpts/Qwen3-*-to-mcore` | **NO — never delete.** |
| 5 | The very latest checkpoint of any **active or paused** run (anything that might be resumed) | **NO.** |
| 6 | The final-iter checkpoint of any frontier run | **NO.** |

## Distance to frontier — which non-frontier runs to delete first

Compute Euclidean distance in (CP, accuracy) space, normalized:

```python
import csv
rows = list(csv.DictReader(open("examples/qwen3/benchmarks/benchmark_results.csv")))
r235 = [x for x in rows if "235b" in x["run_name"].lower() and x.get("limit") == "1000"]

cps = [float(x["eval_crit_path"]) for x in r235 if x.get("eval_crit_path")]
accs = [float(x["benchmark_avg"]) for x in r235 if x.get("benchmark_avg")]
cp_lo, cp_hi = min(cps), max(cps)
ac_lo, ac_hi = min(accs), max(accs)

def pareto(pts):  # pts: [(cp, acc, idx)]
    out = []
    for p in pts:
        if not any(q[0] <= p[0] and q[1] >= p[1] and q != p and (q[0] < p[0] or q[1] > p[1]) for q in pts):
            out.append(p)
    return out

pts = [(float(x["eval_crit_path"]), float(x["benchmark_avg"]), i)
       for i, x in enumerate(r235)
       if x.get("eval_crit_path") and x.get("benchmark_avg")]
front = set(p[2] for p in pareto(pts))

def norm_dist(p, fp):
    return ((p[0]-fp[0])/(cp_hi-cp_lo+1e-9))**2 + ((p[1]-fp[1])/(ac_hi-ac_lo+1e-9))**2

# distance from frontier for non-frontier points
fpts = [pts[i] for i in front]
ranked = sorted([(min(norm_dist(p, fp) for fp in fpts)**0.5, p) for p in pts if p[2] not in front],
                reverse=True)  # farthest first
for d, p in ranked[:15]:
    print(f"d={d:.3f}  CP={p[0]:.0f} acc={p[1]:.2f}  {r235[p[2]]['run_name']}")
```

The top of `ranked` is the **safest to delete first**: farthest from any
frontier point on both axes. Walk down the list deleting until disk
usage is back below 70%.

## How to delete

For each chosen run + iter:

```bash
RUN=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning/<run_name>
CKPT=$(ls -d $RUN/checkpoint/pretrain-mcore-* | head -1)

# 1. Decide WHICH iters to delete (keep at least the final-iter dir
#    even for non-frontier runs, in case we want to re-benchmark later).
KEEP_ITER=$(cat $CKPT/latest_checkpointed_iteration.txt)
for d in $CKPT/iter_*; do
  iter=$(basename $d | sed 's/iter_0*//')
  [ "$iter" = "$KEEP_ITER" ] && { echo "KEEP $d (last iter)"; continue; }
  echo "DELETE $d  ($(du -sh $d | cut -f1))"
  # rm -rf $d   # uncomment after dry-run review
done

# 2. HF dirs are usually safe to delete entirely (regeneratable)
ls -d $RUN/hf_converted_iter*_cp 2>/dev/null
# rm -rf $RUN/hf_converted_iter*_cp
```

**Always dry-run first** (echo only) before adding `rm -rf`. Lustre
deletes are irreversible, and the cluster has had file-loss incidents.

## After deletion

- Note in a markdown log (e.g. `examples/qwen3/benchmarks/disk_cleanup_log.md`)
  what was deleted, when, and why. Future debugging that finds a
  missing checkpoint will look here.
- Re-run `df` / `du` to confirm reclamation.
- If a deleted iter re-becomes interesting later (e.g. a follow-up
  sweep changes the Pareto picture), it can be regenerated by resuming
  the run (skill: `resume-training-run`) — but only if the **last-saved
  iter** is still on disk. This is why Tier-5 (last-iter of paused
  runs) is **never** safe to delete.

## Anti-patterns

- **Bulk `rm -rf` on the whole `output_router_finetuning/`** — destroys
  active runs.
- **Deleting `hf_converted_*` for runs not yet in benchmark_results.csv**
  — you'll need it again in a few hours when the benchmark batch runs.
- **Deleting before refreshing the Pareto chart** — you may delete
  something that is currently on the frontier.
- **Treating "old" as "deletable"** — recency is irrelevant; distance
  from the Pareto frontier is what matters.
