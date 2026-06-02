---
name: cluster-babysitter
description: Stand up and run a persistent SLURM job babysitter that monitors in-flight training/benchmark jobs every ~20 min, refreshes the results CSV, regenerates the Pareto plot when the frontier moves, auto-remediates known failures, and auto-retries preempted jobs. Use when a long-running sweep/benchmark campaign is in flight and must be kept progressing without manual polling.
---

# Cluster babysitter (persistent monitor + auto-remediation)

Goal: keep a multi-hour/multi-day campaign progressing **without being asked**. Two pieces:

## Architecture: supervisor + single-cycle watcher
- **`babysit_supervisor.sh`** (runs on the local/dev box, not the cluster): infinite loop that
  invokes the watcher once per cycle and sleeps `INTERVAL_SEC=1200` (20 min). Wrap each cycle in
  `timeout 900 python3 watcher.py` so a hung cycle can't wedge the loop. PID-file guard prevents
  double-start. Survives watcher crashes (each cycle is a fresh process).
  - Restart: `setsid bash babysit_supervisor.sh >supervisor.log 2>&1 </dev/null &`
  - Check before relaunch: `ps -p "$(cat /tmp/babysit_supervisor.pid)"`.
- **`babysit_watcher.py`** — does ONE idempotent cycle and exits. Per cycle:
  1. ssh cluster, `squeue` + categorize jobs.
  2. Refresh CSV (`collect_results_unified.py`).
  3. **Git-commit the CSV** (the cross-cluster sync mechanism: ORD is the hub, NRT pulls).
  4. Compute the frontier; if it moved, regenerate the Pareto plot (both limit=inf canonical
     and limit=1000 reference). Don't just say "check back later" — regen on movement.
  5. Auto-remediate known failures (see below).
  6. Log every action to a watcher log for later audit.

## Auto-remediate WITHOUT asking (apply known fixes)
- **Broken HF dir** (missing shards / 0-byte `config.json`) → reconvert from the distcp iter dir.
- **Failed lm-eval** from a broken HF → reconvert then re-eval.
- **Crashed wandb sweep** (multiple cells `state=crashed`) → relaunch the sweep agent.
- **Preempted / transient-failed job** (`CANCELLED by <reaper-uid>`, port conflict, "Memory
  required by task not available", cluster didn't form) → fix the deterministic cause if known,
  then **auto-resubmit**. Don't report-and-wait.
- **Idle clusters** → schedule the pending sweeps/benchmarks immediately (see [[gpu-job-dispatch]]).

## Gate these on explicit user OK (do NOT do autonomously)
- Destructive ops (delete distcp, force-cancel running training, downgrade deps).
- Methodology changes (switch EP, change discount-factor/reward defaults).
- Brand-new sweeps with novel axes.

## HF disk-cleanup guard
Only delete a `hf_converted_iter<N>` dir once the CSV has **both** limit=1000 AND limit=inf
**and** a walltime JSON for that (cell,iter) — deleting earlier strands pending walltime jobs.

## Practical cadence
Monitor every ~20 min while jobs are in flight (short enough to catch problems, long enough to
amortize ssh + CSV scrape). For waiting on a specific external state (transfer done, job
formed), prefer an explicit poll-until loop with a completion signal over fixed sleeps.
