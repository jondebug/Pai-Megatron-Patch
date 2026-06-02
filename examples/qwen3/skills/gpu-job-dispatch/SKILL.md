---
name: gpu-job-dispatch
description: Dispatch SLURM GPU jobs to maximize GPU time and minimize queue wait. Use when jobs are stuck PENDING, when a cluster is idle and should be filled, or whenever submitting batches of training/benchmark jobs. Covers multi-partition spreading, interactive partitions, idle-reaper avoidance, and pending-reason diagnosis. Tuned for the NRT (oci-nrt-cs-001) and ORD (cs-oci-ord) clusters.
---

# Dispatching SLURM jobs for maximum GPU time

Goal: get jobs **RUNNING** fast and keep GPUs busy. The #1 mistake is submitting to a
single (contended) partition and letting jobs sit PENDING while idle capacity exists
elsewhere.

## The dispatch checklist (do this every time)

1. **Diagnose why anything is pending** — the reason dictates the fix:
   ```
   squeue -u $USER -o "%i %P %T %r %S"
   ```
   - `Priority` → you're behind in the queue; **spread across more partitions** (incl. interactive). Idle nodes are often held by backfill for higher-priority jobs.
   - `Resources` → not enough free nodes in that partition right now; spread / shrink the ask.
   - `QOSMax*`, `AssocGrp*` → you hit a per-user/QoS cap; use a different partition/QoS.
   - `ReqNodeNotAvail`/`ReservedForMaintenance` → nodes down/reserved; pick another partition.

2. **See where capacity actually is** (idle or mixed = schedulable):
   ```
   sinfo -h -o "%P %a %t %D" | grep -iE "idle|mix"
   ```
   Target partitions showing `idle` or `mix` nodes for your GPU geometry.

3. **Submit spanning ALL eligible partitions** — SLURM runs the job on whichever frees
   first. Use a comma list:
   ```
   sbatch -p interactive,batch_short,batch_block1,batch_long,batch_singlenode my.sh   # NRT
   sbatch -p interactive_singlenode,polar4,polar3,backfill_singlenode my.sh           # ORD single-node
   ```
   For **already-pending** jobs, retarget without resubmitting:
   ```
   scontrol update jobid=<ID> partition=interactive_singlenode,polar4,polar3,backfill_singlenode
   ```

4. **Always include `interactive` partitions.** They have dedicated/idle capacity, often
   higher scheduling priority, faster turnaround, AND are **auto-exempt from the idle-job
   reaper**. On these clusters: NRT `interactive`; ORD `interactive` / `interactive_singlenode`.

5. **Match geometry to partition:**
   - Single-node (≤8 GPU) jobs → prefer `*_singlenode` partitions (there are usually more of
     them idle). Always use `--gpus-per-node=N`, **never `--gpus=N`** — the latter lets SLURM
     spread GPUs across nodes, which both breaks single-node assumptions and trips the
     idle-GPU reaper.
   - Multi-node jobs → only the multi-node partitions (NRT `batch_block1`; ORD `polar3/polar4`,
     `backfill_block1`). Add `--mem=0` so overlapping srun steps can share node memory.

## Never waste allocated GPUs (the idle-job reaper)

An **idle-job reaper** (`DCGM_FI_DEV_GPU_UTIL`/`SM_ACTIVE` ≈ 0 for 30 min) cancels jobs whose
GPUs sit idle — wasted GPU-hours and a lost allocation. Avoid it:
- **Fail-fast on setup that may not complete.** For multi-node jobs that must form a cluster
  (e.g. Ray): poll for the expected GPU count and **exit non-zero within ~6 min** if it
  doesn't form, instead of proceeding with a partial cluster that then idles. Example loop:
  ```
  formed=0; for i in $(seq 1 36); do
    TOT=$(ray status 2>/dev/null | awk -F/ '/GPU/{print $2}' | awk '{printf "%d",$1}')
    [ "${TOT:-0}" -ge "$NEED" ] && { formed=1; break; }; sleep 10; done
  [ "$formed" != 1 ] && { echo "cluster failed to form"; exit 2; }
  ```
- Prefer interactive partitions for fiddly bring-up (reaper-exempt).
- If you legitimately need idle time, request an exemption per the cluster's GPU Idle Time
  Exemption guide.

## Keep the queue full (continuous utilization)

- **Self-chaining**: each job pre-submits its successor with `--dependency=afterany:$SLURM_JOB_ID`
  BEFORE training, so capacity is reused across the 4h walltime cap. Frequent `--save-interval`
  so a chained successor resumes (framework auto-resume from the save dir). ⚠️ A fast-failing
  chain cascades — if jobs die in <a few min, the whole chain burns through; fix the crash
  before relying on `afterany`.
- **Migrate work from contended → free clusters.** If cluster A is saturated and B is idle,
  move the next batch to B (fresh-start sweeps need no data transfer; continuations need the
  checkpoint staged first).
- **Proactively refill idle clusters** — don't wait to be told. An empty queue is wasted GPUs.

## Backfill / preemptible partitions

`backfill*` partitions are **preemptible** (jobs get killed when a higher-priority job needs
the node). Fine for **restartable** work (benchmarks, chained training with frequent saves);
avoid for long jobs that can't cheaply resume.

## Helper

`dispatch.sh` (next to this file) wraps the above: it inspects partition availability and
submits an sbatch script spanning the best GPU partitions (interactive first). Usage:
```
bash dispatch.sh --cluster nrt --nodes 1 -- /path/to/job.sh ARG1 ARG2
bash dispatch.sh --cluster ord --nodes 1 --jobname bench3000 -- /path/to/job.sh --manifest m.json
```
