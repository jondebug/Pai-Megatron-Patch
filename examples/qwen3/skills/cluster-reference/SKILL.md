---
name: cluster-reference
description: Reference for the two GPU clusters used by this project (NRT and ORD) — login hosts, GPU type, storage paths/quotas, SLURM account, the partitions on each and their properties (single- vs multi-node, interactive, preemptible/backfill, idle reaper), and cross-cluster transfer. Use when choosing where/how to submit, where to put data, or how to move checkpoints between clusters.
---

# Cluster reference: NRT & ORD

SLURM account for both: **`nvr_israel_rlop`**. Always `--gpus-per-node=N` (never `--gpus=N`).
Verify live state with `sinfo -h -o "%P %a %t %D"` and `squeue`; partition lists below are the
GPU-relevant ones observed.

## ORD — cs-oci-ord-login-01.nvidia.com (A100 80 GB)
- **Storage:** `/lustre/fsw/portfolios/nvr/users/jonathanp` (project **scne**, quota raised to
  **500 T**). Project area also under `/lustre/fsw/portfolios/nvr/projects/...`. Home dir is
  small — keep caches (`~/.cache`, HF cache) on lustre.
- **GPU partitions:**
  - `polar3`, `polar4` — **bulk multi-node GPU work** (the default for big sweeps). `polar`,
    `grizzly`, `longrun_polar`, `longrun_grizzly` also exist.
  - `interactive`, `interactive_singlenode` — interactive; often have **idle capacity** and are
    **reaper-exempt**. Good for quick turnaround / fiddly bring-up.
  - `batch_singlenode`, `backfill_block1` (multi-node, **preemptible**),
    `backfill_singlenode` (**preemptible**, lots of nodes).
  - CPU-only: `cpu`, `cpu_short`, `cpu_long`, `cpu_interactive`, `cpu_datamover`.
- 235B conversion historically: single-node EP=8 (or 2-node PP=2 EP=8 for the heavier path).

## NRT — oci-nrt-cs-001-login-01.nvidia.com (H100 80 GB)
- **Storage:** `/lustre/fs1/portfolios/nvr/projects/nvr_israel_rlop/users/jonathanp`. Group
  `nvresearch`. This is where NRT training output, ckpts, datasets, containers live.
- **GPU partitions** (all share ~83 GPU nodes, usually `mix`):
  - `batch_block1` — **default, multi-node** (used for EP=16 continuation + EP=8 fresh-start).
  - `batch_large`, `batch_long`, `batch_short`, `batch_singlenode` — vary by walltime/size;
    `batch_short` turns over fastest.
  - `interactive` — **reaper-exempt**, try it when `batch_*` is queue-bound.
  - `backfill` — **preemptible**.
  - CPU-only: `cpu`, `cpu_short`, `cpu_long`, `cpu_interactive`, `cpu_datamover`, `cpu_small`.
- **Idle-job reaper is active**: cancels jobs whose GPUs idle (DCGM util ≈0) for ~30 min. Fail
  fast on cluster bring-up; use interactive partitions for setup. See [[gpu-job-dispatch]].

## Cross-cluster transfer
- **NRT can `ssh`/`rsync` from ORD** after key auth (NRT pulls; ORD is the git/results hub).
- Big checkpoints: **8 parallel rsync streams ≈ 96 MB/s** vs ~13 MB/s single-stream; run under
  `setsid`/`nohup` on the login node so it survives SSH drops; judge progress by shard count.
- When moving a **distcp** checkpoint, include the hidden `.metadata` index and the top-level
  tokenizer/`*.json` files — plain `iter_*/` rsync misses them and load fails. See
  [[accuracy-benchmark]] / [[gpu-memory-management]].
- The mcore base for fresh-start (`Qwen3-235B-A22B-to-mcore`, ~543 G) and HF base
  (`Qwen3-235B-A22B`, ~438 G) must each be present on the cluster you train/eval on.
