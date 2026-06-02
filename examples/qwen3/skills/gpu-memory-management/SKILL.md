---
name: gpu-memory-management
description: Fit large Qwen3 MoE (235B-A22B) router-training into 80 GB GPUs — choose expert-parallelism (EP), activation recompute, and allocator settings; diagnose and fix CUDA OOMs. Use when a 235B job OOMs, when deciding EP/node count, or when a checkpoint won't load due to shard/EP mismatch.
---

# GPU memory management for 235B-A22B MoE training (80 GB GPUs)

## Per-GPU budget (bf16, router-only)
- Non-expert weights (replicated): ~20 GiB. Experts (225B params / EP × 2 B): EP=8 → ~56 GiB,
  EP=16 → ~28 GiB.
- **EP=8 (1 node): ~71 GiB weights → OOMs on 80 GB.** The actual tip-over is the **LM-head
  logits** (`micro_batch×seq × 151936` materialized on the output rank) — even `lm_loss`
  *logging* in router-only training triggers it. `expandable_segments:True` helps but isn't
  enough alone.
- **EP=16 (2 nodes): ~48 GiB weights → comfortable.** This is why **continuation runs (EP=16)
  train fine while fresh-start (forced to EP=8) OOMs.**

## Fixes, in order of preference
1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — reclaims fragmentation; necessary
   but not sufficient at EP=8.
2. **Full activation recompute** (the EP=8 fix that works): run_mcore `AC=full` **and** add
   `--recompute-granularity full`. ⚠️ run_mcore's `full` branch emits `--recompute-method
   uniform` but leaves `--recompute-granularity full` commented, and its `sel` branch emits
   `--recompute-activations` — having both selective+method triggers
   `AssertionError: recompute method is not yet supported for selective recomputing
   granularity`. So use `AC=full` (drops `--recompute-activations`) + supply granularity. Frees
   GBs of activations → the ~38 MB logit overflow fits.
3. **Go EP=16** (durable): needs a 16-way-loadable base (re-convert the legacy 8-shard mcore
   base, or load from distcp which reshards). More headroom, no recompute slowdown.
4. Last resorts (change methodology — get user OK): drop `--rl-ppo-reeval` (second forward),
   lower `--global-batch-size`.

## EP ↔ checkpoint shard layout (load failures)
- A **legacy mcore checkpoint** (`mp_rank_00_000..NNN/model_optim_rng.pt`) has a FIXED shard
  count = its save EP. The base `Qwen3-235B-A22B-to-mcore` is **8 shards (EP=8)** → cannot load
  at EP=16 (ranks 8-15 look for missing `mp_rank_00_008..015`). **distcp / `torch_dist`** saves
  are parallelism-agnostic and reshard to any EP — that's why continuation works at EP=16.

## Diagnosing OOM (and other crashes)
- The real error hides behind torch-elastic `ChildFailedError` (`error_file: <N/A>`). Grep the
  `.err` for the **rank0 Traceback** / `OutOfMemoryError` / the failing `empty_strided_cuda(...)`
  allocation — that line names the tensor (e.g. the `*,151936` logits) and the deficit.
- `--empty-unused-memory-level` accepts **{0,1,2} only**. Setting **3 silently kills every job
  at argparse** with no clear error in the chain wrapper.
