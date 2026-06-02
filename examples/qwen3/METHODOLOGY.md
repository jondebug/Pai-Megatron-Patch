# Qwen3-235B-A22B Router-Training Pareto Project — Methodology & Findings

Living document. Captures the research goal, methods, infrastructure, gotchas, and findings
for the 235B MoE router-only RL training / critical-path Pareto study. Companion runnable
skills live in `examples/qwen3/skills/`.

> Status note: quantitative values marked *(prelim)* are from the in-progress CSV/wandb and may
> shift; numbers without a tag are stable measurements. Anything marked *(TBD)* still needs data.

---

## 1. Goal & core idea

Train **only the routers** of a frozen Qwen3-235B-A22B MoE (94 layers, 128 experts, top-8) with
RL so that token routing is **more load-balanced along the critical path**, reducing inference
step time at large expert-parallelism — at minimal accuracy cost. Deliverable: a **Pareto
frontier of accuracy vs. critical path (CP)**, showing RL load-balancing dominates the classic
aux-loss approach.

- **Router-only**: LM weights frozen; only the per-layer router (gating) weights train. Tiny
  optimizer state, but backprop still flows through the full model (activations dominate memory).
- **Critical path (CP)** = `Σ over MoE layers of (max tokens on any single expert in that layer)`.
  At high EP (experts spread across GPUs), the most-loaded expert per layer sets the step time,
  so CP is the inference-latency-relevant load-imbalance metric. Training logs it as
  `num_tokens_on_critical_path`; the benchmark CSV stores it as `eval_crit_path`.

---

## 2. Reward functions (what `--rl-reward-type` can be)

All are PPO rewards (`--rl-algorithm ppo`, mean baseline, `--rl-ppo-epochs 1`, GAE λ=1,
discount γ via `--rl-discount-factor`). Router-only.

| reward type | idea |
|---|---|
| `critical_path` | directly penalize the max-loaded expert per layer (the CP itself). Most on-target for the latency objective. |
| `per_token_load_weighted` | penalize per-token contribution to expert load (smoother signal; the v5 default). |
| `topn_load` | penalize load on the top-N most-loaded experts. |
| `per_token_topn_binary` | binary variant of the above. |
| `entropy` | reward routing entropy (spread) — indirect balancing. |

Modifiers:
- **LM reward** (`--rl-lm-reward-coeff`): adds an LM-quality term so balancing doesn't wreck
  accuracy (30B-proven). Costs memory (materializes vocab logits — see §4).
- **Gumbel / stochastic routing** (`--rl-stochastic-routing --rl-stochastic-temperature 0.3`):
  exploration during training. On 235B it appears on the low-CP frontier (e.g. `gumbel_t0.3`
  cells r01/r07) — competitive but not dominating *(prelim)*.
- **Aux loss** (`--moe-aux-loss-coeff`): the classic Megatron load-balancing loss; our baseline
  to beat. Swept 0.001–0.02. Higher aux → lower CP but lower accuracy.
- **Discount factor γ** (`--rl-discount-factor`): 30B tested γ∈{0,0.2,0.3,0.5,0.8}. **235B γ data
  is still missing** — the v7 γ sweep crashed on ORD (max iter 843, never benchmarked); being
  re-run on NRT as the migrated gamma sweep *(TBD)*.
- **KL** (`--kl-loss-coeff`): KL-to-reference penalty to prevent the router drifting too far.

---

## 3. Training setup

- Backend: Pai-Megatron-Patch + Megatron-LM-250624, `pai-megatron-patch_25.04` container.
- `run_mcore_qwen3.sh` positional args: `... TP PP CP ETP EP SP DO FL SFT AC OPT_OFFLOAD
  SAVE_INTERVAL ...`. AC (`sel`/`full`/`none`) controls activation recompute.
- **Fresh-start** loads the mcore base (positional `PRETRAIN_CKPT`); **continuation** resumes
  from a cell's own distcp save (`--load` override).
- Chaining: each job pre-submits its `afterany` successor before training and saves frequently,
  so a 4 h walltime cap doesn't lose progress (framework auto-resumes from the save dir).
  ⚠️ a fast-failing chain cascades through all depths in minutes — fix the crash first.

---

## 4. 235B performance & memory requirements (hard-won)

- **Per-GPU weight memory** (bf16, replicated non-expert + sharded experts):
  - EP=8 (1 node): ~71 GiB/GPU → **OOMs on 80 GB**. The tip-over is the **LM-head logits**
    (`128×151936` per micro-batch) materialized on the output rank — even lm_loss *logging*
    triggers it. `expandable_segments:True` alone isn't enough.
  - EP=16 (2 nodes): ~48 GiB/GPU → comfortable. This is why **continuation runs (EP=16) train
    fine but fresh-start (forced to EP=8 by the 8-shard base) OOMs.**
- **Fix to fit EP=8 fresh-start**: full activation recompute — set run_mcore `AC=full` (drops
  `--recompute-activations`, adds `--recompute-method uniform --recompute-num-layers`) **plus**
  `--recompute-granularity full` (run_mcore leaves it commented). Frees GBs of activation memory
  → the ~38 MB logit overflow fits. Validated training past iter 30 at EP=8.
- **EP must match the checkpoint shard layout** for legacy mcore loads. The base
  `Qwen3-235B-A22B-to-mcore` is 8 `mp_rank` shards (EP=8) and cannot load at EP=16; distcp
  (`torch_dist`) saves reshard freely. To run fresh-start at EP=16 you'd re-convert the base to
  16-way (durable alternative to recompute-full).
- Throughput: EP=16 training ≈ 7 s/iter (steady state); EP=8 fresh-start similar once compiled
  (iter-1 includes ~145 s compile). Conversion (distcp→HF, EP=8): ~30–45 min/cell.

---

## 5. Managing results in a CSV

- `collect_results_unified.py` → `benchmark_results.csv`: one row per (run_name, bench_iteration,
  limit). Columns include lm_loss, per-task acc, `benchmark_avg`, `eval_crit_path`, checkpoint
  path, and joined vLLM walltime (`wt_decode_tps/wt_ttft_ms/wt_e2e_ms`). It scrapes CP from
  training logs for old cells lacking wandb CP, and pins the pretrained baseline CP=8800.
- **limit=1000 vs limit=inf**: always evaluate both; full (`inf`) reads ~3–4 pp higher. The
  canonical Pareto plot uses `--limit-filter inf`.
- **Cross-cluster sync via git**: ORD is the hub; the CSV is git-committed each babysitter cycle
  and NRT pulls. ⚠️ **NRT-run lm-eval results are NOT auto-ingested** (collector runs on ORD) —
  copy them over or extend the collector.
- `generate_pareto.py` builds the plot; `ensure_baseline_point()` always injects the pretrained
  baseline (acc 76.74, CP 8800) regardless of limit filter.

---

## 6. Accuracy benchmarks

- lm-eval-harness: HellaSwag, ARC-Challenge, WinoGrande; `acc_norm` (winogrande `acc`).
- **acc ≈ 95.17 − 7.59·lm_loss (r = −0.995)** — near-deterministic; use lm_loss as a cheap
  accuracy proxy to triage which checkpoints deserve a full eval.
- 235B: manifest-driven batch (`submit_batch_benchmark_235b.sh`, EP=8) or NRT lm-eval via vLLM.
  See [[accuracy-benchmark]] skill.

---

## 7. Walltime benchmarks & the impact of EP

- Single-node vLLM (TP=8/EP=8): pretrained 235B = **108.8 tps / 155 ms TTFT / 18.8 s e2e**.
- **Actual throughput DROPS as EP grows across nodes**: EP=8 = 108.8 → **EP=16 = 86.9 tps**.
  Cross-node MoE all-to-all (dispatch/combine) dominates; intra-node NVLink at EP=8 is fastest.
- Multi-node Ray bring-up is fiddly (Ray not in container, head-IP routing, port pinning,
  fail-fast) — see [[walltime-benchmark]] and [[gpu-job-dispatch]].
- **Profiling decomposition** (`VLLM_TORCH_PROFILER_DIR` + `categorize_trace.py`): split each
  decode step into attn / ffn / dispatch+combine / other, then project an **NVLink-domain tps**
  by replacing the inter-node `(K−1)/K` fraction of all-to-all with NVLink-speed. This isolates
  the CP/compute benefit from the inter-node comm penalty *(in progress)*.
- **Simulation (`cp_microbench.py`)** estimates EP{1..128} *expert-compute* step time from
  routing traces. Caveats: models compute only (no comm) and shipped with 30B arch constants —
  must be set to 235B (4096/1536/94) for valid 235B numbers. A first run on r09 showed ~1.0×
  speedup, but that is **suspect** for those two reasons *(prelim)*.

---

## 8. Key findings to date

- **RL load-balancing dominates aux-only** on the acc-vs-CP frontier. RL frontier (limit=inf):
  e.g. r15@3000 (74.44%, CP 4682) and r08@3000 (74.61%, CP 5533) beat aux-only at matched acc
  *(prelim, from CSV)*.
- **Pretrained baseline**: acc 76.74%, CP 8800.
- **iter-3000 continuation is cell-dependent**: it *helped* r08 and r15 (lower CP, on frontier)
  but *hurt* r09 (acc 74.08%@1500 → 72.91%@3000, over-trained into a dominated point). Don't
  assume more iterations help every cell.
- **CP→walltime at 235B is comm-limited**: at EP=16 the CP win doesn't show up in raw throughput
  because cross-node comm dominates — hence the profiling + NVLink-domain projection.
- **γ (discount factor) on 235B: no valid data yet** (v7 crashed; re-running).
- **gumbel/stochastic** cells appear on the low-CP frontier but don't clearly dominate *(prelim)*.

---

## 9. KL debugging & current impact

- We chased a KL-related regression earlier in the project (the `235b-klv*` cells). The KL
  penalty (`--kl-loss-coeff`) is meant to keep the trained router near the reference to protect
  accuracy. The KL-variant cells produced a few benchmarked points (e.g. `235b-klv_ppo_cpb...`
  ≈ 74.0% acc, CP ~4844) but did **not** clearly beat the plain RL+aux frontier.
- **Current impact of KL**: marginal/neutral on 235B so far — it has not unlocked a better
  acc/CP trade than RL+aux without KL. *(This section needs a clean re-derivation from the CSV
  to state the exact KL-on vs KL-off delta — flagged TBD.)*

---

## 10. Cluster operations

- Two clusters: **ORD** (cs-oci-ord, A100-80GB, `/lustre/fsw`, partitions polar3/polar4/
  interactive*) and **NRT** (oci-nrt-cs-001, H100-80GB, `/lustre/fs1`, partition batch_block1/
  interactive*). NRT can `ssh`/`rsync` from ORD after key auth.
- **Idle-job reaper** cancels jobs whose GPUs idle >30 min — fail-fast on cluster bring-up and
  prefer interactive partitions (reaper-exempt). See [[gpu-job-dispatch]].
- **Maximize GPU time** by submitting across many partitions (interactive first), retargeting
  pending jobs with `scontrol update partition=`, and keeping queues full via chaining +
  proactive refill + cross-cluster migration.
- **Babysitter** keeps the campaign progressing unattended — see [[cluster-babysitter]].

---

## 11. Open questions / next

- 235B γ sweep results (migrated to NRT).
- Clean KL on/off delta from the CSV.
- NVLink-domain projection numbers at EP=32/64/128.
- Benchmark the v10 reward-formulation sweep once it reaches iter 1500.
- Re-run `cp_microbench.py` with correct 235B constants on a high-CP-reduction cell (r15@3000).
