# Goal: Qwen3-235B-A22B Pareto Frontier with RL on it

**Start by reading** `.cursor/skills/README.md` (infra invariants) and
`CLAUDE.md` (project context) before doing anything else.

Invoke as `/goal @goals/qwen235b_pareto_search.md`.

---

## Mission

Build a benchmarked Pareto frontier of router-training configurations for
**Qwen3-235B-A22B** that mirrors what we already have for Qwen3-30B-A3B,
specifically finding working points where **RL (alone or RL+aux) is
Pareto-optimal vs pure aux-loss and the pretrained baseline**.

---

## Reference state — 30B (what we want to replicate at 235B scale)

- 30B has **186 benchmarked checkpoints** in
  `examples/qwen3/benchmarks/benchmark_results.csv` across 7 W&B sweeps,
  spanning `bench_iteration ∈ {500, 1500, 2000, 3000, 5000, "final"}`.
- We benchmark **multiple iterations per run** because the optimal iter is
  config-dependent — aggressive load-balancing configs often peak at
  iter 2000 and degrade by 3000–5000.

Best 30B Pareto points (limit=1000, full Pareto across all categories):

| Category | Run (abbreviated) | iter | CP | acc% |
|---|---|---|---|---|
| rl+aux | `pareto_g0...cpb_n1_a0.015_r88` | 2000 | 2602 | 59.70 |
| rl+aux | `pareto_g0...cpb_n1_a0.01_r80` | 2000 | 2659 | 61.17 |
| rl+aux | `pareto_g0.3...cpb_n1_a0.01_r81` | 2000 | 2691 | 62.40 |
| rl+aux | `rl-discovery...cpb...kl0.0003_r20` | 1500 | 3157 | 62.43 |
| **aux_only** | `norl_aux0.01_r08` | 3000 | 3171 | 62.87 |
| rl+aux | `rl-discovery...lm1.0_aux0.001_cpb...kl0.001_r15` | 1500 | 3801 | 63.07 |
| rl+aux | `pareto_g0.8...gumbel_t0.3_gae0.95...kl0.0003_r47` | 2000 | 3831 | 63.57 |
| rl+aux | `explore...gumbel_t0.3_aux0.003_r12` | 3000 | 3998 | 63.83 |
| rl+aux | `explore...gumbel_t0.3_aux0.003_kl0.0003_r28` | 3000 | 4181 | 64.27 |
| rl+aux | `explore...gumbel_t0.3_cos_gae0.95_aux0.003_r15` | 3000 | 4181 | 64.43 |
| rl+aux | `rl-discovery...lm1.0_aux0.001_r11` | 1500 | 4341 | 64.53 |
| rl+aux | `rl-discovery...lm1.0_aux0.001_kl0.001_r14` | 1500 | 4549 | 64.73 |
| **rl_only** | `rl-discovery_rlc2.0...lm1.0_noaux_kl0.001_r09` | 1500 | 4733 | 64.83 |

**Pretrained 30B baseline: CP=4780, acc=64.10%.**

12 of 13 Pareto points are RL-based. RL dominates from the aggressive end
(CP=2602) through the high-accuracy end (CP=4733, +0.73pp over pretrained).
The cleanest 30B Pareto sweep was `qho6ccev` (config:
`examples/qwen3/run_config_jsons/pareto_frontier_sweep.json`).

Master CSV: `examples/qwen3/benchmarks/benchmark_results.csv`.

---

## Current 235B state

- **0 rows** in `benchmark_results.csv` for 235B. Nothing benchmarked yet.
- Five sweep configs already designed under `examples/qwen3/run_config_jsons/`:

  | Config | Purpose | save_interval status |
  |---|---|---|
  | `235b_pareto_sweep.json` | CPB + RL + aux + critic + KL grid | ⚠️ 1500 → fix to 300 |
  | `235b_pareto_nocpb_sweep.json` | Pure aux + RL+aux, no CPB | ✓ 300 |
  | `235b_rl_advantage_sweep.json` | Stochastic-routing ablation | ⚠️ 1500 → fix to 300 |
  | `235b_kl_validation_sweep.json` | KL coefficient sweep | ⚠️ 1500 → fix to 300 |
  | `235b_sanity_sweep.json` | 50-iter sanity check | ✓ 100 |

  **Three configs have `save_interval=1500` with `train_iters=1500` —
  fix to 300 before running** (otherwise no resumable mid-run checkpoint
  on SLURM eviction).

---

## Done criteria (all four must hold)

1. **≥ 30 benchmarked 235B (run, iteration) pairs** in the master CSV at
   `limit=1000`, with coverage:
   - ≥ 20 `rl+aux` runs spanning `rl_loss_coeff ∈ {0.1, 0.5, 1.0}` and CPB on/off.
   - ≥ 6 `aux_only` runs at `moe_aux_loss_coeff ∈ {0.001, 0.005, 0.01, 0.02}`.
   - The pretrained 235B baseline benchmarked once.
   - Multiple iters per run count toward the 30 — encouraged.

2. **Every Pareto-eligible run reached its configured `train_iters`** (typically
   1500). Crashed/OOM/walltime-evicted runs MUST be resumed first
   (skill: `resume-training-run`).

3. **Every Pareto-candidate run benchmarked at multiple iters** (skill:
   `iter-selection-for-pareto`: `i_early / i_mid / i_late`).

4. **Pareto chart shows ≥ 2 RL points strictly Pareto-dominating the
   `aux_only` frontier** (lower CP at matched accuracy, or higher accuracy
   at matched CP).

5. **Report at `examples/qwen3/benchmarks/235b_pareto_report.md`** listing
   Pareto-dominating configs + best iter + W&B URLs, the chart link, and a
   direct comparison vs the 30B Pareto.

Tighter quantified target:
- ≥ 2 distinct `rl_only` points on the frontier.
- ≥ 2 distinct `rl+aux` points on the frontier.
- Frontier spans CP < 0.7× pretrained-235B-baseline CP **and** accuracy ≥
  pretrained − 0.5pp.

---

## Execution plan

### Phase 1 — Audit current 235B state

**Goal**: know exactly what exists before launching anything. Report
findings before proceeding.

1. Read `.cursor/skills/README.md` for infra invariants.
2. Search `benchmark_results.csv` for any 235B rows.
3. For each run dir under `/lustre/.../output_router_finetuning/` matching
   `235b-*` or `235bnp*`:
   - Read `checkpoint/.../latest_checkpointed_iteration.txt`.
   - List `iter_NNNNNNN/` directories present.
   - Record the run's `train_iters` target (from W&B config or sweep JSON).
   - Mark **UNDER-TARGET** if `last_iter < train_iters`.
4. Inventory the five `235b_*.json` configs — note which need
   `save_interval` fix.
5. Check W&B (`nvr-israel/qwen3-router-training`) for any 235B sweep IDs;
   record state (running / finished / crashed).
6. Establish the **235B pretrained baseline CP**: search W&B for any 235B
   run's first eval (`critical_eval/critical_path` at iter 0), or run:
   ```bash
   sbatch examples/qwen3/benchmarks/submit_benchmark.sh \
     --checkpoint-dir /lustre/.../qwen-ckpts/Qwen3-235B-A22B-complete \
     --model-size A22B --run-name pretrained_235b_baseline --limit 1000
   ```
7. **Disk check** (skill: `checkpoint-disk-management`). If usage > 80% of
   quota, clean before any new work.

**Stop and confirm findings with the user before launching any new training.**

### Phase 1.5 — Continue under-target runs FIRST

**Do this before launching new sweeps** so already-spent compute isn't
wasted and the Pareto comparison is fair. Skill: `resume-training-run`.

For each sweep with under-target runs:
1. Confirm no chains are still actively training it (`squeue`).
2. Relaunch the sweep:
   ```bash
   cd examples/qwen3
   ./launch_sweep.sh <SWEEP_ID> --parallel 2 --agents 1 --8gpu
   ```
   Resume is automatic via `wandb_run_id.txt` +
   `latest_checkpointed_iteration.txt`, with cross-sweep fallback by
   config name.
3. Verify within 5 minutes: log shows `RESUME: Found previous wandb run ID`
   or `CROSS-SWEEP RESUME: matched ...` plus Megatron `loading checkpoint
   from iter_NNNNNNN`. If iter 0 instead — stop and investigate (likely
   `fresh_start: true`).
4. If a run's checkpoint dir is entirely missing, it can't resume — skip
   as failed. Edge cases (missing `wandb_run_id.txt` etc.) covered in the
   `resume-training-run` skill.

### Phase 2 — Anchor sweep: aux-only baseline at 235B

**Goal**: establish where pure aux-loss sits on the 235B Pareto frontier
before introducing RL. This is the comparison baseline for phases 3–4.

Pre-launch:
- Fix `save_interval` in any 235B config that needs it (see "Current 235B
  state" table).
- Verify combo count:
  ```bash
  cd examples/qwen3 && python3 -c "
  import sys; sys.path.insert(0, '.')
  from wandb_sweep_config import generate_filtered_sweep_config, load_config
  cfg = load_config('run_config_jsons/235b_pareto_nocpb_sweep.json')
  _, valid, _ = generate_filtered_sweep_config(cfg)
  print(f'Total runs: {len(valid)}')
  for i, c in enumerate(valid): print(f'  [{i:2d}] {c}')
  "
  ```

Launch (`235b_pareto_nocpb_sweep.json` covers both `use_rl_loss=false` and
`use_rl_loss=true` — the `false` rows ARE the aux-only anchor):
```bash
cd examples/qwen3
python3 wandb_sweep_config.py --config run_config_jsons/235b_pareto_nocpb_sweep.json
./launch_sweep.sh <SWEEP_ID> --parallel 2 --agents 1 --8gpu
```

Monitor: within 15 min verify allocation is `R`, log has iter-1 metrics,
`critical_eval/critical_path` is logged after first eval. Check at iters
500 and 1000 that `eval/lm_loss` is stable (not diverging).

Gate: ≥ 4 aux-only runs complete `train_iters`. Record their (CP, LM-loss)
as the aux-only anchor (skill: `training-log-forensics`).

### Phase 3 — Discovery sweep: where does RL shine?

**Goal**: find RL configurations that beat the Phase-2 aux-only anchor.
Test RL × aux × `rl_loss_coeff` × `lm_reward` × CPB.

Sweeps to run (priority order, watch QOS):
- `235b_rl_advantage_sweep.json` — RL vs noRL at aux={0.01, 0.015, 0.02},
  CPB on/off, `lm_reward ∈ {0, 0.3}`, Gumbel t=0.3 on/off.
- `235b_pareto_sweep.json` — full CPB + KL grid.

```bash
cd examples/qwen3
python3 wandb_sweep_config.py --config run_config_jsons/<file>
./launch_sweep.sh <SWEEP_ID> --parallel 2 --agents 1 --8gpu
```

Intermediate analysis (at 500 iters, before final benchmarks):
- Compare W&B `critical_eval/critical_path` for RL vs noRL at matched aux.
- If RL runs consistently show CP ≥ noRL at the same aux after 500 iters,
  apply `hypothesis-reassessment` skill before continuing.

**Decision point**: if after 1000 iters **all** RL runs have CP ≥ all noRL
runs at matched aux+CPB, **surface this finding to the user and propose
Phase-3b before launching anything else.** Do not silently continue.

### Phase 4 — Refinement: dense neighbors around RL-dominant points

**Goal**: fill in the Pareto frontier around the best RL configurations
found in Phase 3.

Design a new sweep config with:
- Dense `rl_loss_coeff` grid around the best value from Phase 3 (e.g., if
  0.5 won, test 0.3, 0.5, 0.7).
- `train_iters=3000` for at least one config (matching 30B training length).
- Matching noRL controls at the same aux level.

Use `slurm-wandb-sweep` skill for config design.

### Phase 5 — Full lm-eval, chart, report

Only after all Pareto-candidate runs reached their target iter and
multi-iter benchmarks (per `iter-selection-for-pareto`) are submitted:

```bash
cd examples/qwen3/benchmarks

python3 collect_benchmark_results.py

python3 generate_pareto.py \
  --limit-filter 1000 --min-accuracy 55 --clean \
  --output ../../../pareto_235b.html

# Full-dataset benchmarks for top 5 Pareto points (omit --limit)
python3 pareto_benchmark.py --sweep-id <SWEEP_ID> --step <best_iter> \
  --benchmark --all --per-job 3
```

**235B conversion caveat**: `pareto_benchmark.py` /
`submit_batch_benchmark.sh` use the 30B 4-GPU converter. For 235B you must
run the 2-node converter **first** for each `(run, iter)`:
```bash
TRAINED_MEGATRON_CKPT=<ckpt_dir> ITER_NUM=<N> \
  sbatch examples/qwen3/benchmarks/cp_latency_test/submit_convert_235b.sh
```
Once `<run>/hf_converted_iter<N>_cp/` exists, the batch benchmark skips
conversion.

Pretrained 235B baseline (once, if not done in Phase 1):
```bash
sbatch examples/qwen3/benchmarks/submit_benchmark.sh \
  --checkpoint-dir /lustre/.../qwen-ckpts/Qwen3-235B-A22B-complete \
  --model-size A22B --run-name pretrained_235b --limit 1000
```

Write `examples/qwen3/benchmarks/235b_pareto_report.md` with:
- Pareto-dominating configs + best iter + W&B URLs.
- Pareto chart link.
- Direct comparison vs 30B Pareto (matched-category table).
- Any configurations where RL fails to dominate, and hypothesis why.

**After the chart is updated, run a disk pass** (skill:
`checkpoint-disk-management`) to delete checkpoints proven non-frontier
and far from it.

---

## Operating rules (always honored)

**Infrastructure**
- SLURM account: `nvr_israel_rlop`. Partition: `interactive` (4h max).
- Every new sbatch you write or copy MUST include `#SBATCH --cpus-per-gpu=2`
  (default eats 31 CPUs/GPU).
- 235B conversion: **always 2 nodes with PP=2 EP=8** (`submit_convert_235b.sh`).
  Single-node 235B conversion OOMs and TP*EP/world_size mismatches.
- 235B training: EP=8 ⇒ `--agents 1 --8gpu` per chain. Never
  `--agents 2 --8gpu` (two agents fight for the same 8 GPUs and stall).
- `save_interval < train_iters` always (use 300 for 1500-iter 235B).
  `fresh_start=false`. `exit_duration_in_mins=230`.

**Methodology**
- Phase 1.5 (resume) before Phase 2 (new sweeps). Always.
- Don't hardcode `iter=N`. The optimum varies by config — use
  `i_early / i_mid / i_late` per run.
- Verify `iter_NNNNNNN/` exists and is non-empty before submitting any
  benchmark for that iter. Skip + log if not.
- **CPB annotation**: RL+CPB runs may appear on the training-time Pareto
  frontier, but `critical_path_bias` is `persistent=False` and the gain
  does not transfer to stock HF inference (skill: `cp-microbench`). Always
  annotate such runs as "training-time only" in the report.
- **Skip KL on LM logits** — proven near-inert for router gradients at 30B
  (Bug B fix established this empirically). Default `kl_loss_coeff=0` in
  new configs unless explicitly testing KL.

**Disk discipline (built into the loop, not optional)**
- Check `df` / quota at the start of each phase and after Phase 5.
- If usage > 80% of quota, run `checkpoint-disk-management`.
- Safe to delete (priority order): regeneratable HF conversions of
  already-benchmarked runs → mid-run checkpoints of non-frontier
  far-from-frontier runs → final-iter checkpoints of completed
  far-from-frontier runs.
- **Never delete**: pretrained checkpoints (`qwen-ckpts/...`), the latest
  checkpoint of any active or paused run, or any frontier run's checkpoints.
- Always dry-run (`echo`) the `rm` plan before executing.
- Log deletions in `examples/qwen3/benchmarks/disk_cleanup_log.md`.

**Hygiene**
- Don't block on foreground sleeps. Submit, capture JOB_ID, poll with the
  pattern in `.cursor/skills/README.md`, work other phases / sweeps in
  parallel.
- Update the TODO list at each phase transition. Report to user when
  (a) a sweep finishes, (b) the chart is updated, (c) the report is written,
  or (d) you hit a blocker needing a human decision.
- Commit nothing unless explicitly asked.

---

## Stop and ask the user if

- Phase 3 shows RL fully dominated by aux-only (CP ≥ noRL at all configs).
- A bug invalidates prior 235B runs (apply `hypothesis-reassessment` +
  `training-bug-investigation` skills first).
- QOS / cluster state blocks a planned phase for >24h.
- A frontier point exceeds the 30B shape (beats 30B Pareto at same CP or
  accuracy).
- Disk cleanup would touch a run that might still be of interest.
- Any SLURM/container/NCCL error you haven't seen documented in the skills.

---

## Between-phase checklist

Before declaring a phase complete and moving to the next:
- [ ] Show a comparison table (`training-log-forensics` skill).
- [ ] Reassess any results that contradict prior findings
      (`hypothesis-reassessment` skill).
- [ ] Update the progress tracker below.
- [ ] Report to user: "Phase N complete. Next: Phase N+1 — [plan summary]."

---

## Progress tracker (update as phases complete)

- [ ] Phase 1 audit complete — 235B baseline CP=_____ acc=_____%
- [ ] Phase 1.5 — under-target runs relaunched: _____ runs
- [ ] Phase 2 — `235b_pareto_nocpb`: sweep_id=_____, _____ runs done
- [ ] Phase 3 — `235b_rl_advantage`: sweep_id=_____, _____ runs done
- [ ] Phase 3 — `235b_pareto_sweep`: sweep_id=_____, _____ runs done
- [ ] Phase 4 — refinement sweep: sweep_id=_____, _____ runs done
- [ ] Phase 5 — benchmarks submitted (multi-iter): _____ (run, iter) pairs
- [ ] Phase 5 — `collect_benchmark_results.py` run, CSV has _____ 235B rows
- [ ] Phase 5 — Pareto chart: pareto_235b.html
- [ ] Phase 5 — Success criteria met? 1:____ 2:____ 3:____ 4:____ 5:____
- [ ] Phase 5 — Report written: 235b_pareto_report.md

---

## Skills (read each `SKILL.md` before invoking)

Index: `.cursor/skills/README.md`. Most relevant for this goal:

| Skill | Used in |
|---|---|
| `launch-sweep` / `slurm-wandb-sweep` | Phases 1.5, 2, 3, 4 |
| `resume-training-run` | Phase 1.5 |
| `iter-selection-for-pareto` | Phase 5 |
| `pareto-benchmark` | Phase 5 |
| `convert-mcore-to-hf` | Phase 5 (235B path) |
| `run-lm-eval-benchmark` | Phase 5 |
| `collect-benchmark-results` | Phase 5 |
| `generate-pareto-chart` | Phase 5 |
| `checkpoint-disk-management` | Continuous; explicitly after Phase 5 |
| `cp-microbench` | CPB-portability annotation in report |
| `training-log-forensics` | Between-phase comparison tables |
| `training-bug-investigation` | When metrics look mechanistically off |
| `hypothesis-reassessment` | When results contradict prior conclusions |

---

**Begin Phase 1 now.** Build a Qwen3-235B-A22B Pareto frontier with RL on
it, matching the density and quality of evidence we already have at 30B
(186 benchmarked configs, RL points dominating non-RL at multiple frontier
locations). Right now `benchmark_results.csv` has 0 rows for 235B. Close
that gap.
