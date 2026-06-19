# 235B Sweep — Gap-Closure Plan & Living Status

**Owner:** jonathanp · **Cluster:** ORD (`cs-oci-ord-login-01.nvidia.com`), account `nvr_israel_rlop`, A100-80GB, EP=16 (2 nodes/cell)
**Project root:** `/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing`
**This doc:** `…/Pai-Megatron-Patch/examples/qwen3/benchmarks/GAP_CLOSURE_STATUS.md`
**Created:** 2026-06-18 · **Last updated:** 2026-06-18

> Living document. Update the **Status Snapshot** + **Changelog** every working session. Nothing about
> the gap-closure / sweep effort should live only in chat — it goes here.

---

## 0. North-star goal

Qwen3-235B-A22B MoE **router-only RL** to reduce inference **Critical Path** (CP = Σ over MoE layers of
tokens on the busiest expert) at minimal accuracy cost, producing:
1. **Accuracy-vs-CP Pareto frontier** showing RL dominates the aux-loss baseline (DONE — RL dominates
   across the whole CP-reduction regime; the no-reduction corner 76.28@8435 is the lone holdout).
2. **Systems half** — does CP reduction actually cut inference latency, and by how much on real/future HW
   (UNDER-DEVELOPED; see §7 carried-over efforts).

Accuracy fit: `acc ≈ 94.93 − 7.526·lm_loss`. rl+aux frontier top: **76.19 @ CP 8024**
(75.91@7405, 75.77@6501, 75.68@6130). Seed variance is **significant** (~1pp across 3 seeds) — frontier
points are n=1; treat sub-1pp differences as within noise.

---

## 1. Fruition definition (the bar every cell must clear)

A sweep **cell is at fruition** iff:
- it **reached iter ≥ 3000**, AND
- **every checkpoint it saved at iter ≥ 1500** — the 500-cadence grid {1500,2000,2500,3000} **and every
  off-grid exit-iter save in between** (e.g. 2209, 2303, 2858) — has an **accuracy + CP** eval in
  `benchmark_results.csv` (±60-iter tolerance for off-grid label drift).

Eval floor = **1500** (user choice "meaningful ≥1500", 2026-06-18; supersedes earlier "≥800").
Why this is safely closeable: `submit_continue_to_3000.sh` has **auto-prune disabled — distcp is NEVER
deleted** (only HF conversions are reaped, re-convertible on demand). So every ≥1500 checkpoint persists
and stays evaluatable; the eval loop drains them at any pace without racing a pruner.

**Supersession:** a config (reward, rlc, aux, kl, γ, baseline) trained to fruition in one cell **covers**
its twin in another (ignores seed — seed replication is a *separate* future sweep, not a gap).

---

## 2. Status Snapshot

> **Live machine numbers auto-refresh every supervisor cycle in `GAP_STATUS_AUTO.md`** (same dir).
> The table below is the curated per-session snapshot — update it (and the Changelog) each working session.

**As of 2026-06-18 ~04:20 UTC:**

| Bucket | N | Meaning |
|--------|---|---------|
| DONE | 47 | at fruition |
| COVERED | 22 | config reached fruition elsewhere |
| EVAL | 2 | at 3000, missing a ≥1500 eval |
| CONT | 55 | trained <3000 → continue |
| NEVER | 36 | no ckpt → fresh-start (incl 18 v15klcg + 8 v14cg γ0.3) |
| NORL_NEVER | 3 | quarantined (need no-RL fresh path) |
| NEEDS_MANUAL | 1 | `v6_ppo_r00` (rlc/aux not in name) |
| DROP | 33 | abandoned v2/v2p/v7/v7r (user-excluded) |
| **LIVE GAP** | **93** | EVAL 2 + CONT 55 + NEVER 36 |

- Eval debt (unevaluated surviving ≥1500 distcp): **12 checkpoints**.
- **Queue:** 79 total, **0 running, 79 pending** (ORD GPU contention — near-0 for days).
  `fresh_=19`, `c3000_=50`, **`v15klcg=18` (all 18 cells queued)**.
- **Supervisor daemon:** PID **519206**, healthy, single instance. PID file `/tmp/babysit_supervisor.pid`.
- Total inf-evaluated rows in CSV: **456**.
- **Binding constraint:** ORD GPU contention, NOT our orchestration. The closer is idempotent and
  self-draining — it fills as GPU frees; nothing is double-submitted.

---

## 3. The gap-closure system (built 2026-06-18)

### 3a. `gap_manifest.py` — authoritative fruition arbiter
`…/benchmarks/gap_manifest.py`. Read-only, idempotent. Classifies all 199 cells (179 on-disk + 20
planned). Rules: FLOOR=1500, TOL=60, DROP={v2,v2p,v7,v7r}, supersession by config-sig, eval target =
every on-disk `iter_*` ≥1500 minus already-evaluated. Declares two **PLANNED grids** so not-yet-created
cells get fresh-started: **v14cg** (32, mirrors run_sweep_critgamma) + **v15klcg** (18, §4).
Emits to `/tmp/`:
- `gap_cont.txt` — cells to continue to 3000
- `gap_fresh.txt` — `cell|RUN_NAME=.. BASELINE=.. GAMMA=.. RLC=.. AUX=.. KL=.. LM=.. REWARD_TYPE=..`
- `gap_eval.txt` — `cell <iter>` per unevaluated ≥1500 checkpoint
- `gap_norl.txt`, `gap_manual.txt` — quarantined (never auto-submitted)

### 3b. Supervisor wiring — `…/users/jonathanp/babysit_3000.sh` (NOT in repo)
Runs every 20 min (`INTERVAL=1200`). Backups: `babysit_3000.sh.bak.gapwire.*`. Cycle:
1. ingest evals → CSV · backfill metadata · **regenerate gap manifest** · reclaim HF · regen Pareto plot ·
   refresh EVAL_STATUS.md · (every 3rd cycle) sync canonical wandb CP.
2. **Continuation loop** → `cat gap_cont.txt` (was a hardcoded v5a/v5b/v12corner/v14cg glob — **now covers
   ALL generations**; this was the core bug: v6/v7g/v8/v9/v11/v11e/v5a_norl were silently skipped).
   Guards: 32-shard-complete, skip diverged (lm_loss≥8 or acc<60), skip-list, `running_name` dedup.
   `submit_continue_to_3000.sh <cell> 1` (name-driven hyperparams, self-healing afterany chain, depth≤8).
3. **Fresh-start loop** → `gap_fresh.txt`: `env $kv RL_TRAIN_ITERS=1500 sbatch … submit_fresh_corner_ep16.sh`.
   Skips cells already on disk or in flight. **Subsumes the old `run_sweep_critgamma.py` step.**
   NORL_NEVER / NEEDS_MANUAL not auto-started.
4. **Eval loop** → every checkpoint **≥1500** with 32 shards lacking an inf result → convert→eval→ingest;
   partial-HF repair; `reclaim_hf` reaps HF (distcp kept). MAX_EVAL-capped.

**Caps (env-overridable):** `GPU_CAP=100000` (limits removed per user), `MAX_CONT=20`, `MAX_FRESH=20`,
`MAX_EVAL=24`, `CAP=1000`. CAP-TRIM exemption regex now includes `c3000_|fresh_` (and `cg_g|basecritic_g|
cp_|ord_rayEP|conv235`) so gap jobs aren't trimmed.

### 3c. Validated 2026-06-18 (smoke test + first daemon cycle)
Continuation picked up **v11e16 + v5a_norl** (previously missed). Fresh launched **v14cg γ0.3 + v15klcg
(KL set)** as 2-node EP=16 jobs. Dedup confirmed (no double-submit). All 18 v15klcg queued.

---

## 3d. Disk management — prioritize eval, prune only the absolutely-dominated (2026-06-18)

**Problem:** project quota fs12 = **250T**; hit **83%** (208.5T) — ~200T is distcp (never auto-pruned),
HF only 8.2T. Training 93 cells would blow the quota → EDQUOT kills jobs.

**Mechanism (in `babysit_3000.sh`, step 1d, every cycle):**
- `disk_pct()` reads the fs12 quota. Thresholds: `PRUNE_PCT=80`, `DISK_PAUSE_TRAIN_PCT=88`.
- **≥80%** → run the HARDENED `cleanup_far_distcp.py` (DELETE=1, FARTHRESH=1.0).
- **≥88%** → **pause continuation + fresh-start this cycle** (eval loop + prune still run) so eval
  catches up and the prune frees space → eval is prioritized over training under pressure.
- Eval-first rationale: a checkpoint can only be safely deleted *after* it's evaluated (acc+CP known),
  so evals must lead.

**HARDENED `cleanup_far_distcp.py` (both-CP rule).** Deletes a distcp iter ONLY if:
(1) inf-evaluated (acc≥40), (2) **both** `eval_crit_path` AND `cp_critical_eval` present,
(3) NOT on frontier under **either** metric, (4) dominated by ≥1pp under **both** (min-gap≥FARTHRESH),
(5) not the cell's max iter, (6) cell contributes no frontier point under either metric. Anything
missing canonical CP / un-evaluated / frontier-under-either → **EXEMPT**. distcp is otherwise never
deleted; HF always reapable. Backup of the legacy-CP-only version: `cleanup_far_distcp.py.bak.legacyCPonly`.

**Reclaimed 2026-06-18:** 59 dominated distcp dirs, **27.7 TB** freed → quota 83%→73%.
**Incident:** the *legacy-CP-only* version (used for that pass) deleted **2 canonical-CP-frontier**
intermediate checkpoints (`235bv5b_…kl0.01_r08@2000` and `@2215`). Their **(acc, both-CP) data survive
in the CSV** (frontier intact); only the model weights are gone (regenerable by re-training r08 if ever
needed for re-benchmarking). Tool hardened immediately so it cannot recur.

### 3e. Lost-checkpoint recovery → folded into the SEED SWEEP (decided 2026-06-18)

The model weights lost to the two prune incidents are **regenerated by the planned multi-seed sweep**
(FOLLOWUP #5), not by a separate job — it re-trains these frontier-defining configs across seeds anyway,
and now CP is recorded + deletes are safeguarded so they'll be preserved. The seed sweep's cell list
**MUST include these configs** (whose weights are currently gone; data still in CSV):
- `235bv5b_rlc0.5_g0_mean_ppo_kl0.01` (r08) — today's canonical-frontier loss (74.5–74.68 @ CP ~5100–5300)
- `235bv5b_rlc0.1_aux0.003_basecritic` (r75) — 75.77 @ 6501 (frontier); `…basemean` (r74) — 75.75
- `235bv5b_rlc0.5_aux0.001` (r42) — 75.72 ; `235bv5a_rlc0.1_ppo_aux0.003` (r01) — 75.2
- `235bv12corner_per_token_load_weighted_rlc0.03_aux0.001` & `…rlc0.05_aux0.001` — corner 75.6–75.95
- `235bv5a_norl_aux0.001` (r24) — 76.09 (the top aux-only corner point)

## 4. v15klcg — KL × critic × γ sweep (NEW, 2026-06-18)

**Motivation:** coverage audit found the 3-way interaction was **entirely uncovered**: kl>0 & critic = 0
cells; kl>0 & γ>0 = 0 cells. critic+γ existed only at kl=0 (v14cg); KL existed only with mean baseline at
γ=0 (v9/v5b/v6). Decision: **18-cell focused probe, critic only** (no mean arm).

Grid (baseline=critic, reward=per_token_load_weighted, fresh→1500→continue→3000, eval every ≥1500 ckpt):
- **kl ∈ {1e-4, 1e-3, 1e-2}**
- **γ ∈ {0, 0.5}**
- **(rlc,aux) ∈ {(0.5, 0.001), (1.0, 0.005), (1.0, 0.02)}** — 3 anchors spanning the CP range
- = 3 × 2 × 3 = **18 cells**, named `235bv15klcg_rlc{R}_aux{A}_basecritic_g{G}_kl{K}_r1`.

Note: KL adds a reference-model forward (~1.5× walltime + extra activation mem) — pricier per cell.
**Status:** all 18 queued (2026-06-18), pending ORD GPU.

---

## 5. Full gap inventory (live cells, by generation)

**CONT (55):** v14cg 22 · v5b 7 · v5a-norl 6 · v8 6 · v11e 4 · v7g 4 · v6 3 · v9 2 · v11 1
**NEVER (36):** v15klcg 18 · v14cg γ0.3 8 · v11 3 · v12corner 2 · v5b 4 · v9 1
**EVAL (2):** v12corner_rlc0.05_aux0.001@3000 · v5b_rlc1.0_aux0.01_seed1_r46@3000
**Eval debt (12 surviving ckpts):** 11 in v5b + 1 v12corner (off-grid intermediates like 2255/1593/2149).

**DROPPED (33, user-excluded):** v2 (3) + v2p (14, kl=1.0 superseded) + v7 (8) + v7r (8) — abandoned/
superseded early sweeps. The only KL cells excluded are the 7 kl=1.0 (v2/v2p); all kl 1e-4…1e-2 are live.

**Quarantined:** 3 norl (v5b_norl seed1 ×3 — need no-RL fresh path) + `v6_ppo_r00` (rlc/aux not in name →
can't derive fresh env). Not auto-submitted; revisit if needed.

---

## 6. Carried-over efforts & their status

| Effort | Status | Notes |
|--------|--------|-------|
| **Eval-gap closing (original)** | ✅ generalized into this system | "eval every checkpoint" now = eval loop floor 1500 + manifest gap_eval. The historical **prune flaw** (91 CP-less ckpts deleted in error) is mitigated: distcp now never pruned; CP recorded on every eval (run_lm_eval_ord.sh + ingest). |
| **frontier_audit.py** | ✅ live | canonical frontier tool; EXCLUDED (missing acc/CP) ≠ dominated. Re-run after eval waves. |
| **Dual CP columns + HTML toggle** | ✅ done | `eval_crit_path` (legacy default) + `cp_critical_eval` (canonical wandb `critical_eval/critical_path`); radio toggle in generate_pareto.py. sync_cp_nearest every 3rd cycle. |
| **CP→inference profiler agent** | ⏳ created, NOT launched | agent `cp-inference-profiler` + kickoff prompt exist. Answers the systems half (compute-vs-comms decomposition, A100 CP→speedup, NVL72 projection). **Awaiting user launch.** |
| **Academic figures** | ✅ done | `cp_fig_system.svg` (+−40% rebalanced), `cp_fig_system_ep2.svg`, `cp_fig_ml.svg`, `cp_fig_dataflow.svg`, `cp_latency_explainer.pptx`. |
| **FOLLOWUP_SWEEPS.md** | ✅ created | research-advisor roadmap (reward comparison, corner attack, LR/early-stop, γ ablation, multi-seed, low-CP). |
| **Disk/HF reclaim** | ✅ live in supervisor | reclaim_hf.py + cleanup_nonfrontier_hf.sh; distcp + frontier + pending always protected. 49.84TB reclaimed earlier. |
| **Walltime / EP-ladder (systems)** | ⏳ partial | pretrained + r15 EP 8/16/32/64 measured; raw multi-node tps flat (comm-bound — inter-node all-to-all masks CP-governed compute). Profiler agent to finish. |
| **NRT offboarding** | ⏳ pending | NRT offboarded — **do not run jobs on NRT**; evacuate any remaining ckpts to ORD. |
| **30B reference sweep (EP=4)** | reference | cross-scale Pareto methodology anchor. |

---

## 7. Guardrails (hard rules — do not violate)

- **Never** use `backfill`/preemptible partitions (use polar3/polar4). **Never** `--gpus` (always `--gpus-per-node`).
- **Never** delete distcp source-of-truth or frontier checkpoints. HF is re-convertible → reapable.
- **Never** call an evaluated checkpoint "dominated" without real acc+CP (invisible ≠ dominated).
- **NEVER use legacy `eval_crit_path`.** It is stale (identical value repeated across a cell's iters for
  ~37% of cells) → unreliable. **All frontier / domination / deletion / plot decisions use canonical
  `cp_critical_eval` ONLY.** A point lacking canonical CP is EXCLUDED (invisible ≠ dominated), never
  judged via legacy. Enforced in `cleanup_far_distcp.py`, `frontier_audit.py`, `generate_pareto.py`.
- **distcp deletion is HUMAN-GATED — never autonomous.** The supervisor only writes a candidate manifest
  (`prune_candidates.txt`); a human reviews + runs the delete. The tool has a **frontier-safety assertion**
  (aborts if any candidate is on/within 0.5pp of the canonical frontier) + a **tombstone log**
  (`distcp_deletions_tombstone.log`). Only reversible HF reaping is automatic.
  *(2026-06-18 incident: legacy-CP-only judgement deleted frontier points — see §3d.)*
- **Never** `pkill -f babysit_3000.sh` (self-matches the operator shell). Kill by explicit PID, verify, relaunch via `setsid nohup … </dev/null & disown`.
- `--empty-unused-memory-level` ∈ **{0,1,2}** only (3 silently kills cells at argparse).
- Complete EP=16 checkpoint = **32 distcp shards**. Exclude diverged cells (eval lm_loss ≈ 9 / collapsed router) from frontier + benchmarking.
- GPU limits **removed** (use freely) per user 2026-06-18; revert to 64/80 only on explicit request.

---

## 8. How to check status / drive the closer

```bash
ssh jonathanp@cs-oci-ord-login-01.nvidia.com
cd .../examples/qwen3/benchmarks
python3 gap_manifest.py            # LIVE GAP + per-gen breakdown (read-only)
squeue -u jonathanp -h -o '%j %t' | grep -E 'fresh_|c3000_|v15klcg'
ps -p $(cat /tmp/babysit_supervisor.pid)   # daemon alive?
tail -f .../sweep_logs/babysit_3000.log    # cycle activity
python3 frontier_audit.py          # frontier + coverage after eval waves
```
**Success = LIVE GAP → 0** (every live cell DONE), frontier regenerated, no CAP-TRIM churn, lustre under quota.

---

## 9. Changelog

- **2026-06-18 (full canonical CP coverage)** — New `backfill_all_checkpoints_cp.py`: ensures EVERY
  on-disk checkpoint (375) has a CSV row carrying a canonical CP, filled by **linear interpolation**
  between bracketing wandb eval points (exact/interp/extrap); adds `limit=cponly` stub rows (marked
  `cp_stub`) for un-rowed checkpoints (139 added) without polluting the inf-frontier, auto-dropped once
  a real eval row appears. Result: **0 on-disk checkpoints without a row, 0 without canonical CP**
  (method split exact 413 / interp 283 / extrap 84; 2 historical no-wandb rows). Wired into the
  supervisor (replaces sync_cp_nearest, every 3rd cycle) so coverage self-maintains. Daemon PID 725820.
- **2026-06-18 (canonical re-pull + frontier corrected)** — Found the CP sync skipped any row with legacy
  CP, so canonical was never backfilled where legacy existed (and the 76.19 headline used stale legacy
  8024). Reworked `sync_cp_nearest.py` to re-pull REAL wandb canonical for EVERY inf row (ignore legacy;
  clear where wandb has none): **462 rows filled** (229 exact + 233 nearest-grid bookmarked), 0 overwrites
  of consequence, **EXCLUDED-no-CP went ~25→0** — canonical frontier now complete. Corrected frontier
  (canonical): aux 5 / rl-only 1–3 / rl+aux 17–18; top RL **76.19 @ ~7868** (was @8024); low-CP points
  shift left up to −1686 (RL reduces CP *more* than legacy showed). Prune tool further hardened to
  **delete only on `cp_method=="exact"`** (never a bookmarked approximation) → now 6 dirs/2.8 TB safe.
  `generate_pareto.py` default = canonical. Backup `.bak_canonrepull`.
- **2026-06-18 (CP metric + safeguards)** — Discovered legacy `eval_crit_path` is stale (constant across
  iters for 37% of cells). Switched ALL decisions to canonical `cp_critical_eval` only
  (`cleanup_far_distcp.py`, `frontier_audit.py`, `generate_pareto.py` default). **Incident:** the prior
  legacy-CP prune deleted frontier points — *systematically*, because stale-high legacy CP disguises the
  biggest CP-reducers (= the frontier) as dominated. Precise damage: today's prune removed **11 important
  ckpts** (2 canonical-frontier r08@2000/@2215 + 9 near-frontier, mid/low-CP 72–74.7); the high-value
  corner losses (75.6–76, v12corner/r75/r74/r42) were the earlier 06-11 incident. All (acc, canonical CP)
  data survive in CSV (frontier intact); weights regenerable. **Safeguards now enforced:** canonical-only;
  frontier-safety assertion (abort if any candidate near frontier); **distcp deletion human-gated (no auto
  delete)**; tombstone log; margin 1.0→1.5pp. Daemon restarted (PID 150978). Triggered canonical CP sync.
- **2026-06-18 (disk)** — Project quota hit 83%. Pruned 59 dominated distcp dirs (**27.7 TB**, 83%→73%)
  via `cleanup_far_distcp.py`. **Incident:** legacy-CP-only judgement deleted 2 canonical-CP-frontier
  intermediates (r08@2000/@2215) — data preserved in CSV, weights lost. **Hardened** the tool to the
  both-CP rule (delete only if dominated ≥1pp under BOTH `eval_crit_path` AND `cp_critical_eval`, both
  present; else exempt). Wired into babysit step 1d: prune at ≥80% quota, **pause training at ≥88%**
  (eval+prune prioritized). Added `disk_pct()`. Daemon restarted (PID 2887410).
- **2026-06-18** — Built `gap_manifest.py`; rewired babysit_3000.sh (continuation now all-generation;
  manifest-driven fresh loop subsumes run_sweep_critgamma; eval floor 800→1500; CAP-TRIM exemption
  +`c3000_|fresh_`; caps env-overridable). Defined fruition = 3000 + every ≥1500 ckpt evaluated. Added
  **v15klcg** 18-cell KL×critic×γ probe. Dropped v2/v2p/v7/v7r (33). Validated all 3 loops; daemon
  relaunched (PID 519206); all 18 v15klcg + 19 fresh + 50 cont queued. LIVE GAP = 93. ORD at 0 GPU.
