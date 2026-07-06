# Cell / Experiment Exploration Backlog

**Canonical record of every sweep we want to explore — hypothesis, status, conclusion.**
Complements `GAP_CLOSURE_STATUS.md` (fruition of launched cells) and `gap_manifest.py` (machine arbiter
+ PLANNED grids). Hand-curated; update when a sweep is launched, concluded, or a new hypothesis is queued.

_Last updated: 2026-06-21 (jonathanp)._

Status legend: **ACTIVE** (training/eval in flight) · **SCHEDULED** (submitted, queued) · **CONCLUDED** (enough
evidence for a verdict) · **PARTIAL** (some cells done, gaps remain) · **QUEUED** (designed, not launched) ·
**DROPPED** (abandoned/superseded/collapsed).

---

## Canonical Pareto frontier (current)

| CP | acc | cell | note |
|---|---|---|---|
| 8435 | **76.28** | `235b-pareto norl aux0.001 r12` | aux corner — RL has not beaten this (open: #49/#52) |
| 7903 | 76.19 | `v5b r111` basecritic g0.5 | RL frontier top |
| 7507 | 76.03 | `v5b r111` basecritic g0.5 | |
| 7405 | 75.91 | `v5b r42` rlc0.5 aux0.001 | |
| 6130 | 75.68 | `v5b r75` basecritic | |
| 5498 | 75.12 | `v5a r02` | |
| 4682 | 74.44 | `v5a r15` | low-CP RL |
| 3669 | 72.12 | `v5b r62` lm0.3 | lowest-CP corner |

RL Pareto-dominates the whole CP-reduction regime; **only the no-reduction corner resists**.

---

## Sweep families — status

| family | design | cells | status | conclusion / note |
|---|---|---|---|---|
| **v5a** | aux×rlc, PPO mean baseline | 28 | CONCLUDED | base Pareto; contributes r02/r05/r11/r15 frontier pts |
| **v5b** | main Pareto + ablations (reward/kl/seed/basecritic) | 52 | PARTIAL/ACTIVE | frontier-bearing (r42/r61/r75/r111/r33/r130015); continuations to 3000 in flight |
| **v6** | KL×LM anti-degradation | 4–6 | CONCLUDED | near-frontier, does not advance |
| **v7g** | clean mean+γ | 4 | PARTIAL | γ-on-mean mostly diverged; confounded (see v16cgr) |
| **v8** | gap-filling | 6 | PARTIAL | |
| **v11e** | `critical_path` reward | 5 | CONCLUDED | ~74–75, dominated |
| **v12corner** | corner configs (per_token, rlc0) | 8 | PARTIAL | near-frontier intermediates, not advancing |
| **v14cg** | **critic×γ×rlc×aux** (+γ0.8 probes) | 30 | **CONCLUDED** | **see Key finding below** |
| **v15klcg** | KL×critic×γ (18-cell probe) | 18 | **ACTIVE** | KL as a stabilizer for critic×γ; in flight |
| **v16cgr** | **low-rlc×γ corrective** | 8 | **SCHEDULED** | tests rlc↓ stabilizes γ>0; jobs 29336840–50 (2026-06-21) |
| 235b-pareto | aux-only corner | 15 | CONCLUDED | owns the 76.28@8435 corner |
| 235b-rladv | critic c256 adv | 16 | PARTIAL | |

---

## Key finding — critic + γ (v14cg)

**Adding a discount factor γ on top of the learned-critic baseline does NOT help, and at rlc=1 it
collapses the router.** Evidence (best acc over iters; cells degrade *with* more training):

- rlc1 × γ0.3: aux0.005→59.8, aux0.008→66, aux0.012→69, aux0.016→69, aux0.02→67 — all diverge.
- rlc1 × γ0.5: aux0.001→**76.0** (frontier-adjacent), then 0.003→74.7, 0.005→74.0, 0.008→71.8, 0.012→69, 0.016→67 — cliff above aux≈0.005.
- rlc0.5 × γ0.5: stable across all aux (72.9–75.4), no collapse.

**CONFOUND:** policy lr was fixed at **1e-4** for the entire grid; rlc was **not** reduced as γ rose;
γ0.3 was tested **only** at rlc=1. γ>0 inflates discounted-return / advantage magnitude → the fixed
lr/rlc is effectively too hot → collapse. So "γ is a dead end" is **not yet a fair conclusion**.

**Actions taken (2026-06-21):**
- 9 collapsed rlc1×γ cells added to `COLLAPSED` in `gap_manifest.py` → supervisor stops auto-continuing them.
- Hard `scancel` of their in-flight jobs **pending user confirmation** (deferred — shared-cluster jobs).
- Launched **v16cgr** to test the fix (see below).
- These 9 collapsed cells are real-evaluated + far-dominated (5–16pp) → disk-reclaim candidates.

---

## v16cgr — corrective sweep (CONCLUDED 2026-07-06) ✅ HYPOTHESIS CONFIRMED

**Verdict:** at rlc=0.1, both γ0.3 and γ0.5 are fully stable 1500→3000 (75.3–76.2, zero collapse) —
the v14cg rlc1×γ collapse was an effective-step-size artifact. **New frontier point:**
`rlc0.1_aux0.003_g0.3 @2961 = 75.90 @ CP6090` (dominates r75 75.68@6130 & 75.77@6501, mid-CP +0.22pp).
High end: 76.21@7970 (aux0.001 g0.3 @1500) ties v14cg-g0.8 within 0.01pp. γ+critic at low rlc is a
productive direction. rlc0.25 arm evals pending. Follow-ups: γ0.8×rlc0.1 probe; low-LR arm still open.

### (original plan, 2026-06-21)

Hypothesis: **reducing rlc stabilizes γ>0.** Grid = γ{0.3,0.5} × rlc{0.1,0.25} × aux{0.001,0.003},
basecritic, fixed lr=1e-4. 8 cells, fresh-start 0→1500, then supervisor continues→3000 + evals.
Registered in `gap_manifest.PLANNED`; launcher `run_sweep_v16cgr.py`.
**Success = γ becomes stable at low rlc AND points sit at/above the r111 corner** → real frontier result.

---

## Open backlog — queued / not yet launched

1. **Low-LR arm of the γ fix** — QUEUED. γ{0.3,0.5} at policy lr {5e-5, 2e-5} (rlc held). Needs the
   fixed `lr-1e-4` name-suffix/lr path in `submit_*` made parameterizable (small plumbing change).
2. **Dominate the aux corner (76.28@8435) with RL** — OPEN (#49/#52). No RL cell beats it yet.
3. **rlc0.5 × γ0.3** — gap in v14cg (γ0.3 only ran at rlc1). Cheap 2-cell fill (aux0.001/0.003).
4. **GAE (λ<1)** — QUEUED. Needs launcher + continue-parser update (`--rl-gae-lambda` currently 1.0).
5. **Seed-variance error bars** on frontier points — PARTIAL (~1pp range across 3 seeds; matters for
   critic-vs-mean edge which is within noise).
6. **Systems half** — CP→walltime ladder (#48), EP16/32/64 microbench (#41), NVLink-domain projection (#50/#54).

---

## Disk / hygiene notes

- 250T quota ~78% after 2026-06-21 cleanup (54 early <1000-iter non-frontier ckpts, 25.4 TB freed).
- Collapsed v14cg rlc1×γ cells (9) are evaluated + far-dominated → safe to delete (pending a sweep).
- Frontier checkpoints NEVER deleted (hard rule). Deletions tombstoned in `distcp_deletions_tombstone.log`.
