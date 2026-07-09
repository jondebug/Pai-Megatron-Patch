# 235B Benchmark-Coverage Priority Report
Generated 2026-06-05. Coverage treats **limit=1000**, **limit=inf**, and **walltime** as THREE
distinct requirements. "Thoroughly evaluated" = has limit=inf accuracy AND walltime.
Source of truth: benchmark_results.csv (ORD) reconciled against on-disk checkpoints (ORD + NRT).

## Data-integrity issues found (fix before trusting the frontier)
- **CSV is STALE (last write 06-02).** On 06-04, ORD ran (a) 12 v11e16 limit=1000 evals
  (17:34-19:20) and (b) 8 walltime jobs for pretrained + r15_cp4682 at EP=8/16/32/64
  (14:20-14:34). NONE are collected into the CSV. ACTION: run the results collector; the
  v11e16 eval dirs have no accuracy_summary.json (results are in logs/results*.json) so the
  collector may need the log-parse path.
- **r13@3000 checkpoint is GONE.** CSV has r13@3000 CP=6102 (from a training-time log) but no
  iter_3000 checkpoint exists on ORD or NRT (both max=1500; the 3000 ckpt was pruned by the
  continuation script's keep-latest logic). Cannot benchmark r13@3000 without retraining.
  r13@1500 (75.08@CP7171, has EP=8 walltime) DOES exist.
- **NRT = training storage only.** 35 cells (v10 x19, NRT-v11e16 x11, v5a r05/r13@1500, base),
  ZERO eval artifacts, ZERO HF, nothing at iter 3000.

## P0 — CRITICAL (defines/extends the frontier or the headline CP->walltime claim)
| checkpoint | gap | why |
|---|---|---|
| 235bv5a_rlc0.1_ppo_aux0.001_r00 @3000 | limit=inf (+wt) | aux-corner candidate; CP drifted 8439->7420. EVAL IN FLIGHT (job 28745520). |
| 235bv5a_rlc1.0_ppo_aux0.01_r15 @3802 | limit=inf (+wt) | frontier cell trained PAST 3000 (3802); @3000 was 74.44 frontier — 3802 may push further. |
| 235b-pareto_norl_aux0.001_r12 @1223 | **walltime** | THE aux corner 76.28@8435 — frontier anchor, no walltime. |
| 235bv5a_rlc1.0_ppo_aux0.005_r14 @2247 | **walltime** | frontier 74.83@5806; (also being pushed to 3000). |
| 235bv5a_rlc0.5_ppo_aux0.005_r08 @3000 | **walltime** | frontier 74.61@5533. |
| 235bv5a_rlc1.0_ppo_aux0.01_r15 @3000 | **walltime** | frontier 74.44@4682 (note: only r15_cp4682 *and* pretrained have the EP ladder so far). |
| 235b-rladv ppo_gumbel_t0.3_aux0.015_r07 @1268 | **walltime** | frontier low-CP 73.11@3736. |
| 235b-rladv ppo_gumbel_t0.3_aux0.01_r01 @1266 | **walltime** | frontier 73.78@4553. |
| 235b-rladv ppo_aux0.015_r06 @1282 | **walltime** | frontier 73.12@3821. |

## P1 — HIGH (frontier extension + the reward/gamma comparison)
- **v5a frontier continuations to 3000** (r01, r06, r07, r14 — LAUNCHED today): each @3000 will need
  limit=inf + walltime. r01/r06/r07 currently have inf@1500 (74.87/75.01/74.80) but no walltime.
- **v11e16 active grid (10 cells @ ~800-896)**: critical_path/per_token/topn/entropy x stoch/gamma/lm.
  Currently only limit=1000 (06-04, UNCOLLECTED). Need: collect 1000 now; limit=inf + walltime once
  pushed to 1500/3000. This is the reward-function + gamma comparison — methodologically required.
- **235bv5a_norl_aux0.001_r24 @1500** (75.47@8741): aux baseline near the corner, no walltime.

## P2 — MEDIUM (fills frontier interior / ablations)
- v5b KL cells (kl0.0001/0.0003/0.001 @~1220, 74.1-74.4@~6400): have inf, lack walltime.
- v6 cells (ppo_lm1.0 / kl0.001 @1500, ~74@~6030): have inf, lack walltime.
- v5a near-frontier @1500 missing walltime: r02 (74.54@6791), r03 (74.23@6034), others.
- v5a higher-pressure @3000 lacking limit=inf: r09 (has inf 73.28@4643; ok), r10/r11/r16/r17/r20/r23
  (@3000, no inf — mostly high-aux/high-pressure, likely low-CP/low-acc; eval a couple to confirm dominated).

## P3 — LOW (skip unless cheap; likely dominated/diverged/superseded)
- v12corner cells @250 (just launched — re-rate after they reach 1500).
- v7g/v8/v9 legacy gap-sweep cells @~800 (gamma/aux variants, early, mostly dominated).
- 235b-rladv *_cpb_n1_* CPB-ablation cells (have inf, dominated; CPB excluded from transferable frontier).
- NRT v10 generation (superseded by v11e16) + NRT-v11e16 duplicates.

## Recommended order of action
1. Collect 06-04 results (v11e16 limit=1000 + r15/pretrained walltime) into CSV; recompute frontier.
2. P0 limit=inf: r00@3000 (running), r15@3802. P0 walltime: the 7 frontier points above (Workstream D EP ladder).
3. P1: as the 4 v5a continuations + v11e16 grid reach 3000, eval limit=inf + walltime.
4. Re-rate P3 after v12corner/v11e16 mature.
