# 235B Pareto — OPEN ITEMS & CLOSING PLAN  (2026-06-06)
Consolidated from the full working session. STATUS: [DONE] [AUTO]=cron-driven [NEEDS-ME]=explicit work [BLOCKED]=needs user.

## A. RESEARCH GOALS (frontier)
- [AUTO] Beat the aux corner 76.28@8435 (best RL now 75.21@8217, gap 1.07pp at lower CP). Live shot = v12corner gentle cells (un-stalled, →3000) + low-pressure configs. Cron never-dry-up targets this.  (#49/#52)
- [NEEDS-ME] Gamma ablation — NO clean γ pair yet. Launched g0_r00@1500 (vs g0.3_r04@1500=73.81). TODO: g0/g0.5/g0.8 triplets for critical_path & per_token rlc1.0_aux0.01 at MATCHED milestone (1500/3000).
- [AUTO] Reward-function comparison (critical_path/per_token/topn/entropy) at fixed iter 3000 — v11e16 grid still maturing (~800-1164); cron drives them to 1500/3000 + inf.
- [AUTO] Low-CP frontier (CP<4500 collapses to 71-73) — anti-degradation sweep; cron next-sweep gap (b).
- [DONE] Frontier-saturation finding documented (2/40 recent evals moved it; rest cluster just under → frontier converged for current reward types).

## B. WALLTIME + PROFILING  (the under-developed area — NOT cron-driven)
- [NEEDS-ME] D1: add compute-vs-comms profiling to cp_vllm_bench.py (attn/dispatch-a2a/expert-ffn/combine-a2a + inter-node comms fraction). GATING STEP — not started.  (#54)
- [NEEDS-ME] D2: EP=64 representative cell (r15 CP4682) + pretrained, with D1 profiling.
- [NEEDS-ME] D3: NVL64-domain projection (replace inter-node a2a with NVLink-bw) to surface CP benefit raw tps hides.  (#50)
- [NEEDS-ME] D4: EP=64 walltime for ALL frontier checkpoints once D1-D3 validate. Only pretrained+r15 have the EP ladder today.
- [NEEDS-ME] r05@3000 reduced-CP walltime ladder.  (#48)

## C. EVAL COVERAGE  (mostly automated)
- [DONE] 7 v5a @3000 evals (r06=75.21 NEW frontier; rest dominated).
- [DONE] Intermediate 1000-3000 assessment (r00@2819=74.86 dominated; concluded intermediates rarely beat endpoints, except r06).
- [AUTO] limit=inf for every relevant top-milestone cell — supervisor targets it, skips diverged + still-training.
- [NEEDS-ME-VERIFY] v5b_g0_r00@1500 convert showed 31/32 distcp — verify HF integrity before trusting its gamma-pair number.

## D. INFRA / HYGIENE
- [DONE] Frontier-checkpoint deletion eliminated; stale-pointer auto-repair; diverged-cell skip; convert-only waste fix; plot auto-regen each cycle; save-interval 250->500; cluster column; CSV ingester (stdout-table); CSV mirrored ORD->NRT each cycle; v12corner un-stalled + added to supervisor.
- [BLOCKED-APPROVAL] Disk: 171TB / frontier-safe cleanup (~77TB, 176 ckpts: diverged-cell + non-milestone exit-saves; keeps all milestones/benchmarked/frontier). Awaiting go-ahead.
- [NEEDS-ME-minor] ingest_evals.py: derive rl/aux metadata from cell name for NEW cells (so future cells aren't blank-meta; CSV backfilled for now).
- [BLOCKED-USER] 64-A100 reservation never found on either cluster — verify with allocation manager.

## E. SYNTHESIS  (NEEDS-ME)
- [NEEDS-ME] 235b_pareto_report.md: write-up + reward-comparison & gamma-comparison tables (at fixed iter 3000) + projected-NVL64 CP->walltime curve (after B).
- [AUTO] Pareto plot regenerated each cycle (current).

## DRIVER SUMMARY
- AUTO (cron, no action): corner chase, reward comparison, low-CP sweep, top-milestone inf evals, CSV/plot/mirror upkeep, never-dry-up.
- NEEDS-ME (will drop if not scheduled): **B (walltime+profiling, esp. D1)**, gamma triplets, synthesis report, v5b_g0_r00 shard verify, ingester metadata.
- BLOCKED: disk cleanup (approval), 64-A100 reservation (user).
