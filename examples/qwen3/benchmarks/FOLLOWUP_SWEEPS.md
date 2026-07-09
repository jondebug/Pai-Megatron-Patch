# 235B Router-RL — Recommended Follow-up Training Sweeps
Source: methodological review of all sweeps + the limit=inf frontier (2026-06-06). Verified against
benchmark_results.csv (95 clean inf 235B points, 11 on the Pareto frontier). Accuracy = mean(hellaswag
acc_norm, arc_challenge acc_norm, winogrande acc); acc ≈ 95.17 − 7.59·eval_lm_loss.

## Where we stand (the basis for these recommendations)
- **RL Pareto-dominates the aux-loss baseline across the whole CP-reduction regime** (every aux point
  with CP≤7702 is dominated by an RL point). The ONLY holdout is the no-reduction corner: aux
  76.28@CP8435, pretrained 76.74@CP8800.
- **Best RL frontier point: 75.28@CP7372** (235bv5a_rlc0.1_ppo_aux0.003_r01 @iter2500 — an INTERMEDIATE
  that beats both its 1500 and 3000 endpoints). Highest RL accuracy overall ~75.95 but at CP~9000
  (dominated by pretrained — not a win). Corner gap ≈ 1.0pp at lower CP.
- **Frontier is saturating for the current reward type** (per_token_load_weighted, γ=0): ~40 recent
  evals, only ~3 nudged the frontier; the rest cluster just under it.
- **Training to 3000 often HURTS** (CP↓ but acc↓) — intermediates (2000–2500) are frequently
  frontier-best, suggesting the cosine-LR-to-3000 schedule overshoots.
- **γ>0 shows no benefit so far** (all evaluated g0.3/g0.5/g0.8 dominated), but the g0.5/g0.8 cells are
  immature. **Low-CP (<4500) collapses to 71–73.**

## Ranked follow-up sweeps

### #1 — Reward-function comparison @ matched iter 3000  *(highest value/cell)*
- **Config:** fix `rlc=1.0, aux=0.005, gamma=0`; sweep `reward_type ∈ {per_token_load_weighted,
  critical_path, topn_binary, entropy}`. Identical LR schedule; eval limit=inf at 1500 AND 3000. 4 cells.
- **Why:** the paper's reward-design contribution is currently UNSUPPORTED — no critical_path/topn/
  entropy reward cell is inf-evaluated at 235B (v10/v11e16 are stubs/immature). **critical_path must be
  in the table** — it directly optimizes the CP metric, so a reviewer will ask why the reward isn't the
  objective itself. Lets us claim "per_token_load_weighted is the best reward formulation" with a
  controlled table instead of an assertion.

### #2 — Corner attack to maturity  *(closes the one open dominance claim)*
- **Config:** gentle grid `rlc ∈ {0.03, 0.05, 0.1} × aux ∈ {0.0005, 0.001, 0.002}`, per_token, γ=0,
  to 3000, with **dense intermediate inf evals at 1000/1500/2000/2500/3000** (best corner point is
  likely an intermediate). Target: any point with acc≥76.3 at CP≤8200. 6–9 cells.
- **Why:** the corner (acc≥76.28 at CP<8435) is the sole region RL doesn't own. Closing it converts
  "RL dominates except the corner" → "RL dominates everywhere" — categorically stronger. v12corner
  (rlc0.05/0.1) is the seed of this; the addition is the matched aux controls + dense eval cadence.
  Success criterion: if after 3000 iters no cell clears acc≥76.0 below CP8435, report the corner as an
  honest, characterized RL limitation rather than leaving it open.

### #3 — LR-schedule / early-stop ablation  *(resolves the intermediate-beats-endpoint confound)*
- **Config:** 2 representative cells (`rlc1.0/aux0.005`, `rlc0.1/aux0.003`) × 3 schedules
  {cosine-to-3000 (current), cosine-to-1500, constant-LR}, matched seed; eval the full iter trace at inf.
  6 cells (cheap to eval — the cosine-3000 arm already exists).
- **Why:** intermediates systematically beat endpoints (r01 75.28@2500 > 74.78@3000; r14 74.83@2247 >
  74.23@3000; r15 74.44@4682@3000 > 73.13@4215@3802). Currently this reads as cherry-picking. Either a
  shorter schedule recovers the intermediate's accuracy at its CP (→ legitimizes the frontier), or
  intermediates become a *stated, pre-registered* early-stopping rule. The current silent
  intermediate-selection is not publishable.

### #4 — Clean gamma ablation (matched triplet)
- **Config:** fix `reward=per_token, rlc=1.0, aux=0.01`; sweep `gamma ∈ {0.0, 0.3, 0.5, 0.8}`, matched
  seed, trained to matched milestones 1500 AND 3000, both inf-evaluated. 3 new cells (g0 exists).
  Optionally repeat for critical_path reward.
- **Why:** "γ>0 doesn't help" currently rests on ONE inf point. Turns it into a one-line defensible
  ablation. Reviewers routinely ask why a non-standard RL choice (γ=0) was made.

### #5 — Multi-seed replication of frontier-defining cells  *(top desk-reject risk)*
- **Config:** pick ~5 frontier-defining cells spanning CP (e.g. the 4682, 5806, 6791, 7372 points + the
  corner attempt), run **2 additional seeds each**, identical config, eval at the same milestone; report
  mean ± std / CI on the Pareto plot. ~8–10 cells, ruthlessly scoped to frontier cells only.
- **Why:** **every frontier point is currently n=1 — there are no seeds anywhere** (`_rNN` is a grid
  index, not a replicate; no seed column). A saturating frontier and a 1.0pp corner gap are within
  plausible seed noise. Without error bars a top-tier reviewer can desk-reject on "n=1, no variance."

### #6 — Low-CP anti-degradation  *(lower priority / exploratory)*
- **Config:** target CP~3500–4500 with accuracy-preserving levers: `rlc ∈ {0.5, 1.0}`, high
  `aux ∈ {0.015, 0.02}`, + KL anchor `kl ∈ {0.0003, 0.001}` and/or lm_reward, γ=0, to 3000, inf-eval.
  4–6 cells.
- **Why:** CP<4500 collapses to 71–73; this extends the dominance region further left. Lower priority —
  KL/LM modifiers have so far been *dominated*, so this may yield a negative result.

## Methodological weaknesses to close before submission
- **W1 — reward comparison unrun at inf** (no critical_path/topn/entropy inf eval at 235B) → #1. *Highest scientific gap.*
- **W2 — no seeds** (every config n=1) → #5. *Highest rejection risk.*
- **W3 — `rl_reward_type` blank on ~53 RL rows;** metadata parse-dependent (being backfilled).
- **W4 — iteration-milestone confound** (frontier points pulled from iters 1266–3802; combined with
  intermediate-beats-endpoint, risks a cherry-picking critique) → #3 + a stated checkpoint-selection rule.
- **W5 — `eval_lm_loss` missing on ~56/125 inf rows;** blocks loss-based calibration of acc≈95.17−7.59·loss.

## Verdict & sequencing
**NO-GO for submission as-is; GO to proceed with a tightly scoped closing campaign.** The core result
(RL dominates aux across the CP-reduction regime) is genuinely strong, but #1, #3, #4, #5 are gating,
and the **walltime/CP→latency mechanism** (the systems half — see the cp-inference-profiler agent) is
what makes "reduce CP" matter to a systems reviewer. Sequencing: **#1 and #2 first** (load-bearing
claims), **#5 in parallel** (gates publishability regardless of new wins), **#3 and #4** are cheap given
existing arms, **#6 last**.
