# Router-RL — Consolidated Reviewer Insights

*Synthesis of three reviewer/advisor messages (2026-08-01 → 08-02). Durable reference for the RL
full-scale-system effort. My status notes and open questions are marked ⟶.*

---

## 0. The reframed goal + corrected success criteria

**Goal:** one full-scale 235B RL cell with *causally verified* learning — not merely `requires_grad=True`.

**Critical correction (all reviewers agree):** a decreasing **signed** REINFORCE/PPO loss is **not** a valid
health metric. With mean-centered advantages and a first-epoch importance ratio ≈ 1, the scalar is
≈ −mean(A) ≈ 0 *by construction* (NeMo's pure-online formulation has ratio≈1 explicitly). **Drop
"loss decreases" for the signed objective.** Track two separate losses:
- `signed_pg_loss` — diagnostic only; may stay ≈ 0.
- `frozen_rollout_audit_loss` — `L_audit(θ) = −E_{a∼π_old}[A_old · log π_θ(a)]`, recompute the *same*
  sampled rollout (fixed old actions + advantages) before/after the update. **This must decrease.** This
  gives the "RL-loss-descent" curve we want without misreading centered cancellation.

**Success = four causal checks, simultaneously:**
1. **Gradient reaches & moves the router** — nonzero RL grad on router weights; scales ~linearly with RLC
   before clipping; nonzero optimizer-attributable Δrouter; measured after DP/EP/PP reductions and
   before/after clipping; predicted improvement `g_RL · Δθ < 0`; track RL/aux/LM/KL grads *separately*
   incl. pairwise cosine.
2. **True reward increases** — use a monotonic reward. Displayed `R_probe = (CP_0 − CP_θ)/CP_0`, measured on
   (a) training batch, (b) fixed training probe, (c) fixed holdout probe, (d) deterministic top-k eval.
   The **fixed-probe and holdout** trends (not noisy on-policy reward) must rise. A *difference* reward can
   naturally approach zero, so its raw mean need not increase — the fixed-probe global objective must improve.
3. **Advantage points the right way** — centered ⇒ mean ≈ 0, so "advantage increases" is the *wrong* test.
   Require `E[Δlogπ(a) | A>0] > 0` and `E[Δlogπ(a) | A<0] < 0`; log adv–ΔlogP correlation, weighted sign
   agreement, reward-by-advantage-quantile (monotone), predicted-vs-realized counterfactual improvement,
   per-layer and globally-synced advantage moments; keep `std(A)` nonzero.
4. **Optimized (audit) loss decreases** — the frozen-rollout audit loss above.

---

## 1. Ranked failure hypotheses (why full-scale RL is destructive)

| P | Hypothesis | Why it fits | Decisive test | ⟶ My status |
|---|-----------|-------------|---------------|-------------|
| **H1** | **Action–policy mismatch**: deterministic top-k is not sampled from the distribution whose log-prob is differentiated; a nonzero gradient can point away from reward | cos(true PG, argmax surrogate)=0.44, wrong-way 17% | Compare implemented gradient vs enumerable Plackett–Luce oracle | ⟶ **DONE** — `pl_verify.py`: PL log-prob is exact (Σπ=1), matches Gumbel-topk & sequential oracle, and yields **unbiased** PG (cos=1.0, ‖diff‖=5.5e-16). The *fix* (Gumbel-topk+PL) is validated; production path still uses argmax. |
| **H2** | **Reward scope wrong**: EP-local (per-rank 1/16) loads disagree with the globally hottest expert → conflicting rank gradients | reward uses local load (code-confirmed) | Replay identical routes under different EP partitions; reward/advantage/gradient must be invariant | ⟶ TODO. `--rl-global-load` exists (all-reduce) but unvalidated/not default. |
| **H3** | **Reduction contract wrong**: old reduction made RL negligible; restoring ~94× exposed a destructive update → missing normalization *plus* estimator errors, not that 94× is correct | norm A/B: 96.9× grad, CP +34% | Log unreduced numerator + every denominator (token, top-k, layer, microbatch, PP, DP) | ⟶ Partially — ×94 confirmed; need the full per-factor log + coefficient-based tuning. |
| **H4** | **Policy shift too large**: a useful small-scale gradient becomes destructive after crossing many top-k boundaries | 0.05 step → 9% flips, IS std 767 | Sweep update magnitude vs router KL, top-k flips, realized reward | ⟶ TODO (gain sweep 1×,4×,16×,32×,94×). |
| H5 | Advantage aggregation destroys structure (global standardization mixes layers/ranks) | — | Compare global vs per-layer vs globally-synced per-layer baselines via grad cosine + realized reward | ⟶ secondary |
| H6 | RL conflicts with LM/aux/KL grads (frozen LM still differentiates through routers; aux may oppose) | — | Separate router grads + pairwise cosine RL/LM/aux/KL | ⟶ secondary (validate aux/KL OFF first) |
| H7 | Residual trajectory corruption (recompute/PP scheduling/stale records) | paused-guard fixed the KL overwrite, but recompute could still mix | Hash actions, old-logprobs, rewards, advantages across every forward/recompute path | ⟶ secondary (Router Replay) |
| H8 | Precision (BF16 perturbs routing boundaries) | — | FP64/FP32/BF16 grads on one captured production batch | ⟶ **secondary, not a first fix** (my BF16 test already refuted log-prob truncation) |

**Priority: H1–H4 *together* (valid sampled actions + global reward + explicit normalization + bounded
policy shift). H5–H8 secondary. BF16 and critic quality are secondary, not first fixes.** This matches my
own sims: the deterministic surrogate, local load, and a wide gain range each *converge in isolation* — so
the divergence is a system-level interaction, not any single defect.

---

## 2. The full-scale configuration (the prescription)

1. **Valid estimator** — Gumbel-top-k *without replacement* for sampling; **ordered Plackett–Luce** exact
   log-prob for the sampled action; deterministic top-k **only for eval**; **one on-policy epoch** initially.
   Restrict Gumbel noise to a **top-32 candidate pool** per token (bound variance across 128 experts).
   No PPO epochs / GAE / critic until the estimator passes oracle tests.
2. **Global reward** — all-reduce expert counts over exactly the ranks holding *unique tokens*. Report
   `R = −Σ_l max_e n_{l,e}`; **train** with a dense smooth-max counterfactual:
   `J_l(n) = (1/β) log Σ_e e^{β n_e}`, `r_{l,t} = J_l(n_l) − J_l(n_l − Δ_src + Δ_dst)`. Dense token credit,
   aligned with global CP. Reward domain must match the CP objective (busiest expert across layers).
   CP is additive across layers ⟶ **layer-local baselines**; GAE not auto-justified (94 layers ≠ temporal
   horizon).
3. **One reduction contract** — `numerator = −(A.detach()·ordered_logprob).sum()`;
   `denominator = global_valid_token_layer_count`; **normalize exactly once**; log numerator, that count,
   #layers, top-k, microbatches, grad-accum, DP factor, final coeff. **Tune strength via a coefficient — not
   hidden ×/÷94.**
4. **Trust region** — top-k flip rate 0.1–1%; router KL ≤ 1e-3 start; bounded update/weight norm;
   IS-ratio p99 ≈ 0.8–1.25; adaptive step scaling or rollback on breach. Router Replay to fix routing
   choices during recomputation (distinguishes recompute mismatch from real policy movement).
5. **FP32 RL island** — full model stays BF16, but compute in FP32: router logits, log-softmax + ordered
   log-prob, global reward, advantage moments, policy-loss reduction, KL, IS ratios, diagnostic gradients.
   Do not switch the whole model to FP16 unless a fixed-batch comparison proves material BF16 grad error.

---

## 3. Validation ladder + graduation (use the real 235B topology throughout)

**Ladder:** (1) no-update capture — verify global counts/rewards/trajectory integrity; (2) one frozen-batch
step — audit loss ↓ + advantage directions agree; (3) five-step fixed-batch overfit — objective descends;
(4) fifty on-policy iters — fixed-probe reward ↑ without excessive route churn; (5) 200 iters × 3 seeds —
deterministic CP improves consistently; (6) 500–1500 iters — add aux, then KL, one at a time.

**Arms:** A = corrected RL only · B = A + trust region · C = B + aux anchor · D = C + KL · E = aux-only control.

**Graduation (one 235B cell, all simultaneously, 500–1500 iters):** RL grad finite & RLC-responsive;
frozen-rollout audit loss ↓; fixed-probe global reward ↑; deterministic CP ↓; positive adv–ΔlogP
correlation; predicted/realized reward agreement; bounded router KL & top-k churn (≤1% flip, IS std ≤0.1);
stable grad & update norms; no holdout-accuracy collapse.

**Immediate milestone: NOT another broad sweep — ONE full-scale RL-only cell passing all four causal checks
at once.**

---

## 4. Consensus vs. tensions (and my resolutions)

- **ST-Gumbel-softmax — apparent contradiction.** One message suggested "differentiable relaxations like
  Gumbel-Softmax / straight-through"; another said **explicitly do NOT** use ST-Gumbel as primary (it's
  another biased estimator). ⟶ **Resolution:** use **hard** Gumbel-top-k *sampling* + exact PL *score
  function* (REINFORCE-style, unbiased — I verified cos=1.0). ST-Gumbel is a biased fallback only. This
  reconciles both. *(Q1 to confirm.)*
- **RSPO now vs. simpler guard first.** One emphasized pairing Gumbel-top-k directly with **RSPO** (per-token
  router-shift ratio, stop-grad + lower-bound floor, soft-rescale IS before clipping). Another said for
  *fresh single-epoch on-policy*, start **simpler** (flip-rate / KL / max-update-ratio / early-stop) and use
  RSPO for off-policy or multi-epoch. ⟶ **Resolution:** start with the simple guard for the first working
  cell; escalate to RSPO if IS ratios still explode. *(Q3.)*
- **Counterfactual reward destination Δ_dst.** `r = J(n) − J(n − Δ_src + Δ_dst)` — the destination is
  under-specified (next-best expert? uniform slot? or source-only marginal `J(n) − J(n − Δ_src)`?). *(Q2.)*
- **External corroboration:** Megatron supports global load balancing and recommends FP32/FP64 router logits
  → backs H2 + the FP32 island.

---

## 5. What I have already validated (evidence in hand)

- **H1 fix works in principle** — PL log-prob exact + unbiased PG (`pl_verify.py`).
- **No single factor reproduces the divergence** — deterministic surrogate (→33), local load (→40), and
  gains 1×–94× all *converge* in clean sims ⟶ supports "H1–H4 jointly," refutes my earlier "C7 is binding."
- **Norm fix mechanism** — `--rl-perlayer-norm` restores gradient 96.9× (≈ ×94) but is harmful alone
  (CP +34%, load ×4) — the H3/H4 warning made concrete.
- **Reference full-system (miniature)** — one iteration of the *entire* prescribed config (Gumbel-top-32 +
  PL + smooth-max counterfactual + per-layer norm) shows audit loss ↓, adv-sign-agree 88.7%, nonzero grad.
  Full trajectory pending (login-node OOM; move to compute node).

---

## 6. Open questions to forward to the reviewer

- **Q1 (estimator):** Confirm the primary should be **hard Gumbel-top-k + exact ordered PL score function**
  (unbiased), with ST-Gumbel-softmax only as a biased fallback if score-function variance proves
  intractable — i.e., we are *not* relying on the differentiable relaxation for the main gradient path?
- **Q2 (reward):** In `r_{l,t} = J_l(n) − J_l(n − Δ_src + Δ_dst)`, what is Δ_dst? Options: (a) the token's
  next-best (argmax-excluding-chosen) expert; (b) a uniform/ideal slot; (c) source-only marginal
  `J_l(n) − J_l(n − Δ_src)` = the token's contribution to the smooth-max bottleneck. Each gives a different
  credit signal — which do you intend for the first working cell?
- **Q3 (trust region):** For the first RL-only cell (fresh, single-epoch, on-policy), is the simple guard
  (flip-rate 0.1–1% + router-KL ≤1e-3 + rollback) sufficient to start, with RSPO added only if IS ratios
  still explode? Or should RSPO be in from the first step because sampled top-k *inherently* shifts routing?
- **Q4 (PL domain vs. candidate pool):** With Gumbel restricted to the top-32 pool, should the PL log-prob's
  normalization (the `logsumexp` denominators) be over the **top-32 pool** (restricted policy — what I
  implemented, keeping sampling and scoring domains consistent) or over all **128** experts with the pool as
  a hard constraint? The former keeps the estimator self-consistent; please confirm.
- **Q5 (aux-free-bias interplay):** For the 4-way comparison {aux-only, aux-free-bias, RL+aux,
  RL+aux-free-bias}, should aux-loss-free bias be *frozen* during the RL-only causal-validation rungs and
  only introduced at rung 6, or can the detached bias run alongside RL from the start without confounding the
  causal checks?
