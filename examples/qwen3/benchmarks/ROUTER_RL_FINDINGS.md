# Router-RL at 235B — Findings, Hypotheses, Fix, Experiments, Results

**TL;DR.** Across the entire Qwen3-235B router-RL campaign (the `235bv21math_*` cells and their
predecessors), the RL policy gradient **never reached the router weights** — RL was a *disconnected
no-op*. Every "RL" result was really just the aux-loss floor doing the work. The cause is a one-line
omission: the KL reference forward runs under `torch.no_grad()` and overwrites the RL trajectory with
detached logits, severing `rl_loss` from the graph. A one-line guard fixes it; a unit test confirms the
RL gradient now flows to `router.weight` and the weights change. This document records the full path.

Owner: jonathanp. Last updated: 2026-07-30. Fix branch: pushed separately (see end).

---

## 0. Goal

Determine whether **router-only RL** (LM frozen, <1% of params) can reduce the MoE inference
**critical path** (CP = Σ over 94 MoE layers of max-tokens-on-any-expert) at 235B while holding
accuracy — and how it compares to the aux load-balancing loss. Math-blend variant: train the router on
a 70% math / 30% general blend and measure transfer to GSM8K/MATH (dev = 10% hash-holdout, test =
official).

---

## 1. Phase 1 — Initial results (what looked true, and was wrong)

Evaluated the iter-500 sweep (aux-coeff sweep + a widened RL sweep over reward type, γ, KL, critic,
GAE, PPO-epochs, rlc):

- **Aux traces a clean accuracy-vs-CP frontier.** Math-test CP reduction vs the untouched-router
  baseline: aux0.001 −8.7%, aux0.005 −30%, aux0.02 −41%, at graceful accuracy cost. Transfer to the
  math distribution confirmed (~30% math-CP reduction vs ~7% for general-corpus-trained routers).
- **Every RL variant landed on top of the aux0.001 floor (~8.5% CP reduction) and was identical to
  every other RL variant** — reward type, γ (0.8/0.99), critic, GAE, PPO2/4, rlc 0.1 vs 1.0 all gave
  the same CP trajectory. Interim (wrong) conclusion: "RL ≤ aux; RL adds nothing."

This uniformity across wildly different RL knobs was the first red flag: a *working* RL system should
respond to rlc, reward type, and γ. It didn't.

---

## 2. Phase 2 — Reward / gradient investigation (hypotheses)

Question posed: *why isn't the RL reward increasing?* Fitting per-step reward slopes on the training
CSVs showed the `diff_lse_load` reward *decaying* (0.178 → ~0.01–0.16), not climbing. Hypotheses
raised (all later shown to be **downstream** of the real bug):

- **H1 — difference-reward decay.** `diff_lse_load` is a per-token *counterfactual difference* reward
  (Dₜ = improvement from moving a token off its hot expert); it decays toward 0 as balance improves.
  So "not increasing" is partly by design.
- **H2 — advantage standardization erases scale.** Advantages are renormalized to unit variance every
  step (`rl_advantage_std ≡ 1.0`), discarding the reward's calibrated magnitude.
- **H3 — misdirected credit.** The loss lowers the summed log-prob of all top-k chosen experts and
  diffuses freed mass to all unchosen experts; it never targets the counterfactual destination `e_cf`.
- **H4 — critic quality.** For critic cells, `rl_value_loss` for GAE-0.95 stayed high (~13) → noisy
  advantages.

Two flag-gated code experiments were built to test H2/H3 (see §6): `--rl-no-advantage-norm` and
`--rl-credit-counterfactual`. **These were the wrong level of the problem** — the gradient was already
severed upstream, so no reward-shaping could matter.

The thread that unraveled it (prompted by "track gradients and critic loss"): the code already computes
a diagnostic `rl_grad_norm_on_logits` (the RL gradient magnitude on the routing logits), but it **logged
0 times** — its gate `rl_loss.requires_grad` was False every step.

---

## 3. Phase 3 — The finding: RL is a disconnected no-op

**Verdict: the RL policy gradient does not reach `router.weight` in any KL-enabled cell (all of them).**
`rl_loss.requires_grad == False` at the point it is added to the training loss, so it contributes
exactly zero gradient.

### Mechanism (exact)

1. The grad-enabled **main forward** populates `RouterTrajectoryTracker.layer_decisions` with the
   router's **in-graph** logits (`requires_grad=True`, `grad_fn` present).
2. Because `kl_loss_coeff > 0`, each step then runs a **KL reference forward** with the frozen router,
   under `with torch.no_grad()` (`helper.py::_run_reference_forward`, sets `tracker.paused = True` at
   entry, resets it in a `finally`).
3. **The live router does not honor `paused`.** `backends/megatron/Megatron-LM-250624/megatron/core/
   transformer/moe/router.py` (~L701) calls `add_layer_decision(...)` on *every* forward, including the
   no-grad reference forward → it **overwrites** the trajectory with detached logits.
4. `loss_func` then builds `rl_loss = compute_reinforce_loss(...)` from the now-detached logits →
   `rl_loss.requires_grad = False`. The additive injection (`helper.py`: `loss = loss + rl_loss·scale`;
   the `RouterRLLossScaler` autograd Function is dead/unused) backprops **zero** into the router.

The `paused` flag exists precisely to prevent this, and the authors already added the analogous guard
for KL's own logit capture (`_kl_state['capture_disabled'] = True`, commented "so the reference forward
does NOT overwrite current_logits ... in the grad graph"). **They simply forgot the same guard for the
RL trajectory.** The `[RL WARNING] routing_logits.requires_grad=False` check never fires because it's
gated on `torch.is_grad_enabled()`, which is False inside the reference forward — a false all-clear.

### Empirical fingerprints (all consistent with a severed gradient)

- `rl_grad_norm_on_logits` logged 0×; `rl_policy_loss ≈ 1e-8` (mean of mean-zero advantages — expected
  regardless of connection); `approx_kl = 0`, `clip_fraction = 0`.
- `num_tokens_on_critical_path` identical across every reward-type / rlc / γ / baseline variant.
- The "two `router.py`" confusion: the `qwen3_moe` router is **dead/shadowed** (megatron.core spec wins
  in `pretrain_qwen.py`); the `backends/megatron` one is live.

**This supersedes every earlier RL conclusion** ("RL ≤ aux", "RL doesn't move CP", "all variants
identical", the H2/H3 estimator diagnosis, and the older "no token credit" note) — all were artifacts of
the gradient being severed, not evidence that RL is weak. **Whether connected RL works is now open and,
for the first time, testable.**

---

## 4. The fix (one line, applied)

`megatron_patch/model/qwen3_moe/moe/rl_trajectory.py`, first line of `add_layer_decision` body:

```python
if getattr(self, 'paused', False):
    return
```

This makes the reference forward stop clobbering the trajectory, so the main forward's in-graph logits
survive → `rl_loss.requires_grad == True` → the RL gradient reaches `router.weight`. Backup:
`rl_trajectory.py.bak_pausedfix`. Applied via read-into-var + atomic rename + `ast.parse` verification.

---

## 5. Verification (unit test — `examples/qwen3/benchmarks/rl_sanity_test.py`)

Reproduces the exact scenario in isolation (tiny router): main forward → `paused=True` reference forward
→ compute RL loss → check gradient + a weight step. Run in the pai-megatron container.

**TEST A (with the fix — the connected path): PASS**
```
after MAIN forward:       stored logits.requires_grad=True  grad_fn=True
after PAUSED ref forward:  stored logits.requires_grad=True   ← guard blocked the overwrite
rl_loss.requires_grad=True  value=-4.6e-02
‖∂(rl_loss)/∂(router.weight)‖ = 1.55                          ← gradient reaches the weights
router.weight change after one RL step = 0.155               ← weights change
```
**TEST B (contrast — deliberately skip the guard = old bug):**
```
stored logits.requires_grad=False
torch.autograd.grad → RuntimeError: "element 0 ... does not require grad and does not have a grad_fn"
```
So the old path literally cannot compute a gradient, and the guard is what flips it. **Gradients flow;
weights change; RL is connected.**

Caveat: the *live-training* diagnostic `rl_grad_norm_on_logits` still isn't surfacing in the `.out` log
(computed into `loss_dict` but swallowed by a bare `except` / not printed) — a telemetry gap, not a
connection gap. TODO: unmask it so the RL gradient can be watched during the connected campaign.

---

## 6. Experiments (queued against the CONNECTED RL)

The 7 experiment cells were reset to fresh-start on the fixed code (throttle idx 24–30):

| cell | config | question |
|---|---|---|
| `ex_pureRL` | AUX=0 KL=0 RLC=1.0 | does RL *alone* balance the router? |
| `ex_pureRLcp` | AUX=0 KL=0 REWARD=critical_path RLC=1.0 | does an *absolute* reward get climbed + cut CP? |
| `ex_pureRLhilr` | AUX=0 KL=0 RLC=1.0 EXTRA_LR=5e-4 | is the router moving too slowly? |
| `ex_auxklNoRL` | RLC=0 | aux0.001+KL control (isolates KL's accuracy edge) |
| `ex_noAdvNorm` | `--rl-no-advantage-norm` | keep the reward's calibrated scale (H2) |
| `ex_creditCF` | `--rl-credit-counterfactual` | directed credit toward `e_cf` (H3) |
| `ex_rlFixed` | both flags | RL's best shot |

Recommended addition: a **connected standard-config** cell (aux0.001 + kl0.001 + rlc0.5 + diff_lse — the
exact config the whole campaign ran, now connected) as the clean "does RL work when actually plugged in?"
baseline. **Result: pending** (cells running as of 2026-07-30). Early note: `ex_pureRL` (aux=0) shows
`load_balancing_loss ≈ 5.77` at iter 313 — gradient flows, but RL-alone balancing is not yet evident;
this is now the real open question, not a bug.

### Estimator-fix flags (implemented, default-OFF = byte-identical current behavior)
- `--rl-no-advantage-norm` — skip unit-variance advantage standardization in both per-token loss paths.
- `--rl-credit-counterfactual` — per-token term = `logP(e_src) − logP(e_cf)` (REINFORCE path).
- Plumbed via `submit_fresh_corner_ep16_math.sh` env `NO_ADV_NORM=1` / `CREDIT_CF=1` (emit only when set).

---

## 7. Auxiliary fixes made along the way

- **MATH dev eval was broken (0.0).** `minerva_math_*_train.yaml` had `num_fewshot: 0`; Minerva-MATH
  needs its 4 hardcoded few-shot examples to emit `\boxed{}`. Fixed to `num_fewshot: 4` (hardcoded
  fewshot ⇒ no dev leakage). Verified: baseline MATH dev 0.0 → **49.51%** (≈ test 51.96%). Note: a
  global `--num_fewshot` can't be used because gsm8k and minerva want different shot counts — the value
  must live in each task YAML.
- **Gradient/critic tracking.** `rl_value_loss` (critic loss) is already logged for critic cells; added
  `grad` + `critic` panels to `plot_train_metrics.py`. `rl_grad_norm_on_logits` will log once the live
  diagnostic is unmasked.
- **Disk / router-save prune.** User hit the 250 TiB quota. `router_save_prune_general.py` (+ self-
  chaining container job) extracts `router_weights.pt` from old general-sweep checkpoints (node-local
  tmp first, since Lustre was over-quota), verifies ≥90 keys, then deletes shards — frontier / current-
  experiment / active-campaign / queued cells protected. Freed 79.5 TB (169 checkpoints), quota 250 →
  178 TiB.

---

## 8. Status & next steps

1. **Unmask the live `rl_grad_norm_on_logits` diagnostic** so the RL gradient can be watched every step.
2. **Run the connected experiment cells + the connected standard-config baseline**; measure whether RL
   actually moves CP now (the first honest test).
3. **Re-derive the RL vs aux comparison** — the prior verdict is void; connected RL must be re-measured.
4. Finish the dev→test **validation-selected** Pareto (MATH dev re-eval landing).

---

## 9. Connected pure-RL DIVERGES — trend analysis + stabilization plan (2026-07-31)

First connected pure-RL run (`ex_pureRL`: aux=0, kl=0, rlc=1.0, diff_lse_load) trained to 1500 on the
fixed code. Verdict: **RL is connected and initially works, then diverges via gradient explosion.**

Binned 0->1500 trend (plots: /workspace/plots/rl_divergence/):

| phase | iters | load_bal | CP | lm_loss | grad_norm |
|---|---|---|---|---|---|
| RL working   | 1-200   | 3.3->3.2  | 9873->8968 | 1.70->1.16 | ~1.5 |
| instability  | 300-500 | oscillate | oscillate  | 1.7->1.9   | 1.6->6.6 |
| grad blow-up | 600-900 | 4.1->5.4  | ->10.7k    | 2.0->2.4   | 8->59 |
| divergence   | 900-1500| ->10.9    | ->11.7k    | ->2.7      | 214->80 |

Key: early on RL genuinely reduced imbalance AND improved lm_loss (1.70->1.16) with aux=0 — proof RL is
connected and doing real work (the old broken campaign never moved CP via RL). Then grad_norm climbs
1.5->6.6->59->214 and load_bal/CP/lm_loss collapse in lockstep. Failure mode: classic RL instability —
rlc=1.0 with NO trust region and NO anchor (aux=0, kl=0), nothing bounds the router's drift. Confound:
the continuation mechanism re-warms the LR at each segment boundary (LR jumps back to ~1e-4 at iter ~991,
coincides with a grad spike) — the continuation should carry ONE continuous LR schedule to 1500; but grad
already grew 1.5->59 within the first segment, so the LR restart is aggravating, not the sole cause.

### Divergence hypotheses
- H-a rlc too hot (overshoot) -> lower rlc.
- H-b no trust region -> re-enable KL to the pretrained reference (now safe post paused-fix).
- H-c no balance anchor -> small aux floor to keep the router in a sane region.
- H-d advantage renorm amplifies noise as the reward decays -> keep the calibrated reward scale.
- H-e LR warmup restart per continuation segment -> make the continuation LR-continuous (infra fix).

### Stabilization experiment plan (queued in math_throttle, fix + rl_grad_diag telemetry, all rlc=0.5 to isolate)
- rlc sweep: 0.1 / 0.5 / 1.0(done) / 2.0 (aux=0,kl=0) -- coefficient / stability curve.
- ex_stab_kl0.001:    AUX=0 KL=0.001 RLC=0.5          -- KL trust region.
- ex_stab_aux0.001:   AUX=0.001 KL=0 RLC=0.5          -- small aux anchor.
- ex_stab_std:        AUX=0.001 KL=0.001 RLC=0.5      -- standard config (both anchors), CONNECTED.
- ex_stab_noadvnorm:  AUX=0 KL=0 RLC=0.5 NO_ADV_NORM=1 -- keep reward calibrated scale.

Success = sustained CP/load-balance reduction to iter 1500 WITHOUT grad_norm blow-up or lm_loss rise,
and rl_grad_norm_on_logits scaling monotonically with rlc (connection + responsiveness).
TODO (infra): fix LR-schedule restart across continuation segments.

---

## §10. PRODUCTION CONFIRMATION under KL>0 (2026-07-31)

The unit test proved the paused-guard fix reconnects the RL gradient in isolation.
This is the **live 235B confirmation under the exact condition that broke it**.

Cell `235bv21math_ex_stab_std_r1` runs the **standard campaign config**:
`AUX=0.001 KL=0.001 RLC=0.5 REWARD=diff_lse_load BASELINE=mean` — the same config
whose RL was a silent no-op across the whole prior sweep, because `kl_coeff>0`
triggers `_run_reference_forward()`, whose detached logits overwrote the RL
trajectory (tracker.paused was set but never honored by the live router).

Fresh telemetry at iter 7:
- `rl_loss_requires_grad = 1.0`  (rl_loss is on the autograd graph)
- `rl_grad_diag = 1.0`           (autograd.grad(rl_loss, router_logits) succeeds, non-None)
- `rl_grad_norm_on_logits = 9.06e-05`  (nonzero gradient on the pre-softmax router logits)

=> The fix holds in production **with KL on** — the precise reference-forward path
that detached RL before. Verdict: RL is now connected in the standard config, not
just in the aux=0 pure-RL cells. All prior "RL == aux" / "RL is a no-op" campaign
conclusions are confirmed VOID and are now being re-measured with a live gradient.

Open question tracked next: does aux-anchor (0.001) + connected RL at rlc=0.5+KL
reduce CP **without diverging**? aux=0 cells (rlc0.1, rlc0.5+KL) are stable
(grad_norm flat ~1.3-1.6) but hold CP ~baseline; stab_std is the aux-anchored test.

---

## §11. ATTRIBUTION: connected RL alone does NOT reduce CP; aux does (2026-07-31)

With RL provably connected (§10), we can finally ask the real question the broken
gradient hid for the whole campaign: **does RL reduce CP?**

Binned mean CP over training (canonical BS=1 seqlen-128 probe, baseline ~9870):

| cell (config)                    | CP trajectory (mean per 150-iter window)     | verdict          |
|----------------------------------|----------------------------------------------|------------------|
| pureRL_rlc0.1 (aux=0, RL, conn.) | 9185→9392→9281→9423→9263→9324 over 0-900     | FLAT ~9300       |
| stab_kl (aux=0, rlc0.5+KL, conn.)| 9216→9455→9370→9528→9373 over 0-750          | FLAT ~9400       |
| aux0.001 ALONE (no RL)           | ...→8116→7887→7887→7693 over 900-1500         | SUSTAINED -22%   |
| stab_std (aux0.001+rlc0.5+KL)    | 9003→8775 over 0-300 (early, declining)       | aux-like so far  |

Two independent aux=0 cells, connected RL, ~750-900 iters each: **mean CP pinned
near baseline.** Per-iter CP swings hard (minCP dips to ~5600, other iters >11000),
so RL *is* perturbing the router — but with NO sustained direction. High-variance
noise, not optimization. Aux, by contrast, drives a clean sustained -22%.

### Reinterpretation of the campaign's "RL == aux" result
Prior "RL" runs were `aux0.001 + DISCONNECTED RL`, i.e. effectively **aux-alone** —
so they matched aux trivially. Now that RL is genuinely connected, **pure RL (aux=0)
is WORSE than aux, not equal.** Fixing the gradient did not make RL beneficial; it
exposed that connected RL *with the current reward* (`diff_lse_load`) is CP-neutral.

### Implication
The bottleneck was never only the disconnected gradient — it is the reward /
credit-assignment signal (matches the original "no token credit on the true
objective" diagnosis). Connecting the gradient is necessary but not sufficient.

Open: does stab_std (aux anchor + connected RL) beat aux-alone's ~7700 at iter
1000-1500? If < 7700 => RL adds value on top of aux. If ~7700 => RL neutral. That
comparison is the next decisive datapoint. Levers if RL stays neutral: (a)
credit_counterfactual + no_advantage_norm flags (already wired, unrun); (b)
CP-direct reward instead of diff_lse_load; (c) critic/GAE for variance reduction;
(d) higher rlc without divergence (needs a trust region stronger than KL alone).

### §11.1 Connection verification of the aux=0 "flat" cells (integrity check)
Doubt raised: the original pureRL_rlc0.1 / stab_kl seg=1 logs (it 0-968) show NO
grad_diag telemetry, while their seg=2 continuations do — were seg=1 actually
connected, or is §11 measuring disconnected cells?

Resolved by working-tree timing (definitive):
- rl_trajectory.py (paused-guard FIX): mtime 2026-07-30 13:00 — present in the
  working tree BEFORE the seg=1 submits (08:05 / 08:13 on 07-31).
- helper.py (grad_diag TELEMETRY): mtime 2026-07-31 08:39 — edited AFTER those
  submits, BEFORE the seg=2 continuations (12:09 / 12:19).
=> seg=1 ran WITH the fix (connected) but WITHOUT the telemetry logging; that is
why grad_diag is absent from their logs yet grad_diag=1 in the continuations.
Corroboration: seg=1 logs print rl_mean_reward + rl_policy_loss (RL loss computed),
and load_bal drifted off pretrained 3.33 (router being updated).

Conclusion: the ~900-iter flat-CP result IS a genuine connected-RL result. §11
holds. (Whether the mechanism is "RL is truly CP-neutral" vs "connected but the
diff_lse_load reward / update magnitude doesn't drive CP down" is unresolved, but
the practical verdict is identical: connected RL at aux=0, rlc in {0.1,0.5}, does
not reduce CP over ~900 iters, while aux drives a clean -22%.)

---

## §12. DEBUG: why connected RL still doesn't reduce CP (reward flat, not a code bug) — 2026-07-31

User concern: "reward should be increasing, rl loss decreasing." Investigated with the
systematic-debugging process (root cause before fixes). Verdict: RL is correctly
implemented and connected; it does not learn because the policy-gradient STEP is too
weak at the campaign's hyperparameters — a tuning/method issue, NOT a bug.

### Evidence
1. Telemetry (pureRL_rlc0.1, aux=0 KL=0, 900 iters): reward FLAT ~0.18, pol_loss ~1e-8,
   load_bal drifts 3.54->3.89 (slightly worse), CP flat ~9300.
2. pol_loss ~1e-8 is EXPECTED, not a bug: REINFORCE loss value = -mean(adv*logpi) with
   mean-0 standardized advantages ≈ -Cov(adv,logpi) ≈ 0. The learning signal is the
   GRADIENT (verified nonzero, ‖grad on router.weight‖=1.55 in the unit test), not this
   scalar. "loss decreasing" is not a valid health metric for standardized REINFORCE.
3. Advantages are healthy: adv_std=1.0, adv_max~1.5 (standardization works; not vanishing).
4. Reward normalizer (--rl-normalize-rewards) is a NO-OP given advantage normalization:
   its affine (reward-m)/s cancels under the subsequent (adv-mean)/std -> final advantage
   = (reward-mean)/std(reward) regardless. Ruled out as a cause of the drift (drift = noise).

### Minimal reproduction (toy: E=16, topk=2, N=256, exact diff_lse_load + both credit branches)
- From EXTREME imbalance (max_load 161, ideal 32): DEFAULT credit balances to ~34 (CV
  1.10->0.04) — so the default per-token credit is NOT fundamentally misaligned. Refuted
  the first hypothesis. Counterfactual credit and no_adv_norm also work.
- lr sweep (DEFAULT credit): lr>=0.01 balances; lr=0.002 -> 161->145; lr=0.0005 -> 161->159.
  A hard LR THRESHOLD below which policy-gradient RL stays FLAT in the available steps —
  reproducing the real symptom.
- Near-balanced start (bias=0.4, like a pretrained router): balances only when lr adequate.

### Root cause
Policy gradient is high-variance and weak per step. At real lr=1e-4 (clip-grad=1.0,
rlc<=0.5), the effective step is far below the threshold needed to move an already
near-balanced 235B router within 1500 iters -> flat CP. Aux succeeds at the SAME lr
because its dense differentiable gradient has vastly better signal-to-noise. Raising the
step (rlc=1.0) causes router collapse (grad explosion) rather than clean learning: RL is
boxed between too-weak (flat) and too-strong (collapse).

### Reinterpretation
The campaign's "RL == aux" was because broken-RL runs were effectively aux-only (§10/§11).
Now: connected RL alone is CP-neutral because it can't learn at these hypers, while aux
optimizes the same objective far more efficiently. RL is not broken — it is dominated by
aux as an optimizer for this near-balanced, tightly-clipped, low-lr regime.

### Next test (decisive, real system)
Widen the step with variance/collapse control and see if CP drops: higher policy lr
(PLR 5e-4..1e-3) and/or higher rlc WITH clip-grad, plus optionally counterfactual credit
(--rl-credit-counterfactual) to sharpen the signal and a critic/GAE to cut variance. If CP
drops without collapse -> step-size root cause confirmed + a working RL config. If it only
oscillates/collapses -> RL is dominated by aux for this problem (report as the finding).

---

## §13. DECISIVE: higher-lr test diverges -> variance is the bottleneck (2026-07-31)

Test of the §12 step-size hypothesis in the real system. Cell ex_pureRLhilr_r1
(AUX=0 KL=0 RLC=0.5, PLR=1e-3 = 10x the campaign lr), clip-grad=1.0.

Result (100-iter binned CP; flat-baseline pure-RL ~9300, real baseline ~9870):
  it   0-99 : meanCP=9444  load_bal=4.20
  it 100-199: meanCP=9742  load_bal=5.32
  it 200-299: meanCP=10333 load_bal=6.77   grad_norm 39.7 (diverging)

10x lr does NOT unlock CP reduction — it DIVERGES: CP rises, load_bal worsens
(4.2->6.8, vs 3.33 baseline), pre-clip grad explodes (1.6->39.7). Router collapsing
even under clip-grad=1.0 (clip bounds the step magnitude but the direction is
consistently toward collapse).

### Conclusion — the RL diagnosis is complete
Pure RL (aux=0) is boxed on both sides, now confirmed in the real system:
  - lr=1e-4  -> too weak  -> FLAT (§11): can't move the near-balanced router in 1500 iters
  - lr=1e-3  -> too strong -> DIVERGES (§13): amplifies the noisy 94-layer policy gradient
There is NO learning-rate sweet spot. The toy (single clean layer) balanced at high lr;
the real 235B policy gradient is averaged over 94 layers x a huge batch = very high
variance, so larger steps amplify noise, not signal. The bottleneck is gradient VARIANCE,
not step size or credit rule or connection.

Aux dominates because its dense differentiable load-balance gradient has vastly better
signal-to-noise than the REINFORCE estimator, and works at lr=1e-4 where RL cannot.

### Only remaining lever for making RL competitive: variance reduction
- learned critic baseline + GAE (per-token value head) to cut estimator variance
- counterfactual credit (--rl-credit-counterfactual) to sharpen per-token signal
- much larger effective batch / reward smoothing
NOTE: the campaign's OLD critic/GAE cells are VOID (ran pre-fix = disconnected RL).
A clean re-run (aux=0, BASELINE=critic, GAE, connected) is the decisive "can RL be
rescued" experiment. If a critic-stabilized RL still can't beat aux -> RL is dominated
for this problem, which is itself the paper's systems finding.

### §13.1 CORRECTION — the higher-lr result above is CONFOUNDED (2026-07-31)
Integrity check: the throttle log shows ex_pureRLhilr_r1 was submitted with
resume_from_iter=997, NOT fresh. Its ckpt dir was created Jul 30 20:43 (prior session,
when the cell was the OLD RLC=1.0 config). The throttle's latest_iter() found that
checkpoint and RESUMED it. So §13 measured "continue an already-diverging RLC=1.0
checkpoint at lr=1e-3" (it started at load_bal 4.2, not the pretrained ~3.3) — NOT a
clean fresh higher-lr test.

=> §13's "higher lr diverges" is INCONCLUSIVE. The step-size-vs-variance question is
NOT yet answered. Re-running pureRLhilr FRESH (cleared stale checkpoint, resume_from_iter=0
from pretrained) to cleanly test whether lr=1e-3 from a clean start diverges (=> variance
is the bottleneck, §12/§13 thesis holds) or reduces CP (=> step-size was the issue after
all, and the campaign's lr=1e-4 was simply too low).

§11 is unaffected: pureRL_rlc0.1 and stab_kl both verified resume_from_iter=0 (fresh,
started at load_bal ~3.3). Only pureRLhilr carried a stale Jul-30 checkpoint.

### §13.2 Follow-up: log evidence + guaranteed-fresh re-run (2026-07-31)
Re-examined: pureRLhilr's log shows "iteration 1/1500", adlr_autoresume=False, and NO
checkpoint-load message -> it most likely ran FRESH from the pretrained base despite the
anomalous resume_from_iter=997 flag (which pointed at an empty/invalid ckpt dir and was
inert). The load_bal 4.2 in bin 0-99 is fast early divergence at 10x lr, not a degraded
start. So §13's "fresh lr=1e-3 diverges" is probably valid — but to remove all doubt the
stale Jul-30 ckpt/log/tensorboard dirs for pureRLhilr AND auxklNoRL were moved aside and
both cells reset to TODO. The guaranteed-fresh re-run (resume_from_iter=0, load_bal starts
~3.3) is the definitive test of step-size vs variance. Watching load_bal from iteration 1.

---

## §14. aux+RL completes at CP~7879 = aux-path; RL adds noise, not improvement (2026-07-31)

stab_std (AUX=0.001 RLC=0.5 KL=0.001, the standard config) ran to iter 1499:
  it 900-999 :  meanCP=8030  load_bal=2.89
  it 1300-1399: meanCP=7781  load_bal=2.85
  it 1400-1499: meanCP=7879  load_bal=2.87  (minCP 5442, maxCP 9711 -> wide RL oscillation)

Final mean CP ~7879 == the aux+RL control (30758349, KL=0) ~7693. The RL component adds
wide per-iter oscillation (5400-9700) but does NOT push the MEAN below the aux-driven
~7700-7900. KL makes no difference (7879 vs 7693).

### RL research conclusion (strongly indicated; pure-aux control auxklNoRL still GPU-queued)
- Aux drives the CP reduction: 9870 -> ~7800 (-20%), clean and sustained.
- RL alone (aux=0): useless — flat at lr=1e-4 (§11), diverges at lr=1e-3 (§13); variance
  bottleneck (§12).
- RL on top of aux: adds noisy oscillation, not a lower mean (§14). aux+RL ~7879 ~=
  aux+RL-control ~7693; both ~= what aux alone is expected to give.
=> Aux dominates RL as the optimizer for router load-balancing / CP on this near-balanced
   235B model. The clean confirmation is the pure-aux (RLC=0) cell auxklNoRL reaching ~7800
   too (=> RL contributes nothing beyond noise). That cell is Priority-queued.

This is the corrected, mechanistically-grounded successor to the campaign's original
"RL == aux" claim: not because RL matches aux, but because RL (once actually connected) is
too high-variance to help, and aux does all the work.

---

## §15. CONFIRMED: pure-aux control descends smoothly to the same floor; RL adds only noise (2026-08-01)

The clean pure-aux control finally scheduled (after a ~6h cluster maintenance drain) and
ran FRESH (verified: started at load_bal 3.28, the pretrained baseline).

ex_auxklNoRL_r1 (AUX=0.001, RLC=0, no RL) binned CP:
  it   0-49 : CP=9096  load_bal=3.41
  it 100-149: CP=8880  load_bal=3.23
  it 150-199: CP=8682  load_bal=3.20
  it 200-249: CP=8328  load_bal=2.91   (snapshot it203 CP=7221)

SMOOTH monotonic descent toward ~7800 with NO oscillation — contrast stab_std (aux+RL)
which reached the same ~7879 mean but with wide per-iter swings (5400-9700). Pure-aux's
descent is if anything smoother/faster than aux+RL at matched iters.

### FINAL RL research verdict (now fully controlled)
| config                         | CP result            | interpretation            |
|--------------------------------|----------------------|---------------------------|
| pure aux (RLC=0)               | smooth 9870->~7800   | aux does all the work     |
| aux + RL (rlc0.5, +/-KL)       | noisy  ~7879         | RL adds oscillation only  |
| pure RL (aux=0), lr=1e-4       | flat ~9300           | too weak to learn (§11)   |
| pure RL (aux=0), lr=1e-3       | diverges             | too strong; variance (§13)|

=> Connected RL contributes NOTHING beyond noise for router CP reduction on this
near-balanced 235B model; the aux loss is a strictly better-conditioned optimizer of the
same objective, and adding RL on top only injects variance (mildly detrimental). This is
the mechanistically-grounded, fully-controlled replacement for the campaign's original
(instrument-confounded) "RL == aux" claim.

Only remaining datapoint: the fresh pureRLhilr (lr=1e-3, resume_from_iter=0) — expected to
diverge from a clean start too, sealing the "no lr sweet spot / variance bottleneck" result.

### §15.1 SEALED: fresh pureRLhilr diverges too (2026-08-01)
The guaranteed-fresh pureRLhilr (AUX=0 KL=0 RLC=0.5 lr=1e-3, resume_from_iter=0) diverges
from a CLEAN start: load_bal 3.94->4.71->5.71->6.32, CP 9377->9879, grad_norm ->47 over
~200 iters. So §13's higher-lr divergence was REAL, not the checkpoint-resume confound.
Confirmed: pure RL has no lr sweet spot (flat at 1e-4, divergent at 1e-3) — gradient
variance is the bottleneck. Cell cancelled after the verdict was unambiguous.

Pure-aux (auxklNoRL) in parallel: smooth 9096->8343 (it 0-350), load_bal 3.41->3.04,
grad 1.1 — clean descent to ~7800, no oscillation. RL-vs-aux question fully closed.

---

## §16. CORRECTION: RL+aux DOES beat aux — on HOLDOUT at matched CP, mid-band (2026-08-01)

§11-§15 concluded "RL adds nothing beyond noise." That was OVER-GENERALIZED from the wrong
regime/metric. User correction: there were real scenarios where RL+aux > aux. Re-examined
the trusted-holdout Pareto (benchmark_results.csv):

Holdout mean by CP band — connected-RL+aux vs aux-only:
  CP 6000-6499:  connRL 80.71  vs  aux 79.53   (+1.2pp)
  CP 6500-6999:  connRL 81.53  vs  aux 80.37   (+1.2pp)
  CP 7000-7499:  connRL 82.34  vs  aux 80.59   (+1.75pp)
  CP 7500-7999:  connRL 82.54  vs  aux 81.94   (+0.6pp)
  CP 3500-4499:  ~parity (deep-end RL edge is weak/n=1, NOT the main effect)

=> The real, repeatable "RL+aux > aux" is the MID-BAND (CP ~6000-7500), ~1-1.75pp holdout,
CONSISTENT across 3 bands. NOT the deep end (parity/noisy).

### Why §11-§15 missed it
1. Metric: measured TRAINING-time CP/load_bal. Aux trivially wins at reducing CP (it
   optimizes balance directly). RL's value is HOLDOUT ACCURACY at matched CP — a different
   axis. CP-reduction != the metric that reveals RL.
2. Reward: used diff_lse_load. The holdout winners use per_token_load_weighted (dense
   "how overloaded is your chosen expert" signal). r33 mid-band used the entropy reward.
3. Regime: tested aux=0/0.001 at moderate CP; the edge lives at connected RL+aux mid-band.

### Mechanism (why RL+aux generalizes better at matched CP)
aux-only, to hit a target CP, drives a BALANCE-OVERFIT router: buys CP + selection-suite
accuracy but DEGRADES holdout generalization. RL+aux hits the same CP, but RL optimizes the
actual discrete load objective WITH EXPLORATION, landing on a better-generalizing router.

### Reconciliation with the disconnection finding (§10)
The BEST holdout cells (top of Pareto) were CONNECTED (KL=0): 235bv5b, 235bv14cg, 235bv16cgr.
The recent campaign focus 235bv15klcg was DISCONNECTED (KL>0) -> its RL was a no-op, so its
"RL" runs couldn't show the edge. The fix matters MOST here: re-running the klcg/v16cgr/v18s
families CONNECTED with per_token_load_weighted, mid-band, should recover/strengthen the
+1-1.75pp holdout edge.

### Corrected next experiment
Connected RL+aux, reward=per_token_load_weighted, aux0.001-0.005 + rlc0.5, target CP
~6000-7500, evaluate HOLDOUT at matched CP vs aux-only seed-replicated controls. Expect
~1-1.75pp edge. (Honest caveats: n small per band; seed noise ~1pp; deep-end edge weak.)

---

## §17. FLAT-LOSS BUG: RL loss collapses to -Cov(ptlp,adv)~1e-8 -> gradient ~0.1% of aux (2026-08-01)

User directive: efficient RL should show reward growing + loss decreasing; if not, find + fix.
It doesn't -> investigated with telemetry.

### Measured facts (production, klcgC per_token_load_weighted, connected, grad_diag=1)
- rl_policy_loss pinned at ~1e-8 (range 2e-9..4e-8), NOT decreasing, for ALL cells/rewards.
- rl_grad_norm_on_logits ~1.0e-3; rl_grad_mean_on_logits ~3e-6 (per-layer, gradient of rl_loss
  wrt router logits).
- Total (aux+RL) grad norm ~1.0-1.5 (clip-grad=1.0). => RL contributes ~0.1% of the router step;
  it literally cannot move the router. Aux (and the LM loss) dominate.

### Mechanism (from the code)
REINFORCE loss = -mean(per_token_log_prob * advantage). Advantage is standardized to EXACTLY
mean-0, so loss = -Cov(per_token_log_prob, advantage). Observed ~1e-8 = an EXACT structural
zero (sampling noise of a real Cov would be ~1e-3, not 1e-8). So per_token_log_prob is either
~constant across tokens, or ~exactly uncorrelated with the standardized advantage.

### What was RULED OUT (controlled CPU sims + static reads)
- Default credit formula: toy gives real loss (-0.12..-0.31) + gradient (0.2-0.4). Not the formula.
- EMA loads, global-vs-per-layer advantage standardization, multi-layer aggregation, content-vs-
  load-correlated routing: every combination still yields healthy loss (~0.01-0.3). None reproduce ~1e-8.
- Live router call site passes RAW logits + binary top-k routing_map (correct); routing_map is a
  mask, not soft probs (so ptlp != -entropy). Reward sign is correct (overloaded -> negative reward).
=> The collapse is PRODUCTION-SPECIFIC (real pretrained router / bf16 / real data), not a formula bug.

### Fix hypothesis (staged, live A/B)
Counterfactual credit (--rl-credit-counterfactual): per_token_log_prob = logP(e_src) - logP(e_cf)
= logit_src - logit_cf (the logsumexp CANCELS), a per-token MARGIN with real variance regardless
of router entropy AND directed at the load-balancing move. Toy CF ptlp std ~1.2 (real). Running
klcgCF_s1234 (CF) vs klcgC_s3031 (default): if CF yields a non-trivial, moving rl_policy_loss +
better CP, the default credit's collapse is the bug and CF is the fix. (GPU-contention-gated.)
NOTE: even with CF, the /total_tokens(94-layer) normalization keeps the RL gradient ~100x below
aux; a scale fix (per-layer norm or higher rlc) is likely also needed for RL to compete.

### Added telemetry (committed 39adcd9)
rl_ptlp_std, rl_cov_ptlp_adv, rl_raw_reward_std -> confirms constant-ptlp vs dead-reward once a
fresh telemetry-enabled cell logs (pending GPU + a metric-registration check).

---

## §18. FIX-TEST RESULT: counterfactual credit does NOT fix it -> the bug is the ADVANTAGE, not the credit (2026-08-01)

Ran klcgCF (CREDIT_CF=1, counterfactual credit) vs klcgC (default), both connected,
per_token_load_weighted, aux0.001+KL0.001, telemetry code.

RESULT: klcgCF rl_policy_loss stays ~1e-8 across 50 iters (1.1e-8, 1.8e-8, 3.3e-8, 1.3e-8,
1.4e-8, 1.9e-9) -- IDENTICAL to the default's ~1e-8. Counterfactual credit does NOT restore
a moving loss.

### Interpretation (decisive redirect)
loss = -Cov(per_token_log_prob, advantage). The ADVANTAGE is identical for both arms; only
ptlp differs (default: sum of chosen log-softmax; CF: logit_src - logit_cf). Both give ~1e-8
=> the collapse is NOT in the credit/ptlp. It is in the ADVANTAGE being ~0, i.e. the
per-token reward carries ~no usable per-token signal in production. The bug is UPSTREAM of
the credit, in the reward/advantage.

### What is CONFIRMED vs OPEN
CONFIRMED:
- RL is connected (grad_diag=1) but not learning; loss ~1e-8, grad ~0.1% of aux (§17).
- The credit formula is NOT the cause (CF refuted; §18).
- => the advantage is degenerate; the per-token reward gives no learnable signal.
OPEN (blocked):
- The EXACT production reason. My controlled toys (single/multi-layer, EMA, global-std,
  content-vs-load routing, low-vs-high load entropy) ALL produce healthy reward std (~0.12)
  and loss (~0.1-1). None reproduce the ~1e-8 collapse -> it is specific to the real
  pretrained router / bf16 / real load dynamics.
- The added diagnostic telemetry (rl_raw_reward_std, rl_ptlp_std, rl_cov_ptlp_adv) is NOT
  reaching the training log line (the rl-metrics logging path apparently doesn't carry the
  new loss_dict keys) -> could not read the raw-reward spread directly. Needs the logging
  path fixed, then a fresh cell, to see whether raw_reward_std ~= 0 (dead reward) definitively.

### Next steps to fully resolve (GPU + logging gated)
1. Fix the rl-metrics logging so rl_raw_reward_std / rl_ptlp_std reach the log; relaunch one
   cell -> definitively see if the per-token reward variance is ~0 in production.
2. If reward variance is dead: design a reward with genuine per-token signal on a near-
   balanced router (e.g. per-token PRIMARY-expert load percentile / rank, or a sharper
   counterfactual on the max-load bottleneck), rather than an average-over-k-experts load.
3. Independent of reward: the RL gradient is also /total_tokens(94-layer)-normalized to ~100x
   below aux; a per-layer normalization or much higher rlc is needed for RL to compete even
   with a good reward.

Honest status: the flat-loss bug is LOCALIZED (advantage/reward, not connection or credit)
but not yet ROOT-CAUSED to a specific line; both remaining steps are blocked on GPU
contention + the metrics-logging gap.

---

## §19. CORRECTION to §17-§18 + REAL bug: normalization suppresses RL by 94x (2026-08-01)

External expert review (correctly) refuted §17-§18's inference. The near-zero signed policy
loss (~1e-8) is EXPECTED under centered on-policy advantages and PPO's first-epoch ratio~=1
(loss ~= -mean(A) ~= 0 by construction); it is NOT evidence of a dead advantage/reward.
Decisive counter-evidence I already had: rl_grad_norm_on_logits ~1e-3 is NONZERO -> advantage
is NOT ~0. Reward degeneracy is UNPROVEN. My §18 "reward is dead" conclusion is WITHDRAWN.

### The real, confirmed bug: reduction over-normalization (single-process FP64 test)
loss = -mean over ALL (layer,token) of (A * per_token_log_prob), i.e. total_loss/total_tokens
where total_tokens = 94 layers x tokens. Analytic-gradient + reduction-scaling test:
  A production /total_tokens          -> ||RL weight-grad|| = 0.0147
  B sum (no averaging)                -> 44.2   (B/A = 3008 = total_tokens, exact)
  C /tokens only                      -> 1.38   (C/A = 94 = num_layers)
  D per-layer mean, then SUM layers   -> 1.38   (D/A = 94 = num_layers)
Autograd == analytic to 1e-17 (gradient math correct). => /total_tokens divides the RL
gradient by an EXTRA 94x vs the intended per-layer treatment (each of the 94 layers is a
separate routing decision). Aux loss is accumulated PER LAYER (summed over 94), so RL ends up
~94x weaker than aux at matched coeff -> RL optimizer-visible update ~0.1% of aux (matches
the measured rl_grad_norm_on_logits 0.001 vs total 1.0-1.5).

### Second real issue (methodological): deterministic top-k is not a REINFORCE sample
per_token_log_prob is the log-prob of the argmax top-k action, not an action sampled from the
categorical policy. So the "policy gradient" is a biased straight-through surrogate, not valid
on-policy PG. Fix = sample top-k without replacement during training, OR explicitly label the
objective a routing surrogate.

### Fix plan
1. Normalization: reduce RL loss PER LAYER (mean over that layer's tokens) then SUM over layers
   -> matches aux's per-layer accumulation, ~94x larger RL gradient (verify vs aux scale, and
   check Megatron's [loss,denom] handling doesn't divide again by microbatches/tokens).
2. Add unsigned-signal telemetry (term_abs_mean, term_rms, per-layer cov) + component grad norms
   (RL/aux/KL/LM) + ||dW||/||W|| before clipping (per reviewer).
3. (Separate) evaluate stochastic top-k sampling vs surrogate labeling.
NOTE: cells run PPO (_compute_ppo_loss_per_token), so the fix goes there (and REINFORCE).
The global-load/reward work (§ earlier) is DEPRIORITIZED (reward degeneracy unproven).

---

## §20. SYSTEMATIC CLAIM BATTERY — every hypothesis tested in isolation (2026-08-01)

Compiled all claims from three sources (this investigation, external reviewer, external
research report) and tested each separately (single-process, FP64 where analytic).

| # | Claim | Source | VERDICT (test) |
|---|-------|--------|----------------|
| C1 | RL grad severed by KL ref-forward | mine | CONFIRMED + fixed (unit + prod grad_diag=1) |
| C2 | Flat loss ~1e-8 => dead advantage/reward | mine §18 | **REFUTED** (nonzero grad; C4) -> §18 withdrawn |
| C3 | Flat loss = BF16 log_softmax truncation | report | **REFUTED**: BF16 (stable AND naive) preserves ptlp_std to conf-gap 50, 0% chosen-logprob rounds to 0. PyTorch log_softmax upcasts; router is high-entropy anyway |
| C4 | Flat signed loss EXPECTED (centered adv + PPO ratio=1) | reviewer | **CONFIRMED**: loss=1.7e-17 but grad=0.078 (alive) |
| C5 | RL grad suppressed 94x by /total_tokens | reviewer | **CONFIRMED**: sum-vs-mean ratio exactly 94 = num_layers; analytic grad matches to 1e-17 |
| C6 | 94-layer REINFORCE variance => weak/divergent | report | CONFIRMED (empirical: flat@lr1e-4, diverge@lr1e-3; mechanism = C11) |
| C7 | Deterministic top-k != REINFORCE sample => biased surrogate | both | **CONFIRMED**: cos(true PG, argmax surrogate)=0.44 mean, 17% anti-aligned |
| C8 | Advantage standardization erases scale | early | PARTIAL: changes grad direction ~11deg (cos 0.98) — a symptom of C7 (baseline only cancels for a SAMPLED action) |
| C9 | Default credit misdirects token credit | early | **REFUTED**: lowers hot-expert logit for 100% of overloaded tokens (== CF) |
| C10 | Reward uses local per-rank load (no EP all-gather) | mine | code-CONFIRMED; signal-degradation UNPROVEN (toy reward std ~0.12, not degenerate) -> minor |
| C11 | Router-shift => volatile IS => divergence | report (RSPO) | **CONFIRMED**: step 0.05 -> 9.3% top-k flip, IS std 767, max 34573, 88% out of clip band |
| C13 | Mid-band RL+aux +1-1.75pp holdout edge | mine §16 + report | CONFIRMED (historical holdout data) |

### The real, tested diagnosis (why connected RL fails to optimize CP)
Three CONFIRMED causes, in priority:
1. **Over-normalization (C5)**: /total_tokens divides the RL gradient by an extra num_layers=94x
   vs aux's per-layer accumulation -> RL update ~0.1% of aux. FIX = --rl-perlayer-norm (done).
2. **Biased surrogate (C7)**: deterministic argmax top-k is not sampled from the categorical
   policy, so the "policy gradient" is only ~0.44-cos aligned with the true PG (17% wrong-way).
   FIX = stochastic top-k sampling (Gumbel/sequential) during training, or label a surrogate.
3. **Variance + router-shift (C6/C11)**: high-variance estimator; tiny param steps flip top-k ->
   IS ratios explode -> bursty clipping -> divergence at any lr strong enough to matter.
   FIX = RSPO (router-shift trust region) + variance reduction (critic/GAE, larger effective batch).

REFUTED as causes: dead advantage (C2), BF16 (C3), credit misdirection (C9). The near-zero
signed loss is benign (C4). Minor: standardization (C8, symptom of C7), local load (C10).

### Strategic alternative (report §7): auxiliary-loss-free load balancing (DeepSeek-V3)
Detached, graph-free per-expert BIAS updated by a fixed step from measured load — zero gradient
interference, preserves specialization, sidesteps the REINFORCE-variance trap entirely. Strong
candidate to replace the RL-vs-aux tradeoff outright.

### Fix order to validate end-to-end (GPU): (1) --rl-perlayer-norm; (2) stochastic top-k; (3) RSPO.

### §20 addendum — code-level corroboration + empirical baseline (2026-08-01)

**C7 corroborated at the code level.** `--rl-stochastic-routing` (Gumbel-noise-before-topk) and
`--rl-stochastic-temperature` exist in arguments.py but are a **defined-but-unwired STUB** — zero
consumers in backends/ router.py or megatron_patch/model/. So the production forward genuinely uses
deterministic argmax top-k (confirms C7 independently of the analytic test). The C7 fix is therefore
unimplemented, not merely disabled.

**Empirical C5 baseline** (klcgC_s3031: connected RL, DEFAULT /total_tokens norm, per_token_load_weighted,
aux0.001 kl0.001 rlc0.5, seed3031, 1362 iters): `rl_grad_diag=1.0` (connected ✓), but
`rl_grad_norm_on_logits≈1.2e-3` (the 94×-suppressed gradient), `rl_mean_reward 0.35→0.16` and
`load_balancing_loss 3.20→3.94` (balance DEGRADING over training), `num_tokens_on_critical_path 8763→8221`
(CP falls only ~6%, aux-driven). This is the C5 failure mode made visible: RL is connected but too weak to
help. The `--rl-perlayer-norm` A/B (NORM1 vs GL0) should show ~94× larger RL gradient and a steeper CP drop
with improving (not degrading) balance. Analysis harness: benchmarks/../norm_ab_analyze.py.

---

## §21. Reviewer response + isolated-sim corroboration + verified C7 fix (2026-08-02)

External reviewer returned a detailed assessment. Confirmed: C5's ×94 is experimentally established;
near-zero-loss and BF16 correctly rejected; full-strength RL is demonstrably destructive. **Key correction
(accepted):** the norm A/B does NOT isolate C7 — it *jointly* exposes deterministic-action bias (C7),
router shift (C11), variance (C6), and local-reward mismatch (C10). Our "decisive confirmation that C7 is
binding" was too strong. Withdrawn.

### Isolated sims corroborate the "jointly exposes" caveat — no single factor reproduces divergence
Single-process load-balancing games (shared linear router, per_token_load_weighted reward, deployed
objective = global argmax max-load):
- **c7_fix_test**: deterministic-argmax surrogate CONVERGES (max-load 113→33), even beating sampled PG
  (→44). Surrogate *bias ≠ divergence*.
- **c10_test**: local per-rank (1/16) load also CONVERGES (→40), only mildly worse than global (→33).
- **gain_sweep**: the clean model is ROBUST across effective gains 1×–94× (only mild wobble at high
  base_lr×gain); never the GPU's ×4-load / +34%-CP explosion.
**Conclusion:** the GPU divergence is a *system-level interaction* (top-k=8 + PPO clip + EMA loads + reward
normalization + KL/aux co-train + 94 coupled layers), not any one isolated defect. This is exactly why the
reviewer's incremental order (one validated component at a time) is right, and why our single-factor
attribution was premature.

### C7 fix CORE verified (reviewer spec): hard Gumbel-top-k + ordered Plackett-Luce score function
`log π(e_1..e_k) = Σ_j [ z_{e_j}/τ − logsumexp_{e∉{e_1..e_{j-1}}}(z_e/τ) ]`. `pl_verify.py` (E=8,k=3,
exact enumeration of all 336 ordered tuples):
- (a) Σπ = 1.0000000000 — valid distribution.
- (b) Gumbel-top-k and sequential-sampling oracle both match closed-form π (TV≈0.065 = sampling noise) —
  same distribution, as the reviewer predicted.
- (c) ‖E_π[∇log π]‖ = 1.5e-16 (score integrates to 0); grad E[R] vs score-fn estimator cos=1.00000000,
  ‖diff‖=5.5e-16 → **exactly unbiased policy gradient**.
Ready to port into the router forward. Deploy/eval deterministic argmax; anneal τ; monitor sampled-vs-argmax
overlap. Do NOT use straight-through Gumbel-softmax (another biased estimator).

### Reviewer's ORDERED fix sequence (supersedes our normalize-first plan)
1. Policy-consistent sampling (C7, Gumbel-top-k + PL) — math verified.
2. Global EP-aggregated loads (C10) — before interpreting any further full-strength run.
3. Verify sampled-action log-prob analytically — DONE (pl_verify.py).
4. Per-layer normalization as a GAIN SWEEP (1×,4×,16×,32×,94×), scaled back — not a jump to 94×.
5. Router-shift trust region — for fresh single-epoch on-policy, start SIMPLER than RSPO: per-token
   old-vs-new categorical KL, top-k flip rate, max update ratio, early-stop on flip threshold.
6. Critic only if measured variance still requires it.

### Reward redesign (reviewer): dense counterfactual, global
Not local-dense-vs-global-sparse. EP-all-reduce counts; per-layer `J_l = LSE_β(n_l,1..n_l,E)` (or exact
max); detached counterfactual `r_{l,t} = J_l(n_l) − J_l(n_l − Δ_src + Δ_dst)`. Preserves token credit,
aligns with global CP. CP is additive across layers → layer-LOCAL baselines; GAE not auto-justified.

### Aux-loss-free bias + publication bar
Run aux-loss-free bias (DeepSeek-V3) as a PRIMARY systems baseline. Decisive 4-way at matched deterministic
CP: {aux-only, aux-free-bias, RL+aux, RL+aux-free-bias} → compare holdout. For the mid-band +1–1.75pp RL
edge to be publishable: ≥3–5 paired seeds, identical start ckpt+data order, dev-selected/official-test-once,
deterministic routing for every eval, CIs, continuous matched-CP regression (not coarse bins), ablations
(sampling/global-reward/norm/trust-region), and evidence RL changes specialization rather than adding
selection noise. Strongest framing: *auxiliary objectives are the better-conditioned CP mechanism; RL may
add holdout generalization at matched mid-band CP only when routing + estimator validity are handled
correctly* — a contribution inside the CP–accuracy Pareto story, not yet a standalone RL result.
