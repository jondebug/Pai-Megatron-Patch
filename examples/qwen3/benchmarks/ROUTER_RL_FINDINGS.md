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
