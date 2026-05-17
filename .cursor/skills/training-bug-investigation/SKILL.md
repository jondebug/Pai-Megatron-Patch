---
name: training-bug-investigation
description: Diagnose suspected silent training bugs (loss term not affecting outcome, regularizer that does nothing, hook not firing, gradient not flowing). Use when a loss component appears active in the log but changing its coefficient produces no measurable effect, or when the user suspects a regularizer is broken.
disable-model-invocation: true
---

# Training-Bug Investigation Pattern

A "silent" training bug is when a loss term **logs nonzero values** but **changing its coefficient doesn't change downstream metrics**. The pattern below is what reliably flushes these out. It applies to KL constraints, auxiliary losses, regularizers, custom hooks, and frozen-weight setups.

## The five-layer checklist

Investigate in order. Don't skip layers — each one rules out a class of bugs.

### Layer 1: Is the value being computed at all?

- Add a print with the **value, dtype, requires_grad, and `id()`** of the loss tensor at the call site.
- Confirm it's nonzero and roughly the right magnitude.
- Confirm `requires_grad=True` (a `.detach()` somewhere upstream is the most common silent bug).

### Layer 2: Is it actually in the backward graph?

- After `.backward()`, check that the parameters you expect to be updated have a nonzero `.grad`:
  ```python
  for n, p in model.named_parameters():
      if p.grad is not None and p.grad.abs().sum() > 0 and "router" in n:
          print(n, p.grad.abs().mean().item())
  ```
- If `param.grad` is `None` for the supposedly-affected params, the loss is not connected to them. Two common causes:
  - The forward path used to produce the loss tensor went through a `.no_grad()` block or a `.detach()`.
  - The hook captured an output **after** detach, e.g. captured `output.detach()` instead of `output`.

### Layer 3: Is what you captured what you think you captured?

For hook-based capture (e.g. `output_layer` forward hook): the hook fires on **every forward**, including the reference-model forward. If you store into a single shared dict slot, the second forward overwrites the first.

**Symptom:** the loss is suspiciously small and constant (e.g. `kl_loss ≈ 1e-7` regardless of coefficient) because you're computing `KL(x, x)`.

**Diagnostic prints** to add (cheap, run for 5 iters then remove):
```python
print(f"cur_logits id={id(cur)} mean={cur.float().mean().item():.6f}")
print(f"ref_logits id={id(ref)} mean={ref.float().mean().item():.6f}")
print(f"same_tensor={cur.data_ptr() == ref.data_ptr()}")
```

If `same_tensor=True` or the means match to many digits, you have an aliasing bug. Fix with a `capture_disabled` flag in the hook that the reference-forward call sets before calling the model and clears in a `finally` block.

### Layer 4: Does varying the coefficient actually change a measurable downstream value?

This is the **only** real test for "is this loss term doing anything?"

Run a **3-point sweep**: coefficient = 0, default, default × 100. Each run for ≥ 50 iters at small scale (30B is fine).

A **healthy** loss term will, by `coeff × 100`:
- Visibly suppress the value of its own loss (KL with `coeff=100` should drive `kl_loss` close to zero).
- Move at least one secondary metric — e.g. `output_entropy`, gradient norms, or a behavioral metric.

If `coeff = 0` and `coeff = 100` produce **identical** secondary metrics within noise, **the term is mechanistically inert** even if its log values look reasonable. Suspect Layer 2 or Layer 3.

### Layer 5: Does the term affect the actual objective you care about?

Even after Layers 1–4 confirm the gradient is real, the term might still **not affect the outcome metric you care about**. Example: KL on LM logits is mathematically real but, on a router-only training problem, the gradient hits the LM head + a bit of the last block and basically nothing of the routers — so even a huge KL coefficient barely moves Critical Path. This is **not a bug** — it's an inert-by-design constraint.

Always separate:
- "The gradient flows" (mechanism, Layer 4)
- "The gradient affects the metric I care about" (outcome, Layer 5)

In your report, **state both findings explicitly** so the user doesn't conflate them.

## Workflow checklist

```
- [ ] L1: Print value, requires_grad at call site
- [ ] L2: Confirm nonzero param.grad on the params you expect
- [ ] L3: Print id() + data_ptr() of any hook-captured tensors
- [ ] L4: 3-point coefficient sweep (0, default, 100×default) at small scale
- [ ] L5: Verify outcome metric responds, not just the loss-term value
- [ ] Document mechanism-vs-outcome distinction in the writeup
```

## Common silent-bug patterns

| Symptom | Likely bug | Fix |
|---|---|---|
| Loss logs reasonable values but coeff doesn't matter | Hook overwriting current with reference (Layer 3) | Add `capture_disabled` flag during reference forward |
| Loss is exactly zero on all iters | Hook not registered on the right module | Walk model hierarchy: `model.module.module.<target>` for DDP+Float16Module wrapping |
| Loss value matches across coefficients | `.detach()` somewhere in the loss expression | `grep -n detach` in the loss file |
| Coefficient matters at small scale but not at large scale | Compute overhead consumes iterations budget | Measure iter time at each coeff; report wallclock-normalized comparison |
| LM loss explodes when adding regularizer | Coefficient mis-scaled relative to LM loss magnitude | Print both losses at iter 1; aim for regularizer ≈ 0.01–0.1× LM loss |

## Reporting template

When summarizing the investigation:

```markdown
## Findings

**Mechanism**: <does the gradient flow? L1–L4>
- Evidence: kl_loss at coeff=0 is X, at coeff=100 is Y; output_entropy went Z→W
- Verdict: gradient is/isn't active

**Outcome**: <does it move the metric the user cares about? L5>
- Evidence: critical_path at coeff=0 is X, at coeff=100 is Y (Δ = small/large)
- Verdict: term does/doesn't matter for the actual objective

**Recommendation**: <keep / remove / replace with different formulation>
```

The mechanism-vs-outcome split is the most important thing in this writeup. **Don't merge them into one verdict.**
