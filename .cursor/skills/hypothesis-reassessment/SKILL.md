---
name: hypothesis-reassessment
description: Re-examine an earlier conclusion when the user pushes back, when new data contradicts it, or when a bug fix invalidates the runs that produced it. Use when the user says things like "are you sure?", "that doesn't match what I saw", "but we observed X earlier", or when a code fix invalidates prior experimental results.
disable-model-invocation: true
---

# Hypothesis Reassessment

When an earlier conclusion is challenged — by the user, by new evidence, or by a bug fix that invalidates prior runs — work the following checklist before defending or restating the old conclusion. **Defending a stale conclusion is the failure mode this skill prevents.**

## Trigger phrases

User language that should activate this skill:
- "Are you sure?"
- "That contradicts what we saw before."
- "But [run X] showed [opposite]."
- "Why did [old result] then?"
- "Wasn't [Y] the case last time?"

A bug fix counts as a trigger too: any conclusion drawn from runs that **predate** a now-fixed bug needs reassessment.

## The three-question audit

Before responding, answer all three for yourself.

### 1. Was the old conclusion based on runs that are now invalidated?

If yes (e.g. the fix changed the gradient that was being studied), **all conclusions from pre-fix runs are unreliable** until reproduced post-fix. State this explicitly.

Example: "The earlier observation that 'KL helps Critical Path' came from runs where KL was computed against itself (same-tensor bug). Those runs measured **compute overhead**, not the regularization effect, so the conclusion they support is invalidated."

### 2. Was the old conclusion confounded by something the new data isolates?

A common confounder pattern: an intervention that adds compute (e.g. reference-model forward pass) **slows iter time**, so the run completes fewer iterations within wallclock — and "fewer iterations" produces different downstream metrics than "more iterations". The intervention looks like it's regularizing when really it's just throttling training.

Always check: did the intervention change **wallclock-normalized iters/min**? If yes, prior conclusions about "what the intervention does" are confounded with "what running for fewer iterations does".

### 3. Is there a third explanation that fits both old and new observations?

Don't just pick the new observation over the old. Look for a hypothesis consistent with both:

| Old observation | New observation | Reconciling hypothesis |
|---|---|---|
| KL on/off matters | KL coefficient doesn't matter | KL forward pass adds compute → fewer iters → different outcome (compute confound) |
| Regularizer helped | Removing regularizer didn't hurt | Effect was driven by an unrelated bug fix in the same diff |
| Run A beat run B | Re-run shows tie | Original gap was within seed noise; run more seeds |

The reconciling hypothesis is usually more useful than either of the two endpoint hypotheses.

## Output format

When responding to a challenge:

1. **Acknowledge the challenge directly** — don't restate the old conclusion as if no challenge happened.
2. **Run the three-question audit out loud** (briefly).
3. **State the updated conclusion** with calibrated confidence:
   - "Confirmed by post-fix data" → high confidence
   - "Best current hypothesis but not directly tested" → medium, flag what would test it
   - "Could go either way" → low, propose the experiment
4. **Flag what's invalidated** — explicitly list any prior writeups, commit messages, or doc edits that should be updated to match the new conclusion.

## Anti-patterns

- **Reflexively defending the old conclusion** because it's already in the chat. The chat is not evidence; runs are.
- **Reflexively flipping** to the user's new view without an audit. Sometimes the old conclusion is right and the user's observation is the misreading.
- **Updating verbally but not on disk.** If a CLAUDE.md / README / writeup contains the stale conclusion, **edit it** as part of the reassessment.
- **Reporting the new view without flagging that it invalidates prior decisions.** If 3 sweeps were launched based on the old hypothesis, those sweeps may need to be cancelled or their interpretation revised — say so.

## Quick decision flow

```
challenge received
   ├── audit Q1: are old runs invalidated by a bug fix?
   │     └── yes → discard pre-fix conclusions, replan from scratch
   ├── audit Q2: is there a confound (compute, seeds, wallclock)?
   │     └── yes → reconciling hypothesis, design experiment to disambiguate
   └── audit Q3: third explanation that fits both?
         └── usually yes; this is your best new hypothesis
update on-disk docs that referenced the stale conclusion
state updated view with calibrated confidence
```
