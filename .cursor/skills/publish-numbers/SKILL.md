---
name: publish-numbers
description: Audit and label every number before it appears in a stakeholder doc, slide, table, or chat-visible summary. Use whenever about to put concrete numbers (CP, accuracy, speedup, latency, percentages) in front of the user or a stakeholder, when constructing comparison tables, when updating docs/*.md, or when the user asks "are these numbers real?". Catches the four recurring failure modes: training-eval CP cited as inference CP, limit=1000 accuracy compared against full-dataset baseline, "best cell" cited as average, and projected/simulated numbers cited as measurements.
disable-model-invocation: true
---

# Publishing Numbers — Audit and Label Before Citing

The recurring failure mode this skill prevents: pulling numbers from
the master CSV / a JSON artifact / a log into a stakeholder-facing
table without (a) re-deriving them from the source, (b) labeling their
provenance, and (c) checking that the comparison slice is consistent
across rows. Three of the worst incidents on this project came from
skipping one of those steps; the user has explicitly asked "are these
numbers fictional???" after seeing such a table.

## When to read this

- About to put a numeric claim in `docs/*.md`, an email, a slide, or
  any chat-visible summary (table, headline bullet, paragraph).
- About to compare a trained checkpoint to the pretrained baseline.
- About to cite a speedup, CP reduction, or accuracy delta to a
  stakeholder.
- The user is challenging a number you previously cited.

## The four provenance categories

Every number you cite belongs to exactly one of these. The category
**must be labeled** when the numbers are mixed in the same table.

| Category | Where it comes from | Typical column / metric | Transfers to inference? |
|---|---|---|---|
| **Measured (inference)** | `vllm_ngc_*.json` or matching `.out`; an actual lm-eval run | `e2e_ms_mean`, `ttft_ms`, `decode_tps`, `hellaswag`/`arc`/`winogrande` accuracy | Yes — this *is* inference. |
| **Measured (training-eval)** | `benchmark_results.csv` `eval_crit_path`, `eval_lm_loss`; per-step Megatron eval | Mostly CP. | **Only for non-CPB runs.** RL+CPB regressed by +26–32% in cp-microbench on stock HF. |
| **Projected (simulation)** | `microbench_*.json` `ep_speedup` field | EP=4/8/32/64/128 step-time speedup | Directional only. Ignores all-to-all comms, scheduler, vLLM engine. |
| **Derived (computed from above)** | Δ-acc, Δ-CP %, "avg over cells", "best cell" | Anything you computed in a one-liner. | Inherits the provenance of its inputs; bs=1 ≈ 1.00× must be disclosed. |

## Pre-publish checklist

Run this on every table before it goes in front of the user:

```
- [ ] Every cell labeled (or grouped under a labeled header):
      measured / training-eval / projected / derived
- [ ] CP column: if a row uses --moe-router-critical-path-bias,
      label it "training-eval, NON-TRANSFERABLE" (see cp-microbench skill)
- [ ] Accuracy comparisons: same slice across all rows?
      → both limit=1000, OR both full dataset. Never mix.
- [ ] Speedup statements: which cells are averaged?
      → spell out "avg over bs ≥ 8" or "best cell at plen=X bs=Y"
- [ ] Source artifact path cited in a footnote or caption
- [ ] EP layout disclosed when speedup is cited (EP=8 single-node ≠ EP=128 production)
- [ ] bs=1 carve-out: if you're averaging speedups, state that bs=1 is ≈ 1.00×
      (single-request decode is bottlenecked on something other than expert imbalance)
```

## Specific incidents this skill exists to prevent

### Incident 1: limit=1000 vs full baseline

`benchmark_results.csv` contains both:

```
pretrained_baseline       limit=1000  acc=64.10
pretrained_baseline_full  limit=""    acc=67.83
```

A trained run benchmarked at `limit=""` (acc=68.04) compared against
the **limit=1000** baseline produces a spurious +3.94 pp delta. The
real delta vs the matched-slice baseline is **+0.21 pp**. **Always
match the slice.**

Practical check before drawing a Δ-acc:

```python
import csv
rows = list(csv.DictReader(open('examples/qwen3/benchmarks/benchmark_results.csv')))
trained = next(r for r in rows if r['run_name'] == TARGET_RUN and r['bench_iteration'] == TARGET_ITER)
trained_limit = trained['limit']      # '' or '1000'
baseline = next(r for r in rows if r['category'] == 'pretrained' and r['limit'] == trained_limit)
delta = float(trained['benchmark_avg']) - float(baseline['benchmark_avg'])
```

### Incident 2: "best cell" cited as average

The vLLM JSON has per-cell speedups across 9 cells (plen × bs). The
"best cell" (plen=1024 bs=32) was **1.161×** for B1; the average over
all 9 cells was **1.064×**; the average over bs ≥ 8 cells was
**1.094×**. All three are correct values. A table column labeled just
"speedup" without disclosing the slice will be read as "average over
all cells" and is misleading either direction.

Compute the canonical four numbers and cite the right one:

```python
import json, statistics as S
d = json.load(open(VLLM_JSON))
base, target = d['models'][0], d['models'][1]   # baseline + first trained
speedups, speedups_bs8, speedups_bs1 = [], [], []
for cell in base['cells']:
    bm = base['cells'][cell]['end_to_end_ms_mean']
    tm = target['cells'][cell]['end_to_end_ms_mean']
    bs = base['cells'][cell]['batch_size']
    sp = bm / tm
    speedups.append(sp)
    (speedups_bs8 if bs >= 8 else speedups_bs1).append(sp)
print(f"best cell: {max(speedups):.3f}×")
print(f"avg all 9 cells: {S.mean(speedups):.3f}×")
print(f"avg bs≥8 (6 cells): {S.mean(speedups_bs8):.3f}×")
print(f"avg bs=1  (3 cells): {S.mean(speedups_bs1):.3f}×   (expect ≈ 1.00×)")
```

### Incident 3: training-eval CP cited as inference CP

`benchmark_results.csv` `eval_crit_path` is the **training-time eval
CP** logged by the Megatron training script. It is *not* the CP
observed by stock HF or vLLM at inference. For non-CPB runs the two
agree closely (verified by `cp-microbench`); for RL+CPB runs they
diverge by +26–32 % because `critical_path_bias` is
`register_buffer(persistent=False)` and is not in the saved
checkpoint.

When publishing a CP number in an inference context, either:

- Restrict to **non-CPB rows only** and label the column "Training-eval
  CP (transfers for these non-CPB runs, verified)", OR
- Run `cp-microbench` on each CPB-using checkpoint and publish the
  microbench-measured CP instead, labeling it "Stock-HF CP".

### Incident 4: projections cited as measurements

`cp-microbench` simulates per-step latency from real router traces +
real FFN kernel timings. It is **not wall-clock** — it omits
all-to-all dispatch/combine, queueing, scheduler overhead, and engine
effects. Microbench EP=128 speedup may be 2× the EP=8 number; that 2×
amplification is a projection of compute, not a measurement of
deployment throughput.

Doctrine for stakeholder docs (per user, May 17 2026): **drop
projected numbers, keep training + test numbers.** If a projection
needs to appear at all, put it under a separate header
("Projections / model") with explicit "not measured" wording, and
never in the headline summary.

## Recommended writeup pattern

When constructing a headline comparison table:

```markdown
| Model | Eval CP† | Δ CP | Acc (full) | Δ acc | Best vLLM cell | Avg vLLM speedup (bs ≥ 8) |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained Qwen3-30B-A3B | 4780 | — | 67.83% | — | 1.000× | 1.000× |
| RL+aux (no CPB), iter 5000 | 3942 | −17.5% | 68.04% | +0.21 pp | 1.161× (plen=1024, bs=32) | 1.094× (+9.4%) |

> †Training-time eval CP; transfers to inference for these non-CPB runs
> (verified by cp-microbench). Would NOT transfer for RL+CPB runs.
> vLLM config: NGC vllm/vllm-openai:latest, TP=8 + enable_expert_parallel=True
> (EP=8 active), max_tokens=256, 10 trials. Speedup at bs=1 is ≈1.00× across
> all prompt lengths (single-request decode-bound). Source:
> `cp_latency_results/vllm_ngc_<TIMESTAMP>.json`.
```

Note what's explicit:
- Run identification (model name + iter)
- CP provenance footnote with transferability statement
- vLLM config (container, EP layout, max_tokens, trials)
- bs=1 carve-out
- Source artifact path

## When the user pushes back ("are these fictional?")

This is the cheapest moment to redo the audit. Follow this sequence:

1. **Re-derive every number from the raw artifact**, not from the
   table. Print the JSON/CSV values.
2. **Identify which category each cell belongs to** (measured /
   training-eval / projected / derived). Cite the source file path.
3. **Acknowledge each unlabeled or mis-labeled cell explicitly.** Do
   not defend the existing table — rewrite it with provenance labels.
4. **Update the on-disk doc**, not just the chat reply. See
   `hypothesis-reassessment` skill for the same principle applied to
   conclusions.

A challenge of "are these real?" almost always means: the numbers are
real, but the labeling is missing. The fix is rarely "produce new
numbers"; it's "annotate the existing ones honestly."
