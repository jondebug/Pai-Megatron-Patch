
### §17 correction note (2026-06-21)

§17 reported "any-idle local expert" rates aggregated across all 64 ranks × 94 layers. A natural
follow-up question (asked by user): the per-rank wall time is bounded by the SLOWEST rank, so
what matters isn't the average across all ranks — it's whether the **slowest rank specifically**
has an idle expert.

Re-analyzing the same dumps to find per-layer's slowest rank (by GEMM-curve-predicted FFN time)
and looking at THAT rank's local pair:

```
At bs=1 prefill, EP=64, layer-averaged stats for the SLOWEST rank per layer:

                    pretrained        r15
slowest n_active    1.99 / 2          2.00 / 2     ← BOTH essentially always have both active
slowest total M     3544 ± 698        2501 ± 558
slowest max-M       2763 ± 481        1747 ± 535
slowest min-M       781 ± 703         753 ± 312
predicted T_rank    263 µs ± 40       192 µs ± 33  ← r15 -27% FASTER predicted at slowest rank
```

**At prefill, the slowest rank in BOTH cells has both local experts active.** §17's "idle expert"
mechanism applies to the average/light ranks but NOT to the slowest rank. And the measured GEMM
curve predicts r15's slowest rank should be 27% FASTER than pretrained's, not 11% slower.

This is a contradiction with the §16 trace measurement (r15 +11% per-rank expert_ffn at EP=64
DECODE). Resolution: the trace is decode-regime (4096 token-expert assignments per step) while
the routing dump is prefill (65536 per forward). At decode, the slowest rank's pair likely DOES
look different from prefill — possibly pretrained's slowest decode rank has one fat expert and
one idle, while r15's has two small both-active. This needs decode-shape routing data to confirm.

**Decode routing dump submitted (job 29338631)** — will resolve the contradiction.

