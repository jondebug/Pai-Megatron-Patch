# CP-RL Inference Campaign on GB200 — Executive Summary

2026-07-08. Question: does critical-path-reducing router RL (CP-RL) on Qwen3-235B-A22B
(128 experts, topK=8, 94 MoE layers) speed up vLLM inference on HSG GB200 NVL72?
Companion docs: `HSG_GB200_FINDINGS.md` (§A-I, full evidence), `METHODOLOGY_LESSONS.md`,
`GB200_VLLM_OPERATIONS.md`, `SPORK_METHODS.md`.

## The answer, in three layers

1. **e2e latency: no demonstrated win in any tested configuration.** Bounded ≲2-3% after
   artifact control, across EP ∈ {8,16,32,64}, prefill + decode, batch 1-512, in both
   TP=EP+AllReduce and DP+EP dispatch/combine (naive backend) serving, swap and full
   weights (32 router-swap cells + 6 full checkpoints, ~120 A/B jobs).

2. **Mechanism: real and large, in the right regime.** Above the GEMM knee at EP=64
   (prefill bs=16/32, ~8k tokens/expert, 2 experts/GPU), the most aggressive CP cell
   (r05, ~57% train-scale CP cut) reduces mean per-rank MoE-FFN kernel time by **33%**
   and busiest-GPU FFN by **50%**, uniformly across 15 nodes. Below the knee the effect
   inverts (flattened routing lifts light GPUs to the busiest GPU's fixed-cost floor).

3. **Why the mechanism never reaches e2e (yet):**
   - TP=EP+AR serving is insensitive to routing balance by construction (fixed AR cost;
     4-16 experts bucketed per rank average the imbalance away), and the step is
     host-idle + comms dominated (prefill wall: ~2% compute / 40-45% exposed comms /
     ~55% host; decode: 3-9% / 15-32% / 60-81%).
   - The load-proportional configs where balance CAN matter (dispatch/combine) are
     blocked at optimized-kernel level on GB200: upstream DeepEP is MNNVL-incompatible
     (known), FlashInfer NVLink backends require nvfp4 / crash on bf16; only the naive
     torch a2a works, and it is host/launch-bound.

## The decisive strategic fact (from the Nemotron Ultra comms study)

a2a traffic ratio AG+RS/dispatch-combine = **EP / topK**. Qwen3's topK=8 puts the
crossover at EP=8 — token-routing dispatch/combine is the winning backend for this model
at essentially every production EP. That is exactly the config class where CP balance
has a mechanism to act. **The CP-RL question gets its fair fight only on a working
MNNVL dispatch backend (hybrid_ep)** — see GB200_VLLM_OPERATIONS.md for the container.

## Open follow-ups (priority order)

1. Obtain hybrid_ep container (deci_handoff, owner I. Rosenfeld Rauch) → rerun the
   3-cell ladder at EP=32/64 decode+prefill with real dispatch/combine kernels.
2. EPLB comparison: expert-placement rebalancing may capture the same balance win with
   zero retraining — the go/no-go question for router-RL as a technique.
3. nvfp4-quantized serving (production GB200 mode; unlocks flashinfer_nvlink_one_sided).
4. Global cross-rank late-arrival measurement (end-time alignment across all EP ranks)
   to quantify straggler-wait reduction directly.

## Journey (claims made and retracted — kept for honesty)

| date | claim | fate |
|---|---|---|
| 07-04 | "frontier cells win 3-9% e2e" | RETRACTED — selection-from-noise (duplicate disagreement + cross-EP sign flips) |
| 07-06 | "CP cuts don't reduce FFN time" | REVISED — true sub-knee; inverts above knee (r05 −33/−50%) |
| 07-07 | "CUDA graphs force NCCL to RING_LL" | RETRACTED — blockX=640 proves NVLS+SIMPLE; kernel names lie |
| 07-08 | "naive-a2a decode −2.6-3.3% win" | RETRACTED — A/A null shows −2.8% order bias |

Every retraction came from a control experiment (duplicates, ladders, A/A nulls,
launch-signature checks). The methodology is the durable asset — see
METHODOLOGY_LESSONS.md.
