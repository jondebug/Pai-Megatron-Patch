
---

## §24 HSG GB200 NVL72 measurements — initial pretrained results (2026-07-02)

### Migration to GB200 NVL72

Project moved to `oci-hsg-cs-001` cluster: NVIDIA GB200 NVL72 aarch64 (Grace CPU + Blackwell GPU).
Key hardware:
- **4 GPUs/node**, 18 nodes per NVL72 rack = 72 GPUs/rack all NVLink5-connected (~1.8 TB/s aggregate per chip)
- **189 GB HBM per GPU** (vs H100's 80 GB, 2.4×)
- 942 GB RAM per node, 144 CPU cores, aarch64

Container: `vllm-openai-v0.20.0-cu130-ubuntu2404-nsys-v2.sqsh` (from `coreai_comparch_infbench/roverman`),
built for aarch64, includes nsys 2026.3.0.

### GEMM saturation knee shifted RIGHT on B200

Direct microbench at Qwen3-235B FFN shape (K=4096, N=1536, bf16):

| M | H100 TFLOPS/s | B200 TFLOPS/s | speedup |
|---|---|---|---|
| 128 | 39 | 47 | 1.2× |
| 512 | 111 | 181 | 1.6× |
| 1024 | 166 | 337 | 2.0× |
| 2048 | 205 | 563 | 2.7× |
| 4096 | 226 | 855 | 3.8× |
| 8192 | 240 | 1094 | 4.6× |

- **B200 asymptote: 1094 TFLOPS/s (4.6× H100's 240)**
- **B200 knee (90% asymptote): M = 6144** (H100 was M = 4096, **shifted right**)

I initially predicted the knee would move LEFT on B200 (faster FLOPs → saturate sooner). Wrong.
Compute scaled ~4.6× but HBM didn't scale proportionally, so the arithmetic-intensity threshold
to hit peak is higher on B200. **Below-knee regime is bigger on B200.** At M=512, B200 is at
16% of peak; H100 was at 46%.

### Pretrained EP=8 prefill on GB200

`hsg_phaseA_pre_A_3808341`: 2 nodes × 4 GPU (EP=8), plen=8192, max_tok=4, n=30 trials.

| bs | pre TTFT (ms) | ORD H100 TTFT | GB200 speedup |
|---|---|---|---|
| 1  | **31.46 ± 0.74** | 53.98 ± 0.27 | **1.72×** |
| 2  | 49.08 | 89.28 | 1.82× |
| 4  | 69.68 | 116.91 | 1.68× |
| 8  | 92.11 | 155.44 | 1.69× |
| 16 | 136.82 | 196.46 | 1.44× |

TTFT tightly bounded (bs=1 std=0.74 ms). **1.4-1.8× speedup from GB200 vs H100.** Prefill scales
sub-linearly with bs (bs=16 e2e only 2.5× bs=1 e2e — GB200 is capacity-rich).

### Pretrained EP=8 decode on GB200

`hsg_dec_pre_3808375`: same config, plen=256, max_tok=32.

| bs | TTFT (ms) | TPOT (ms/tok) | ORD H100 TPOT | GB200 TPOT speedup |
|---|---|---|---|---|
| 1    | 15.98 | 7.74 | 111.09 | **14.4×** |
| 8    | 33.90 | 10.66 | 112.61 | 10.6× |
| 64   | 133.41 | 15.10 | 112.39 | 7.4× |
| 256  | 180.26 | 22.00 | – | – |
| 512  | 318.55 | 32.72 | 226.59 | 6.9× |
| 1024 | 536.46 | 127.73 | 440.83 | 3.4× (memory pressure) |
| 1024 decode_tps | | | | 12293 (bs=512) / 7288 (bs=1024) |

**Massive decode TPOT speedup**: bs=1 is **14×** faster on GB200 than H100. Drops at bs=1024
(memory pressure / KV eviction). Peak throughput at bs=512: 12293 tok/sec.

### Comparison: r15 substitute is unavailable

The original ORD Phase A ran `r15 = 235bv5a_rlc1.0_ppo_aux0.01_r15/.../hf_converted_iter3000_cp/`.
**That specific HF-converted checkpoint has been deleted from ORD storage.** Only Megatron distcp
remains at the training checkpoint dir. Copying distcp doesn't help — vLLM can't load Megatron
format directly.

Pivoting to a substitute fine-tuned cell that still has HF safetensors on ORD:
`235bv5b_rlc1.0_aux0.015_kl0.001_r63/.../hf_converted_iter1000_cp/` — 118 safetensors, ~435 GB,
RL-trained checkpoint. Different training config (v5b vs v5a, r63 vs r15, iter1000 vs iter3000)
so **not directly comparable to ORD r15 results**, but valid as "some fine-tuned Qwen3-235B" to
compare against pretrained.

Rsync in progress with 8-way parallel stripe.

### Preliminary conclusion (pretrained-only)

GB200 offers substantial per-request latency improvements over H100:
- 1.4-1.8× TTFT speedup at moderate prefill bs
- 3-14× TPOT speedup at decode

At the LOW bs interactive-serving regime (bs=1 prefill, bs=1-256 decode), GB200 makes single-user
serving dramatically more responsive.

At HIGH bs batch regime (bs≥512 decode), GB200 still 3-7× faster but the ratio narrows as we
approach memory-bandwidth limits.

The r15-vs-pretrained comparison to test the CP-RL question on GB200 requires the r63 rsync to
complete (~40+ min at current rate). Once complete, we can run the pretrained/r63 A/B and see
if:
1. GB200's shifted knee (M=6144 vs H100's 4096) makes the below-knee r15-hurts-more effect worse
2. GB200's faster NVLink5 comm shrinks the EP=64 decode +15% TPOT regression from ORD
3. The +6.84% bs=1 prefill mystery ("outside kernel layer" per §13) survives the hardware change

### Files

- Config: `hsg_phaseA_pre_A_3808341.json`, `hsg_dec_pre_3808375.json`, `hsg_gemm_knee.json`, `hsg_smoke_3807832.json`
- Nsys traces: submitted (jobs 3809157, 3809158) for pretrained bs=1 prefill and bs=512 decode
- Launchers: `hsg_run_phase_A_pre.sh`, `hsg_decode_pre.sh`, `hsg_run_nsys.sh`, `hsg_gemm_knee.py`, `hsg_smoke_test.sh`

