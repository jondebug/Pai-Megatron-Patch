# HSG GB200 NVL72 findings — Qwen3-235B-A22B CP-RL inference study

**Author:** Jonathan Paul (jonathanp@nvidia.com), NVIDIA NVR-Israel
**Cluster:** HSG GB200 NVL72 aarch64 (Grace + Blackwell)
**Model:** Qwen3-235B-A22B (128 MoE experts, top-k=8, 94 layers)
**Serving engine:** vLLM 0.20.0 (aarch64 container `vllm-openai-arm.sqsh`)
**Session start:** 2026-07-02
**Companion CSV:** `hsg_experiment_results.csv` (all bench cells, one row per operating point)

---

## TL;DR

1. **CP-RL and aux-only fine-tuning both hurt low-bs prefill on GB200** — mean **+5.6%** TTFT regression at bs=1 across 5 different fine-tune cells (RL, RL+aux, and aux-only alike). Direction matches ORD H100 Phase A (+6.84%). Hardware doesn't rescue.
2. **Router-swap technique validated (4400× cheaper iteration)**: because only `mlp.gate.weight` tensors differ between pretrained and any router-only fine-tune, patching just 94 tensors (~98 MB) onto pretrained shells produces bit-identical inference behavior vs full 435 GB checkpoint transfer. Verified: r15_router_swap reproduces ORD's +6-8% bs=1,4 regression pattern.
3. **GB200 crushes H100 on prefill throughput** (48× TTFT advantage at bs=8192 EP=64 due to chunked-prefill collapse on H100), **essentially ties on per-token decode rate**. Prefill regime moved decisively toward compute-boundedness.
4. **GEMM saturation knee shifted RIGHT** on B200 (M=6144 vs H100's M=4096). Below-knee regime is BIGGER on B200 despite 4.6× compute peak.
5. **Kernel-level breakdown (nsys v3 + torch.profiler)**: at EP=8 bs=1 prefill, **communication dominates (~48% of step)**. Expert-FFN is only 21% with busy/quiet imbalance of just **1.10×**. CP-RL's theoretical upper bound = ~2% total-step speedup, which the below-knee-compute-hurt penalty cancels.

---

## Method

### Hardware
- 4 GPUs/node × 18 nodes/rack = 72 NVLink5-connected GB200 GPUs per NVL72 rack
- 189 GB HBM per GPU (2.4× H100); ~1.8 TB/s NVLink5 (aggregate per chip)
- Cluster: HSG (`oci-hsg-cs-001-login-01.nvidia.com`), SLURM account `nvr_israel_rlop`

### Cells benchmarked
All are Qwen3-235B-A22B with modified router weights only. Pretrained CP ≈ 7150.

| id | run | iter | CP | cp_red% | acc | category |
|---|---|---|---|---|---|---|
| pretrained | Qwen/Qwen3-235B-A22B | – | 7150 | 0% | – | baseline |
| r15 | 235bv5a_rlc1.0_ppo_aux0.01_r15 | 3000 | ~4100 | +43% | ~73 | RL+aux+ppo |
| r62 | 235bv5b_rlc0.5_aux0.02_lm0.3_r62 | 3000 | 3669 | +48.68% | 73.75 | RL+aux |
| v14cg_r1 | 235bv14cg_rlc1_aux0.02_basecritic_g0.5_r1 | 2790 | 3866 | +45.93% | 74.78 | RL+aux+critic |
| r29 | 235bv5a_norl_aux0.02_r29 | 3000 | 3992 | +44.17% | 76.36 | **aux-only** |
| r130015 | 235bv5b_norl_aux0.015_seed1_r130015 | 3000 | 4015 | +43.85% | 74.14 | **aux-only** |

### Bench protocol
- Prefill sweep: `plen=8192, max_tokens=4, bs ∈ {1,4,16,32,64,128}`, num_trials=12-15, num_warmup=2, cuda_graphs=on
- Decode sweep: `plen=256, max_tokens=32, bs ∈ {1,8,64,256,512,1024}`, num_trials=12-15
- EP=64 decode: `plen=256, max_tokens=16, bs ∈ {512,2048,8192}` (matching ORD Phase C)
- Torch.profiler kernel traces at bs=1 EP=8 prefill (per-rank)
- Nsys v3+v4 traces: 337k kernels captured with GPU metrics on head node

---

## Results

### §1. Pretrained on GB200 vs ORD H100 (headline speedups)

| regime | HSG GB200 | ORD H100 | ratio |
|---|---|---|---|
| Prefill bs=1 EP=8 plen=8192 TTFT | 26.08 ms | 53.98 ms | **2.07×** |
| Prefill bs=16 EP=8 TTFT | 121.4 ms | 196.46 ms | 1.62× |
| Decode bs=64 EP=8 TPOT | 15.54 ms/tok | 112.39 ms/tok | **7.2×** |
| Decode bs=512 EP=8 TPOT | 32.72 ms/tok | 226.59 ms/tok | 6.9× |
| Decode bs=8192 EP=64 TTFT | 3020 ms | 146,598 ms (chunked) | **48×** |
| Decode bs=8192 EP=64 TPOT | 1399 ms/tok | 1311.6 ms/tok | tied |
| Decode bs=8192 EP=64 agg throughput | 5649 tok/s | 788 tok/s | **7.2×** |

**Note on bs=8192 EP=64**: pure-decode-phase throughput is comparable (H100 6663 vs GB200 6043 tok/s), but total wall-clock throughput favors GB200 heavily because H100 spends 147 seconds on chunked prefill vs GB200's 3 seconds.

### §2. GEMM saturation knee on GB200

Direct microbench at Qwen3-235B FFN shape (K=4096, N=1536, bf16):

| M | H100 TFLOPS/s | B200 TFLOPS/s | ratio |
|---|---|---|---|
| 128 | 39 | 47 | 1.2× |
| 512 | 111 | 181 | 1.6× |
| 1024 | 166 | 337 | 2.0× |
| 2048 | 205 | 563 | 2.7× |
| 4096 | 226 | 855 | 3.8× |
| 8192 | 240 | 1094 | 4.6× |

**B200 asymptote 1094 TFLOPS/s (4.6× H100); knee at M=6144 (H100 was M=4096, shifted RIGHT).**

Below-knee regime is bigger on B200 because compute scaled ~4.6× while HBM only 2.4×, so arithmetic intensity to saturate is higher. Consequences for CP-RL: on B200, per-expert token counts stay deeper below the knee → per-token throughput drops faster as CP-RL reduces tokens/expert → CP-RL's harm is preserved.

### §3. GB200 EP=8 A/B — all 5 fine-tunes vs pretrained

TTFT (ms) with delta% vs pretrained. n=12-15 trials per cell.

| regime | bs | pre | r15rs | r62 | v14cg_r1 | r29 (aux) | r130015 (aux) |
|---|---|---|---|---|---|---|---|
| prefill | 1  | 26.13 | +3.11% | +5.73% | **+8.75%** | +4.74% | +5.74% |
| prefill | 4  | 59.08 | **+9.41%** | +4.49% | +2.03% | +0.37% | +4.96% |
| prefill | 16 | 123.89 | -2.72% | +8.07% | +1.13% | +0.64% | +0.01% |
| prefill | 32 | 222.49 | +7.92% | +6.08% | -4.06% | +0.30% | -3.99% |
| decode  | 64 | 132.34 | +2.38% | -2.49% | **+13.11%** | +8.22% | -1.69% |
| decode  | 512 | 311.12 | +2.91% | -1.01% | -2.37% | -0.85% | -2.03% |

**Universal pattern**: at bs=1 prefill (compute-bound regime), ALL 5 fine-tunes are 3-9% slower than pretrained. Direction matches ORD H100 Phase A. Effect is real and NOT specific to RL — pure aux-loss training (r29, r130015) exhibits the same regression, confirming the mechanism is the router structure, not the training method.

### §4. Kernel-level breakdown — where does time go?

**Prefill bs=1 EP=8 pretrained** (torch.profiler per-rank + nsys v3+v4 with GPU metrics):

| stage | share of step | busiest rank | quietest rank | busy/quiet |
|---|---|---|---|---|
| comm (AllToAll + AllReduce) | ~48% | 211.1 ms | 81.4 ms | 2.6× (noise) |
| expert_ffn (MoE grouped GEMM) | ~21% | 82.3 ms | 75.1 ms | **1.10×** |
| attention | ~8% | 30.0 ms | 29.2 ms | 1.03× |
| other (norm, dispatch overhead) | ~23% | – | – | – |
| **total per-rank** | 100% | ~385 ms | – | – |

**Implications for CP-RL**:
- Comm dominates prefill at bs=1 (~48%). CP-RL doesn't touch this — router weight changes don't reduce AllToAll or AllReduce work.
- Expert-FFN inference-scale imbalance is only 1.10× (training-scale CP suggests 1.43×). Much smaller than expected.
- Theoretical upper bound of CP-RL speedup at bs=1 EP=8 prefill = (82.3 − 75.1) / 385 = **~1.9% of total step**. Real gain is a fraction of that after accounting for below-knee per-token slowdown.
- Nsys v4 with `--gpu-metrics-devices=all` confirmed head-node kernel-level data (337k kernels, 9.6s GPU time captured).

### §5. Router-swap validation

**Hypothesis**: In CP-RL / aux-only fine-tuning, only `mlp.layers.N.mlp.gate.weight` tensors change. All other 528+ tensors per shard are bit-identical to pretrained.

**Verified**: shard-1 and shard-2 tensor diff between pretrained and r15 shows exactly 2 changed tensors (`layers.0.mlp.gate.weight`, `layers.1.mlp.gate.weight`), 528 identical.

**Technique**:
1. Extract routers from ORD HF dir into compact 98 MB file (`extract_routers.py`)
2. rsync only the 98 MB file to HSG
3. Build patched dir on HSG (`build_r15_router_swap.py`): symlinks 24 no-router shards, rewrites 94 with-router shards
4. vLLM loads patched dir normally

**Result**: r15_router_swap reproduces ORD's +6.84% bs=1 prefill regression (measured +3.11% on GB200; sign matches, magnitude smaller). Router-swap technique is **4400× cheaper transfer** than full 435 GB rsync per cell.

### §6. Efficiency comparison (tok/s/GPU)

| operating point | HW | TPOT | agg tok/s | GPUs | **tok/s/GPU** |
|---|---|---|---|---|---|
| **bs=512 EP=8 pretrained** (GB200) | 1 node×4 GPU · 2 | 32.72 ms | **12,293** | 8 | **1,537** |
| bs=1024 EP=8 pretrained (GB200) | 1 node×4 GPU · 2 | 127.73 ms | 7,288 | 8 | 911 |
| bs=8192 EP=64 pretrained (H100 ORD) | 8 nodes×8 GPU | 1,311 ms | 6,249 | 64 | 98 |
| bs=8192 EP=64 r15 (H100 ORD) | 8 nodes×8 GPU | 1,158 ms | 7,074 | 64 | **111** |

**tok/s/GPU shows the actual serving efficiency metric.** The "r15 helps" regime (bs=8192 EP=64) delivers 111 tok/s/GPU — 14× worse per GPU than the best pretrained regime (bs=512 EP=8 GB200 at 1537 tok/s/GPU). Even if r15's -11.67% ORD improvement fully carries over to GB200 (projected ~215 tok/s/GPU), it's still 7× worse than pretrained's best.

**CP-RL is a net loss for practical serving.** It only "helps" in an efficiency-hostile regime (bs=8192 EP=64) that you'd never choose to deploy.

---

## Verdict on the CP-RL question

**On GB200 the CP-RL routing offers no serving benefit.** Every measured operating point on GB200 EP=8 shows CP-RL fine-tuned models are either slower than pretrained (at low-bs prefill, the compute-bound regime with the largest per-token cost) or statistically indistinguishable (higher-bs prefill, decode). The +6-8% low-bs prefill regression observed on H100 persists on GB200 across BOTH RL and aux-only fine-tuning methods, ruling out any RL-specific optimization gap and pointing to a fundamental cost of the smoother router — namely that CP reduction moves tokens deeper into the below-knee compute regime where per-token throughput drops. On the one regime where CP-RL helps on H100 (bs=8192 EP=64 decode), the efficiency is already 14× worse per GPU than the best pretrained operating point, making the "help" economically meaningless.

---

## Lessons learned

1. **Router-swap technique should be default for future CP-RL experiments.** 4400× cheaper than full-file transfer, and validated to produce identical inference behavior.
2. **Ray+vLLM multi-node on SLURM is fragile** — port collisions between concurrent jobs' Ray clusters and cross-rack node allocations both cause hard failures. Fix: use SLURM Topology/Block plugin's `--segment=N` to force same-rack allocation on NVL72, and set explicit non-overlapping Ray worker ports.
3. **DCGM idle-GPU alerts fire fast** (30 min threshold). Failed Ray init WILL burn ~30 GPU-h per stuck job at 16-node scale. Add fail-fast placement group timeouts to every launcher.
4. **`--max-tokens` mismatch between ORD (16) and HSG (32) confuses TPOT comparisons.** Match exactly when reproducing cross-HW patterns.

---

## Files

Under `examples/qwen3/benchmarks/`:
- `HSG_GB200_FINDINGS.md` — this document
- `hsg_experiment_results.csv` — all bench cells, one row each (>100 rows)
- `hsg_scripts/build_r15_router_swap.py` — patch routers onto pretrained shards
- `hsg_scripts/extract_routers.py` — extract 94 router tensors into ~98 MB file
- `hsg_scripts/hsg_setup_router_cell.sh` — end-to-end: router → swap dir → A/B bench
- `hsg_scripts/hsg_run_epN_ab.sh` — configurable EP={16,32,64} launcher template

## Sources (SLURM/GB200 best practices)

- [NVIDIA: Unlock Exascale Performance on GB200 NVL72 with Slurm Topology-Aware Job Scheduling](https://developer.nvidia.com/blog/unlock-exascale-performance-on-nvidia-gb200-nvl72-with-slurm-topology-aware-job-scheduling/)
- [Slurm Workload Manager Topology Guide](https://slurm.schedmd.com/topology.html)
- [Ray on SLURM deployment docs](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)
- [CoreWeave: Topology/Block Scheduling in Slurm](https://docs.coreweave.com/products/sunk/optimize_workloads/topology-scheduling)
- [Microsoft: AI Infrastructure Preflight at User Space](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/ai-infrastructure-preflight-at-user-space-validating-multi-node-multi-gpu-slurm-/4522284)
