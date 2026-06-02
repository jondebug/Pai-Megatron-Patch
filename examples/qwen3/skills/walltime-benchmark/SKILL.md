---
name: walltime-benchmark
description: Measure 235B MoE inference walltime (decode tok/s, TTFT, e2e) with vLLM — single-node TP=8 and multi-node Ray expert-parallel (EP=16/32/64/128) — plus per-step profiling (attn/ffn/dispatch/combine) and an NVLink-domain projection. Use for the CP→latency contribution: showing how routing balance affects step time at large EP.
---

# Walltime benchmarking (vLLM, single- and multi-node EP)

Script: `examples/qwen3/benchmarks/cp_latency_test/cp_vllm_bench.py`
Outputs `decode_tps`, `ttft_ms_mean`, `end_to_end_ms_mean` per (prompt_len, batch_size) cell.

## Single-node (TP=8 = EP=8, 1 node) — easy, reliable
```
python3 cp_vllm_bench.py --models <name>=<HF_DIR> --tp-size 8 \
  --prompt-lengths 256 --batch-sizes 8 --max-tokens 256 --num-trials 10 --output out.json
```
Container = `vllm-openai-latest.sqsh`; `enforce_eager=True`, `gpu_memory_utilization=0.85`,
`enable_expert_parallel=True`. Pretrained 235B H100 ≈ **108.8 tps / 155 ms TTFT**.

## Multi-node Ray (EP = NODES×8) — the working recipe
`tensor_parallel_size = NODES×8`, `distributed_executor_backend=ray`. The vLLM container has
**no Ray**, so build the cluster yourself. Every one of these was a separate failure mode —
all are required:
1. **`pip install ray==2.48.0`** at runtime on every node. <2.48 lacks
   `ray.experimental.channel.accelerator_context` which vLLM V1's ray executor imports.
2. **Co-locate the driver with the Ray head in ONE srun/container** (shared `/tmp/ray`), else
   the vLLM client can't find the cluster session and sees only the local 8 GPUs.
3. **Head publishes its on-node IP** (`hostname -I | awk '{print $1}'`) to a shared lustre
   file; workers read it. The login-resolved (`getent`) IP is often **not worker-routable**.
4. **Pin every Ray port below the ephemeral range (≥32768)** so random agent ports can't land
   in the worker range: `--min-worker-port 20002 --max-worker-port 20200
   --node-manager-port 20301 --object-manager-port 20302 --runtime-env-agent-port 20303
   --dashboard-agent-grpc-port 20304 --dashboard-agent-listen-port 20305
   --metrics-export-port 20306 --include-dashboard=false`.
5. **`--overlap --mem=0`** on the srun steps so the blocking `ray start` step and the driver
   step can share node resources/memory.
6. **Wait-for-cluster with FAIL-FAST**: poll `ray status` GPU count; if it doesn't reach
   NODES×8 within ~6 min, `exit` — never proceed with a partial cluster that then idles into
   the reaper. See [[gpu-job-dispatch]].

Reference launcher: `run_nrt_walltime_rayEP.sh` (set `--nodes=N`).
**Finding:** actual throughput *drops* with EP across nodes (EP8=108.8 → EP16=86.9 tps) —
cross-node all-to-all dispatch/combine dominates. That motivates the profiling + NVLink
projection below.

## Per-step profiling (attn / ffn / dispatch+combine / other)
Set `VLLM_TORCH_PROFILER_DIR=<dir>`; the bench's `--profile` mode wraps a decode burst in
`llm.start_profile()/stop_profile()` → per-worker Chrome traces. Then
`categorize_trace.py <dir> --ep N` buckets GPU-kernel time by name (flash/attn → attn;
gemm/moe/grouped → ffn; nccl/alltoall → dispatch+combine; rest → other) and prints the split.
**NVLink-domain projection:** the dispatch+combine bucket is the inter-node-sensitive cost;
`(NODES-1)/NODES` of the all-to-all crosses nodes — replace that fraction's time with
NVLink-speed (≈900 vs ≈50 GB/s) to project tps in a single large NVLink domain.

## Simulation alternative (no GPUs at high EP)
`cp_microbench.py` replays captured routing traces and simulates per-step **expert-compute**
time at EP∈{1..128} (step = Σ_layers max-loaded-GPU FFN time). Caveats: (a) models ONLY expert
compute — no comm — so it's a *different quantity* than actual walltime; (b) verify the arch
constants match the model (it shipped with **30B** values: HIDDEN_SIZE 2048, MOE_INTERMEDIATE
768, 48 layers — 235B is 4096/1536/94, which changes the result materially).
