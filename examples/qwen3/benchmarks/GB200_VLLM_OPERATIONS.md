# GB200 NVL72 + vLLM Operations Guide (HSG cluster)

Operational knowledge from the CP-RL campaign (2026-06/07). Cluster: HSG
(`oci-hsg-cs-001-login-01`), 4 GPUs/node, 189GB HBM, MNNVL rack fabric, aarch64,
container `vllm-openai-arm.sqsh` (vLLM 0.2x, NCCL 2.28.9, nsys 2026.3.0, CUDA 13).

## all2all / EP backend status matrix (bf16 Qwen3-235B, this container)

| backend | status | notes |
|---|---|---|
| TP=EP + AllReduce | works (campaign default) | NVLS+SIMPLE inside graphs (verify via blockX=640, not kernel names); balance-insensitive |
| allgather_reducescatter | works (TEP shapes) | workspace ≈ 315KiB × DP × max_model_len → chunked-prefill + mnbt=2048 bounds it at ~5GiB |
| naive | works | plain torch a2a; load-proportional semantics but host/launch-bound |
| deepep_high_throughput / low_latency | **broken** | `deep_ep.cpp:226 invalid resource handle` — upstream Buffer is MNNVL-incompatible (KNOWN); NVSHMEM rebuilds don't fix it |
| hybrid_ep | absent from container | THE fix: MNNVL-aware HybridEPBuffer. Validated container: `/lustre/share/hw_nsw_misc/irauch/deci_handoff/` (their cluster) or registry `networking-insights/dev-tools/vllm_nsys_nccl:0.18_v2.29.7-1_arm64_hybridep`. Env: `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1`, `NCCL_MNNVL_ENABLE=1`, `NCCL_CUMEM_ENABLE=1`; drop `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` |
| flashinfer_nvlink_one_sided | nvfp4-only | errors cleanly on bf16 |
| flashinfer_nvlink_two_sided | crashes on bf16 | `NoneType .dim` in MoE runner |
| nccl_ep (nccl_low_latency) | not in container | alternative to DeepEP fork; needs NCCL 2.30 symlink bake-in + `NCCL_NET_PLUGIN=none` (HPC-X 2.19v8 segfault) |

## Serving geometry on NVL72

- **Wide TP-AR (TP=32/64) is the wrong shape**: NVLS is clique-limited at width;
  cross-trunk NVLink-SHARP is future hardware (Quantum-5/Vera Rubin). symm-mem AR
  supports TP ∈ {2,4,6,8} only — "world size 64 not supported" is by design.
- **Right shape: TP≤8 clique + EP/DP** (TEP pattern; TEP4 validated internally with
  symm-mem AR under graphs on NCCL 2.30.1). With TP>1+EP, vLLM uses sequence
  parallelism: tokens/GPU = ceil(tokens_per_DP_engine / TP).
- Backend selection for MoE dispatch: ratio AG+RS / dispatch-combine = EP/topK.
  Qwen3 topK=8 → dispatch/combine favored from EP=8.
- CUDA graphs: ~4× e2e win at small-medium batch (launch-bound otherwise). DeepEP-HT
  auto-disables graphs; naive keeps them.
- NCCL: prefer 2.30.x (NVLS-under-graphs fixes); `NCCL_NVLS_ENABLE=1` explicit;
  `NCCL_GRAPH_REGISTER=1` + ncclMemAlloc for NCCL-AR under capture; never force
  NCCL_PROTO. FlashInfer mnnvl AR backend is vLLM's intended multi-node TP path on
  GB200NVL (auto-selected in 0.18+).

## Multi-node launch patterns (SLURM)

- Ray cluster (TP=EP): pin ports per job — worker 10002-19999, metrics 22222, agents
  22223/22224, dashboard-listen 52365 — or concurrent jobs collide.
- vLLM offline DP (multiproc): one process per DP rank; set `VLLM_DP_RANK`(+`_LOCAL`),
  `VLLM_DP_SIZE`, `VLLM_DP_MASTER_IP/PORT`. Disable SLURM GPU binding
  (`--gpu-bind=none`, `CUDA_VISIBLE_DEVICES=0,1,2,3`) — vLLM indexes devices by
  dp_rank_local itself.
- Sequential engines in one Ray cluster at 64 ranks can hit second-engine `ray.get`
  startup timeouts (checkpoint-dependent, reproducible). Workaround: one model per job.
- Rack locality: `--segment=N` (= node count). 16-node segments queue 12-48h; 4-8 node
  jobs flow much faster — shape experiments accordingly.
- NEVER `pip install --user` torch/C-ext packages on login nodes: ~/.local shadows
  container site-packages and breaks all in-flight jobs (broke a night of EP=64 runs).

## nsys tracing of Ray-based vLLM (recipe that works)

See SPORK_METHODS.md for the full recipe. Essence: `nsys profile -t cuda,nvtx -s none
--cpuctxsw=none ... ray start --block` on every worker (keeps raylet + vLLM workers in
the traced tree); sentinel file on lustre; SIGINT nsys to finalize (--block never exits
on its own; allow 25 min); batch script MUST wait on the worker srun step; gate each
run by exporting one rep → sqlite and checking kernels + nccl NVTX counts.
Note: this disproves "Ray actors are not traceable by nsys" — worth sharing with the
networking-insights team (their workaround was the mp backend + cudaProfilerStart patch).

## Known failure signatures (grep-able)

| signature | meaning | action |
|---|---|---|
| `deep_ep.cpp:226 invalid resource handle` | upstream DeepEP on MNNVL | use hybrid_ep container |
| `vllm/_C.abi3.so: undefined symbol _ZN2at...` | user-site torch shadowing container | `pip uninstall -y torch` from ~/.local |
| `DP adjusted local rank N out of bounds` | SLURM GPU binding vs vLLM DP device indexing | `--gpu-bind=none` + all GPUs visible |
| `only supports nvfp4 activation quantization` | flashinfer one-sided on bf16 | two_sided (also broken) or quantize |
| second-engine `ray.get` timeout at EP=64 | sequential engine init race | one model per job |
| worker nsys rep 12MB / no kernels | daemonized ray start traced, not workers | use --block recipe |
| SPORK `no such table: X` | tables_to_read includes absent tables | trim advanced_config (no missing-table tolerance) |
| Excel `sheet too large` in SPORK | >1,048,576 kernel timestamps | ignore; HTML + summary sheets unaffected |

## 2026-07-09 addendum: rebuilt hybrid_ep container + serve-mode ops

- Registry hybridep tag = mixed-commit, unbootable. Rebuild via
  `spork_configs/hsg_build_deci_handoff.sh` (see HSG_GB200_FINDINGS §J for traps).
- Offline LLM() DP path is broken in the fork; use `vllm serve` + `--headless`
  + `--data-parallel-start-rank 4i` per extra node. VLLM_ENGINE_READY_TIMEOUT_S=1800
  (cold lustre load at DP=16 exceeds the 600s default).
- Rebuilt container has NO nsys; stage from vllm-openai-arm.sqsh to
  $BASE/tools/nsys_pkg (binary under target-linux-sbsa-armv8/).
- sbatch --export strips quotes: JSON flags (--compilation-config) must be
  embedded single-quoted in the script, not passed via --export. -O.field dot
  notation is NOT in this fork (-O = optimization_level enum).
- hybrid_ep: eager-only in practice (ignores cudagraph_mode); expect ~2× worse
  ITL than AG+RS at small batch. hybrid_ep comms are DeepEP kernels (ag_nvl_kernel,
  dispatch/combine_kernel), NOT NCCL — SPORK collectives parser sees ~nothing;
  classify by kernel name instead.
- Rack-locality: launcher now asserts single nvl72 rack prefix (--segment gives it).
