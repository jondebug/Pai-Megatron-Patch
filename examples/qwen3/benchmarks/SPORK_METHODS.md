# SPORK Methods: nsys tracing of multi-node Ray+vLLM on HSG GB200

2026-07-06. How we produce SPORK-processable Nsight Systems traces from multi-node
Ray-parallel vLLM benchmarks on HSG (GB200 NVL72), and how the traces were analyzed.
Companion results in `HSG_GB200_FINDINGS.md` (§ Mechanism study).

## Why the obvious approaches fail

| attempt | failure |
|---|---|
| `nsys profile python3 bench.py` on the head only | Ray workers on other nodes run the GPU kernels; head trace has no CUPTI kernels / no NVTX. SPORK: `no such table: NVTX_EVENTS`. |
| single-node EP=8 trace | GB200 node uses NVLink/CUDA-IPC, no NCCL → SPORK's parser requires NCCL-tagged NVTX (`experiment_processor.py` filters `name.contains('nccl')`). Multi-node (EP≥16) required. |
| `nsys profile ray start ...` on workers | `ray start` daemonizes and exits; nsys traces the 2-second CLI (12 MB rep, empty). |
| `... ray start --block` + wait for exit after `ray stop` | `--block` never exits after `ray stop`; nsys never finalizes (job 4060919 hung 29 min). |
| SIGINT nsys with 10-min cap, default sampling | Default CPU sampling makes reports huge (7M callchain rows); conversion was killed at 75% (job 4063844). |

## Working recipe (launcher `hsg_scripts/hsg_run_nsys_ab_v3.sh`)

1. **Workers**: `nsys profile -t cuda,nvtx -s none --cpuctxsw=none --output=<lustre>/worker_$HOST ray start --address=$HEAD:6379 --block ...` — `--block` keeps the raylet
   and every vLLM worker it spawns inside nsys's traced process tree.
2. **Head** runs the bench under its own nsys (optional; worker traces carry the data),
   then touches a **sentinel file on lustre**.
3. Workers poll the sentinel; on it: `ray stop`, then **`kill -INT $NSYS_PID`** — nsys
   finalizes its report on SIGINT (`--block` will not exit on its own). Allow up to 25
   min before a last-resort TERM. `-s none` keeps finalization ~1 min.
4. The **batch script must `wait` on the worker srun step** (backgrounded with `&`),
   or SLURM kills nsys mid-write at teardown (job 4059123).
5. Built-in gate: first worker exports its rep → sqlite and checks
   `CUPTI_ACTIVITY_KIND_KERNEL > 1000` and NVTX rows containing `nccl` > 0; prints
   `SPORK_GATE=PASS/FAIL` into the job log.
6. Keep the Ray port pinning (worker ports 10002-19999; metrics 22222; agents
   22223/22224) — required for multi-job coexistence on HSG.

Validated: EP=16 4-node test (2.18M kernels, 340k NCCL NVTX per worker) and 6× EP=32
8-node ladder jobs, all gates PASS; ~500 MB per worker rep at `-s none`.

## SPORK processing

- Export on a compute node (container has nsys 2026.3.0):
  `nsys export --type=sqlite <rep> --output <sqlite>` (≈1.3 GB per worker).
- SPORK: repo `nsight-analysis`, venv on lustre (`.venv-hsg`; NEVER `pip install --user`
  on login nodes — it shadows container torch and kills running jobs).
- Config deltas from the repo test config (see `hsg_scripts/spork_config*.json`):
  - `"input_type": "sqlite"`
  - `tables_to_read` must list only tables that exist — drop `TARGET_INFO_GPU_METRICS`
    + `GPU_METRICS` (absent without `--gpu-metrics-devices`), add `NVTX_EVENTS`,
    `CUPTI_ACTIVITY_KIND_{RUNTIME,MEMCPY,MEMSET}`. The sqlite reader has no
    missing-table tolerance.
- Run per job dir (7 rank sqlites = multi-rank analysis):
  `python spork/engine/analysis.py --mode=bulk_report --config=config_hsg.json
  --input_path=<dir> --output=<out>`.
- Known xlsx caveat: decode traces exceed Excel's 1,048,576-row sheet limit on the
  kernel-timestamp sheets; HTML output and the summary sheets are unaffected.

## Bench-window attribution (A/B split)

A single trace spans both models (pretrained bench, then cell bench). SPORK's session
totals mix them. `hsg_scripts/analyze_ladder_ffn.py` splits per-rank kernel activity:
segment NCCL AllReduce timestamps at every gap > 60 s and take the two **densest**
segments (bench A, bench B) — do NOT use the single largest gap (it lands in the model-A
load phase and merges both benches). Validation: AR kernel counts match A vs B within
0.1-9%; kernel-time sums are normalized by AR count before comparing.

Kernel classes: `moe_ffn` (moe/grouped/expert/silu/topk), `gemm` (gemm/matmul/cutlass/
nvjet), `attention`, `nccl`, `other`.
