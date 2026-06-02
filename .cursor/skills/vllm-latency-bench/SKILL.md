---
name: vllm-latency-bench
description: Run a real end-to-end vLLM latency benchmark comparing N MoE checkpoints across multiple (prompt_length, batch_size) cells, with TP+EP enabled. Use when the user wants real wall-time speedup numbers (not simulated), publication-grade latency comparisons, or to validate that a checkpoint's reduced critical path translates to actual inference TTFT/throughput gains.
---

# vLLM end-to-end latency bench

## When

The user wants real wall-clock speedup numbers across multiple
checkpoints — not the simulated EP step time from `cp-microbench`.

## Script + submitter

- `examples/qwen3/benchmarks/cp_latency_test/cp_vllm_bench.py` — core (accepts N `name=path` models, runs each across the prompt×batch grid, reports e2e/TTFT/TPS + speedup vs the first model).
- `examples/qwen3/benchmarks/cp_latency_test/submit_vllm_bench_ngc.sh` — 1 node × 8 GPU, vLLM container.

**Container**: `/lustre/fsw/portfolios/nvr/users/jonathanp/containers/vllm-openai-latest.sqsh`
(vLLM ≥ 0.7; the standard `pai-megatron-patch_25.04.sqsh` cannot install
vLLM cleanly — PyTorch 2.7 conflict).

## Typical invocation

```bash
MODELS="pretrained=/lustre/.../qwen-ckpts/Qwen3-30B-A3B-complete,B1=/lustre/.../<run1>/.../hf_converted,aux_only=/lustre/.../<run2>/.../hf_converted_iter1500" \
TP_SIZE=8 \
PROMPT_LENGTHS="256,1024,4096" \
BATCH_SIZES="1,8,32" \
MAX_TOKENS=256 \
NUM_TRIALS=10 \
TAG="$(date +%Y%m%d_%H%M%S)" \
sbatch examples/qwen3/benchmarks/cp_latency_test/submit_vllm_bench_ngc.sh
```

The first model in `MODELS` is the baseline; others are reported as
speedup vs it.

## Parallelism

vLLM is launched with `tensor_parallel_size=8` + `enable_expert_parallel=True`,
which gives **TP=8 + EP=8** active on 8 GPUs. This is the canonical
layout for both 30B-A3B and 235B-A22B in this project. Don't pass `TP=1`
unless you specifically want a sharding ablation — most CP-reduction
gains show up only when EP is active.

## Smoke tests (before a real run)

- `submit_vllm_smoke.sh` — 1 model, 1 cell, 2 trials. Validates the container + plumbing.
- `submit_vllm_smoke_235b.sh` — 235B baseline only, `gpu_memory_utilization=0.92` patched inline.

Always smoke-test once after pulling a new vLLM container.

## Output

`/lustre/.../cp_latency_results/vllm_ngc_<TAG>.json` — per-cell, per-model
mean/std for `e2e_ms`, `ttft_ms`, `decode_tps`; plus speedup % vs the
first model.

**Important: trust the `.out` log first.** A previous run wrote a JSON file
with `e2e_ms_mean=0` and `decode_tps_mean=0` even though the stdout summary
contained valid end-to-end and decode numbers. Always inspect the matching
`/lustre/.../cp_latency_logs/vllm_*_<JOB_ID>.out` before concluding that only
TTFT was measured. Treat the JSON as a convenience artifact, not the source of
truth, until the serialization path is fixed.

A reasonable summary print from the authoritative `.out` file:

```bash
python3 - /lustre/.../cp_latency_logs/vllm_ngc_<JOB_ID>.out <<'PY'
import re, sys
log = sys.argv[1]
cell = None
for line in open(log):
    m = re.match(r"\s*##\s+(plen\d+_bs\d+)", line)
    if m:
        cell = m.group(1)
        print(f"\n{cell}")
        continue
    if cell and re.match(r"\s*(pretrained|B1|aux|[A-Za-z0-9_+-]+)", line):
        parts = line.split()
        if len(parts) >= 6 and parts[-1].endswith("x"):
            print("  " + line.strip())
PY
```

## Validated headline results (30B, EP=8)

Source: `cp_latency_logs/vllm_ngc_27853483.out`, not the JSON file.

| Model | Best e2e speedup | Avg e2e speedup over bs≥8 cells | Accuracy delta |
|---|---:|---:|---:|
| `B1_rlaux_iter5000` (RL+aux, no CPB) | **1.161×** at plen=1024 bs=32 | **1.094×** (+9.4%) | +0.21 pp |
| `aux_only_iter1500` | **1.111×** at plen=1024 bs=32 | **1.047×** (+4.7%) | +0.09 pp |

Speedup is largest at moderate prompt length + high batch — both increase
per-step expert occupancy and surface the imbalance. At bs=1 the effect is
near zero, so don't summarize only single-request latency.

Before publishing any of these numbers in a doc / table / slide, run the
**`publish-numbers` checklist** — label the slice ("best cell" vs
"avg over bs ≥ 8" vs "avg over all 9 cells"), disclose the EP layout, and
cite the source JSON path. Mixing "best cell" with "average" in the same
column has caused user pushback before.

## Gotchas

- vLLM API drift: the script's `_build_engine` tries `enable_expert_parallel=True` and falls back if the kwarg is rejected. Don't "fix" the try/except — it's intentional for cross-version compat.
- Engine load time alone for 235B at TP=8 is ~5 min — budget accordingly.
- 235B vLLM needs `gpu_memory_utilization` ≈ 0.92 on 8 × 80 GB; the default 0.85 will OOM mid-run.
- Always sanity-generate a single token before the full benchmark (the script already does this) — catches bad checkpoints fast.
- If the JSON and `.out` disagree, use the `.out` summary and file a follow-up
  to fix `cp_vllm_bench.py` serialization. Do **not** tell the user e2e/decode
  was never measured just because the JSON fields are zero.
