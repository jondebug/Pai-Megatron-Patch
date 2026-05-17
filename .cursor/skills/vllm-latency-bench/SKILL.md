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

A reasonable summary print:

```bash
python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
for cell in d['cells']:
    print(f'plen={cell[\"prompt_len\"]} bs={cell[\"batch_size\"]}')
    for m in cell['models']:
        sp = m.get('e2e_speedup_pct'); print(f'  {m[\"name\"]:25s} e2e={m[\"e2e_ms\"]:.1f}ms ttft={m[\"ttft_ms\"]:.1f}ms', f'speedup={sp:+.1f}%' if sp is not None else '(baseline)')
" /lustre/.../cp_latency_results/vllm_ngc_<TAG>.json
```

## Validated headline results (30B, EP=8)

- `B1` (RL + aux, no CPB): **+16.1 % e2e** at plen=1024 bs=32, +0.21 pp accuracy.
- `aux_only`: +11.1 % at the same cell, +0.09 pp.
- Speedup is largest at moderate prompt length + high batch — both increase per-step expert occupancy and surface the imbalance.

## Gotchas

- vLLM API drift: the script's `_build_engine` tries `enable_expert_parallel=True` and falls back if the kwarg is rejected. Don't "fix" the try/except — it's intentional for cross-version compat.
- Engine load time alone for 235B at TP=8 is ~5 min — budget accordingly.
- 235B vLLM needs `gpu_memory_utilization` ≈ 0.92 on 8 × 80 GB; the default 0.85 will OOM mid-run.
- Always sanity-generate a single token before the full benchmark (the script already does this) — catches bad checkpoints fast.
