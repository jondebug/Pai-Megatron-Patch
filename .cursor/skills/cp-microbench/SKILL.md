---
name: cp-microbench
description: Run the CP→latency microbenchmark that captures real MoE routing decisions from a HuggingFace checkpoint and simulates per-EP-layout step latency from real FFN kernel timings. Use when the user wants to test whether a trained checkpoint actually reduces critical path on a dataset, project EP=8/EP=128 inference-step speedup from a routing trace, or compare multiple checkpoints' CP behavior outside of a real inference engine.
---

# CP → latency microbenchmark

## When

The user wants to know "does this checkpoint's CP advantage actually
translate to step-latency speedup, and at what EP layout?" without
spinning up vLLM. Cheap (1-2 GPUs, ~30-90 min).

## Script + submitters

- `examples/qwen3/benchmarks/cp_latency_test/cp_microbench.py` — core logic (hooks router, captures per-layer per-expert token counts, times FFN kernels, simulates EP step time).
- `submit_microbench.sh` — WikiText (out-of-distribution sanity).
- `submit_microbench_megatrondata.sh` — **Megatron mmap training data** (primary, on-distribution evaluation).
- `submit_microbench_debug.sh` — 1 GPU, tiny workload, for plumbing tests.

## Recommended invocation (training distribution)

```bash
TRAINED_MODEL="/lustre/.../<run>/checkpoint/<pretrain-...>/hf_converted_iter<N>" \
BASELINE_MODEL="/lustre/.../qwen-ckpts/Qwen3-30B-A3B-complete" \
TRAINED_NAME="<short_name>_megdata" \
BASELINE_NAME="pretrained_megdata" \
NUM_BATCHES=64 BATCH_SIZE=4 SEQ_LENGTH=2048 \
TAG="<run>_$(date +%H%M%S)" \
sbatch examples/qwen3/benchmarks/cp_latency_test/submit_microbench_megatrondata.sh
```

Output: `/lustre/.../cp_latency_results/microbench_<TAG>.json` containing per-model:

- captured CP (max-tokens-per-expert summed across 48 layers)
- mean / std of CP per layer
- simulated step time at EP ∈ {4, 8, 32, 128}
- projected speedup vs baseline at each EP

## Reading results

```bash
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); 
[print(m['name'], 'CP=', m['captured_cp'], 'EP128_speedup=', m['ep_speedup']['128']) for m in d['models']]" \
  /lustre/.../cp_latency_results/microbench_<TAG>.json
```

## Major finding to remember (don't re-discover)

**RL + CPB together is the problematic combination**. Models trained with
both flags show CP **regression** on stock HF inference (no CPB hook),
because `critical_path_bias` is a `register_buffer(persistent=False)` —
not saved in the checkpoint — and the policy learned to depend on the
biased logits. Standalone CPB (no RL) and pure aux-loss models *do*
generalize to stock inference and show speedup. Always note which
combination is being benchmarked when interpreting results.

## Gotchas

- `--cpus-per-gpu=2` is mandatory (defaults would eat ~60 CPUs/GPU).
- The dataset path is sensitive: the script supports Megatron mmap (`.idx`+`.bin` siblings, no extension) and local `.arrow` / `.parquet`. Do **not** rely on `datasets.load_dataset("Salesforce/wikitext")` — older `datasets` in the container chokes on its glob patterns and compute nodes block S3.
- For 235B, you'll need a converted `hf_converted_iter<N>_cp/` dir first (see `convert-mcore-to-hf`).
- The `ep_speedup` numbers are projections from FFN kernel timings only — they ignore comms. Use `vllm-latency-bench` for real wall-time.
