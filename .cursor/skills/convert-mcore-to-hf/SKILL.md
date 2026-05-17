---
name: convert-mcore-to-hf
description: Convert a trained Megatron-Core checkpoint to HuggingFace format so it can be benchmarked with lm-eval or served by vLLM. Use when the user wants to benchmark a checkpoint that only has a Megatron iter_NNNNNNN/ directory, prepare a checkpoint for vLLM inference, or convert 235B-A22B checkpoints (which require special 2-node handling).
---

# Convert Megatron-Core → HuggingFace

## When

The HF dir (`<ckpt>/hf_converted*/`) is missing or has no `*.safetensors`.
Required as a prerequisite for `run-lm-eval-benchmark` and
`vllm-latency-bench`.

## Decision: which submitter

| Model | Submitter | Resources | Layout |
|---|---|---|---|
| 30B A3B, single iter | (inline in `examples/qwen3/benchmarks/submit_benchmark.sh`, Step 1) | 1 node × 4 GPU | TP=1 PP=1 EP=4 |
| 30B A3B, many iters | (inline in `examples/qwen3/benchmarks/submit_batch_benchmark.sh`) | 1 node × 4 GPU | TP=1 PP=1 EP=4 |
| 235B A22B | `examples/qwen3/benchmarks/cp_latency_test/submit_convert_235b.sh` | **2 nodes × 8 GPU** | TP=1 PP=2 EP=8 |

For 30B the converter is almost always invoked **as part of** the
benchmark submit scripts (which skip conversion if safetensors already
exist). Only the 235B case has a stand-alone submitter.

## 30B convert (standalone, if ever needed)

Reuse the converter directly:

```bash
srun --account=nvr_israel_rlop --partition=interactive \
     --nodes=1 --gpus-per-node=4 --cpus-per-gpu=2 --time=01:00:00 \
     --container-image="/lustre/.../pai-megatron-patch_25.04.sqsh" \
     --container-mounts="$HOME:$HOME,/lustre/.../rl_token_routing:/lustre/.../rl_token_routing" \
     --container-workdir="$REPO_ROOT/toolkits/distributed_checkpoints_convertor" \
     bash -c "
       export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/backends/megatron/Megatron-LM-250624:\$PWD/impl:\${PYTHONPATH:-}
       export MODEL_PARALLEL_ARGS='--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4'
       export KUBERNETES_CONTAINER_RESOURCE_GPU=4
       bash scripts/qwen3/run_8xH20.sh A3B '<MCORE_CKPT_DIR>' '<HF_OUTPUT_DIR>' true true bf16 \
         '/lustre/.../qwen-ckpts/Qwen3-30B-A3B-complete'
     "
```

**Selecting a specific iteration**: the converter reads
`<MCORE_CKPT_DIR>/latest_checkpointed_iteration.txt`. Temporarily
overwrite it with the desired iter number, run the conversion, then
restore it. `submit_batch_benchmark.sh` and `submit_convert_235b.sh`
already do this with a `trap` on EXIT.

## 235B convert

```bash
TRAINED_MEGATRON_CKPT=/lustre/.../output_router_finetuning/<run>/checkpoint/<pretrain-...>/ \
ITER_NUM=1500 \
sbatch examples/qwen3/benchmarks/cp_latency_test/submit_convert_235b.sh
```

Output: `<TRAINED_MEGATRON_CKPT>/hf_converted_iter${ITER_NUM}_cp/`.
Runtime: ~1–2 h once allocated.

## Why 2-node for 235B

The 235B weights (~470 GB) + the HF-format target buffer + EP shards do
not fit on 8 × 80 GB H100/H20s on a single node — load step OOMs. PP=2
splits the layer stack across 2 nodes, halving the per-GPU peak. Single
node with TP=8 EP=8 fails with `TP*EP != world_size` (the Megatron
converter requires this product to equal the actual GPU count).

## Gotchas / things that broke before

- **Don't** retry single-node 235B conversion — both OOM and the TP/EP/world_size mismatch are dead ends.
- **PYTHONPATH** must include `backends/megatron/Megatron-LM-250624` — the converter script defaults to a different version path.
- **`KUBERNETES_CONTAINER_RESOURCE_GPU`** must equal `gpus-per-node`.
- Each converted iter goes in a **separate** `hf_converted_iter<N>/` dir so multiple iters of the same run can coexist.
- After conversion, verify with `ls <HF_OUTPUT_DIR>/*.safetensors | wc -l` — should be ~14 files for 30B, ~80+ for 235B.
