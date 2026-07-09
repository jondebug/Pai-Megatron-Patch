#!/usr/bin/env python3
"""cp_vllm_bench.py -- End-to-end inference latency comparison via vLLM.

Tests the same hypothesis as `cp_microbench.py` but with a real production
inference engine, so the numbers include attention, KV-cache, all-to-all
dispatch, kernel-fusion effects, etc.

Method:
  - Boot vLLM with `tensor_parallel_size=8` and `enable_expert_parallel=True`
    (== 8 GPUs, all 128 experts split into 16-expert shards across the 8 GPUs).
  - For each prompt set, measure:
      * TTFT (time to first token)
      * TPS (tokens per second at decode steady state)
      * End-to-end latency (mean / p50 / p95)
  - Same prompt set, same engine config, only the model weights differ.

Run on a single 8-GPU node. Requires vLLM (auto-installed at job startup if
not already present in the container).

Usage:
    python cp_vllm_bench.py \
        --baseline-model /lustre/.../Qwen3-30B-A3B-complete \
        --trained-model  /lustre/.../hf_converted_iter2000 \
        --output ./vllm_results.json \
        [--prompt-lengths 256,1024] [--max-tokens 256] \
        [--batch-sizes 1,8,32]
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser(description="vLLM end-to-end CP→latency comparison")
    # Either provide --baseline + --trained (legacy 2-model mode) or --models
    # (multi-model mode). The first model in --models is treated as the
    # baseline for speedup ratios.
    p.add_argument("--baseline-model", default=None,
                   help="(legacy) baseline HF model path")
    p.add_argument("--trained-model", default=None,
                   help="(legacy) trained HF model path")
    p.add_argument("--baseline-name", default="baseline")
    p.add_argument("--trained-name", default="trained")
    p.add_argument("--models", default=None,
                   help="Comma-separated 'name=path' specs, e.g. "
                        "'pretrained=/path1,B1=/path2,aux_only=/path3'. "
                        "First entry is treated as the baseline for speedup "
                        "ratios. Overrides --baseline-model / --trained-model.")
    p.add_argument("--output", default="./vllm_results.json")
    p.add_argument("--tp-size", type=int, default=8,
                   help="vLLM tensor_parallel_size (also acts as EP size for MoE)")
    p.add_argument("--prompt-lengths", default="256,1024",
                   help="Comma-separated prompt lengths in tokens")
    p.add_argument("--batch-sizes", default="1,8,32",
                   help="Comma-separated batch sizes (concurrent requests)")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="Tokens to generate per request")
    p.add_argument("--num-warmup", type=int, default=3,
                   help="Warmup iterations to discard")
    p.add_argument("--num-trials", type=int, default=10,
                   help="Timed iterations per (prompt_len, batch_size) cell")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", action="store_true", help="capture a torch-profiler decode burst (needs VLLM_TORCH_PROFILER_DIR)")
    p.add_argument("--profile-steps", type=int, default=32)
    p.add_argument("--prefill-profile", action="store_true",
                   help="when --profile is set, profile a SINGLE max_tokens=1 generation "
                        "(pure prefill forward) instead of a decode burst, so the trace "
                        "isolates the compute-bound prefill expert-FFN. Uses prompt_lengths[0] x batch_sizes[-1].")
    p.add_argument("--cuda-graphs", action="store_true",
                   help="enable CUDA graphs (enforce_eager=False). Default is enforce_eager=True.")
    p.add_argument("--distributed-executor-backend", default=None,
                   help="e.g. 'ray' for cross-node TP")
    return p.parse_args()


def _resolve_model_list(args):
    """Return ordered list of (name, path) from CLI args."""
    if args.models:
        out = []
        for spec in args.models.split(","):
            spec = spec.strip()
            if "=" not in spec:
                raise ValueError(f"--models entries must be 'name=path', got {spec!r}")
            name, path = spec.split("=", 1)
            out.append((name.strip(), path.strip()))
        if not out:
            raise ValueError("--models is empty after parsing")
        return out
    if args.baseline_model and args.trained_model:
        return [(args.baseline_name, args.baseline_model),
                (args.trained_name, args.trained_model)]
    raise SystemExit("Provide either --models or both --baseline-model and --trained-model")


def _build_engine(model_path: str, args):
    """Build a vLLM engine. Falls back gracefully across vLLM API versions."""
    from vllm import LLM

    common = {
        "model": model_path,
        "tensor_parallel_size": args.tp_size,
        **({"distributed_executor_backend": args.distributed_executor_backend} if getattr(args, "distributed_executor_backend", None) else {}),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "enforce_eager": (not getattr(args, "cuda_graphs", False)),
        "max_model_len": max(2048, max(int(s) for s in args.prompt_lengths.split(",")) + args.max_tokens + 64),
        "gpu_memory_utilization": 0.85,
        "seed": args.seed,
    }
    # vLLM 0.20.x: profiling is enabled via profiler_config (NOT the VLLM_TORCH_PROFILER_DIR
    # env var anymore). Pass it at engine construction when --profile + a dir are set.
    import os as _os
    _pdir = _os.environ.get("VLLM_TORCH_PROFILER_DIR")
    if getattr(args, "profile", False) and _pdir:
        try:
            from vllm.config import ProfilerConfig  # noqa: F401
            common["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": _pdir,
                                         "torch_profiler_use_gzip": True}
            print(f"  profiler_config set -> torch trace dir {_pdir}")
        except Exception as _e:
            print(f"  Note: ProfilerConfig unavailable ({_e}); profiling may not capture.")
    # vLLM EP-API has changed across versions:
    #   - 0.7-0.8: enable_expert_parallel=True (kwarg)
    #   - 0.9+:    automatic when MoE + TP>1; enable_expert_parallel may be removed/renamed
    # Try the explicit flag first; on any kwarg-related failure, fall back to defaults.
    # Inject compilation_config override from env (added for combo-kernel disable test)
    import json as _json, os as _os2
    _gmu = _os2.environ.get("VLLM_GPU_MEMORY_UTILIZATION")
    if _gmu:
        try:
            common["gpu_memory_utilization"] = float(_gmu)
            print("  gpu_memory_utilization override:", _gmu)
        except Exception as _e:
            print("  Failed to parse VLLM_GPU_MEMORY_UTILIZATION:", _e)
    _mns = _os2.environ.get("VLLM_MAX_NUM_SEQS")
    if _mns:
        try:
            common["max_num_seqs"] = int(_mns)
            print("  max_num_seqs override:", _mns)
        except Exception as _e:
            print("  Failed to parse VLLM_MAX_NUM_SEQS:", _e)
    _mnbt = _os2.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS")
    if _mnbt:
        try:
            common["max_num_batched_tokens"] = int(_mnbt)
            common["enable_chunked_prefill"] = False
            print("  max_num_batched_tokens override:", _mnbt, "chunked_prefill=False")
        except Exception as _e:
            print("  Failed to parse VLLM_MAX_NUM_BATCHED_TOKENS:", _e)
    _cc_json = _os2.environ.get("VLLM_COMPILATION_CONFIG_JSON")
    if _cc_json:
        try:
            common["compilation_config"] = _json.loads(_cc_json)
            print("  compilation_config override:", _cc_json)
        except Exception as _e:
            print("  Failed to parse VLLM_COMPILATION_CONFIG_JSON:", _e)

    try:
        return LLM(enable_expert_parallel=True, **common)
    except (TypeError, ValueError) as e:
        print(f"  Note: enable_expert_parallel=True rejected ({type(e).__name__}: {e}); "
              "falling back to default MoE layout (TP-only sharding).")
        return LLM(**common)


def _generate(llm, prompts, sp):
    """Compatibility wrapper: vLLM renamed `use_tqdm` flag at some point."""
    try:
        return llm.generate(prompts, sp, use_tqdm=False)
    except TypeError:
        # Newer/older vLLM may not have use_tqdm — try without it.
        return llm.generate(prompts, sp)


def _make_prompts(tokenizer, prompt_len: int, batch_size: int, seed: int):
    """Generate `batch_size` deterministic prompts of EXACTLY `prompt_len` tokens.

    Prefers vLLM `TokensPrompt` (token IDs passed straight through, so the prompt
    length is exact — decoding random IDs to text then re-tokenizing is NOT length-
    preserving and was overflowing max_model_len). Falls back to decoded text if
    TokensPrompt is unavailable."""
    import torch
    g = torch.Generator().manual_seed(seed)
    id_lists = []
    for i in range(batch_size):
        ids = torch.randint(low=1000, high=50000, size=(prompt_len,), generator=g).tolist()
        id_lists.append(ids)
    try:
        from vllm import TokensPrompt
        return [TokensPrompt(prompt_token_ids=ids) for ids in id_lists]
    except Exception:
        # Fallback: decode then truncate-re-encode to clamp <= prompt_len.
        out = []
        for ids in id_lists:
            txt = tokenizer.decode(ids)
            re_ids = tokenizer.encode(txt)[:prompt_len]
            out.append(tokenizer.decode(re_ids))
        return out


def _time_one_cell(llm, sampling_params, prompts, num_warmup, num_trials) -> Dict:
    from vllm import SamplingParams
    # Warmup
    for _ in range(num_warmup):
        _ = _generate(llm, prompts, sampling_params)

    # Timed
    end_to_end = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        outs = _generate(llm, prompts, sampling_params)
        t1 = time.perf_counter()
        end_to_end.append((t1 - t0) * 1000.0)

    # Approximate TTFT from runs with max_tokens=1.
    ttft_sp = SamplingParams(
        max_tokens=1,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        ignore_eos=True,
    )
    ttft_samples = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = _generate(llm, prompts, ttft_sp)
        t1 = time.perf_counter()
        ttft_samples.append((t1 - t0) * 1000.0)

    total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    mean_e2e_s = statistics.mean(end_to_end) / 1000.0
    return {
        "end_to_end_ms_mean": float(statistics.mean(end_to_end)),
        "end_to_end_ms_p50":  float(statistics.median(end_to_end)),
        "end_to_end_ms_p95":  float(sorted(end_to_end)[int(0.95 * (len(end_to_end) - 1))]),
        "end_to_end_ms_std":  float(statistics.stdev(end_to_end) if len(end_to_end) > 1 else 0.0),
        "ttft_ms_mean":       float(statistics.mean(ttft_samples)),
        "ttft_ms_std":        float(statistics.stdev(ttft_samples) if len(ttft_samples) > 1 else 0.0),
        "ttft_ms_n":          int(len(ttft_samples)),
        "ttft_ms_p95":        float(sorted(ttft_samples)[int(0.95 * (len(ttft_samples) - 1))]),
        "decode_tps":         float(total_gen_tokens / mean_e2e_s) if mean_e2e_s > 0 else 0.0,
        "total_gen_tokens":   int(total_gen_tokens),
    }


def benchmark_model(name: str, path: str, args) -> Dict:
    from vllm import SamplingParams
    from transformers import AutoTokenizer

    print(f"\n{'=' * 80}")
    print(f"BUILDING vLLM ENGINE for {name}: {path}")
    print(f"{'=' * 80}")

    t_load_start = time.time()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    llm = _build_engine(path, args)
    t_load = time.time() - t_load_start
    print(f"  Engine ready in {t_load:.1f}s")

    # Sanity: tiny generation to confirm vLLM actually works before the full bench.
    sanity_sp = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
    try:
        sanity_out = _generate(llm, ["Hello, this is a sanity check."], sanity_sp)
        print(f"  Sanity gen OK: {sanity_out[0].outputs[0].text!r}")
    except Exception as e:
        print(f"  WARNING: sanity generation failed: {type(e).__name__}: {e}")

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, ignore_eos=True)

    prompt_lens = [int(x) for x in args.prompt_lengths.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    cells = {}
    for plen in prompt_lens:
        for bs in batch_sizes:
            print(f"\n  → prompt_len={plen}, batch_size={bs}, max_tokens={args.max_tokens}")
            prompts = _make_prompts(tok, plen, bs, args.seed)
            stats = _time_one_cell(llm, sp, prompts,
                                   num_warmup=args.num_warmup,
                                   num_trials=args.num_trials)
            print(f"    e2e mean={stats['end_to_end_ms_mean']:.1f}ms "
                  f"(p50={stats['end_to_end_ms_p50']:.1f}, p95={stats['end_to_end_ms_p95']:.1f}); "
                  f"ttft={stats['ttft_ms_mean']:.1f}ms; "
                  f"decode={stats['decode_tps']:.1f} tok/s")
            cells[f"plen{plen}_bs{bs}"] = {"prompt_len": plen, "batch_size": bs, **stats}

    if getattr(args, "profile", False):
        import os as _os
        _pdir = _os.environ.get("VLLM_TORCH_PROFILER_DIR", "(unset)")
        _pr = _make_prompts(tok, prompt_lens[0], batch_sizes[-1], args.seed)
        if getattr(args, "prefill_profile", False):
            # Pure prefill: a single max_tokens=1 generation. The whole trace is one
            # prefill forward (plus a trivial 1-token decode), so per-rank expert_ffn
            # self-time = the COMPUTE-BOUND prefill FFN. Warm the engine first so the
            # captured forward is steady-state (graphs already captured for decode).
            print("\n  >>> PROFILING PREFILL forward (plen=%d, bs=%d, max_tokens=1) -> %s"
                  % (prompt_lens[0], batch_sizes[-1], _pdir))
            _psp = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, ignore_eos=True)
            for _ in range(max(2, args.num_warmup)):
                _ = _generate(llm, _pr, _psp)
        else:
            print("\n  >>> PROFILING decode burst (%d tok) -> %s" % (args.profile_steps, _pdir))
            _psp = SamplingParams(max_tokens=args.profile_steps, temperature=0.0, top_p=1.0, ignore_eos=True)
        try:
            llm.start_profile()
            _ = _generate(llm, _pr, _psp)
            llm.stop_profile()
            print("  >>> profile trace written")
        except Exception as _e:
            print("  >>> profiling FAILED: %s: %s" % (type(_e).__name__, _e))

    # Tear down the engine so we can load the next model.
    del llm
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"name": name, "model_path": path, "cells": cells}


def _print_summary(model_results: List[Dict]):
    """Print per-cell e2e + per-model summary. First model is the baseline for
    speedup ratios."""
    if not model_results:
        return
    baseline = model_results[0]
    print()
    print("=" * 110)
    print("vLLM END-TO-END LATENCY SUMMARY")
    print("=" * 110)
    for m in model_results:
        print(f"  - {m['name']}: {m['model_path']}")
    print(f"  Baseline for speedup ratios: {baseline['name']}")
    print()

    cells = sorted(baseline["cells"].keys())
    # Per-cell table: one row per cell, one column block per non-baseline model.
    for cell in cells:
        bcell = baseline["cells"][cell]
        print(f"\n  ## {cell} (prompt_len={bcell['prompt_len']}, batch_size={bcell['batch_size']})")
        print(f"  {'model':<32s} {'e2e_mean_ms':>12s} {'e2e_p95_ms':>11s} "
              f"{'ttft_ms':>9s} {'decode_tps':>11s} {'speedup_vs_base':>16s}")
        print("  " + "-" * 96)
        for m in model_results:
            mc = m["cells"].get(cell)
            if mc is None:
                continue
            sp = bcell["end_to_end_ms_mean"] / max(mc["end_to_end_ms_mean"], 1e-9)
            print(f"  {m['name'][:32]:<32s} {mc['end_to_end_ms_mean']:>12.1f} "
                  f"{mc['end_to_end_ms_p95']:>11.1f} {mc['ttft_ms_mean']:>9.1f} "
                  f"{mc['decode_tps']:>11.1f} {sp:>15.4f}x")
    print("=" * 110)


def main():
    args = parse_args()

    # Ensure vLLM importable (the SLURM wrapper installs it before running, but
    # we double-check to give a clean error if the install failed).
    try:
        import vllm  # noqa: F401
    except ImportError:
        print("ERROR: vLLM not installed. Run `pip install vllm` first "
              "(or use the SLURM submitter which does this automatically).",
              file=sys.stderr)
        sys.exit(1)

    model_list = _resolve_model_list(args)
    print(f"Configuration:")
    print(f"  TP / EP size      : {args.tp_size}")
    print(f"  Prompt lengths    : {args.prompt_lengths}")
    print(f"  Batch sizes       : {args.batch_sizes}")
    print(f"  Max tokens        : {args.max_tokens}")
    print(f"  Warmup / trials   : {args.num_warmup} / {args.num_trials}")
    print(f"  Models ({len(model_list)}):")
    for name, path in model_list:
        print(f"    - {name}: {path}")

    model_results = []
    for name, path in model_list:
        model_results.append(benchmark_model(name, path, args))

    _print_summary(model_results)

    out = {
        "config": {
            "models": [{"name": n, "path": p} for n, p in model_list],
            "tp_size": args.tp_size,
            "prompt_lengths": [int(x) for x in args.prompt_lengths.split(",")],
            "batch_sizes": [int(x) for x in args.batch_sizes.split(",")],
            "max_tokens": args.max_tokens,
            "num_warmup": args.num_warmup,
            "num_trials": args.num_trials,
            "seed": args.seed,
        },
        "models": model_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
