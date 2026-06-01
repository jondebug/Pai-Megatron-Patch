#!/usr/bin/env python3
"""
Profile a single vLLM MoE forward to measure per-GPU CP impact.

Records per (layer, step, rank):
  - local_num_tokens (after dispatch all-to-all)
  - per_expert_topk_counts (number of tokens routed to each expert globally)
  - moe_forward_ms (wall time of the full MoE forward at that layer)

Records per (rank) globally:
  - torch.profiler trace JSON (kernel-level, viewable in chrome://tracing / perfetto)

Usage:
  python3 cp_vllm_profile.py --model <HF dir> --name <label> [options]
"""
import argparse, json, os, time
from pathlib import Path

DEFAULT_OUT = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

# This script is exec()d on each worker via vLLM's collective_rpc to install the hook.
HOOK_TEMPLATE = r'''
import time, json, os
from pathlib import Path
import torch

def _install_moe_hook(out_path):
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except ImportError:
        try:
            from vllm.model_executor.layers.fused_moe import FusedMoE
        except ImportError:
            print("[profile_hook] could not import FusedMoE — skipping token-count instrumentation")
            return

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "a", buffering=1)
    orig_forward = FusedMoE.forward
    counter = {"step": 0, "layer_ids": {}}

    def patched(self, hidden_states, router_logits, *args, **kwargs):
        layer_id = counter["layer_ids"].setdefault(id(self), len(counter["layer_ids"]))
        if hidden_states.dim() == 2:
            num_tokens = hidden_states.shape[0]
        else:
            num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        out = orig_forward(self, hidden_states, router_logits, *args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        per_expert = None
        if router_logits is not None and router_logits.dim() == 2:
            try:
                topk = int(getattr(self, "top_k", 8))
                idx = router_logits.topk(topk, dim=-1).indices
                per_expert = torch.bincount(idx.flatten(), minlength=router_logits.shape[1]).tolist()
            except Exception:
                per_expert = None
        f.write(json.dumps({
            "step": counter["step"],
            "layer_id": layer_id,
            "num_input_tokens": int(num_tokens),
            "moe_ms": (t1 - t0) / 1e6,
            "rank": int(os.environ.get("RANK", -1)),
            "per_expert_topk_counts": per_expert,
        }) + "\n")
        return out

    FusedMoE.forward = patched
    print(f"[profile_hook] FusedMoE.forward patched; logging to {out_path}")

_install_moe_hook("OUTPATH")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir")
    ap.add_argument("--name", required=True, help="Short label for output filenames")
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="Decode tokens; small to keep trace manageable")
    ap.add_argument("--num-warmup", type=int, default=2)
    ap.add_argument("--num-record-steps", type=int, default=2)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from torch.profiler import profile, ProfilerActivity, record_function

    print(f"[profile] model={args.model}")
    print(f"[profile] name={args.name} tp={args.tp_size} plen={args.prompt_len} bs={args.batch_size}")

    from vllm import LLM, SamplingParams
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=max(2048, args.prompt_len + args.max_tokens + 64),
        gpu_memory_utilization=0.85,
        seed=args.seed,
    )
    try:
        llm = LLM(enable_expert_parallel=True, **llm_kwargs)
    except (TypeError, ValueError) as e:
        print(f"  enable_expert_parallel rejected ({e}); using default MoE layout")
        llm = LLM(**llm_kwargs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hook_out_pattern = str(out_dir / f"profile_{args.name}_rank__RANK___moe.jsonl")

    # ============================================================
    # Install per-worker MoE hook via collective_rpc if available
    # ============================================================
    installed = False
    try:
        engine = llm.llm_engine
        rpc_fn = None
        for attr_chain in [
            ("engine_core", "collective_rpc"),
            ("model_executor", "collective_rpc"),
        ]:
            obj = engine
            ok = True
            for a in attr_chain:
                if hasattr(obj, a): obj = getattr(obj, a)
                else: ok = False; break
            if ok and callable(obj):
                rpc_fn = obj; break

        if rpc_fn is None:
            print("  [profile] no collective_rpc attr found; hook not installed")
        else:
            template = HOOK_TEMPLATE
            pattern = hook_out_pattern  # closure capture
            def _install_fn():
                import os as _os
                rank = int(_os.environ.get("RANK", -1))
                out_path = pattern.replace("__RANK__", str(rank))
                code = template.replace("OUTPATH", out_path)
                exec(compile(code, "<moe_hook>", "exec"), {})
            rpc_fn(_install_fn)
            installed = True
            print(f"  [profile] MoE hook installed on all ranks via collective_rpc")
    except Exception as e:
        print(f"  [profile] could not install per-worker hook: {type(e).__name__}: {e}")

    if not installed:
        print("  [profile] WARNING — only torch.profiler trace will be captured (no token-count log)")

    # ============================================================
    # Generate deterministic prompts
    # ============================================================
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    g = torch.Generator().manual_seed(args.seed)
    prompts = []
    for i in range(args.batch_size):
        ids = torch.randint(low=1000, high=50000, size=(args.prompt_len,), generator=g).tolist()
        prompts.append(tok.decode(ids))

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    # ============================================================
    # Warmup
    # ============================================================
    print(f"  [profile] warmup ({args.num_warmup} calls)")
    for _ in range(args.num_warmup):
        llm.generate(prompts, sp, use_tqdm=False)

    # ============================================================
    # Profile capture
    # ============================================================
    trace_path = str(out_dir / f"profile_{args.name}_trace.json")
    print(f"  [profile] capturing {args.num_record_steps} generate() calls — trace → {trace_path}")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for step_i in range(args.num_record_steps):
            with record_function(f"generate_step_{step_i}"):
                llm.generate(prompts, sp, use_tqdm=False)

    prof.export_chrome_trace(trace_path)
    print(f"  [profile] trace: {trace_path}")
    summary_path = str(out_dir / f"profile_{args.name}_kernels.txt")
    with open(summary_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=120))
    print(f"  [profile] kernel summary: {summary_path}")

    # ============================================================
    # Aggregate per-rank MoE logs
    # ============================================================
    moe_logs = list(out_dir.glob(f"profile_{args.name}_rank*_moe.jsonl"))
    print(f"  [profile] {len(moe_logs)} per-rank MoE logs found")
    per_rank = {}
    for p in moe_logs:
        try:
            rank = int(p.name.split("rank")[-1].split("_")[0])
        except ValueError:
            continue
        per_rank.setdefault(rank, {"layer_count": 0, "total_moe_ms": 0.0, "total_input_tokens": 0,
                                    "per_layer_tokens": [], "per_layer_moe_ms": []})
        with open(p) as fh:
            for line in fh:
                try: e = json.loads(line)
                except Exception: continue
                if "err" in e: continue
                per_rank[rank]["layer_count"] += 1
                per_rank[rank]["total_moe_ms"] += e.get("moe_ms", 0)
                per_rank[rank]["total_input_tokens"] += e.get("num_input_tokens", 0)
                per_rank[rank]["per_layer_tokens"].append(e.get("num_input_tokens", 0))
                per_rank[rank]["per_layer_moe_ms"].append(e.get("moe_ms", 0))
    agg_path = str(out_dir / f"profile_{args.name}_summary.json")
    with open(agg_path, "w") as f:
        json.dump({"per_rank": per_rank}, f, indent=2)
    print(f"  [profile] aggregate summary: {agg_path}")

    if per_rank:
        toks = [v["total_input_tokens"] for v in per_rank.values()]
        moe_ms = [v["total_moe_ms"] for v in per_rank.values()]
        print(f"\n  === PER-RANK SUMMARY ({len(per_rank)} ranks) ===")
        print(f"  Total input tokens (sum over layers/steps) per rank:")
        for r in sorted(per_rank):
            print(f"    rank {r}: tokens={per_rank[r]['total_input_tokens']:>8}  moe_total_ms={per_rank[r]['total_moe_ms']:.1f}")
        if min(toks) > 0:
            print(f"\n  IMBALANCE  tokens: min={min(toks)} max={max(toks)} ratio={max(toks)/min(toks):.3f}x")
        if min(moe_ms) > 0:
            print(f"  IMBALANCE  moe_ms: min={min(moe_ms):.1f} max={max(moe_ms):.1f} ratio={max(moe_ms)/min(moe_ms):.3f}x")

if __name__ == "__main__":
    main()
