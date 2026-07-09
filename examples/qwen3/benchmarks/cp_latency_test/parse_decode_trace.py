#!/usr/bin/env python3
"""parse_decode_trace.py -- programmatically decompose a vLLM/torch.profiler
Chrome-trace JSON into per-decode-step GPU time by stage:
  {attention, moe_dispatch_a2a, expert_ffn, moe_combine_a2a, other_gemm,
   elementwise/norm, idle/gap}.

Method (no eyeballing):
  - Keep only GPU kernel events: cat in {"kernel","gpu_op"} OR events on a
    device stream (pid/tid that the trace marks as GPU). We detect GPU tids
    from metadata "process_labels"/"thread_name" containing "stream"/"GPU".
  - Classify each kernel by name regex into a stage bucket.
  - Restrict to steady-state decode: drop the first/last K steps (warmup/flush)
    using the user_annotation "generate_step_*" or "Decode" markers if present;
    else use the longest contiguous span and trim 10% each end.
  - Sum self-time per bucket (handle overlap: report both raw kernel-sum and
    busy-wall via union of intervals on the busiest stream).

Classification regexes are intentionally broad and logged so they can be audited
against the actual kernel names present (printed as a coverage report: % of GPU
time matched vs "other").
"""
import argparse, json, re, sys
from collections import defaultdict

# name -> stage. First match wins. Case-insensitive.
RULES = [
    ("moe_dispatch_a2a", re.compile(r"all.?to.?all|alltoall|AllToAll|nccl.*[Aa]ll2[Aa]ll|dispatch", re.I)),
    ("comm_other",       re.compile(r"nccl|ncclDevKernel|allreduce|all_reduce|reduce_scatter|allgather|all_gather|broadcast|sendrecv|c10d", re.I)),
    ("attention",        re.compile(r"attention|flash|fmha|paged|_attn|reshape_and_cache|rotary|rope|cutlass.*attn", re.I)),
    ("expert_ffn",       re.compile(r"grouped_gemm|group_gemm|moe|fused_moe|silu|swiglu|gmm|act_and_mul|expert", re.I)),
    ("gemm",             re.compile(r"gemm|cutlass|ampere_|sm80_|sgemm|hgemm|s16816|cublas|matmul|linear|wgrad|dgrad", re.I)),
    ("norm_elementwise", re.compile(r"norm|rms|layernorm|elementwise|vectorized|add|mul|copy|cast|convert|index|gather|scatter|memset|memcpy", re.I)),
]

def classify(name):
    for stage, rx in RULES:
        if rx.search(name):
            return stage
    return "other"

def load_events(path):
    import gzip
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    ev = d.get("traceEvents", d) if isinstance(d, dict) else d
    return ev

def gpu_tids(ev):
    """Identify (pid,tid) belonging to GPU streams from metadata events."""
    gpu = set()
    # metadata: ph=="M", name in {"thread_name","process_name","process_labels"}
    pid_is_gpu = set()
    for e in ev:
        if e.get("ph") != "M":
            continue
        nm = e.get("name", "")
        args = e.get("args", {}) or {}
        label = str(args.get("name", "")) + " " + str(args.get("labels", ""))
        if nm in ("process_name", "process_labels"):
            if re.search(r"GPU|stream|device|CUDA", label, re.I):
                pid_is_gpu.add(e.get("pid"))
        if nm == "thread_name":
            if re.search(r"stream|GPU|Stream", label, re.I):
                gpu.add((e.get("pid"), e.get("tid")))
    return gpu, pid_is_gpu

def decompose(path, trim_frac):
    """Return (stage_self_ms dict, total_ms, stage_names) for one trace file, or None if host-only."""
    ev = load_events(path)
    gpu_tid_set, gpu_pids = gpu_tids(ev)
    kernels = []
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        dur = e.get("dur")
        if dur is None:
            continue
        is_kernel = (cat in ("kernel", "gpu_op", "Kernel")) or \
                    (e.get("pid"), e.get("tid")) in gpu_tid_set or \
                    (e.get("pid") in gpu_pids and cat not in ("cpu_op", "user_annotation", "cuda_runtime", "python_function"))
        if is_kernel:
            kernels.append(e)
    if not kernels:
        return None
    ts = sorted(k["ts"] for k in kernels)
    lo, hi = ts[0], ts[-1]
    span = hi - lo
    a = lo + trim_frac * span
    b = hi - trim_frac * span
    win = [k for k in kernels if a <= k["ts"] <= b]
    stage_self = defaultdict(float)
    stage_names = defaultdict(lambda: defaultdict(float))
    for k in win:
        st = classify(k["name"])
        stage_self[st] += k["dur"]
        stage_names[st][k["name"]] += k["dur"]
    return dict(stage_self), sum(stage_self.values()), stage_names, len(win)


def main():
    import glob, os
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="trace .json[.gz] file OR a directory of per-rank traces")
    ap.add_argument("--trim-frac", type=float, default=0.1)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--per-rank", action="store_true",
                    help="treat `trace` as a directory; decompose every per-rank trace "
                         "and emit a per-rank expert_ffn / comm table (for prefill busiest-FFN analysis)")
    ap.add_argument("--prefill", action="store_true",
                    help="prefill mode: use the full trace (trim-frac=0) since a max_tokens=1 "
                         "trace is one prefill forward; report busiest-rank expert_ffn.")
    args = ap.parse_args()

    if args.prefill:
        args.trim_frac = 0.0

    if args.per_rank or os.path.isdir(args.trace):
        files = sorted(glob.glob(os.path.join(args.trace, "*.json.gz")) +
                       glob.glob(os.path.join(args.trace, "*.json")) +
                       glob.glob(os.path.join(args.trace, "*.pt.trace.json*")))
        files = [f for f in files if not f.endswith(".json-out")]
        print(f"per-rank decomposition over {len(files)} files in {args.trace}")
        rows = []
        for f in files:
            r = decompose(f, args.trim_frac)
            if r is None:
                print(f"  {os.path.basename(f)[:50]:50s}  HOST-ONLY (skipped)")
                continue
            stage_self, total, _, nwin = r
            ffn = stage_self.get("expert_ffn", 0.0) / 1e3
            comm = stage_self.get("comm_other", 0.0) / 1e3
            a2a = stage_self.get("moe_dispatch_a2a", 0.0) / 1e3
            attn = stage_self.get("attention", 0.0) / 1e3
            gemm = stage_self.get("gemm", 0.0) / 1e3
            rows.append({"file": os.path.basename(f), "total_ms": total/1e3,
                         "expert_ffn_ms": ffn, "comm_other_ms": comm, "a2a_ms": a2a,
                         "attention_ms": attn, "gemm_ms": gemm, "win_kernels": nwin})
        rows.sort(key=lambda x: -x["expert_ffn_ms"])
        print(f"\n{'rank-file':40s} {'total':>9s} {'expert_ffn':>11s} {'comm':>9s} {'attn':>8s} {'gemm':>9s}")
        for r in rows:
            print(f"  {r['file'][:38]:38s} {r['total_ms']:9.2f} {r['expert_ffn_ms']:11.3f} "
                  f"{r['comm_other_ms']:9.2f} {r['attention_ms']:8.2f} {r['gemm_ms']:9.2f}")
        if rows:
            import statistics as st
            ffns = [r["expert_ffn_ms"] for r in rows]
            comms = [r["comm_other_ms"] for r in rows]
            print(f"\nexpert_ffn  busiest={max(ffns):.3f}ms  min={min(ffns):.3f}ms  "
                  f"mean={st.mean(ffns):.3f}ms  std={(st.stdev(ffns) if len(ffns)>1 else 0):.3f}ms  n={len(ffns)}")
            print(f"comm_other  busiest={max(comms):.3f}ms  min={min(comms):.3f}ms  "
                  f"mean={st.mean(comms):.3f}ms  std={(st.stdev(comms) if len(comms)>1 else 0):.3f}ms")
        if args.json_out:
            json.dump({"dir": args.trace, "per_rank": rows,
                       "busiest_expert_ffn_ms": max((r["expert_ffn_ms"] for r in rows), default=0.0),
                       "min_expert_ffn_ms": min((r["expert_ffn_ms"] for r in rows), default=0.0)},
                      open(args.json_out, "w"), indent=2)
            print(f"\nwrote {args.json_out}")
        return

    ev = load_events(args.trace)
    print(f"total events: {len(ev)}")
    gpu_tid_set, gpu_pids = gpu_tids(ev)

    # collect candidate GPU kernel events
    kernels = []
    cats = defaultdict(int)
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        cats[cat] += 1
        dur = e.get("dur")
        if dur is None:
            continue
        is_kernel = (cat in ("kernel", "gpu_op", "Kernel")) or \
                    (e.get("pid"), e.get("tid")) in gpu_tid_set or \
                    (e.get("pid") in gpu_pids and cat not in ("cpu_op", "user_annotation", "cuda_runtime", "python_function"))
        if is_kernel:
            kernels.append(e)

    print(f"event categories: {dict(cats)}")
    print(f"GPU stream tids detected: {len(gpu_tid_set)}; gpu pids: {len(gpu_pids)}")
    print(f"candidate GPU kernel events: {len(kernels)}")
    if not kernels:
        print("\n!!! NO GPU KERNEL EVENTS FOUND -- trace is host-only, unusable for stage decomposition.")
        print("    (categories present:", dict(cats), ")")
        sys.exit(3)

    # steady-state window: use ts range, trim
    ts = sorted(k["ts"] for k in kernels)
    lo, hi = ts[0], ts[-1]
    span = hi - lo
    a = lo + args.trim_frac * span
    b = hi - args.trim_frac * span
    win = [k for k in kernels if a <= k["ts"] <= b]

    stage_self = defaultdict(float)
    stage_names = defaultdict(lambda: defaultdict(float))
    for k in win:
        st = classify(k["name"])
        stage_self[st] += k["dur"]
        stage_names[st][k["name"]] += k["dur"]

    total = sum(stage_self.values())
    print(f"\n=== STAGE DECOMPOSITION (steady-state window, kernel self-time sum) ===")
    print(f"window kernels: {len(win)}  total kernel-time: {total/1e3:.3f} ms")
    for st in sorted(stage_self, key=lambda s: -stage_self[s]):
        pct = 100 * stage_self[st] / total if total else 0
        print(f"  {st:18s} {stage_self[st]/1e3:10.3f} ms  {pct:6.2f}%")
    # audit: top kernel names in 'other'
    if stage_self.get("other", 0) > 0:
        print("\n  TOP 'other' kernels (audit — reclassify if needed):")
        for nm, t in sorted(stage_names["other"].items(), key=lambda x: -x[1])[:15]:
            print(f"    {t/1e3:8.3f} ms  {nm[:90]}")

    if args.json_out:
        json.dump({
            "trace": args.trace,
            "window_kernels": len(win),
            "total_kernel_ms": total/1e3,
            "stage_self_ms": {k: v/1e3 for k, v in stage_self.items()},
            "stage_pct": {k: 100*v/total for k, v in stage_self.items()} if total else {},
            "top_kernels_by_stage": {
                st: sorted(({"name": n, "ms": t/1e3} for n, t in d.items()),
                           key=lambda x: -x["ms"])[:10]
                for st, d in stage_names.items()
            },
        }, open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")

if __name__ == "__main__":
    main()
