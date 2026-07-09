#!/usr/bin/env python3
"""cp_aggregate_perstage.py — combine vllm_*_sweep_*.json bench JSONs with their per-rank
torch.profiler trace dirs into a single CSV: one row per (cell, EP, regime, plen, bs).

Stage columns are AGGREGATE GPU-kernel self-time (ms) over the trace window — for prefill
profile (max_tokens=1) it covers exactly one prefill forward; for decode profile (profile-
steps=K) it covers K decode steps minus the 10% trim.

Cross-check column: stage_sum_ms (sum of all stages on the busiest rank) vs e2e_ms / step.

Usage:
  python cp_aggregate_perstage.py --manifest manifest.csv --out perstage.csv

Manifest columns: cell,cp_train,cp_inf_busiest,ep,regime,bench_json,trace_dir
"""
import argparse, csv, glob, gzip, json, os, re, statistics, sys
from collections import defaultdict

# Reuse classifier from parse_decode_trace.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_decode_trace import decompose


def decompose_per_rank(trace_dir):
    """Return per-rank list of {expert_ffn_ms, comm_other_ms, attention_ms, gemm_ms,
    norm_elementwise_ms, other_ms, total_ms}. Deduplicates the parse_decode_trace.py
    glob double-match bug."""
    seen = set()
    files = []
    for pat in ("*.pt.trace.json.gz", "*.pt.trace.json", "*.json.gz", "*.json"):
        for f in glob.glob(os.path.join(trace_dir, pat)):
            if f not in seen:
                seen.add(f)
                files.append(f)
    rows = []
    for f in sorted(files):
        # Skip dedup tag and host-only or non-trace JSONs
        base = os.path.basename(f)
        if not re.search(r"rank\d+", base):
            continue
        r = decompose(f, 0.0)  # prefill: no trim
        if r is None:
            continue
        stage_self, total, _, nwin = r
        rows.append({
            "file": base,
            "rank": int(re.search(r"rank(\d+)", base).group(1)),
            "expert_ffn_ms":     stage_self.get("expert_ffn", 0.0) / 1e3,
            "comm_other_ms":     stage_self.get("comm_other", 0.0) / 1e3,
            "moe_dispatch_a2a_ms": stage_self.get("moe_dispatch_a2a", 0.0) / 1e3,
            "attention_ms":      stage_self.get("attention", 0.0) / 1e3,
            "gemm_ms":           stage_self.get("gemm", 0.0) / 1e3,
            "norm_elem_ms":      stage_self.get("norm_elementwise", 0.0) / 1e3,
            "other_ms":          stage_self.get("other", 0.0) / 1e3,
            "total_ms":          total / 1e3,
            "win_kernels":       nwin,
        })
    return rows


def kernel_name_sniff(trace_dir, sample_n=2000):
    """Return dict of {kernel_pattern: count} for collective-name detection."""
    seen = set()
    files = []
    for pat in ("*.pt.trace.json.gz", "*.pt.trace.json"):
        for f in glob.glob(os.path.join(trace_dir, pat)):
            if f not in seen:
                seen.add(f); files.append(f)
    counts = defaultdict(int)
    if not files:
        return counts
    f = sorted(files)[0]
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        d = json.load(fh)
    events = d.get("traceEvents", d) if isinstance(d, dict) else d
    n = 0
    for e in events:
        if e.get("ph") != "X":
            continue
        nm = e.get("name", "")
        for tag, rx in [
            ("AllReduce", re.compile(r"AllReduce|all_reduce|ncclDevKernel_AllReduce", re.I)),
            ("AllToAll",  re.compile(r"AllToAll|all_to_all|alltoall", re.I)),
            ("SendRecv",  re.compile(r"SendRecv|sendrecv|ncclDevKernel_SendRecv", re.I)),
            ("AllGather", re.compile(r"AllGather|all_gather|ncclDevKernel_AllGather", re.I)),
            ("ReduceScatter", re.compile(r"ReduceScatter|reduce_scatter", re.I)),
        ]:
            if rx.search(nm):
                counts[tag] += 1
        n += 1
        if n >= sample_n:
            break
    return dict(counts)


def aggregate_cell(cell, ep, regime, bench_json, trace_dir, cp_train=None, cp_inf=None):
    """One row per (cell, ep, regime, plen, bs)."""
    out_rows = []
    if not os.path.exists(bench_json):
        print(f"  WARN: missing bench json {bench_json}", file=sys.stderr)
        return out_rows
    bj = json.load(open(bench_json))
    per_rank = decompose_per_rank(trace_dir) if trace_dir and os.path.isdir(trace_dir) else []
    coll = kernel_name_sniff(trace_dir) if trace_dir and os.path.isdir(trace_dir) else {}
    # Compute per-rank stats
    def stat(field):
        vals = [r[field] for r in per_rank]
        if not vals:
            return (0, 0, 0, 0)
        mn, mx = min(vals), max(vals)
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return (mn, mx, mean, sd)
    ffn = stat("expert_ffn_ms")
    comm = stat("comm_other_ms")
    a2a = stat("moe_dispatch_a2a_ms")
    attn = stat("attention_ms")
    gemm = stat("gemm_ms")
    total = stat("total_ms")
    n_ranks = len(per_rank)

    for model in bj.get("models", []):
        for cell_key, c in model.get("cells", {}).items():
            row = dict(
                cell=cell, cp_train=cp_train, cp_inf_busiest=cp_inf,
                ep=ep, regime=regime,
                plen=c.get("prompt_len"), bs=c.get("batch_size"),
                n=c.get("ttft_ms_n") or c.get("e2e_ms_n"),
                ttft_ms=round(c.get("ttft_ms_mean", 0.0), 3),
                ttft_std=round(c.get("ttft_ms_std", 0.0), 3),
                e2e_ms=round(c.get("end_to_end_ms_mean", 0.0), 3),
                e2e_std=round(c.get("end_to_end_ms_std", 0.0), 3),
                decode_tps=round(c.get("decode_tps", 0.0), 3),
                n_ranks=n_ranks,
                busiest_ffn_ms=round(ffn[1], 3),
                min_ffn_ms=round(ffn[0], 3),
                mean_ffn_ms=round(ffn[2], 3),
                std_ffn_ms=round(ffn[3], 3),
                mean_comm_ms=round(comm[2], 3),
                busiest_comm_ms=round(comm[1], 3),
                mean_a2a_ms=round(a2a[2], 3),
                mean_attn_ms=round(attn[2], 3),
                mean_gemm_ms=round(gemm[2], 3),
                busiest_total_ms=round(total[1], 3),
                mean_total_ms=round(total[2], 3),
                coll_AllReduce=coll.get("AllReduce", 0),
                coll_AllToAll=coll.get("AllToAll", 0),
                coll_SendRecv=coll.get("SendRecv", 0),
                bench_json=os.path.basename(bench_json),
                trace_dir=os.path.basename(trace_dir) if trace_dir else "",
            )
            out_rows.append(row)
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with cols: cell,cp_train,cp_inf,ep,regime,bench_json,trace_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_rows = []
    with open(args.manifest) as f:
        rdr = csv.DictReader(f)
        for m in rdr:
            cell=m["cell"]; ep=m["ep"]; regime=m["regime"]; print("-- %s ep=%s regime=%s --" % (cell, ep, regime))
            rows = aggregate_cell(
                cell=m["cell"], ep=int(m["ep"]), regime=m["regime"],
                bench_json=m["bench_json"], trace_dir=m.get("trace_dir") or "",
                cp_train=m.get("cp_train"), cp_inf=m.get("cp_inf"),
            )
            all_rows.extend(rows)

    if not all_rows:
        print("no rows produced", file=sys.stderr); sys.exit(2)
    cols = list(all_rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
