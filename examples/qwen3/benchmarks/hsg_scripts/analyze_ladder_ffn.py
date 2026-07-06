"""CP-ladder mechanism analysis: per-rank expert-FFN kernel time, pre vs cell.

Each nsys_v3 job traces ONE Ray worker node across the whole session, which
contains TWO benches back-to-back (model A = pretrained, then model B = cell).
We split the trace at the largest gap in NCCL AllReduce activity (model swap =
minutes of load between benches), then sum kernel time by class within each
window. Output: per-rank and per-class totals for window A and B + deltas.

Run inside the HSG lustre venv:  python analyze_ladder_ffn.py <job_dir> ...
"""
import sqlite3
import sys
import json
import os
from collections import defaultdict

MIN_SPLIT_GAP_NS = 30e9   # model swap gap is minutes; require >=30s to split

def classify(name: str) -> str:
    n = name.lower()
    if "nccl" in n:
        return "nccl"
    if any(k in n for k in ("moe", "grouped", "group_gemm", "expert", "topk", "softmax_topk", "fused_experts", "silu")):
        return "moe_ffn"
    if any(k in n for k in ("gemm", "matmul", "nvjet", "cutlass", "s16816", "wgmma")):
        return "gemm"
    if any(k in n for k in ("attn", "attention", "flash", "fmha", "paged")):
        return "attention"
    return "other"

def analyze_worker(sqlite_path: str) -> dict:
    conn = sqlite3.connect(sqlite_path)
    rows = conn.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName = s.id"
    ).fetchall()
    conn.close()
    if not rows:
        return {}

    # Bench-step marker: NCCL AllReduce bf16 kernels (TP-AR each layer)
    ar = sorted(r[0] for r in rows if "nccl" in r[2].lower() and "allreduce" in r[2].lower())
    if len(ar) < 100:
        return {"error": f"only {len(ar)} AllReduce kernels"}

    # Segment AR activity at every gap > 60s (loads/init emit sparse AR bursts;
    # the two benches are the two DENSEST segments). 4142038 lesson: taking the
    # single largest gap picked an init-phase boundary and merged both benches.
    SEG_GAP_NS = 60e9
    segments = [[ar[0], ar[0], 1]]  # [lo, hi, count]
    for t in ar[1:]:
        if t - segments[-1][1] > SEG_GAP_NS:
            segments.append([t, t, 1])
        else:
            segments[-1][1] = t
            segments[-1][2] += 1
    dense = sorted(segments, key=lambda s: -s[2])[:2]
    if len(dense) < 2:
        return {"error": f"only {len(segments)} AR segments; cannot split"}
    dense.sort(key=lambda s: s[0])  # chronological: A = pre, B = cell
    (a_lo, a_hi, a_n), (b_lo, b_hi, b_n) = dense

    out = {"windows": {"A": [a_lo, a_hi], "B": [b_lo, b_hi]},
           "ar_counts": {"A": a_n, "B": b_n},
           "n_segments": len(segments)}
    for label, lo, hi in (("A", a_lo, a_hi), ("B", b_lo, b_hi)):
        by_class = defaultdict(float)
        for start, end, name in rows:
            if lo <= start <= hi:
                by_class[classify(name)] += (end - start) / 1e6  # ms
        out[label] = dict(by_class)
        out[label]["window_s"] = (hi - lo) / 1e9
    return out

def main():
    results = {}
    for job_dir in sys.argv[1:]:
        job = os.path.basename(job_dir.rstrip("/"))
        results[job] = {}
        for f in sorted(os.listdir(job_dir)):
            if f.startswith("worker_") and f.endswith(".sqlite"):
                r = analyze_worker(os.path.join(job_dir, f))
                results[job][f.replace(".sqlite", "")] = r
                err = r.get("error", "")
                if err:
                    print(f"{job}/{f}: {err}")
                else:
                    a, b = r["A"], r["B"]
                    print(f"{job}/{f}: A moe={a.get('moe_ffn',0):.0f}ms gemm={a.get('gemm',0):.0f}ms "
                          f"nccl={a.get('nccl',0):.0f}ms | B moe={b.get('moe_ffn',0):.0f}ms "
                          f"gemm={b.get('gemm',0):.0f}ms nccl={b.get('nccl',0):.0f}ms")
    out_path = os.environ.get("OUT_JSON", "/tmp/ladder_ffn.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"WROTE {out_path}")

if __name__ == "__main__":
    main()
