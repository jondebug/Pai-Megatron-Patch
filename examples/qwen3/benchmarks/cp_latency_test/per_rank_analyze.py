import json, numpy as np, sys

def analyze(path, label):
    d = json.load(open(path))
    rows = d["per_rank"]
    seen = set(); unique = []
    for r in rows:
        k = r["file"]
        if k not in seen:
            seen.add(k); unique.append(r)
    print(f"\n=== {label} (n={len(unique)} unique ranks) ===")
    def stats(arr, ll):
        a = np.array(arr)
        print(f"  {ll:15s}: min={a.min():7.1f}  p25={np.percentile(a,25):7.1f}  p50={np.percentile(a,50):7.1f}  p75={np.percentile(a,75):7.1f}  p90={np.percentile(a,90):7.1f}  max={a.max():7.1f}  mean={a.mean():7.1f}±{a.std():.1f}")
    for f in ["total_ms", "expert_ffn_ms", "comm_other_ms", "attention_ms", "gemm_ms", "a2a_ms"]:
        if any(f in r for r in unique):
            stats([r.get(f, 0) for r in unique], f)
    ffn = np.array([r["expert_ffn_ms"] for r in unique])
    comm = np.array([r["comm_other_ms"] for r in unique])
    print(f"  corr(expert_ffn, comm_other) = {np.corrcoef(ffn, comm)[0,1]:+.3f}  (negative = barrier-wait pattern: low-ffn rank gets long comm-wait)")
    return unique

pre = analyze(sys.argv[1], "PRE")
if len(sys.argv) > 2:
    r15 = analyze(sys.argv[2], "R15")
    if pre and r15:
        # Cross-compare
        print("\n=== CROSS-CELL DELTA ===")
        for f in ["total_ms", "expert_ffn_ms", "comm_other_ms", "attention_ms"]:
            p = np.array([r[f] for r in pre]); r = np.array([rr[f] for rr in r15])
            print(f"  {f:15s}  pre busiest={p.max():7.1f}  r15 busiest={r.max():7.1f}  Δbusiest={r.max()-p.max():+7.1f}ms ({(r.max()-p.max())/p.max()*100:+.1f}%)")
            print(f"  {' ':15s}  pre mean   ={p.mean():7.1f}  r15 mean   ={r.mean():7.1f}  Δmean   ={r.mean()-p.mean():+7.1f}ms ({(r.mean()-p.mean())/p.mean()*100:+.1f}%)")
