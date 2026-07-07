"""Wall-clock exposure split per bench window: compute-busy / exposed-comms / idle.

For each worker sqlite: find the two dense AR segments (bench A=pre, B=cell) as in
analyze_ladder_ffn.py; within each, isolate the STEADY portion (last sub-burst after
capture/warmup, found by splitting at >2s AR gaps); then per GPU compute interval
unions: compute kernels vs nccl kernels; exposed_nccl = nccl-union minus compute-union.
Percentages are of the steady-burst wall time. Usage: python analyze_exposure.py <dirs>
"""
import sqlite3, sys, json, os
from collections import defaultdict

def union_len(intervals, clip_lo, clip_hi, subtract=None):
    """Total length of union of intervals within [clip_lo, clip_hi], optionally
    minus the union of `subtract` intervals."""
    ivs = sorted((max(s, clip_lo), min(e, clip_hi)) for s, e in intervals
                 if e > clip_lo and s < clip_hi)
    merged = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    if subtract is None:
        return sum(e - s for s, e in merged)
    sub = sorted((max(s, clip_lo), min(e, clip_hi)) for s, e in subtract
                 if e > clip_lo and s < clip_hi)
    smerged = []
    for s, e in sub:
        if smerged and s <= smerged[-1][1]:
            smerged[-1][1] = max(smerged[-1][1], e)
        else:
            smerged.append([s, e])
    total, j = 0, 0
    for s, e in merged:
        cur = s
        while j < len(smerged) and smerged[j][1] <= cur:
            j += 1
        k = j
        while k < len(smerged) and smerged[k][0] < e:
            if smerged[k][0] > cur:
                total += smerged[k][0] - cur
            cur = max(cur, smerged[k][1])
            k += 1
        if cur < e:
            total += e - cur
    return total

def analyze(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    rows = conn.execute(
        "SELECT k.start, k.end, s.value, k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName = s.id").fetchall()
    conn.close()
    ar = sorted(r[0] for r in rows if "nccl" in r[2].lower() and "allreduce" in r[2].lower())
    if len(ar) < 100:
        return {"error": "too few AR"}
    # dense segments (benches)
    segs = [[ar[0], ar[0], 1]]
    for t in ar[1:]:
        if t - segs[-1][1] > 60e9:
            segs.append([t, t, 1])
        else:
            segs[-1][1] = t; segs[-1][2] += 1
    dense = sorted(segs, key=lambda s: -s[2])[:2]
    dense.sort(key=lambda s: s[0])
    out = {}
    for label, (lo, hi, n) in zip("AB", dense):
        # steady burst = last sub-burst (>=3s) after splitting at >2s AR gaps
        sub = [[lo, lo]]
        for t in ar:
            if not (lo <= t <= hi):
                continue
            if t - sub[-1][1] > 2e9:
                sub.append([t, t])
            else:
                sub[-1][1] = t
        steady = [s for s in sub if s[1] - s[0] > 3e9]
        if not steady:
            out[label] = {"error": "no steady burst"}
            continue
        s_lo, s_hi = steady[-1]
        wall = s_hi - s_lo
        per_gpu = {}
        by_dev = defaultdict(lambda: ([], []))  # dev -> (compute_ivs, nccl_ivs)
        for start, end, name, dev in rows:
            if end < s_lo or start > s_hi:
                continue
            if "nccl" in name.lower():
                by_dev[dev][1].append((start, end))
            else:
                by_dev[dev][0].append((start, end))
        for dev, (comp, nccl) in by_dev.items():
            c = union_len(comp, s_lo, s_hi)
            x = union_len(nccl, s_lo, s_hi, subtract=comp)
            per_gpu[str(dev)] = {
                "compute_pct": round(c / wall * 100, 1),
                "exposed_comms_pct": round(x / wall * 100, 1),
                "idle_pct": round((wall - c - x) / wall * 100, 1),
            }
        out[label] = {"steady_wall_s": wall / 1e9, "per_gpu": per_gpu}
    return out

def main():
    results = {}
    for job_dir in sys.argv[1:]:
        job = os.path.basename(job_dir.rstrip("/"))
        results[job] = {}
        for f in sorted(os.listdir(job_dir)):
            if f.startswith("worker_") and f.endswith(".sqlite"):
                r = analyze(os.path.join(job_dir, f))
                results[job][f.replace(".sqlite", "")] = r
                for lab in "AB":
                    if lab in r and "per_gpu" in r[lab]:
                        g = list(r[lab]["per_gpu"].values())
                        cp = sum(x["compute_pct"] for x in g) / len(g)
                        xp = sum(x["exposed_comms_pct"] for x in g) / len(g)
                        ip = sum(x["idle_pct"] for x in g) / len(g)
                        print(f"{job}/{f} {lab}: wall={r[lab]['steady_wall_s']:.0f}s "
                              f"compute={cp:.0f}% exposed_comms={xp:.0f}% idle={ip:.0f}%")
                break  # first worker per job is representative; drop for full run
    with open(os.environ.get("OUT_JSON", "/tmp/exposure.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("EXPOSURE_DONE")

if __name__ == "__main__":
    main()
