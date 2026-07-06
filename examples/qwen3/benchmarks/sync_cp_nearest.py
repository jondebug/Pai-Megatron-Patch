#!/usr/bin/env python3
# Pull REAL canonical CP (wandb critical_eval/critical_path) into cp_critical_eval for inf rows
# that lack it. REWRITTEN 2026-07-01: the old version iterated api.runs(project) and silently
# no-op'd (match bug); this version queries per-cell with filters={"display_name": cell} —
# the approach that actually filled 42 rows on 2026-06-30/07-01. Nearest-iter within TOL=400.
# Fill-missing-only; never overwrites an existing canonical value. Prints a summary line always.
import wandb, csv, os, tempfile
B=os.path.dirname(os.path.abspath(__file__))
CSV=os.path.join(B,"benchmark_results.csv")
PROJ="qwen3-router-training"; TOL=400
def f(x):
    try: return float(x)
    except: return None
rows=list(csv.DictReader(open(CSV))); cols=list(rows[0].keys())
need={}
for r in rows:
    if r["limit"].strip()=="inf" and f(r.get("benchmark_avg")) and not f(r.get("cp_critical_eval")):
        it=r.get("bench_iteration") or r.get("train_iters")
        try: need.setdefault(r["run_name"].strip(),set()).add(int(float(it)))
        except: pass
print("sync_cp: %d cells / %d rows need canonical CP"%(len(need),sum(len(v) for v in need.values())))
if not need: raise SystemExit(0)
api=wandb.Api(timeout=120)
cellcp={}
for cell in need:
    if "pretrained" in cell.lower(): continue           # base model has no training run
    pairs=[]
    try:
        for run in api.runs(PROJ, filters={"display_name": cell}):
            try:
                h=run.history(keys=["critical_eval/critical_path","iteration"],pandas=False,samples=10000)
                for p in h:
                    cp=p.get("critical_eval/critical_path"); it=p.get("iteration")
                    if cp is not None and it is not None: pairs.append((int(it),float(cp)))
            except Exception: pass
    except Exception as e:
        print("  query fail %s: %r"%(cell,e)); continue
    if pairs: cellcp[cell]=pairs
filled=0
for r in rows:
    if r["limit"].strip()!="inf" or not f(r.get("benchmark_avg")) or f(r.get("cp_critical_eval")): continue
    pairs=cellcp.get(r["run_name"].strip())
    if not pairs: continue
    it=r.get("bench_iteration") or r.get("train_iters")
    try: it=int(float(it))
    except: continue
    best=min(pairs,key=lambda p:abs(p[0]-it))
    if abs(best[0]-it)<=TOL:
        r["cp_critical_eval"]="%.1f"%best[1]; r["cp_method"]="wandb_real(gap=%d)"%abs(best[0]-it); filled+=1
print("sync_cp: FILLED %d rows (wandb_real)"%filled)
if filled:
    fd,tmp=tempfile.mkstemp(dir=B)
    with os.fdopen(fd,"w",newline="") as o:
        w=csv.DictWriter(o,fieldnames=cols); w.writeheader(); w.writerows(rows)
    os.replace(tmp,CSV)
    print("sync_cp: CSV written")
