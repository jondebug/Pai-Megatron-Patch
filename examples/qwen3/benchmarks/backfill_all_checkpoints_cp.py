#!/usr/bin/env python3
# Ensure EVERY on-disk 235B checkpoint has a CSV row carrying a canonical CP (cp_critical_eval):
#  - canonical CP = wandb critical_eval/critical_path, by LINEAR INTERPOLATION between the two
#    bracketing eval-grid points (exact if within <=40 iters; nearest/extrapolated beyond the range).
#  - existing rows: (re)fill cp_critical_eval with the interpolated value + set cp_method
#    (exact|interp|extrap). Legacy eval_crit_path is never read/written.
#  - checkpoints with NO row: add a stub row (limit="cponly", comments="cp_stub") so it's represented
#    without polluting the inf-frontier; stubs are dropped automatically once a real eval row exists.
# DRY-RUN by default; APPLY=1 writes (backup .bak_cpbackfill).
import wandb, csv, os, glob, re, tempfile
from collections import defaultdict
api=wandb.Api(timeout=90)
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
APPLY=os.environ.get("APPLY","0")=="1"
NEW="cp_critical_eval"; METH="cp_method"
def f(x):
    try: return float(x)
    except: return None
rows=list(csv.DictReader(open(CSV))); hdr=list(rows[0].keys())
for c in (NEW,METH):
    if c not in hdr: hdr.append(c)
for r in rows: r.setdefault(NEW,""); r.setdefault(METH,"")
itcol=[c for c in hdr if c.lower()=="bench_iteration"][0] if any(c.lower()=="bench_iteration" for c in hdr) else "bench_iteration"
def gen(n):
    m=re.match(r"235bv[0-9]+[a-z]*",n); return m.group(0) if m else None
# on-disk complete checkpoints
disk=set()
for itd in glob.glob(ROOT+"/235bv*/checkpoint/*/iter_*"):
    if not glob.glob(itd+"/*.distcp"): continue
    m=re.search(r"/(235bv[^/]+)/checkpoint/.*/iter_0*(\d+)$",itd)
    if m: disk.add((m.group(1),int(m.group(2))))
# wandb canonical history per cell
needed=set(c for c,_ in disk)|set(r["run_name"].strip() for r in rows if gen(r["run_name"].strip()))
runs_by_name=defaultdict(list)
for r in api.runs("qwen3-router-training"):
    if r.name in needed: runs_by_name[r.name].append(r)
cache={}
def hist(name):
    if name in cache: return cache[name]
    pts={}
    for run in runs_by_name.get(name,[]):
        try: h=run.history(keys=["critical_eval/critical_path","iteration"],pandas=False,samples=5000)
        except Exception: continue
        for p in h:
            cp=p.get("critical_eval/critical_path"); it=p.get("iteration")
            if cp is not None and it is not None: pts[int(it)]=float(cp)
    cache[name]=sorted(pts.items()); return cache[name]
def canon(cell,it):
    pts=hist(cell)
    if not pts: return None,None
    # exact
    nit,ncp=min(pts,key=lambda t:abs(t[0]-it))
    if abs(nit-it)<=40: return round(ncp,1),"exact"
    # interpolate between bracketing points
    lo=[p for p in pts if p[0]<=it]; hi=[p for p in pts if p[0]>=it]
    if lo and hi:
        (i0,c0)=lo[-1]; (i1,c1)=hi[0]
        if i1==i0: return round(c0,1),"exact"
        cp=c0+(c1-c0)*(it-i0)/(i1-i0); return round(cp,1),"interp"
    return round(ncp,1),"extrap"   # beyond the logged range -> nearest endpoint
# index existing rows by (cell,iter) and track real (non-stub) rows
def isstub(r): return (r.get("comments") or "").strip()=="cp_stub"
real_keys=set();
for r in rows:
    c=r["run_name"].strip()
    try: it=int(float(r.get(itcol) or r.get("train_iters")))
    except: continue
    if not isstub(r): real_keys.add((c,it))
# (1) refresh canonical CP on every 235B row; drop stubs that now have a real row
out=[]; updated=0; dropped=0; nowandb=0
for r in rows:
    c=r["run_name"].strip(); g=gen(c)
    try: it=int(float(r.get(itcol) or r.get("train_iters")))
    except: it=None
    if isstub(r) and it is not None and (c,it) in real_keys:
        dropped+=1; continue   # real eval row now exists -> stub no longer needed
    if g and it is not None:
        cp,meth=canon(c,it)
        if cp is not None:
            if str(r.get(NEW,""))!=str(cp): updated+=1
            r[NEW]=cp; r[METH]=meth
        else:
            nowandb+=1
    out.append(r)
rows=out
# (2) add stub rows for on-disk checkpoints with no row at all
have=set()
for r in rows:
    c=r["run_name"].strip()
    try: have.add((c,int(float(r.get(itcol) or r.get("train_iters")))))
    except: pass
added=0; added_nowandb=0
for (cell,it) in sorted(disk):
    if (cell,it) in have: continue
    cp,meth=canon(cell,it)
    stub={k:"" for k in hdr}
    stub["run_name"]=cell; stub[itcol]=str(it); stub["train_iters"]=str(it)
    stub["limit"]="cponly"; stub["comments"]="cp_stub"
    if cp is not None: stub[NEW]=cp; stub[METH]=meth; added+=1
    else: stub[METH]="no_wandb"; added_nowandb+=1
    rows.append(stub)
print("=== backfill canonical CP for ALL checkpoints (interp) ===")
print("on-disk complete checkpoints: %d"%len(disk))
print("existing rows: canonical refreshed/updated=%d | no-wandb=%d | stubs dropped(now real)=%d"%(updated,nowandb,dropped))
print("stub rows ADDED for un-rowed checkpoints: %d (of which no-wandb-canonical=%d)"%(added,added_nowandb))
# method distribution over 235B rows
from collections import Counter
mc=Counter(r.get(METH,"") for r in rows if gen(r["run_name"].strip()))
print("cp_method distribution (235B rows):",dict(mc))
if APPLY:
    os.system("cp %s %s.bak_cpbackfill"%(CSV,CSV))
    fd,tmp=tempfile.mkstemp(dir=os.path.dirname(CSV),suffix=".csv"); os.close(fd)
    with open(tmp,"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=hdr,extrasaction="ignore"); w.writeheader(); w.writerows(rows); fo.flush(); os.fsync(fo.fileno())
    os.replace(tmp,CSV); print("WROTE CSV (backup .bak_cpbackfill)")
else:
    print("DRY-RUN (APPLY=1 to write).")
