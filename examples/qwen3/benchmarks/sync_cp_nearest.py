# Re-pull REAL canonical CP (wandb critical_eval/critical_path, nearest-grid) into cp_critical_eval for
# EVERY inf row, independent of legacy. Overwrites contaminated values (canon copied from legacy) with the
# real wandb value; CLEARS cp_critical_eval where wandb has no canonical (so canonical-only correctly
# excludes it). Legacy eval_crit_path is NEVER read and NEVER written. Exact (<=40 iters) marked "exact",
# else "nearest_grid(gap=N)" (bookmarked for accurate re-measurement), else "no_wandb".
import wandb, csv, os, tempfile
from collections import defaultdict
api=wandb.Api(timeout=90)
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
APPLY=os.environ.get("APPLY","0")=="1"
NEWCOL="cp_critical_eval"; METHCOL="cp_method"
def f(x):
    try: return float(x)
    except: return None
rows=list(csv.DictReader(open(CSV))); hdr=list(rows[0].keys())
for c in (NEWCOL,METHCOL):
    if c not in hdr: hdr.append(c)
for r in rows: r.setdefault(NEWCOL,""); r.setdefault(METHCOL,"")
needed=set(r["run_name"].strip() for r in rows if r["limit"].strip()=="inf")
runs_by_name=defaultdict(list)
for r in api.runs("qwen3-router-training"):
    if r.name in needed: runs_by_name[r.name].append(r)
cache={}
def merged(name):
    if name in cache: return cache[name]
    pts={}
    for run in runs_by_name.get(name,[]):
        try: h=run.history(keys=["critical_eval/critical_path","iteration"],pandas=False,samples=5000)
        except Exception: continue
        for p in h:
            cp=p.get("critical_eval/critical_path"); it=p.get("iteration")
            if cp is not None and it is not None: pts[int(it)]=float(cp)
    cache[name]=sorted(pts.items()); return cache[name]
def nearest(pts,target):
    if not pts: return None,None
    it,cp=min(pts,key=lambda t:abs(t[0]-target))
    return cp,abs(it-target)
exact=approx=nomatch=cleared=overwritten=0
for r in rows:
    if r["limit"].strip()!="inf": continue
    cell=r["run_name"].strip()
    it=r.get("bench_iteration") or r.get("train_iters")
    try: it=int(float(it))
    except: continue
    prev=(r.get(NEWCOL) or "").strip()
    cp,gap=nearest(merged(cell),it)
    if cp is None:                                   # wandb has no canonical -> exclude (clear stale)
        if prev: r[NEWCOL]=""; cleared+=1
        r[METHCOL]="no_wandb"; nomatch+=1; continue
    newv=round(cp,1)
    if prev and abs(f(prev)-newv)>0.5: overwritten+=1
    r[NEWCOL]=newv
    if gap<=40: r[METHCOL]="exact"; exact+=1
    else: r[METHCOL]="nearest_grid(gap=%d)"%gap; approx+=1
print("REAL-canonical re-pull: exact=%d approx(bookmarked)=%d no_wandb(cleared/excluded)=%d overwrote_contaminated=%d cleared_stale=%d"
      %(exact,approx,nomatch,overwritten,cleared))
if APPLY:
    os.system("cp %s %s.bak_canonrepull"%(CSV,CSV))
    fd,tmp=tempfile.mkstemp(dir=os.path.dirname(CSV),suffix=".csv"); os.close(fd)
    with open(tmp,"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=hdr); w.writeheader(); w.writerows(rows); fo.flush(); os.fsync(fo.fileno())
    os.replace(tmp,CSV); print("WROTE CSV (canonical re-pulled from wandb); backup .bak_canonrepull")
else:
    print("DRY-RUN (APPLY=1 to write). Legacy eval_crit_path untouched.")
