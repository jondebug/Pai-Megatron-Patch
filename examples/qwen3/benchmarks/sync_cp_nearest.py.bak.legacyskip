# Attach nearest-grid wandb critical_eval CP to inf rows that lack CP within +/-40 iters.
# BOOKMARK: records cp_method = "nearest_grid(gap=N)" so these approximate points are flagged
# for accurate re-measurement later. Exact (<=40) matches are marked "exact". eval_crit_path untouched.
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
for r in rows:
    r.setdefault(NEWCOL,""); r.setdefault(METHCOL,"")
needed=set(r["run_name"].strip() for r in rows if r["limit"].strip()=="inf")
runs_by_name=defaultdict(list)
for r in api.runs("qwen3-router-training"):
    if r.name in needed: runs_by_name[r.name].append(r)
def merged(name):
    pts={}
    for run in runs_by_name.get(name,[]):
        try: h=run.history(keys=["critical_eval/critical_path","iteration"],pandas=False,samples=5000)
        except Exception: continue
        for p in h:
            cp=p.get("critical_eval/critical_path"); it=p.get("iteration")
            if cp is not None and it is not None: pts[int(it)]=float(cp)
    return sorted(pts.items())
def nearest(pts,target):
    if not pts: return None,None
    it,cp=min(pts,key=lambda t:abs(t[0]-target))
    return cp,abs(it-target)
cache={}; exact=0; approx=0; nomatch=0
for r in rows:
    if r["limit"].strip()!="inf": continue
    cell=r["run_name"].strip()
    it=r.get("bench_iteration") or r.get("train_iters")
    try: it=int(float(it))
    except: continue
    have=f(r.get("eval_crit_path")) or f(r.get(NEWCOL))
    if have and have>1000:    # already has CP -> mark exact if within 40 of a grid pt, else leave
        if cell not in cache: cache[cell]=merged(cell)
        cp,gap=nearest(cache[cell],it)
        if gap is not None and gap<=40 and not (r.get(METHCOL) or "").strip(): r[METHCOL]="exact"
        continue
    if cell not in cache: cache[cell]=merged(cell)
    cp,gap=nearest(cache[cell],it)
    if cp is None: nomatch+=1; continue
    r[NEWCOL]=round(cp,1)
    if gap<=40: r[METHCOL]="exact"; exact+=1
    else: r[METHCOL]="nearest_grid(gap=%d)"%gap; approx+=1
print("filled exact:",exact,"| filled APPROX(bookmarked):",approx,"| no wandb match:",nomatch)
if APPLY and (exact+approx)>0:
    os.system("cp %s %s.bak_nearest"%(CSV,CSV))
    fd,tmp=tempfile.mkstemp(dir=os.path.dirname(CSV),suffix=".csv"); os.close(fd)
    with open(tmp,"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=hdr); w.writeheader(); w.writerows(rows); fo.flush(); os.fsync(fo.fileno())
    os.replace(tmp,CSV); print("WROTE CSV (+%d CP, %d bookmarked approx); backup .bak_nearest"%(exact+approx,approx))
else: print("DRY-RUN")
