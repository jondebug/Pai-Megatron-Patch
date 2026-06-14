import wandb, csv, os, tempfile
from collections import defaultdict
api = wandb.Api(timeout=90)
CSV = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
APPLY = os.environ.get("APPLY","0") == "1"
NEWCOL = "cp_critical_eval"
def f(x):
    try: return float(x)
    except: return None

rows = list(csv.DictReader(open(CSV)))
hdr = list(rows[0].keys())
if NEWCOL not in hdr: hdr.append(NEWCOL)
for r in rows: r.setdefault(NEWCOL, "")

needed = set(r["run_name"].strip() for r in rows if r["limit"].strip()=="inf")
runs_by_name = defaultdict(list)
for r in api.runs("qwen3-router-training"):
    if r.name in needed: runs_by_name[r.name].append(r)

def merged_history(name):
    pts = {}
    for run in runs_by_name.get(name, []):
        try:
            h = run.history(keys=["critical_eval/critical_path","eval/num_tokens_on_critical_path","iteration","_step"], pandas=False, samples=5000)
        except Exception:
            continue
        for p in h:
            cp = p.get("critical_eval/critical_path")
            if cp is None: cp = p.get("eval/num_tokens_on_critical_path")
            it = p.get("iteration"); it = it if it is not None else p.get("_step")
            if cp is not None and it is not None: pts[int(it)] = float(cp)
    return sorted(pts.items())

def nearest(pts, target, tol=80):
    if not pts: return None
    it, cp = min(pts, key=lambda t: abs(t[0]-target))
    return cp if abs(it-target) <= tol else None

cache={}; pop=0; nomatch=0; agree=0; differ=0
for r in rows:
    if r["limit"].strip()!="inf": continue
    cell=r["run_name"].strip()
    it=r.get("bench_iteration") or r.get("train_iters")
    try: it=int(float(it))
    except: continue
    if cell not in cache: cache[cell]=merged_history(cell)
    cp=nearest(cache[cell], it)
    if cp is None: nomatch+=1; continue
    r[NEWCOL]=round(cp,1); pop+=1
    ex=f(r["eval_crit_path"])
    if ex and ex>1000:
        if abs(ex-cp)/ex>0.10: differ+=1
        else: agree+=1
print("populated %s: %d | inf rows w/o wandb match: %d | vs legacy eval_crit_path: agree=%d differ=%d" % (NEWCOL,pop,nomatch,agree,differ))
print("(eval_crit_path column left UNTOUCHED)")
if APPLY and pop>0 and len(rows)>100:
    os.system("cp %s %s.bak_newcol" % (CSV,CSV))
    fd,tmp=tempfile.mkstemp(dir=os.path.dirname(CSV),suffix=".csv"); os.close(fd)
    with open(tmp,"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=hdr); w.writeheader(); w.writerows(rows); fo.flush(); os.fsync(fo.fileno())
    os.replace(tmp,CSV); print("WROTE CSV (new column %s, %d populated); backup .bak_newcol" % (NEWCOL,pop))
else:
    print("DRY-RUN (APPLY=1 to write)")
