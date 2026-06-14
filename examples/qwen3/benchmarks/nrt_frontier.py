import wandb, csv, collections, os
from cell_metadata import parse_meta
api=wandb.Api(timeout=90)
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
BASE=8800.0
def f(x):
    try: return float(x)
    except: return 0.0
# NRT checkpoints (cell,iter,nshards)
nrt=[]
for ln in open("/tmp/nrt_ckpts.csv"):
    p=ln.strip().split(",")
    if len(p)==3: nrt.append((p[0],int(p[1]),int(p[2])))
nrt_cells=sorted(set(c for c,_,_ in nrt))
# CSV: accuracy by (cell,iter); existing evaluated points for combined frontier
rows=[r for r in csv.DictReader(open(CSV)) if r["limit"].strip()=="inf"]
acc_by={}; existing=collections.defaultdict(list)  # cat -> [(cp,acc,label)]
for r in rows:
    try: it=int(float(r.get("bench_iteration") or r.get("train_iters")))
    except: continue
    a=f(r.get("benchmark_avg"))
    if a>0: acc_by[(r["run_name"],it)]=a
    cp=f(r.get("cp_critical_eval")) or f(r.get("eval_crit_path"))
    if a>0 and cp>1000:
        existing[(r.get("category") or parse_meta(r["run_name"])["category"])].append((cp,a,r["run_name"][:30]+"@%d"%it))
# wandb CP grid per NRT cell
runs=collections.defaultdict(list)
for r in api.runs("qwen3-router-training"):
    if r.name in nrt_cells: runs[r.name].append(r)
def grid(cell):
    pts={}
    for run in runs.get(cell,[]):
        try: h=run.history(keys=["critical_eval/critical_path","iteration"],pandas=False,samples=5000)
        except Exception: continue
        for x in h:
            if x.get("critical_eval/critical_path") is not None and x.get("iteration") is not None: pts[int(x["iteration"])]=float(x["critical_eval/critical_path"])
    return sorted(pts.items())
def nearest(g,t):
    if not g: return None,None
    it,cp=min(g,key=lambda z:abs(z[0]-t)); return cp,abs(it-t)
cache={}; out=[]
for cell,it,n in nrt:
    if cell.startswith("235bbase") or n==0: continue
    if cell not in cache: cache[cell]=grid(cell)
    cp,gap=nearest(cache[cell],it)
    a=acc_by.get((cell,it))
    cat=parse_meta(cell)["category"]
    out.append({"cell":cell,"it":it,"n":n,"acc":a,"cp":cp,"gap":gap,"cat":cat})
# combined frontier per category: existing + NRT-with-both; mark NRT non-dominated
def dominated(cp,a,pool):
    cpr=(BASE-cp)
    for (cp2,a2) in pool:
        if (BASE-cp2)>=cpr and a2>=a and ((BASE-cp2)>cpr or a2>a): return True
    return False
print("=== NRT checkpoint frontier classification (assume nothing) ===")
status=collections.Counter()
frontier_hits=[]
for o in out:
    if o["acc"] and o["cp"]:
        pool=[(cp,a) for cp,a,_ in existing.get(o["cat"],[])]+[ (x["cp"],x["acc"]) for x in out if x is not o and x["acc"] and x["cp"] and x["cat"]==o["cat"]]
        dom=dominated(o["cp"],o["acc"],pool)
        o["status"]="DOMINATED" if dom else "FRONTIER"
        if not dom: frontier_hits.append(o)
    elif not o["acc"]:
        o["status"]="NEEDS_EVAL(incomplete<32)" if o["n"]<32 else "NEEDS_EVAL(loadable)"
    else:
        o["status"]="needs_CP"
    status[o["status"]]+=1
for k,v in status.most_common(): print("  %-26s %d"%(k,v))
print("=== NRT FRONTIER candidates (non-dominated, evaluated) ===")
for o in sorted(frontier_hits,key=lambda x:-x["acc"]): print("  acc=%.2f cp=%.0f red=%.1f%% gap=%s %s@%d"%(o["acc"],o["cp"],100*(BASE-o["cp"])/BASE,o["gap"],o["cell"][:42],o["it"]))
print("=== cells entirely NEEDS_EVAL (no evaluated iters) ===")
bycell=collections.defaultdict(list)
for o in out: bycell[o["cell"]].append(o)
for c,os_ in sorted(bycell.items()):
    if all(not x["acc"] for x in os_): print("  %-46s iters_unevaluated=%d (complete>=32: %d)"%(c[:46],len(os_),sum(1 for x in os_ if x["n"]>=32)))
