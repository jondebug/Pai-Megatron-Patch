import wandb, csv, collections, math
from cell_metadata import parse_meta
api=wandb.Api(timeout=90)
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
BASE=8800.0
def f(x):
    try: return float(x)
    except: return None
rows=[r for r in csv.DictReader(open(CSV)) if r["limit"].strip()=="inf"]
# 1) FIT acc = a + b*lmloss on real (benchmark_avg, eval_lm_loss) pairs
xs=[];ys=[]
for r in rows:
    lm=f(r.get("eval_lm_loss")); a=f(r.get("benchmark_avg"))
    if lm and a and a>0 and 0<lm<8: xs.append(lm); ys.append(a)
n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
a0=my-b*mx
resid=[y-(a0+b*x) for x,y in zip(xs,ys)]
rmse=math.sqrt(sum(e*e for e in resid)/n)
print("FIT acc = %.3f + %.3f*lmloss  (n=%d, RMSE=%.3f pp, max|resid|=%.2f)"%(a0,b,n,rmse,max(abs(e) for e in resid)))
import os as _os
MARGIN=float(_os.environ.get("MARGIN", max(1.5, 3*rmse)))   # delete only if predicted-acc is this far BELOW frontier
print("delete margin (pp below frontier):",round(MARGIN,2))
# real frontier (per category) from evaluated CSV points
pts=collections.defaultdict(list)
for r in rows:
    a=f(r.get("benchmark_avg")); cp=f(r.get("cp_critical_eval")) or f(r.get("eval_crit_path"))
    if a and a>0 and cp and cp>1000:
        pts[(r.get("category") or parse_meta(r["run_name"])["category"])].append((BASE-cp,a))  # (cp_reduction, acc)
def frontier_acc_at(cat, red):
    # best acc among frontier points with cp_reduction >= red (monotone upper envelope)
    cand=[a for (rr,a) in pts.get(cat,[]) if rr>=red-1e-9]
    return max(cand) if cand else (max(a for _,a in pts.get(cat,[])) if pts.get(cat) else 0)
# 2) NRT checkpoints
nrt=[]
for ln in open("/tmp/nrt_ckpts.csv"):
    p=ln.strip().split(",")
    if len(p)==3 and not p[0].startswith("235bbase"): nrt.append((p[0],int(p[1]),int(p[2])))
cells=sorted(set(c for c,_,_ in nrt))
runs=collections.defaultdict(list)
for r in api.runs("qwen3-router-training"):
    if r.name in cells: runs[r.name].append(r)
def grid(cell):
    pts={}
    for run in runs.get(cell,[]):
        try: h=run.history(keys=["critical_eval/critical_path","critical_eval/lm_loss","iteration"],pandas=False,samples=5000)
        except Exception: continue
        for x in h:
            cp=x.get("critical_eval/critical_path"); lm=x.get("critical_eval/lm_loss"); it=x.get("iteration")
            if cp is not None and lm is not None and it is not None: pts[int(it)]=(float(cp),float(lm))
    return sorted(pts.items())
def nearest(g,t):
    if not g: return None,None,None
    it,(cp,lm)=min(g,key=lambda z:abs(z[0]-t)); return cp,lm,abs(it-t)
cache={}; rec=[]
for cell,it,nsh in nrt:
    if cell not in cache: cache[cell]=grid(cell)
    cp,lm,gap=nearest(cache[cell],it)
    if cp is None: rec.append((cell,it,nsh,None,None,None,"NO_WANDB_LMLOSS")); continue
    pa=a0+b*lm; red=BASE-cp; fr=frontier_acc_at(parse_meta(cell)["category"],red)
    below=fr-pa
    verdict="DELETE_far(%.1fpp below)"%below if below>MARGIN else ("KEEP_near(%.1fpp)"%below if below>0 else "KEEP_on/above")
    rec.append((cell,it,nsh,round(cp),round(lm,3),round(pa,2),verdict))
import collections as C
vc=C.Counter(r[6].split("(")[0] for r in rec)
print("=== verdicts ==="); 
for k,v in vc.most_common(): print("  %-22s %d"%(k,v))
# per-cell: if ALL iters DELETE_far -> whole cell deletable
bycell=C.defaultdict(list)
for r in rec: bycell[r[0]].append(r)
print("=== cells where EVERY checkpoint is DELETE_far (whole-cell deletable) ===")
ndel=0
for c,rs in sorted(bycell.items()):
    if all(x[6].startswith("DELETE_far") for x in rs):
        ndel+=1; print("  %-46s iters=%d maxPredAcc=%.2f"%(c[:46],len(rs),max(x[5] for x in rs)))
print("whole-deletable cells:",ndel,"/",len(bycell))
with open("/tmp/nrt_delete_cells.txt","w") as fo:
    for c,rs in sorted(bycell.items()):
        if all(x[6].startswith("DELETE_far") for x in rs): fo.write(c+"\n")
print("wrote deletable list -> /tmp/nrt_delete_cells.txt")
with open("/tmp/nrt_delete_ckpts.txt","w") as fo:
    for r in rec:
        if r[6].startswith("DELETE_far"): fo.write("%s,%d\n"%(r[0],r[1]))
print("wrote per-checkpoint DELETE_far list -> /tmp/nrt_delete_ckpts.txt (%d ckpts)"%sum(1 for r in rec if r[6].startswith("DELETE_far")))
print("=== cells with a KEEP (near/above frontier) - DO NOT DELETE ===")
for c,rs in sorted(bycell.items()):
    keeps=[x for x in rs if x[6].startswith("KEEP")]
    if keeps: print("  %-46s keep_iters=%s bestPredAcc=%.2f"%(c[:46],[x[1] for x in keeps],max(x[5] for x in keeps)))
