import csv, re, os, shutil, glob
from collections import defaultdict
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
APPLY=os.environ.get("APPLY","0")=="1"
def f(x):
    try: return float(x)
    except: return 0.0
rows=[r for r in csv.DictReader(open(CSV)) if r["limit"].strip()=="inf"]
# (cell,iter) that have inf accuracy
def itof(r):
    try: return int(float(r.get("bench_iteration") or r.get("train_iters")))
    except: return None
inf_acc=set()
pts=defaultdict(list)  # category -> [(cell,iter,cp,acc)]
for r in rows:
    it=itof(r);
    if it is None: continue
    if f(r.get("benchmark_avg"))>0: inf_acc.add((r["run_name"],it))
    cp=f(r.get("cp_critical_eval")) or f(r.get("eval_crit_path"))
    if f(r.get("benchmark_avg"))>0 and cp>1000:
        pts[(r.get("category") or "").strip()].append((r["run_name"],it,cp,f(r.get("benchmark_avg"))))
# frontier keep-set: non-dominated (cp_red high == cp low, acc high) per category
keep=set()
for cat in ("aux_only","rl_only","rl+aux","pretrained"):
    P=pts.get(cat,[])
    for i,(n,it,cp,a) in enumerate(P):
        dom=False
        for j,(n2,it2,cp2,a2) in enumerate(P):
            if i==j: continue
            if cp2<=cp and a2>=a and (cp2<cp or a2>a): dom=True; break
        if not dom: keep.add((n,it))
print("inf-evaluated (cell,iter):",len(inf_acc),"| frontier keep-set:",len(keep))
# scan HF dirs
todelete=[]; protected_frontier=0; kept_pending=0
for hf in glob.glob(ROOT+"/*/checkpoint/*/hf_converted_iter*"):
    m=re.search(r"hf_converted_iter0*([0-9]+)",hf)
    if not m: continue
    it=int(m.group(1)); cell=hf.split("/output_router_finetuning/")[1].split("/")[0]
    key=(cell,it)
    if key not in inf_acc:        # not yet inf-evaluated -> KEEP (pending eval)
        kept_pending+=1; continue
    if key in keep:               # on frontier -> KEEP (for walltime)
        protected_frontier+=1; continue
    todelete.append(hf)
def dsize(p):
    try: return sum(os.path.getsize(os.path.join(d,fn)) for d,_,fs in os.walk(p) for fn in fs)
    except: return 0
print("HF dirs: to-delete=%d | kept(frontier)=%d | kept(pending-eval)=%d"%(len(todelete),protected_frontier,kept_pending))
# SAFETY: never touch anything that isn't an hf_converted_iter dir
assert all("hf_converted_iter" in p for p in todelete)
if APPLY:
    freed=0
    for p in todelete:
        sz=dsize(p); shutil.rmtree(p); freed+=sz
    print("DELETED %d HF dirs, freed %.2f TB (distcp untouched)"%(len(todelete),freed/1e12))
else:
    tot=sum(dsize(p) for p in todelete[:9999])
    print("DRY-RUN would free ~%.2f TB across %d dirs"%(tot/1e12,len(todelete)))
    for p in todelete[:5]: print("  e.g.",p.split("/output_router_finetuning/")[1][:80])
