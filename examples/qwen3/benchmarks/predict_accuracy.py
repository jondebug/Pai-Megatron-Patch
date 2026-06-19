#!/usr/bin/env python3
# Predict accuracy for every 235B checkpoint row from lm_loss, into NEW columns (pred_acc, pred_acc_src);
# real benchmark_avg / eval_lm_loss are never modified.
#  - PRIMARY: eval_lm_loss present -> acc ~= a1 + b1*eval_lm_loss  (fit on clean 235B, RMSE~0.4pp)
#  - PROXY:   else use wandb 'train/lm loss' (nearest within +-60 iters) -> acc ~= a2 + b2*train_loss (RMSE~1.8pp)
#  - else: blank.  Both fits are RE-DERIVED from current data each run (self-calibrating).
# DRY-RUN default; APPLY=1 writes (backup .bak_predacc).
import wandb, csv, os, re, tempfile
from collections import defaultdict
api=wandb.Api(timeout=90)
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
APPLY=os.environ.get("APPLY","0")=="1"
def f(x):
    try: return float(x)
    except: return None
def is235(n): return n.startswith("235bv")
rows=list(csv.DictReader(open(CSV))); hdr=list(rows[0].keys())
for c in ("pred_acc","pred_acc_src"):
    if c not in hdr: hdr.append(c)
for r in rows: r.setdefault("pred_acc",""); r.setdefault("pred_acc_src","")
def ols(p):
    n=len(p); sx=sum(x for x,y in p); sy=sum(y for x,y in p); sxx=sum(x*x for x,y in p); sxy=sum(x*y for x,y in p)
    b=(n*sxy-sx*sy)/(n*sxx-sx*sx); a0=(sy-b*sx)/n; res=[y-(a0+b*x) for x,y in p]
    return a0,b,(sum(e*e for e in res)/n)**0.5
def it_of(r):
    try: return int(float(r.get("bench_iteration") or r.get("train_iters")))
    except: return None
# (1) eval-lm_loss fit on clean 235B
ev_pairs=[(f(r["eval_lm_loss"]),f(r["benchmark_avg"])) for r in rows
          if r["limit"].strip()=="inf" and is235(r["run_name"]) and f(r.get("benchmark_avg")) and f(r.get("eval_lm_loss"))
          and f(r["benchmark_avg"])>=55 and 0<f(r["eval_lm_loss"])<6]
a1,b1,rmse1=ols(ev_pairs)
# (2) train-loss proxy: pull wandb train/lm loss per cell
need=set(r["run_name"].strip() for r in rows if is235(r["run_name"]))
runs=defaultdict(list)
for r in api.runs("qwen3-router-training"):
    if r.name in need: runs[r.name].append(r)
cache={}
def tl_hist(name):
    if name in cache: return cache[name]
    pts={}
    for run in runs.get(name,[]):
        try: h=run.history(keys=["train/lm loss","iteration"],pandas=False,samples=5000)
        except Exception: continue
        for p in h:
            l=p.get("train/lm loss"); i=p.get("iteration")
            if l is not None and i is not None: pts[int(i)]=float(l)
    cache[name]=sorted(pts.items()); return cache[name]
def tl_at(cell,it):
    pts=tl_hist(cell)
    if not pts: return None
    ni,nl=min(pts,key=lambda t:abs(t[0]-it))
    return nl if abs(ni-it)<=60 else (None if abs(ni-it)>400 else nl)  # within 60 exact-ish, up to 400 ok, else None
# fit train proxy on evaluated 235B
tp_pairs=[]
for r in rows:
    if r["limit"].strip()!="inf" or not is235(r["run_name"]): continue
    a=f(r.get("benchmark_avg")); i=it_of(r)
    if a is None or a<55 or i is None: continue
    t=tl_at(r["run_name"].strip(),i)
    if t is not None: tp_pairs.append((t,a))
a2,b2,rmse2=ols(tp_pairs)
print("eval-lm_loss fit:  acc = %.3f %+.4f*L  (n=%d RMSE=%.3f)"%(a1,b1,len(ev_pairs),rmse1))
print("train-loss proxy:  acc = %.3f %+.4f*T  (n=%d RMSE=%.3f)"%(a2,b2,len(tp_pairs),rmse2))
# (3) predict for every 235B row
n_eval=n_proxy=n_none=0
for r in rows:
    if not is235(r["run_name"]): continue
    L=f(r.get("eval_lm_loss")); i=it_of(r)
    if L is not None and 1.5<L<6:
        p=a1+b1*L; src="eval_lmloss(rmse%.2f)"%rmse1; n_eval+=1
    else:
        t=tl_at(r["run_name"].strip(),i) if i is not None else None
        if t is not None and 1.5<t<7:
            p=a2+b2*t; src="train_proxy(rmse%.2f)"%rmse2; n_proxy+=1
        else:
            r["pred_acc"]=""; r["pred_acc_src"]="none"; n_none+=1; continue
    r["pred_acc"]=round(max(0,min(100,p)),2); r["pred_acc_src"]=src
print("predicted: eval-lm_loss=%d | train-proxy=%d | none=%d"%(n_eval,n_proxy,n_none))
if APPLY:
    os.system("cp %s %s.bak_predacc"%(CSV,CSV))
    fd,tmp=tempfile.mkstemp(dir=os.path.dirname(CSV),suffix=".csv"); os.close(fd)
    with open(tmp,"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=hdr,extrasaction="ignore"); w.writeheader(); w.writerows(rows); fo.flush(); os.fsync(fo.fileno())
    os.replace(tmp,CSV); print("WROTE CSV (pred_acc/pred_acc_src); backup .bak_predacc")
else: print("DRY-RUN (APPLY=1 to write).")
