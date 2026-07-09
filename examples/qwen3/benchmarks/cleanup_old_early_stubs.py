#!/usr/bin/env python3
# Delete the N (default 50) LEAST-helpful un-evaluated early checkpoints:
#   - iter < MAXITER (default 1000),
#   - NOT modified in the last 7 days (mtime older than 7d -> "not from the last week"),
#   - un-evaluated (no real benchmark_avg>=40 for this cell@iter),
#   - rank by FARTHEST from frontier (gap = best_real_acc_at(cp<=) - pred_acc), descending; take top N.
# Safety guards: canonical CP + pred_acc must exist; never the cell's latest on-disk ckpt (continuation
# source); never a cell with an in-flight squeue job; never a point on the frontier. Tombstone (STUB_OLD).
# DRY-RUN default; DELETE=1 writes.
import csv, os, glob, re, shutil, subprocess, time, datetime as dt
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
TOMB=os.path.join(os.path.dirname(CSV),"distcp_deletions_tombstone.log")
MAXITER=int(os.environ.get("MAXITER","1000")); N=int(os.environ.get("N","50")); AGEDAYS=int(os.environ.get("AGEDAYS","7"))
MINGAP=float(os.environ.get("MINGAP","2.0"))   # only delete points at least this far below frontier (protect near-frontier)
DELETE=os.environ.get("DELETE","0")=="1"
now=time.time(); age_cut=now-AGEDAYS*86400
def f(x):
    try: return float(x)
    except: return None
rows=list(csv.DictReader(open(CSV)))
# real-evaluated canonical frontier
pts=[]
for r in rows:
    if r["limit"].strip()!="inf" or not r["run_name"].startswith("235bv"): continue
    a=f(r.get("benchmark_avg")); c=f(r.get("cp_critical_eval"))
    if a is not None and a>=40 and c is not None: pts.append((c,a))
def onf(cp,acc): return not any(q0<=cp and q1>=acc and (q0<cp or q1>acc) for q0,q1 in pts)
def ceil(cp):
    cc=[a for c,a in pts if c<=cp+1e-6]; return max(cc) if cc else None
# cells that contribute ANY real-eval frontier point -> exempt their WHOLE trace (incl early intermediates)
frontier_cells=set()
for r in rows:
    if r["limit"].strip()!="inf" or not r["run_name"].startswith("235bv"): continue
    a=f(r.get("benchmark_avg")); c=f(r.get("cp_critical_eval"))
    if a and a>=40 and c and onf(c,a): frontier_cells.add(r["run_name"].strip())
# per (cell,iter): real acc?, pred_acc, canonical cp
realacc=set(); pa={}; cp={}
for r in rows:
    if not r["run_name"].startswith("235bv"): continue
    try: it=int(float(r.get("bench_iteration") or r.get("train_iters")))
    except: continue
    k=(r["run_name"].strip(),it)
    if f(r.get("benchmark_avg")) and f(r.get("benchmark_avg"))>=40: realacc.add(k)
    if f(r.get("pred_acc")) is not None: pa[k]=f(r.get("pred_acc"))
    if f(r.get("cp_critical_eval")) is not None: cp[k]=f(r.get("cp_critical_eval"))
# in-flight cells
jobs=subprocess.run("squeue -u jonathanp -h -o '%j'",shell=True,capture_output=True,text=True).stdout
def inflight(cell): return cell in jobs
# on-disk iter dirs + per-cell max
disk={}
for itd in glob.glob(ROOT+"/235bv*/checkpoint/*/iter_*"):
    if not glob.glob(itd+"/*.distcp"): continue
    m=re.search(r"/(235bv[^/]+)/checkpoint/.*/iter_0*(\d+)$",itd)
    if m: disk.setdefault(m.group(1),{})[int(m.group(2))]=itd
cellmax={c:max(v) for c,v in disk.items()}
cands=[]
for cell,iters in disk.items():
    if cell in frontier_cells: continue   # never trim a frontier cell's trace (incl early intermediates)
    mx=cellmax[cell]                       # the cell's LATEST is the continuation/running-write target -> kept (it==mx below);
                                           # old non-latest intermediates of a queued cell ARE safe to delete
    for it,itd in iters.items():
        if it>=MAXITER: continue                       # iter < MAXITER only
        if it==mx: continue                            # never the cell's latest (continuation source)
        k=(cell,it)
        if k in realacc: continue                      # only UN-evaluated
        if k not in pa or k not in cp: continue        # need pred_acc + canonical CP to judge
        try:
            if os.path.getmtime(itd) > age_cut: continue   # modified in last 7d -> keep
        except OSError: continue
        cap=ceil(cp[k])
        if cap is None: continue
        gap=cap-pa[k]
        if gap<MINGAP or onf(cp[k],pa[k]): continue     # must be at least MINGAP below frontier (protect near-frontier)
        cands.append((gap,cell,it,pa[k],cp[k],itd))
cands.sort(reverse=True)                                # farthest-from-frontier first
sel=cands[:N]
def dsz(p):
    try: return int(subprocess.check_output(["du","-sb",p]).split()[0])
    except: return 0
print("=== %s least-helpful old(<%dd not touched) early(<%d) un-evaluated stubs: top %d of %d eligible ==="
      %("DELETE" if DELETE else "DRY-RUN",AGEDAYS,MAXITER,len(sel),len(cands)))
freed=0;n=0
for gap,cell,it,p,c,itd in sel:
    sz=dsz(itd); age=(now-os.path.getmtime(itd))/86400
    print("  %s %s@%d pred_acc=%.1f cp=%.0f gap=%.2fpp age=%.1fd %.2fTB"%("DEL" if DELETE else "would",cell[:40],it,p,c,gap,age,sz/1e12))
    if DELETE:
        with open(TOMB,"a") as t: t.write("%s\t%s\t%d\tpred_acc=%.1f\tcanon_cp=%.0f\tgap=%.2f\tSTUB_OLD\t%d\t%s\n"%(dt.datetime.utcnow().isoformat(),cell,it,p,c,gap,sz,itd))
        try: shutil.rmtree(itd); freed+=sz; n+=1
        except Exception as e: print("   rm failed",e)
    else: freed+=sz; n+=1
print("=== %s: %d dirs, %.1f TB ==="%("DELETED" if DELETE else "WOULD DELETE",n,freed/1e12))
