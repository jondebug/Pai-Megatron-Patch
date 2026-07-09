#!/usr/bin/env python3
# Clean up cp_stub (un-evaluated) checkpoints that are SAFE to drop:
#   (1) the cell has a LATER on-disk checkpoint (it < cell_max)  -> never the continuation source,
#   (2) the cell has NO active job in the queue (not part of a current/in-flight sweep),
#   (3) FAR from frontier: predicted gap > GAP pp (default 3.0; conservative vs the ~1.8pp pred error),
#   (4) canonical CP present, pred_acc present, and the point is not itself on the frontier.
# Tombstone log (marked STUB). DRY-RUN default; DELETE=1 writes.
import csv, os, glob, re, shutil, subprocess, datetime as dt
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
TOMB=os.path.join(os.path.dirname(CSV),"distcp_deletions_tombstone.log")
GAP=float(os.environ.get("GAP","3.0"))
DELETE=os.environ.get("DELETE","0")=="1"
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
# in-flight cells (any squeue job name containing the cell name)
jobnames=subprocess.run("squeue -u jonathanp -h -o '%j'",shell=True,capture_output=True,text=True).stdout
def in_flight(cell): return cell in jobnames
# on-disk map + cell max
disk={}
for itd in glob.glob(ROOT+"/235bv*/checkpoint/*/iter_*"):
    if not glob.glob(itd+"/*.distcp"): continue
    m=re.search(r"/(235bv[^/]+)/checkpoint/.*/iter_0*(\d+)$",itd)
    if m: disk.setdefault(m.group(1),{})[int(m.group(2))]=itd
cellmax={c:max(v) for c,v in disk.items()}
# stub info
stub_pa={}; stub_cp={}
for r in rows:
    if (r.get("comments") or "").strip()!="cp_stub": continue
    try: it=int(float(r.get("bench_iteration") or 0))
    except: continue
    pa=f(r.get("pred_acc")); c=f(r.get("cp_critical_eval"))
    if pa is not None: stub_pa[(r["run_name"].strip(),it)]=pa
    if c is not None: stub_cp[(r["run_name"].strip(),it)]=c
cands=[]; skip_inflight=skip_latest=skip_near=0
for (cell,it),pa in stub_pa.items():
    c=stub_cp.get((cell,it))
    if c is None: continue
    if cell not in disk or it not in disk[cell]: continue          # not on disk
    if it>=cellmax.get(cell,0): skip_latest+=1; continue           # (1) no later checkpoint -> keep
    if in_flight(cell): skip_inflight+=1; continue                 # (2) current sweep -> keep
    if onf(c,pa): continue
    cap=ceil(c)
    if cap is None or (cap-pa)<=GAP: skip_near+=1; continue         # (3) not far enough -> keep
    cands.append((cap-pa,cell,it,pa,c,disk[cell][it]))
cands.sort(reverse=True)
def dsz(p):
    try: return int(subprocess.check_output(["du","-sb",p]).split()[0])
    except: return 0
print("=== %s STUB cleanup (later-ckpt + not-in-flight + pred-gap>%.1fpp) ==="%("DELETE" if DELETE else "DRY-RUN",GAP))
print("skipped: latest-of-cell=%d in-flight(current sweep)=%d near-frontier(<=%.1f)=%d | candidates=%d"%(skip_latest,skip_inflight,GAP,skip_near,len(cands)))
freed=0;n=0
for gap,cell,it,pa,c,itd in cands:
    sz=dsz(itd)
    print("  %s %s@%d pred_acc=%.1f cp=%.0f gap=%.2fpp %.2fTB"%("DEL" if DELETE else "would",cell[:44],it,pa,c,gap,sz/1e12))
    if DELETE:
        with open(TOMB,"a") as t: t.write("%s\t%s\t%d\tpred_acc=%.1f\tcanon_cp=%.0f\tgap=%.2f\tSTUB\t%d\t%s\n"%(dt.datetime.utcnow().isoformat(),cell,it,pa,c,gap,sz,itd))
        try: shutil.rmtree(itd); freed+=sz; n+=1
        except Exception as e: print("   rm failed",e)
    else: freed+=sz; n+=1
print("=== %s: %d stub dirs, %.1f TB ==="%("DELETED" if DELETE else "WOULD DELETE",n,freed/1e12))
