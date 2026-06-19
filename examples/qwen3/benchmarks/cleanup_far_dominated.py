#!/usr/bin/env python3
# Aggressive-but-safe prune of checkpoints FAR below the canonical frontier.
# Delete a distcp iter ONLY if ALL hold:
#   - inf-evaluated (acc>=40) AND canonical CP present (cp_critical_eval, any method),
#   - NOT on the canonical frontier, AND its cell contributes NO frontier point (frontier cells exempt),
#   - dominated by a margin: EXACT-CP points need gap >= FAR_EXACT (default 2.0pp);
#     BOOKMARKED/interp-CP points need the larger gap >= FAR_APPROX (default 3.0pp) to cover CP error,
#   - it is NOT the cell's last on-disk checkpoint, UNLESS the cell already reached iter>=3000
#     (a finished cell needs no continuation source; a still-training cell keeps its latest).
# Frontier-safety ASSERTION: aborts (deletes nothing) if any candidate is within 0.5pp of the frontier.
# Tombstone log of every deletion. DRY-RUN default; DELETE=1 writes.
import csv, os, glob, re, shutil, subprocess, datetime as dt
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
TOMB=os.path.join(os.path.dirname(CSV),"distcp_deletions_tombstone.log")
FAR_EXACT=float(os.environ.get("FAR_EXACT","2.0"))
FAR_APPROX=float(os.environ.get("FAR_APPROX","3.0"))
DELETE=os.environ.get("DELETE","0")=="1"
def f(x):
    try: return float(x)
    except: return None
rows=list(csv.DictReader(open(CSV)))
inf={}; pts=[]
for r in rows:
    if r["limit"].strip()!="inf": continue
    acc=f(r["benchmark_avg"]); cc=f(r.get("cp_critical_eval","")); meth=(r.get("cp_method") or "").strip()
    try: it=int(float(r.get("bench_iteration") or r.get("train_iters")))
    except: continue
    if acc is None or acc<40 or cc is None: continue
    inf[(r["run_name"].strip(),it)]=(acc,cc,meth); pts.append((cc,acc))
def onf(cp,acc): return not any(q0<=cp and q1>=acc and (q0<cp or q1>acc) for q0,q1 in pts)
def ceil(cp):
    c=[a for cc_,a in pts if cc_<=cp+1e-6]; return max(c) if c else None
frontier_cells=set(cell for (cell,it),(acc,cc,meth) in inf.items() if onf(cc,acc))
disk={}
for itd in glob.glob(ROOT+"/235bv*/checkpoint/*/iter_*"):
    if not glob.glob(itd+"/*.distcp"): continue
    m=re.search(r"/(235bv[^/]+)/checkpoint/.*/iter_0*(\d+)$",itd)
    if m: disk.setdefault(m.group(1),{})[int(m.group(2))]=itd
cands=[]
for cell,iters in disk.items():
    if cell in frontier_cells: continue
    mx=max(iters); done=mx>=3000
    for it,itd in iters.items():
        if (cell,it) not in inf: continue
        acc,cc,meth=inf[(cell,it)]
        if onf(cc,acc): continue
        cap=ceil(cc)
        if cap is None: continue
        gap=cap-acc
        need=FAR_EXACT if meth=="exact" else FAR_APPROX
        if gap < need: continue
        if it==mx and not done: continue          # keep last ckpt of a still-training cell
        cands.append((gap,cell,it,acc,cc,meth,itd,(it==mx)))
cands.sort(reverse=True)
# frontier-safety assertion
bad=[c for c in cands if onf(c[4],c[3]) or (ceil(c[4])-c[3])<0.5]
if bad:
    print("!!! ABORT: %d candidate(s) within 0.5pp of frontier — deleting nothing."%len(bad))
    for b in bad[:10]: print("   VIOLATION",b[1],b[2],"acc=%.2f"%b[3])
    raise SystemExit(2)
def dsz(p):
    try: return int(subprocess.check_output(["du","-sb",p]).split()[0])
    except: return 0
freed=0;n=0
print("=== %s FAR-DOMINATED prune (FAR_EXACT=%.1f, FAR_APPROX=%.1f) | frontier cells exempt: %d ==="
      %("DELETE" if DELETE else "DRY-RUN",FAR_EXACT,FAR_APPROX,len(frontier_cells)))
for gap,cell,it,acc,cc,meth,itd,ismax in cands:
    sz=dsz(itd)
    print("  %s %s@%d acc=%.2f cp=%.0f gap=%.2fpp %s%s %.2fTB"%(
        "DEL" if DELETE else "would",cell[:42],it,acc,cc,gap,meth,"(cell-max,done)" if ismax else "",sz/1e12))
    if DELETE:
        with open(TOMB,"a") as t: t.write("%s\t%s\t%d\tacc=%.2f\tcanon_cp=%.0f\tgap=%.2f\t%s\t%d\t%s\n"%(dt.datetime.utcnow().isoformat(),cell,it,acc,cc,gap,meth,sz,itd))
        try: shutil.rmtree(itd); freed+=sz; n+=1
        except Exception as e: print("   rm failed",e)
    else: freed+=sz; n+=1
print("=== %s: %d dirs, %.1f TB ==="%("DELETED" if DELETE else "WOULD DELETE",n,freed/1e12))
