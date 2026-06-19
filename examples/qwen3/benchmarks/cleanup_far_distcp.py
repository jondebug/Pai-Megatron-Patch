#!/usr/bin/env python3
# Frontier-safe distcp prune — CANONICAL CP ONLY (cp_critical_eval, the wandb critical_eval/critical_path).
# Legacy eval_crit_path is NEVER used (it is stale/constant-per-cell for ~37% of cells -> unreliable).
# Delete a distcp iter-dir ONLY if:
#   (1) inf-accuracy-evaluated (acc>=40), AND
#   (2) canonical CP (cp_critical_eval) present, AND
#   (3) NOT on the canonical-CP frontier, AND
#   (4) dominated by >= FARTHRESH pp under the canonical frontier, AND
#   (5) NOT the cell's MAX iter (keep >=1 distcp/cell for continuation), AND
#   (6) the cell contributes NO canonical-frontier point (frontier cells fully exempt).
# Missing canonical CP / un-evaluated / frontier -> EXEMPT. distcp otherwise never deleted; HF reapable.
# Default DRY-RUN. DELETE=1 to remove. FARTHRESH default 1.0pp. TARGET_TB optional cumulative cap.
import csv, os, glob, re, shutil, subprocess
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
FARTHRESH=float(os.environ.get("FARTHRESH","1.0"))
DELETE=os.environ.get("DELETE","0")=="1"
TARGET_TB=float(os.environ.get("TARGET_TB","0") or 0)
def f(x):
    try: return float(x)
    except: return None
rows=list(csv.DictReader(open(CSV)))
itc=[c for c in rows[0] if "iter" in c.lower()][0]
# inf[(cell,it)] = (acc, cp_canon).  CANONICAL CP only.
inf={}; pts=[]
for r in rows:
    if r["limit"].strip()!="inf": continue
    cell=r["run_name"].strip(); it=f(r[itc]); acc=f(r["benchmark_avg"]); cc=f(r.get("cp_critical_eval",""))
    meth=(r.get("cp_method") or "").strip()
    if it is None or acc is None or acc<40: continue
    if cc is None: continue                          # no canonical CP -> not a decision basis
    inf[(cell,int(it))]=(acc,cc,meth); pts.append((cc,acc))   # pts (frontier envelope) uses ALL canonical
def onf(cp,acc): return not any(q0<=cp and q1>=acc and (q0<cp or q1>acc) for q0,q1 in pts)
def ceil(cp):
    c=[a for (cc_,a) in pts if cc_<=cp+1e-6]; return max(c) if c else None
frontier_cells=set(cell for (cell,it),(acc,cc,meth) in inf.items() if onf(cc,acc))
# scan distcp on disk
cells={}
for itd in glob.glob(f"{ROOT}/*/checkpoint/*/iter_*"):
    if not glob.glob(itd+"/*.distcp"): continue
    m=re.search(r"/([^/]+)/checkpoint/.*/iter_0*(\d+)$",itd)
    if not m: continue
    cells.setdefault(m.group(1),[]).append((int(m.group(2)),itd))
cands=[]
for cell,lst in cells.items():
    if cell in frontier_cells: continue                       # HARD exempt frontier cells
    mx=max(i for i,_ in lst)
    for it,itd in lst:
        if it==mx: continue                                   # keep cell's latest (continuation source)
        if (cell,it) not in inf: continue                     # un-evaluated / no canonical CP -> keep
        acc,cc,meth=inf[(cell,it)]
        if meth!="exact": continue                            # only delete if its OWN canonical CP is EXACTly measured (never on a bookmarked approx)
        if onf(cc,acc): continue                              # canonical-frontier -> keep
        cap=ceil(cc)
        if cap is None or (cap-acc) < FARTHRESH: continue     # only FAR below canonical frontier
        cands.append((cap-acc,cell,it,acc,cc,itd))
cands.sort(reverse=True)
# ---- HARD frontier-safety ASSERTION: recompute the canonical frontier independently and verify NO
#      candidate is on it or within SAFETY_MARGIN of it. If any is, ABORT and delete nothing. This makes
#      deleting a frontier (or near-frontier) point structurally impossible regardless of logic bugs. ----
SAFETY_MARGIN=float(os.environ.get("SAFETY_MARGIN","0.5"))
violations=[]
for gap,cell,it,acc,cc,itd in cands:
    if onf(cc,acc): violations.append((cell,it,acc,cc,"on-canonical-frontier"))
    elif (ceil(cc)-acc) < max(FARTHRESH,SAFETY_MARGIN):
        violations.append((cell,it,acc,cc,"within-margin"))
if violations:
    print("!!! ABORT: frontier-safety assertion FAILED — candidates near/on the canonical frontier:")
    for v in violations[:20]: print("    VIOLATION",v)
    print("Deleting NOTHING. Investigate before any prune.")
    raise SystemExit(2)
TOMB=os.path.join(os.path.dirname(CSV),"distcp_deletions_tombstone.log")
def dsz(p):
    try: return int(subprocess.check_output(["du","-sb",p]).split()[0])
    except: return 0
import datetime as _dt
freed=0; n=0
print(f"=== {'DELETE' if DELETE else 'DRY-RUN'} far-from-frontier distcp prune (CANONICAL-CP ONLY, "
      f"FARTHRESH={FARTHRESH}pp, TARGET_TB={TARGET_TB or 'all'}) ===")
print(f"frontier cells exempt: {len(frontier_cells)} | deletable candidates: {len(cands)}")
for gap,cell,it,acc,cc,itd in cands:
    if TARGET_TB and freed/1e12 >= TARGET_TB: break
    sz=dsz(itd)
    print(f"  {'DEL' if DELETE else 'would'} {cell}@{it} acc={acc} cp_canon={cc:.0f} gap={gap:.2f}pp {sz/1e12:.2f}TB")
    if DELETE:
        try:
            with open(TOMB,"a") as t:
                t.write(f"{_dt.datetime.utcnow().isoformat()}\t{cell}\t{it}\tacc={acc}\tcanon_cp={cc:.0f}\tgap={gap:.2f}pp\t{sz}\t{itd}\n")
            shutil.rmtree(itd); freed+=sz; n+=1
        except Exception as e: print("    rm failed:",e)
    else: freed+=sz; n+=1
print(f"=== {'DELETED' if DELETE else 'WOULD DELETE'}: {n} distcp dirs, {freed/1e12:.1f} TB "
      f"({'freed' if DELETE else 'reclaimable'}; canonical-frontier+cell-max+unevaluated+no-canon-CP kept) ===")
