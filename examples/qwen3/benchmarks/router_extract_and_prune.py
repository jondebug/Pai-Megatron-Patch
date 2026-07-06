#!/usr/bin/env python3
# Router-preserving prune of dominated >=1.5pp checkpoints (runs INSIDE pai-megatron container).
# Target: real inf eval + canonical CP, gap>=GAPMIN(1.5pp), not frontier point, not frontier-cell,
# not active sweeps (v16cgr/v15klcg), cell-latest only if iter==3000.
# Per target: extract model router tensors from distcp -> <iter_dir>/router_weights.pt (+meta),
# VERIFY (>=90 keys, nonzero norms) -> tombstone DOM15_ROUTER_SAVED -> delete shards/large files,
# keep router_weights.pt. DRY-RUN default; DELETE=1 executes. LIMIT=N caps count (pilot: LIMIT=1).
import csv, glob, re, os, sys, shutil, subprocess, datetime as dt
import torch
from torch.distributed.checkpoint import FileSystemReader
import torch.distributed.checkpoint as dcp
B="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks"
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
TOMB=B+"/distcp_deletions_tombstone.log"
GAPMIN=float(os.environ.get("GAPMIN","1.5")); DELETE=os.environ.get("DELETE","0")=="1"
LIMIT=int(os.environ.get("LIMIT","10000"))
ACTIVE=("235bv16cgr","235bv15klcg")
def f(x):
    try: return float(x)
    except: return None
rows=list(csv.DictReader(open(B+"/benchmark_results.csv")))
inf=[r for r in rows if r["limit"].strip()=="inf" and f(r.get("benchmark_avg")) and f(r.get("benchmark_avg"))>=40]
pts=[(f(r["cp_critical_eval"]),f(r["benchmark_avg"])) for r in inf if f(r.get("cp_critical_eval"))]
def onf(cp,acc): return not any(q0<=cp and q1>=acc and (q0<cp or q1>acc) for q0,q1 in pts)
def ceil(cp):
    cc=[a for c,a in pts if c<=cp+1e-6]; return max(cc) if cc else None
fcells=set(); fpoints=set(); acc={}; cpv={}
for r in inf:
    a=f(r["benchmark_avg"]); c=f(r.get("cp_critical_eval"))
    it=r.get("bench_iteration") or r.get("train_iters")
    try: it=int(float(it))
    except: continue
    cell=r["run_name"].strip(); acc[(cell,it)]=a
    if c is not None:
        cpv[(cell,it)]=c
        if onf(c,a): fcells.add(cell); fpoints.add((cell,it))
disk={}
for d in glob.glob(ROOT+"/235bv*/checkpoint/*/iter_*"):
    if len(glob.glob(d+"/*.distcp"))<32: continue
    m=re.search(r"/(235bv[^/]+)/checkpoint/.*/iter_0*([0-9]+)$",d)
    if m: disk.setdefault(m.group(1),{})[int(m.group(2))]=d
cellmax={c:max(v) for c,v in disk.items()}
targets=[]
for cell,iters in disk.items():
    if cell.startswith(ACTIVE) or cell in fcells: continue
    for it,itd in iters.items():
        k=(cell,it)
        if k in fpoints or k not in acc or k not in cpv: continue
        gap=(ceil(cpv[k]) or acc[k])-acc[k]
        if gap<GAPMIN: continue
        if it==cellmax[cell] and it!=3000: continue      # latest deletable only at 3000
        targets.append((gap,cell,it,itd))
targets.sort(reverse=True)
targets=targets[:LIMIT]
print("=== %s: %d targets (gap>=%.1fpp) ==="%("DELETE" if DELETE else "DRY-RUN",len(targets),GAPMIN))
def extract_router(itd):
    r=FileSystemReader(itd); md=r.read_metadata()
    keys=[k for k in md.state_dict_metadata.keys()
          if "router" in k.lower() and "optimizer" not in k.lower() and "_extra_state" not in k.lower()]
    if len(keys)<90: return None,"only %d router keys"%len(keys)
    sd={}
    for k in keys:
        m=md.state_dict_metadata[k]
        try: sd[k]=torch.empty(m.size,dtype=m.properties.dtype)
        except Exception as e: return None,"alloc %s: %r"%(k,e)
    try:
        dcp.load(state_dict=sd,storage_reader=r)   # single-process (no_dist auto)
    except Exception as e:
        return None,"dcp.load: %r"%e
    bad=[k for k,v in sd.items() if not torch.isfinite(v.float()).all() or float(v.float().abs().sum())==0.0]
    if bad: return None,"zero/nonfinite: %s"%bad[:3]
    return sd,None
freed=0; done=0
for gap,cell,it,itd in targets:
    tag="%s@%d gap=%.2f"%(cell[:44],it,gap)
    if not DELETE:
        print("  would %s  (%s)"%(tag,itd[len(ROOT)+1:60])); continue
    sd,err=extract_router(itd)
    if err: print("  SKIP %s — extract failed: %s"%(tag,err)); continue
    out=os.path.join(itd,"router_weights.pt")
    meta={"cell":cell,"iteration":it,"acc":acc[(cell,it)],"cp_canonical":cpv[(cell,it)],
          "gap_pp":gap,"extracted":dt.datetime.utcnow().isoformat(),"source":"distcp","keys":len(sd)}
    torch.save({"meta":meta,"router":{k:v.cpu() for k,v in sd.items()}},out)
    chk=torch.load(out,map_location="cpu")
    if len(chk.get("router",{}))<90: print("  SKIP %s — verify failed"%tag); continue
    try: sz=int(subprocess.check_output(["du","-sb",itd]).split()[0])
    except: sz=0
    with open(TOMB,"a") as t:
        t.write("%s\t%s\t%d\tacc=%.2f\tcanon_cp=%.0f\tgap=%.2f\tDOM15_ROUTER_SAVED\t%d\t%s\n"
                %(dt.datetime.utcnow().isoformat(),cell,it,acc[(cell,it)],cpv[(cell,it)],gap,sz,itd))
    for p in os.listdir(itd):
        fp=os.path.join(itd,p)
        if p=="router_weights.pt": continue
        (shutil.rmtree if os.path.isdir(fp) else os.remove)(fp)
    freed+=sz; done+=1
    print("  DONE %s — router saved (%d keys, %.0f MB), %.2f TB freed"
          %(tag,len(sd),os.path.getsize(out)/1e6,sz/1e12))
print("=== %s: %d/%d processed, %.1f TB freed ==="%("DELETED" if DELETE else "WOULD",done if DELETE else len(targets),len(targets),freed/1e12))
