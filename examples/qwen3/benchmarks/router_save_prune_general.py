#!/usr/bin/env python3
# Router-SAVE prune of OLD general-sweep checkpoints (runs INSIDE pai-megatron container).
# For every on-disk checkpoint that is NOT protected, extract the router tensors ->
# <iter_dir>/router_weights.pt, VERIFY (>=90 keys, finite, nonzero), tombstone ROUTER_SAVED_GEN,
# then delete every other file in the iter dir (distcp shards + .metadata), keeping router_weights.pt.
# PROTECTED (never touched): 235bv21math_* (current experiment), the benchmark+holdout Pareto
# frontier cells, the active campaign families v15klcg/v16cgr/v18s (+ their continuation sources),
# and any cell whose name appears in squeue (running/pending).
# DRY-RUN default; DELETE=1 executes. LIMIT=N caps count (pilot: LIMIT=1). SHOWKEEP=1 lists protected.
import csv, glob, re, os, sys, shutil, subprocess, datetime as dt
import torch
from torch.distributed.checkpoint import FileSystemReader
import torch.distributed.checkpoint as dcp

N="/lustre/fsw/portfolios/nvr/users/jonathanp"
B=N+"/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks"
ROOT=N+"/rl_token_routing/output_router_finetuning"
TOMB=B+"/distcp_deletions_tombstone.log"
DELETE=os.environ.get("DELETE","0")=="1"
LIMIT=int(os.environ.get("LIMIT","1000000"))
SHOWKEEP=os.environ.get("SHOWKEEP","0")=="1"
ACTIVE_PREFIX=("235bv21math_","235bv16cgr","235bv15klcg","235bv18s")

def f(x):
    try: return float(x)
    except: return None

# ---- frontier protect-set (benchmark + holdout), both non-dominated ----
rows=list(csv.DictReader(open(B+"/benchmark_results.csv")))
def frontier(ps):
    return set(n for c,a,n in ps if not any((cc<=c and aa>a) or (cc<c and aa>=a) for cc,aa,_ in ps))
bp=[]; hp=[]; acc={}; cpv={}
for r in rows:
    if r["limit"].strip()!="inf": continue
    n=r["run_name"].strip(); c=f(r.get("cp_critical_eval"))
    if "pretrained" in n.lower(): continue
    it=r.get("bench_iteration") or r.get("train_iters")
    try: it=int(float(it))
    except: it=None
    a=f(r.get("benchmark_avg"))
    if a is not None and it is not None: acc[(n,it)]=a
    if c is not None and it is not None: cpv[(n,it)]=c
    if a and a>=40 and c is not None: bp.append((c,a,n))
    ah=f(r.get("holdout_avg"))
    if ah and ah>=40 and c is not None and (r.get("mmlu_pro") or "").strip(): hp.append((c,ah,n))
FRONTIER=frontier(bp)|frontier(hp)

# ---- queued / running (protect continuation sources & active work) ----
JOBS=subprocess.run("squeue -u jonathanp -h -o %j",shell=True,capture_output=True,text=True).stdout

def protected(cell):
    if cell.startswith(ACTIVE_PREFIX): return "active-family"
    if cell in FRONTIER: return "frontier"
    if cell in JOBS: return "in-queue"
    return None

# ---- verbatim extract_router() from router_extract_and_prune.py ----
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
        dcp.load(state_dict=sd,storage_reader=r)
    except Exception as e:
        return None,"dcp.load: %r"%e
    bad=[k for k,v in sd.items() if not torch.isfinite(v.float()).all() or float(v.float().abs().sum())==0.0]
    if bad: return None,"zero/nonfinite: %s"%bad[:3]
    return sd,None

# ---- enumerate on-disk checkpoints ----
targets=[]; kept={}
for d in glob.glob(ROOT+"/*/checkpoint/*/iter_*"):
    if len(glob.glob(d+"/*.distcp"))<32: continue           # incomplete or already-pruned -> skip
    m=re.search(r"/([^/]+)/checkpoint/.*/iter_0*([0-9]+)$",d)
    if not m: continue
    cell=m.group(1); it=int(m.group(2))
    why=protected(cell)
    if why: kept.setdefault(why,set()).add(cell); continue
    targets.append((cell,it,d))
targets.sort()
targets=targets[:LIMIT]

print("=== ROUTER-SAVE PRUNE (%s) | protected: frontier=%d active-family=%d in-queue=%d ==="
      %("DELETE" if DELETE else "DRY-RUN",len(kept.get("frontier",[])),
        len(kept.get("active-family",[])),len(kept.get("in-queue",[]))))
print("=== %d on-disk checkpoints TARGETED for router-save+prune ==="%len(targets))
if SHOWKEEP:
    for why,cells in kept.items():
        print("  PROTECT[%s] (%d cells): %s"%(why,len(cells),", ".join(sorted(cells)[:6])+(" ..." if len(cells)>6 else "")))

freed=0; done=0; fail=0
for cell,it,itd in targets:
    tag="%s@%d"%(cell[:50],it)
    if not DELETE:
        print("  would prune %s"%tag); continue
    # FAST-PATH: a valid router_weights.pt already exists (prior extract pass) -> skip the slow
    # DCP re-extraction and just free the shards.
    existing=os.path.join(itd,"router_weights.pt")
    if os.path.exists(existing):
        try:
            _chk=torch.load(existing,map_location="cpu"); _nk=len(_chk.get("router",{}))
        except Exception:
            _nk=0
        if _nk>=90:
            try: sz=int(subprocess.check_output(["du","-sb",itd]).split()[0])
            except Exception: sz=0
            for p in os.listdir(itd):
                if p=="router_weights.pt": continue
                fp=os.path.join(itd,p)
                (shutil.rmtree if os.path.isdir(fp) else os.remove)(fp)
            with open(TOMB,"a") as t:
                t.write("%s\t%s\t%d\tacc=%s\tcanon_cp=%s\tROUTER_KEPT_GEN\t%d\t%s\trouter=%s\n"
                        %(dt.datetime.utcnow().isoformat(),cell,it,acc.get((cell,it)),cpv.get((cell,it)),sz,itd,existing))
            freed+=sz; done+=1
            print("  DONE(fast) %s — existing router kept (%d keys), %.3f TB freed"%(tag,_nk,sz/1e12))
            continue
    sd,err=extract_router(itd)   # router now in memory
    if err: print("  SKIP %s — extract failed: %s"%(tag,err)); fail+=1; continue
    # Save router to NODE-LOCAL /tmp FIRST (Lustre is over-quota; no headroom for a 100MB write yet).
    # Then free the Lustre shards, THEN move the router back into the (now-freed) iter dir.
    meta={"cell":cell,"iteration":it,"acc":acc.get((cell,it)),"cp_canonical":cpv.get((cell,it)),
          "extracted":dt.datetime.utcnow().isoformat(),"source":"distcp","keys":len(sd)}
    tmpout="/tmp/rw_%d_%d.pt"%(os.getpid(),done)
    try:
        torch.save({"meta":meta,"router":{k:v.cpu() for k,v in sd.items()}},tmpout)
        chk=torch.load(tmpout,map_location="cpu")
    except Exception as e:
        print("  SKIP %s — tmp save failed: %r"%(tag,e)); fail+=1
        if os.path.exists(tmpout): os.remove(tmpout)
        continue
    if len(chk.get("router",{}))<90:
        print("  SKIP %s — verify failed (post-save)"%tag); fail+=1; os.remove(tmpout); continue
    try: sz=int(subprocess.check_output(["du","-sb",itd]).split()[0])
    except: sz=0
    # delete Lustre shards (frees ~sz) — router is safe on node-local /tmp
    for p in os.listdir(itd):
        fp=os.path.join(itd,p)
        (shutil.rmtree if os.path.isdir(fp) else os.remove)(fp)
    # move router back to the now-freed iter dir; fall back to leaving it on /tmp path in the tombstone
    final=os.path.join(itd,"router_weights.pt")
    try:
        shutil.move(tmpout,final)
    except Exception as e:
        # Lustre quota accounting can lag right after the shard delete, so the copy often still
        # landed full-size; accept it if so, else leave the router on node-local /tmp.
        if os.path.exists(final) and os.path.getsize(final)>=os.path.getsize(tmpout)-4096:
            try: os.remove(tmpout)
            except Exception: pass
        else:
            print("  WARN %s — router left at %s: %r"%(tag,tmpout,e)); final=tmpout
    with open(TOMB,"a") as t:
        t.write("%s\t%s\t%d\tacc=%s\tcanon_cp=%s\tROUTER_SAVED_GEN\t%d\t%s\trouter=%s\n"
                %(dt.datetime.utcnow().isoformat(),cell,it,acc.get((cell,it)),cpv.get((cell,it)),sz,itd,final))
    freed+=sz; done+=1
    print("  DONE %s — router saved (%d keys), %.3f TB freed"%(tag,len(sd),sz/1e12))
print("=== %s: %d/%d pruned, %d failed, %.2f TB freed ==="
      %("DELETED" if DELETE else "WOULD",done if DELETE else len(targets),len(targets),fail,freed/1e12))
