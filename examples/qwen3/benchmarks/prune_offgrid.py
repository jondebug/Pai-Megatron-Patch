#!/usr/bin/env python3
# GRID-ONLY retention (user 2026-07-06): delete off-grid exit-save checkpoints (iter % 500 != 0)
# from 235bv* cells. Protections:
#   - never the cell's max iter (continuation source, even if off-grid)
#   - never an iter with a real inf eval (evaluated points are data — dominated-prune owns those)
#   - never an iter with an in-flight conveval/lmeval job
#   - only complete checkpoints (>=32 shards) — incomplete ones are handled elsewhere
# Tombstoned OFFGRID_GRIDONLY. DRY-RUN default; DELETE=1 executes. No router extraction (user choice).
import csv, glob, re, os, shutil, subprocess, datetime as dt
B="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks"
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
TOMB=B+"/distcp_deletions_tombstone.log"
DELETE=os.environ.get("DELETE","0")=="1"
def f(x):
    try: return float(x)
    except: return None
evald=set()
for r in csv.DictReader(open(B+"/benchmark_results.csv")):
    if r["limit"].strip()=="inf" and f(r.get("benchmark_avg")) and f(r.get("benchmark_avg"))>=40:
        it=r.get("bench_iteration") or r.get("train_iters")
        try: evald.add((r["run_name"].strip(),int(float(it))))
        except: pass
jobs=subprocess.run("squeue -u jonathanp -h -o %j",shell=True,capture_output=True,text=True).stdout
disk={}
for d in glob.glob(ROOT+"/235bv*/checkpoint/*/iter_*"):
    if len(glob.glob(d+"/*.distcp"))<32: continue
    m=re.search(r"/(235bv[^/]+)/checkpoint/.*/iter_0*([0-9]+)$",d)
    if m: disk.setdefault(m.group(1),{})[int(m.group(2))]=d
cellmax={c:max(v) for c,v in disk.items()}
targets=[]
for cell,iters in disk.items():
    for it,itd in iters.items():
        if it%500==0: continue                       # grid point -> keep
        if it==cellmax[cell]: continue               # latest -> keep (continuation source)
        if (cell,it) in evald: continue              # evaluated -> keep (data point)
        # in-flight eval job for this (cell,iter)? job names embed _<iter> suffix
        tagpats=["_%d"%it]
        if any(("conveval" in j or "lmeval" in j) and j.strip().endswith("_%d"%it) and True for j in jobs.split()):
            # conservative: skip any iter number currently appearing in an eval job name
            if any(j.strip().endswith("_%d"%it) for j in jobs.split() if "eval" in j): continue
        targets.append((cell,it,itd))
targets.sort()
print("=== %s: %d off-grid checkpoints (grid-only retention) ==="%("DELETE" if DELETE else "DRY-RUN",len(targets)))
freed=0; n=0
for cell,it,itd in targets:
    try: sz=int(subprocess.check_output(["du","-sb",itd]).split()[0])
    except: sz=0
    if DELETE:
        with open(TOMB,"a") as t:
            t.write("%s\t%s\t%d\tOFFGRID_GRIDONLY\tunevaluated_exit_save\t%d\t%s\n"
                    %(dt.datetime.utcnow().isoformat(),cell,it,sz,itd))
        try: shutil.rmtree(itd); freed+=sz; n+=1
        except Exception as e: print("  rm FAILED",cell,it,e); continue
    else: freed+=sz; n+=1
    print("  %s %-52s @%d %.2fTB"%("DEL" if DELETE else "would",cell[:52],it,sz/1e12))
print("=== %s: %d dirs, %.1f TB ==="%("FREED" if DELETE else "would free",n,freed/1e12))
