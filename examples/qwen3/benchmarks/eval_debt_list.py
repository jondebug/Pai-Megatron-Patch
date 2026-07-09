#!/usr/bin/env python3
# Print eval debt as "ckroot|iter|cell" lines: on-disk checkpoints >=1500 with 32 complete distcp shards
# and no inf eval yet. ckroot = the pretrain-mcore dir (containing iter_NNNNNNN/), as the convert script expects.
import csv,glob,re,os
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
def f(x):
    try: return float(x)
    except: return None
evald=set()
for r in csv.DictReader(open(CSV)):
    if r["limit"].strip()=="inf" and f(r.get("benchmark_avg")) and f(r.get("benchmark_avg"))>=40:
        it=r.get("bench_iteration") or r.get("train_iters")
        try: evald.add((r["run_name"].strip(),int(float(it))))
        except: pass
seen=set(); out=[]
for d in glob.glob(ROOT+"/235bv*/checkpoint/*/iter_*"):
    if len(glob.glob(d+"/*.distcp"))<32: continue
    m=re.search(r"/(235bv[^/]+)/checkpoint/.*/iter_0*([0-9]+)$",d)
    if not m: continue
    cell=m.group(1); it=int(m.group(2)); ckroot=os.path.dirname(d)
    if it<1500 or (cell,it) in evald or (cell,it) in seen: continue
    seen.add((cell,it)); out.append((cell,it,ckroot))
for cell,it,ckroot in sorted(out): print("%s|%d|%s"%(ckroot,it,cell))
