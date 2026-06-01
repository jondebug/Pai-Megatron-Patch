#!/bin/bash
# Delete hf_converted_iter{N}_cp dirs once the CSV has BOTH limit=1000 AND limit=inf
# accuracy for that (cell, iter). distcp source-of-truth is always preserved.
# Idempotent. Runs anywhere (pure filesystem + CSV read). No GPU needed.
set -e
cd /lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks
python3 - <<'PYEOF'
import csv, re, shutil
from pathlib import Path
from collections import defaultdict
CSV=Path("benchmark_results.csv")
ROOT=Path("/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning")
rows=list(csv.DictReader(open(CSV)))
# (cell, iter) -> set of limits with a benchmark_avg
limits=defaultdict(set)
for r in rows:
    if not r.get("benchmark_avg"): continue
    cell=re.sub(r"_iter\d+$","",r["run_name"])
    try: it=int(r["bench_iteration"])
    except: continue
    limits[(cell,it)].add(r.get("limit",""))
import glob
WT=Path("/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results")
wt_done=set()
for f in WT.glob("vllm_pretrained_235b_vs_*.json"):
    mm=re.search(r"_vs_(.+?)_iter(\d+)", f.name)
    if mm: wt_done.add((mm.group(1), int(mm.group(2))))
deletable=[(c,i) for (c,i),L in limits.items() if "1000" in L and "inf" in L and (c,i) in wt_done]
freed=0; n=0
for cell,it in deletable:
    cdir=ROOT/cell/"checkpoint"
    if not cdir.is_dir(): continue
    for sub in cdir.iterdir():
        hf=sub/f"hf_converted_iter{it}_cp"
        if hf.is_dir():
            sz=sum(f.stat().st_size for f in hf.rglob("*") if f.is_file())
            shutil.rmtree(hf)
            freed+=sz; n+=1
            print(f"deleted {hf} ({sz/1e9:.0f} GB)")
print(f"Cleanup: removed {n} HF dirs, freed {freed/1e12:.2f} TB (criterion: 1000+inf both present)")
PYEOF
