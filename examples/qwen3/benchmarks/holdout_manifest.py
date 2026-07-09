#!/usr/bin/env python3
# Key-model list for the HOLDOUT evaluation (model-card benchmarks). Emits "ckroot|iter|cell" lines
# (same convention as eval_debt_list.py) for on-disk 32-shard checkpoints; skips ones that already
# have a holdout_avg in the CSV (re-runnable/idempotent). The pretrained baseline is handled
# separately by the driver (existing HF, no conversion).
# Sets: canonical frontier (re-evaluable) U corner-comparison families U per-sweep best points.
import csv, glob, re, os
B="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks"
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
def f(x):
    try: return float(x)
    except: return None
rows=list(csv.DictReader(open(B+"/benchmark_results.csv")))
have_hold=set()
if rows and "mmlu_pro" in rows[0]:   # 4-task suite marker (2026-07-09): re-run 2-task-era targets
    for r in rows:
        if (r.get("mmlu_pro") or "").strip():
            it=r.get("bench_iteration") or r.get("train_iters")
            try: have_hold.add((r["run_name"].strip(),int(float(it))))
            except: pass
inf=[r for r in rows if r["limit"].strip()=="inf" and f(r.get("benchmark_avg")) and f(r.get("cp_critical_eval"))]
pts=[(f(r["cp_critical_eval"]),f(r["benchmark_avg"]),r["run_name"].strip(),r.get("bench_iteration") or r.get("train_iters")) for r in inf]
fr=[p for p in pts if not any(q[0]<=p[0] and q[1]>=p[1] and (q[0]<p[0] or q[1]>p[1]) for q in pts)]
want=set()
for c,a,n,it in fr:
    try: want.add((n,int(float(it))))
    except: pass
# corner-comparison families
CORNER=[
    ("235b-pareto_norl_aux0.001_r12",[1223]),
    ("235bv5a_norl_aux0.001_r24",[2500,3000]),
    ("235bv15klcg_rlc0.5_aux0.001_basecritic_g0_kl0.001_r1",[1602,2500,3000]),
    ("235bv15klcg_rlc0.5_aux0.001_basecritic_g0_kl0.001_seed2027_r1",[1500,3000]),
    ("235bv15klcg_rlc0.5_aux0.001_basecritic_g0_kl0.001_seed2028_r1",[1500,3000]),
]
# sweep-best ablation points (per family best acc among evaluated, on the CP-relevant side)
BESTS=[
    ("235bv16cgr_rlc0.1_aux0.003_basecritic_g0.3_r1",[2961,3000]),
    ("235bv16cgr_rlc0.1_aux0.001_basecritic_g0.3_r1",[1500]),
    ("235bv14cg_rlc0.5_aux0.001_basecritic_g0.8_r1",[2361]),
    ("235bv15klcg_rlc0.5_aux0.001_basecritic_g0.5_kl0.0001_r1",[1596]),
    ("235bv15klcg_rlc0.5_aux0.001_basecritic_g0_kl0.0001_r1",[1591]),
]
for cell,its in CORNER+BESTS:
    for it in its: want.add((cell,it))
out=[]; missing=[]
for cell,it in sorted(want):
    if (cell,it) in have_hold: continue
    g=glob.glob(ROOT+"/"+cell+"/checkpoint/*/iter_0*%d"%it)
    g=[d for d in g if len(glob.glob(d+"/*.distcp"))>=32 and d.endswith("%07d"%it)]
    if not g: missing.append("%s@%d"%(cell[:40],it)); continue
    out.append("%s|%d|%s"%(os.path.dirname(g[0]),it,cell))
for l in out: print(l)
import sys
print("## %d holdout targets (%d unavailable: %s)"%(len(out),len(missing),missing[:6]),file=sys.stderr)
