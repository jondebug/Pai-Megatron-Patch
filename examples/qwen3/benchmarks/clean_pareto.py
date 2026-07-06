#!/usr/bin/env python3
# Clean per-class Pareto frontier: aux-only / RL-only / RL+aux frontier LINES only (no dominated cloud),
# + 235B pretrained baseline star. x = CP reduction % vs 235B pretrained (9320). No 30B anywhere.
import csv, re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
B="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks"
BASE_CP=9320.0; BASE_ACC=76.74   # 235B pretrained (earliest-measured wandb CP; acc from pretrained_235b inf eval)
def f(x):
    try: return float(x)
    except: return None
def cls(n):
    if "norl" in n: return "aux"
    m=re.search(r"aux([0-9.]+)",n)
    return "rl+aux" if (m and float(m.group(1))>0) else "rl-only"
pts={"aux":[],"rl-only":[],"rl+aux":[]}
for r in csv.DictReader(open(B+"/benchmark_results.csv")):
    if r["limit"].strip()!="inf": continue
    a=f(r.get("benchmark_avg")); c=f(r.get("cp_critical_eval"))
    if not a or not c or a<40 or "pretrained" in r["run_name"].lower(): continue
    pts[cls(r["run_name"].strip())].append((c,a))
def frontier(ps):  # non-dominated: no other pt with cp<= and acc>= (strictly better once)
    fr=[p for p in ps if not any(q[0]<=p[0] and q[1]>=p[1] and (q[0]<p[0] or q[1]>p[1]) for q in ps)]
    return sorted(set(fr))
red=lambda c:(BASE_CP-c)/BASE_CP*100
plt.figure(figsize=(9,6))
style={"aux":("#2ca02c","D","Aux-only"),"rl-only":("#d62728","^","RL-only"),"rl+aux":("#1f77b4","o","RL + Aux")}
for k in ("aux","rl-only","rl+aux"):
    fr=frontier(pts[k])
    if not fr: continue
    xs=[red(c) for c,a in fr]; ys=[a for c,a in fr]
    col,mk,lab=style[k]
    plt.plot(xs,ys,marker=mk,color=col,lw=2,ms=7,label="%s (%d pts)"%(lab,len(fr)))
plt.scatter([0],[BASE_ACC],marker="*",s=320,color="black",zorder=5,label="Qwen3-235B-A22B pretrained")
plt.annotate("pretrained\n%.1f%% acc, CP=%.0f (0%% reduction)"%(BASE_ACC,BASE_CP),(0,BASE_ACC),
             textcoords="offset points",xytext=(12,-8),fontsize=8)
plt.xlabel("Critical Path Reduction (%)  vs 235B pretrained (CP=9320)")
plt.ylabel("Benchmark Accuracy (%)  (HellaSwag/ARC-C/WinoGrande mean)")
plt.title("Qwen3-235B-A22B: Accuracy vs Critical-Path Reduction — per-class Pareto frontiers")
plt.grid(alpha=0.3); plt.legend(loc="lower left", fontsize=9)
plt.tight_layout(); plt.savefig(B+"/pareto_235b_clean.png",dpi=150)
print("wrote pareto_235b_clean.png")
for k in ("aux","rl-only","rl+aux"):
    fr=frontier(pts[k]); print("  %s frontier: %d pts, top acc=%.2f @cp=%.0f (%.1f%% red)"%(k,len(fr),max(a for c,a in fr),min(c for c,a in fr if a==max(aa for cc,aa in fr)),red(min(c for c,a in fr))) if fr else (k,0))
