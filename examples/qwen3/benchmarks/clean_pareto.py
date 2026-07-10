#!/usr/bin/env python3
# Clean per-class Pareto frontier: aux-only / RL-only / RL+aux frontier LINES only (no dominated cloud),
# + 235B pretrained baseline star. x = CP reduction % vs 235B pretrained (9320). No 30B anywhere.
# --holdout : same style, but y = 4-task model-card holdout mean (MMLU/GSM8K/MMLU-Pro/BBH) —
#             never used for selection, so retroactively unbiased. Only rows with the full 4-task
#             suite (mmlu_pro present) are plotted; baseline star = pretrained holdout row.
import csv, re, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
B="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks"
HOLDOUT = "--holdout" in sys.argv
BASE_CP=9320.0; BASE_ACC=76.74   # 235B pretrained (earliest-measured wandb CP; acc from pretrained_235b inf eval)
CORNER_SIG="v15klcg_rlc0.5_aux0.001_basecritic_g0_kl0.001"   # corner config family (orig + seed reps)
def f(x):
    try: return float(x)
    except: return None
def cls(n):
    if "norl" in n: return "aux"
    m=re.search(r"aux([0-9.]+)",n)
    return "rl+aux" if (m and float(m.group(1))>0) else "rl-only"
pts={"aux":[],"rl-only":[],"rl+aux":[]}
corner=[]   # holdout mode: corner-family @3000 points for the n-seed variability bar
for r in csv.DictReader(open(B+"/benchmark_results.csv")):
    if r["limit"].strip()!="inf": continue
    n=r["run_name"].strip()
    if HOLDOUT and not (r.get("mmlu_pro") or "").strip(): continue   # require the full 4-task suite
    a=f(r.get("holdout_avg") if HOLDOUT else r.get("benchmark_avg")); c=f(r.get("cp_critical_eval"))
    if HOLDOUT and ("pretrained" in n.lower() or n=="Qwen3-235B-A22B") and a:
        BASE_ACC=a; continue   # holdout baseline anchor comes from the CSV row itself
    if not a or not c or a<40 or "pretrained" in n.lower(): continue
    if HOLDOUT and CORNER_SIG in n:
        it=f(r.get("bench_iteration") or r.get("train_iters"))
        if it and abs(it-3000)<60: corner.append((c,a))
    pts[cls(n)].append((c,a))
if HOLDOUT and BASE_ACC==76.74: BASE_ACC=82.01   # fallback: our-harness pretrained 4-task mean
def frontier(ps):  # non-dominated: no other pt with cp<= and acc>= (strictly better once)
    fr=[p for p in ps if not any(q[0]<=p[0] and q[1]>=p[1] and (q[0]<p[0] or q[1]>p[1]) for q in ps)]
    return sorted(set(fr))
red=lambda c:(BASE_CP-c)/BASE_CP*100
plt.figure(figsize=(9,6))
style={"aux":("#2ca02c","D","Aux-only"),"rl-only":("#d62728","^","RL-only"),"rl+aux":("#1f77b4","o","RL + Aux")}
ally=[]
for k in ("aux","rl-only","rl+aux"):
    fr=frontier(pts[k])
    if not fr: continue
    xs=[red(c) for c,a in fr]; ys=[a for c,a in fr]; ally+=ys
    col,mk,lab=style[k]
    plt.plot(xs,ys,marker=mk,color=col,lw=2,ms=7,label="%s (%d pts)"%(lab,len(fr)))
plt.scatter([0],[BASE_ACC],marker="*",s=320,color="black",zorder=5,label="Qwen3-235B-A22B pretrained")
if HOLDOUT:
    if len(corner)>=2:   # seed-variability bar recomputed on the holdout axis
        cs=[c for c,a in corner]; ys=[a for c,a in corner]
        mc,ma=sum(cs)/len(cs),sum(ys)/len(ys)
        plt.errorbar([red(mc)],[ma],yerr=[(max(ys)-min(ys))/2],xerr=[(max(cs)-min(cs))/2/BASE_CP*100],
                     fmt="s",color="dimgray",ms=6,capsize=4,zorder=4,
                     label="corner config, n=%d seeds @3000 (mean±½range)"%len(corner))
else:
    # seed-variability bar: the 76.37-config family, n=3 seeds @3000 (mean 75.73 +- 0.32 acc, CP 7950 +- 198)
    _sx=(BASE_CP-7950.0)/BASE_CP*100; _sxe=198.0/BASE_CP*100
    plt.errorbar([_sx],[75.73],yerr=[0.32],xerr=[_sxe],fmt="s",color="dimgray",ms=6,capsize=4,zorder=4,
                 label="corner config, n=3 seeds @3000 (mean±½range)")
plt.annotate("pretrained\n%.1f%% acc, CP=%.0f (0%% reduction)"%(BASE_ACC,BASE_CP),(0,BASE_ACC),
             textcoords="offset points",xytext=(12,-8),fontsize=8)
plt.xlabel("Critical Path Reduction (%)  vs 235B pretrained (CP=9320)")
if HOLDOUT:
    plt.ylabel("Holdout Accuracy (%)  (MMLU/GSM8K/MMLU-Pro/BBH mean)")
    plt.title("Qwen3-235B-A22B: Holdout accuracy vs Critical-Path Reduction — per-class Pareto frontiers")
    lo=min(ally) if ally else BASE_ACC-4
    plt.ylim(lo-0.4, BASE_ACC+0.6)
else:
    plt.ylabel("Benchmark Accuracy (%)  (HellaSwag/ARC-C/WinoGrande mean)")
    plt.title("Qwen3-235B-A22B: Accuracy vs Critical-Path Reduction — per-class Pareto frontiers")
    plt.ylim(72.0, 77.2)   # explicit floor at 72 (frontier min 72.12); no dead padding below
plt.grid(alpha=0.3); plt.legend(loc="lower left", fontsize=9)
OUT = "pareto_235b_holdout.png" if HOLDOUT else "pareto_235b_clean.png"
plt.tight_layout(); plt.savefig(B+"/"+OUT,dpi=150)
print("wrote "+OUT)
for k in ("aux","rl-only","rl+aux"):
    fr=frontier(pts[k]); print("  %s frontier: %d pts, top acc=%.2f @cp=%.0f (%.1f%% red)"%(k,len(fr),max(a for c,a in fr),min(c for c,a in fr if a==max(aa for cc,aa in fr)),red(min(c for c,a in fr))) if fr else (k,0))
