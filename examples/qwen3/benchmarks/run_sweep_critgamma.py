import glob, re, os, subprocess
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
Q="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3"
FRESH=Q+"/submit_fresh_corner_ep16.sh"
APPLY=os.environ.get("APPLY","0")=="1"
SWEEPLOG="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs"
GPU_CAP=int(os.environ.get("GPU_CAP","64")); MAX_FRESH=int(os.environ.get("MAX_FRESH","2"))
# ---- grid ----
GAMMAS=[0.3,0.5]; RLCS=[0.5,1.0]; AUXS=[0.001,0.003,0.005,0.008,0.012,0.016,0.02]
grid=[(g,rlc,aux) for g in GAMMAS for rlc in RLCS for aux in AUXS]
grid+=[(0.8,0.5,aux) for aux in (0.001,0.006,0.012,0.02)]   # gamma0.8 ceiling probes
def fmt(x): return ("%g"%x)
def maxiter(cell):
    its=[int(m.group(1)) for d in glob.glob(ROOT+"/"+cell+"/checkpoint/*/iter_*") for m in [re.search(r"iter_0*([0-9]+)$",d)] if m]
    return max(its) if its else None
# existing critic cells already covering a (gamma,rlc,aux) config (ANY prefix)
existing=[]
for d in glob.glob(ROOT+"/*/"):
    n=d.rstrip("/").split("/")[-1]
    if "basecritic" not in n: continue
    g=re.search(r"_g([0-9.]+)",n); rlc=re.search(r"rlc([0-9.]+)",n); aux=re.search(r"_aux([0-9.]+)",n)
    if g and rlc and aux: existing.append((float(g.group(1)),float(rlc.group(1)),float(aux.group(1)),n))
def covered(g,rlc,aux):
    for (g2,r2,a2,n) in existing:
        if abs(g2-g)<1e-9 and abs(r2-rlc)<1e-9 and abs(a2-aux)<1e-9: return n
    return None
# in-flight jobs (avoid re-submitting a fresh-start that's queued/running but has no checkpoint yet)
jobnames=subprocess.run("squeue -u jonathanp -h -o '%j'",shell=True,capture_output=True,text=True).stdout.split()
def inflight(jobtag,cell): return any(jobtag in j or cell in j for j in jobnames)
plan={"skip_done":[],"continue":[],"inflight":[],"covered":[],"fresh":[]}
cmds=[]
for (g,rlc,aux) in grid:
    cell="235bv14cg_rlc%s_aux%s_basecritic_g%s_r1"%(fmt(rlc),fmt(aux),fmt(g))
    jobtag="cg_g%s_rlc%s_aux%s"%(fmt(g),fmt(rlc),fmt(aux))
    cov=covered(g,rlc,aux)
    if cov:                                   # an existing critic cell already has this config
        mi=maxiter(cov)
        (plan["skip_done"] if (mi or 0)>=3000 else plan["continue"]).append((cov,mi)); continue
    if inflight(jobtag,cell):                 # already submitted (queued/running) -> do NOT resubmit
        plan["inflight"].append(cell); continue
    mi=maxiter(cell)
    if mi is not None and mi>=3000: plan["skip_done"].append((cell,mi)); continue
    if mi is not None: plan["continue"].append((cell,mi)); continue   # started -> supervisor continues
    plan["fresh"].append(cell)
    cmds.append("RUN_NAME=%s BASELINE=critic GAMMA=%s RLC=%s AUX=%s REWARD_TYPE=per_token_load_weighted RL_TRAIN_ITERS=1500 "
                "sbatch --parsable -J %s --output=%s/fresh_%s_%%j.out --error=%s/fresh_%s_%%j.err %s"
                %(cell,fmt(g),fmt(rlc),fmt(aux),jobtag,SWEEPLOG,cell,SWEEPLOG,cell,FRESH))
print("=== SWEEP PLAN (%d configs): skip>=3000=%d continue=%d in-flight=%d fresh-pending=%d ==="
      %(len(grid),len(plan["skip_done"]),len(plan["continue"]),len(plan["inflight"]),len(plan["fresh"])))
for c,mi in plan["continue"]: print("   cont   %s @%s"%(c[:50],mi))
for c in plan["inflight"]: print("   inflt  %s"%c[:54])
for c in plan["fresh"]: print("   fresh  %s"%c[:56])
if APPLY:
    Q_CAP=int(os.environ.get("Q_CAP","36"))   # max total queued jobs (matches supervisor CAP)
    com_nodes=subprocess.run("squeue -u jonathanp -h -t R,PD -o '%D'",shell=True,capture_output=True,text=True).stdout
    gpus_committed=sum(int(x)*8 for x in com_nodes.split() if x.strip().isdigit())
    qn=len([l for l in subprocess.run("squeue -u jonathanp -h",shell=True,capture_output=True,text=True).stdout.splitlines() if l.strip()])
    # Reserve a committed slice for the sweep: gate on THIS sweep's own committed GPU vs
    # SWEEP_BUDGET, not total committed. So the sweep always keeps up to SWEEP_BUDGET/16 cells
    # queued even when continuations fill the rest; CAP-TRIM (which protects cg_g) holds total
    # committed <= GPU_CAP by trimming non-sweep pending. Result: sweep never starves, never >cap.
    SWEEP_BUDGET=int(os.environ.get("SWEEP_BUDGET","32"))   # ~2 EP=16 cells reserved for the sweep
    cg_lines=subprocess.run("squeue -u jonathanp -h -t R,PD -o '%D %j'",shell=True,capture_output=True,text=True).stdout.splitlines()
    cg_committed=sum(int(l.split()[0])*8 for l in cg_lines if l.strip() and ("cg_g" in l or "235bv14cg" in l))
    nslots=max(0, min(MAX_FRESH, (SWEEP_BUDGET-cg_committed)//16, Q_CAP-qn))
    print("sweep committed=%d (budget %d) | total committed=%d | queued=%d -> submitting %d of %d pending"%(cg_committed,SWEEP_BUDGET,gpus_committed,qn,nslots,len(cmds)))
    for cmd in cmds[:nslots]:
        jid=subprocess.run(cmd,shell=True,capture_output=True,text=True).stdout.strip()
        print("  SUBMIT",cmd.split("RUN_NAME=")[1].split()[0][:50],"->",jid)
else:
    print("=== DRY-RUN (APPLY=1 submits up to MAX_FRESH=%d/run, GPU-capped at %d) ==="%(MAX_FRESH,GPU_CAP))
