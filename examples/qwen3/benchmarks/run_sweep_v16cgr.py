#!/usr/bin/env python3
# v16cgr CORRECTIVE sweep: low-rlc x gamma. Tests the hypothesis that the rlc1xgamma collapse in
# v14cg was an effective-step-size problem -- reducing rlc (0.1, 0.25) should stabilize gamma>0.
# Fixed-lr arm of the fix (the launcher keeps the 1e-4 policy lr; a separate low-lr arm would need
# the lr plumbing change). Fresh-start 0->1500 from the EP=16 mcore-dist base; supervisor then
# continues to 3000 and evals each >=1500 save. Job name = run_name (contains 'basecritic_g' ->
# CAP-TRIM exempt). DRY-RUN default; APPLY=1 submits.
import glob, re, os, subprocess
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
Q="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3"
FRESH=Q+"/submit_fresh_corner_ep16.sh"
SWEEPLOG="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/sweep_logs"
APPLY=os.environ.get("APPLY","0")=="1"; MAX_FRESH=int(os.environ.get("MAX_FRESH","8"))
def g(x): return "%g"%x
GAMMAS=[0.3,0.5]; RLCS=[0.1,0.25]; AUXS=[0.001,0.003]
grid=[(gm,rlc,aux) for gm in GAMMAS for rlc in RLCS for aux in AUXS]
def maxiter(cell):
    its=[int(m.group(1)) for d in glob.glob(ROOT+"/"+cell+"/checkpoint/*/iter_*")
         for m in [re.search(r"iter_0*([0-9]+)$",d)] if m]
    return max(its) if its else None
jobnames=subprocess.run("squeue -u jonathanp -h -o '%j'",shell=True,capture_output=True,text=True).stdout.split()
fresh=[]; skip=[]
for (gm,rlc,aux) in grid:
    cell="235bv16cgr_rlc%s_aux%s_basecritic_g%s_r1"%(g(rlc),g(aux),g(gm))
    if any(cell in j for j in jobnames): skip.append((cell,"in-flight")); continue
    mi=maxiter(cell)
    if mi is not None: skip.append((cell,"on-disk@%s"%mi)); continue
    fresh.append((gm,rlc,aux,cell))
print("=== v16cgr corrective sweep: %d configs | %d fresh-needed | %d skip ==="%(len(grid),len(fresh),len(skip)))
for c,why in skip: print("   skip  %-46s (%s)"%(c,why))
for gm,rlc,aux,c in fresh: print("   fresh %s"%c)
if APPLY:
    os.makedirs(SWEEPLOG,exist_ok=True)
    for gm,rlc,aux,cell in fresh[:MAX_FRESH]:
        cmd=("RUN_NAME=%s BASELINE=critic GAMMA=%s RLC=%s AUX=%s REWARD_TYPE=per_token_load_weighted RL_TRAIN_ITERS=1500 "
             "sbatch --parsable -J %s --output=%s/fresh_%s_%%j.out --error=%s/fresh_%s_%%j.err %s"
             %(cell,g(gm),g(rlc),g(aux),cell,SWEEPLOG,cell,SWEEPLOG,cell,FRESH))
        jid=subprocess.run(cmd,shell=True,capture_output=True,text=True).stdout.strip()
        print("  SUBMIT %-46s -> %s"%(cell,jid))
else:
    print("=== DRY-RUN (APPLY=1 MAX_FRESH=%d to submit) ==="%MAX_FRESH)
