#!/bin/bash
# Reap hf_converted_iter{N}_cp dirs for inf-EVALUATED checkpoints that are NOT close to the
# accuracy-vs-CP frontier. distcp source-of-truth is NEVER touched; only the regenerable HF copy.
# Frontier-close = within MARGIN pp of the best accuracy achievable at the point's CP-or-lower
# (envelope over ALL inf points, so both the RL frontier and the aux corner are protected).
# Only points that ALREADY have an inf result are eligible (so freshly-converted-not-yet-evaluated
# HF is never deleted). Default DRY-RUN; set DELETE=1 to actually remove.
set -uo pipefail
ROOT=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning
CSV=/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv
MARGIN="${MARGIN:-0.5}"     # keep HF if acc within MARGIN pp of the envelope at its CP
DELETE="${DELETE:-0}"

python3 - "$ROOT" "$CSV" "$MARGIN" "$DELETE" <<"PYEOF"
import csv, os, glob, re, shutil, sys
ROOT, CSV, MARGIN, DELETE = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]=="1"
def f(x):
    try: return float(x)
    except: return None
rows=list(csv.DictReader(open(CSV)))
itcol=[c for c in rows[0] if "iter" in c.lower()][0]
def is_rl(r): return str(r.get("rl_enabled","")).strip().lower() in ("true","1","yes")
# inf points: (cell,iter)->(cp,acc,rl); the RL-only envelope for the "RL frontier" ceiling
inf={}; rlpts=[]
for r in rows:
    if r["limit"].strip()!="inf": continue
    cell=r["run_name"].strip(); it=f(r[itcol]); cp=f(r["eval_crit_path"]); acc=f(r["benchmark_avg"])
    if it is None or cp is None or acc is None or acc<40: continue
    inf[(cell,int(it))]=(cp,acc,is_rl(r))
    if is_rl(r): rlpts.append((cp,acc))
def ceiling(cp):
    # best RL accuracy achievable at CP-or-lower (RL frontier is non-decreasing in CP)
    cand=[a for (c,a) in rlpts if c<=cp+1e-6]
    return max(cand) if cand else -1
kept=deleted=freed=0; dellist=[]; exempt_nonrl=0
for (cell,it),(cp,acc,rl) in sorted(inf.items()):
    hfs=glob.glob(f"{ROOT}/{cell}/checkpoint/*/hf_converted_iter{it}_cp")
    hfs=[h for h in hfs if os.path.isdir(h)]
    if not hfs: continue
    if not rl:               # aux-only / norl / baseline comparison points -> always keep HF
        exempt_nonrl+=len(hfs); kept+=len(hfs); continue
    close = acc >= ceiling(cp) - MARGIN
    for h in hfs:
        if close:
            kept+=1; continue
        sz=sum(os.path.getsize(os.path.join(dp,fn)) for dp,_,fns in os.walk(h) for fn in fns if os.path.exists(os.path.join(dp,fn)))
        dellist.append((cell,it,cp,acc,ceiling(cp),sz,h))
        deleted+=1; freed+=sz
        if DELETE:
            shutil.rmtree(h, ignore_errors=True)
mode="DELETED" if DELETE else "WOULD DELETE (dry-run)"
print(f"=== HF cleanup ({mode}); MARGIN={MARGIN}pp vs RL frontier; kept HF={kept} (incl {exempt_nonrl} non-RL baseline) ===")
for cell,it,cp,acc,ceil,sz,h in dellist:
    print(f"  {mode}: {cell}@{it}  acc={acc:.2f} cp={cp:.0f} (envelope@cp={ceil:.2f}, gap={ceil-acc:.2f}pp)  {sz/1e9:.0f}GB")
print(f"=== {mode}: {deleted} HF dirs, {freed/1e12:.2f} TB {'freed' if DELETE else 'reclaimable'} (distcp untouched) ===")
PYEOF