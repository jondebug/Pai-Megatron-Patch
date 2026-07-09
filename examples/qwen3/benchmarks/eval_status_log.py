#!/usr/bin/env python3
# Generate EVAL_STATUS.md: a complete log of every 235B checkpoint in [1000,3000] and its
# limit=inf eval status (OK / PENDING-convert / PENDING-HFready / INCOMPLETE / DIVERGED).
# Re-run any time (and wired into babysit_3000.sh so it stays fresh each cycle).
import csv, os, glob, re, sys
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
B="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks"
CSV=f"{B}/benchmark_results.csv"
OUT=f"{B}/EVAL_STATUS.md"
STAMP=sys.argv[1] if len(sys.argv)>1 else "(unstamped)"
def f(x):
    try: return float(x)
    except: return None
rows=list(csv.DictReader(open(CSV)))
itcol=[c for c in rows[0] if "iter" in c.lower()][0]
infidx={}; diverged=set()
for r in rows:
    cell=r["run_name"].strip(); it=f(r[itcol]); acc=f(r["benchmark_avg"]); cp=f(r["eval_crit_path"]); ll=f(r.get("eval_lm_loss"))
    if (ll and ll>=8) or (acc and 0<acc<60): diverged.add(cell)
    if r["limit"].strip()=="inf" and it is not None and acc and acc>=40:
        infidx[(cell,int(it))]=(acc,cp)
cells=sorted(set(os.path.basename(p.rstrip("/")) for pat in ["235bv5a_*","235bv5b_*","235bv6_*","235bv11e16_*","235bv12corner_*"] for p in glob.glob(f"{ROOT}/{pat}/")))
tot=ev=pend=hfready=incomplete=divskip=0
pending_rows=[]; per_cell=[]
for cell in cells:
    its={}
    for itd in glob.glob(f"{ROOT}/{cell}/checkpoint/*/iter_*"):
        nshard=len(glob.glob(itd+"/*.distcp"))
        if nshard==0: continue
        m=re.search(r"iter_0*(\d+)",itd)
        if not m: continue
        it=int(m.group(1))
        if 1000<=it<=3000: its[it]=(os.path.dirname(itd), nshard)
    if not its: continue
    isdiv=cell in diverged
    parts=[]
    for it,(ckdir,nshard) in sorted(its.items()):
        tot+=1
        if isdiv: divskip+=1; parts.append(f"{it}:DIVERGED"); continue
        if nshard<32: incomplete+=1; parts.append(f"{it}:INCOMPLETE({nshard}/32)"); continue
        if (cell,it) in infidx:
            ev+=1; acc,cp=infidx[(cell,it)]
            parts.append(f"{it}:OK({acc:.2f}@{cp:.0f})" if cp else f"{it}:OK({acc:.2f})")
        else:
            pend+=1
            hf=len(glob.glob(f"{ckdir}/hf_converted_iter{it}_cp/*.safetensors"))
            state="PENDING-HFready" if hf>=100 else "PENDING-needConvert"
            if hf>=100: hfready+=1
            parts.append(f"{it}:{state}")
            pending_rows.append((cell,it,state))
    per_cell.append((cell, isdiv, parts))
L=[]
L.append(f"# 235B checkpoint eval-status log\nGenerated: {STAMP}\n")
L.append(f"Scope: every checkpoint on disk with iter in [1000,3000].\n")
L.append("## Totals\n")
L.append(f"- relevant checkpoints: **{tot}**")
L.append(f"- inf-evaluated (OK): **{ev}**")
L.append(f"- **PENDING eval: {pend}**  (HF-ready: {hfready}, need convert: {pend-hfready})")
L.append(f"- incomplete (<32 shards, skipped): {incomplete}")
L.append(f"- in diverged cells (excluded): {divskip}\n")
L.append("## PENDING checkpoints (created, not yet inf-evaluated)\n")
for cell,it,state in pending_rows:
    L.append(f"- {cell} @ iter {it}  — {state}")
L.append("\n## Full per-cell status\n")
for cell,isdiv,parts in per_cell:
    tag=" [DIVERGED]" if isdiv else ""
    L.append(f"**{cell}**{tag}\n    "+"  ".join(parts))
open(OUT,"w").write("\n".join(L)+"\n")
print(f"wrote {OUT}")
print(f"TOTALS: relevant={tot} OK={ev} PENDING={pend} (HFready={hfready}) incomplete={incomplete} diverged_excl={divskip}")
