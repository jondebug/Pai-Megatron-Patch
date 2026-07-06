#!/usr/bin/env python3
"""
gap_manifest.py — authoritative "fruition" arbiter for the 235B EP=16 sweep universe.

Fruition := cell reached iter >= 3000 AND every checkpoint it saved at iter >= FLOOR
(grid AND off-grid exit saves) has an accuracy eval in benchmark_results.csv (+-TOL tolerance).

Classifies every 235bv* cell (on-disk) PLUS the declared PLANNED grids (v14cg, v15klcg) into:
  DONE / COVERED / EVAL / CONT / NEVER / NORL_NEVER / DROP
and emits actionable lists for the supervisor loops:
  gap_cont.txt   : cell                         (continue to 3000)
  gap_fresh.txt  : cell|RUN_NAME=.. BASELINE=.. GAMMA=.. RLC=.. AUX=.. KL=.. LM=.. REWARD_TYPE=..
  gap_eval.txt   : cell <iter>                   (one line per unevaluated >=FLOOR checkpoint)
Read-only; never submits. Idempotent. Loud about DROP / NORL_NEVER / supersession.
"""
import os, csv, re, glob, argparse, subprocess, datetime
from collections import defaultdict

ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing"
R=os.path.join(ROOT,"output_router_finetuning")
BCSV=os.path.join(ROOT,"Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv")
TCSV=os.path.join(ROOT,"Pai-Megatron-Patch/examples/qwen3/benchmarks/training_results.csv")
OUT=os.environ.get("OUTDIR","/tmp")
DROP={"235bv2","235bv2p","235bv7","235bv7r"}   # abandoned/superseded early sweeps (user-approved)
COLLAPSED={  # rlc1xgamma cells w/ confirmed router collapse (acc<72, degrade w/ training); see CELL_BACKLOG.md
 "235bv14cg_rlc1_aux0.005_basecritic_g0.3_r1","235bv14cg_rlc1_aux0.008_basecritic_g0.3_r1",
 "235bv14cg_rlc1_aux0.012_basecritic_g0.3_r1","235bv14cg_rlc1_aux0.016_basecritic_g0.3_r1",
 "235bv14cg_rlc1_aux0.02_basecritic_g0.3_r1","235bv14cg_rlc1_aux0.008_basecritic_g0.5_r1",
 "235bv14cg_rlc1_aux0.012_basecritic_g0.5_r1","235bv14cg_rlc1_aux0.016_basecritic_g0.5_r1",
 "235bv14cg_rlc1_aux0.02_basecritic_g0.5_r1",
 # triage 2026-07-01: latest real eval acc<70 (collapsed) -> do not continue
 "235bv5b_rlc1.0_aux0.005_basecritic_r73","235bv11e16_critical_path_rlc0.5_aux0.01_stoch",
 "235bv11e16_topn_load_rlc0.5_aux0.01_stoch"}

FLOOR=1500
TOL=60          # off-grid label drift: an eval within +-TOL of an on-disk iter counts as that ckpt
REWARDS=("critical_path","per_token_load_weighted","per_token","topn","entropy")

# ---------- declared PLANNED grids (fresh-start targets that may not exist on disk yet) ----------
def g(x): return ("%g"%x)
PLANNED={}   # cell -> dict(fresh env)
# v14cg: critic x gamma x rlc x aux  (kl=0)  -- mirrors run_sweep_critgamma.py
for gm in (0.3,0.5):
    for rlc in (0.5,1.0):
        for aux in (0.001,0.003,0.005,0.008,0.012,0.016,0.02):
            c="235bv14cg_rlc%s_aux%s_basecritic_g%s_r1"%(g(rlc),g(aux),g(gm))
            PLANNED[c]=dict(BASELINE="critic",GAMMA=g(gm),RLC=g(rlc),AUX=g(aux),KL="0",LM="0",
                            REWARD_TYPE="per_token_load_weighted")
for aux in (0.001,0.006,0.012,0.02):
    c="235bv14cg_rlc0.5_aux%s_basecritic_g0.8_r1"%g(aux)
    PLANNED[c]=dict(BASELINE="critic",GAMMA="0.8",RLC="0.5",AUX=g(aux),KL="0",LM="0",
                    REWARD_TYPE="per_token_load_weighted")
# v15klcg: KL x critic x gamma  (the named hole) -- 18 cells
for kl in (0.0001,0.001,0.01):
    for gm in (0,0.5):
        for (rlc,aux) in ((0.5,0.001),(1.0,0.005),(1.0,0.02)):
            c="235bv15klcg_rlc%s_aux%s_basecritic_g%s_kl%s_r1"%(g(rlc),g(aux),g(gm),g(kl))
            PLANNED[c]=dict(BASELINE="critic",GAMMA=g(gm),RLC=g(rlc),AUX=g(aux),KL=g(kl),LM="0",
                            REWARD_TYPE="per_token_load_weighted")
# v16cgr: low-rlc x gamma CORRECTIVE probe -- tests whether reducing rlc stabilizes gamma>0
# (rlc1xgamma collapsed; lr was fixed at 1e-4 across the whole v14cg grid). 8 cells.
for gm in (0.3,0.5):
    for rlc in (0.1,0.25):
        for aux in (0.001,0.003):
            c="235bv16cgr_rlc%s_aux%s_basecritic_g%s_r1"%(g(rlc),g(aux),g(gm))
            PLANNED[c]=dict(BASELINE="critic",GAMMA=g(gm),RLC=g(rlc),AUX=g(aux),KL="0",LM="0",
                            REWARD_TYPE="per_token_load_weighted")
# seed-replication of the corner-dominating config (76.37@8179, 2026-07-01): confirm across seeds
for sd in ("2027","2028"):
    c="235bv15klcg_rlc0.5_aux0.001_basecritic_g0_kl0.001_seed%s_r1"%sd
    PLANNED[c]=dict(BASELINE="critic",GAMMA="0",RLC="0.5",AUX="0.001",KL="0.001",LM="0",
                    REWARD_TYPE="per_token_load_weighted",SEED=sd)

def gen(n):
    m=re.match(r"(235bv[0-9]+[a-z]*)",n); return m.group(1) if m else None

# ---------- scan disk ----------
disk=defaultdict(set)
ondisk=set()
for d in sorted(os.listdir(R)):
    if not gen(d) or not os.path.isdir(os.path.join(R,d)): continue
    ondisk.add(d)
    for p in glob.glob(os.path.join(R,d,"**","iter_*"),recursive=True):
        m=re.search(r"iter_0*(\d+)$",p)
        if m and os.path.isdir(p): disk[d].add(int(m.group(1)))

# ---------- scan evals ----------
ev=defaultdict(set)
with open(BCSV) as f:
    for row in csv.DictReader(f):
        rn=row.get("run_name","").strip()
        if not gen(rn): continue
        it=row.get("bench_iteration") or row.get("train_iters") or ""
        try: it=int(float(it))
        except: continue
        if row.get("benchmark_avg","").strip() not in ("","nan"): ev[rn].add(it)
tl={}
if os.path.exists(TCSV):
    with open(TCSV) as f:
        for row in csv.DictReader(f):
            try: tl[row.get("run_name","").strip()]=int(float(row.get("latest_iter","") or 0))
            except: pass

universe=sorted(ondisk | set(PLANNED))

def maxiter(c):
    v=[0]+[max(s) for s in (disk[c],ev[c]) if s]
    if c in tl: v.append(tl[c])
    return max(v)
def is_evaled(c,it): return any(abs(it-e)<=TOL for e in ev[c])
def missing(c):      return sorted(i for i in disk[c] if i>=FLOOR and not is_evaled(c,i))
def is_norl(c):      return "norl" in c
def feats(c):
    rw="per_token_load_weighted"
    for r in REWARDS:
        if r in c: rw=r; break
    base="critic" if ("basecritic" in c or re.search(r"_critic[_$]|critic_ppo",c)) else "mean"
    return (rw,
            (re.search(r"rlc([0-9.]+)",c) or [None,"0"])[1] if re.search(r"rlc([0-9.]+)",c) else "0",
            (re.search(r"_aux([0-9.]+)",c).group(1) if re.search(r"_aux([0-9.]+)",c) else "0"),
            (re.search(r"_kl([0-9.]+)",c).group(1) if re.search(r"_kl([0-9.]+)",c) else "0"),
            (re.search(r"_g([0-9.]+)",c).group(1) if re.search(r"_g([0-9.]+)",c) else "0"),
            base)
def sig(c):
    rw,rlc,aux,kl,gm,base=feats(c); return (rw,rlc,aux,kl,gm,base)
def fresh_env(c):
    if c in PLANNED: return dict(PLANNED[c])
    rw,rlc,aux,kl,gm,base=feats(c)
    return dict(BASELINE=base,GAMMA=gm,RLC=rlc,AUX=aux,KL=kl,LM=
                (re.search(r"_lm([0-9.]+)",c).group(1) if re.search(r"_lm([0-9.]+)",c) else "0"),
                REWARD_TYPE=rw)

recs=[]
for c in universe:
    m=maxiter(c); miss=missing(c)
    fr = m>=3000 and not miss
    recs.append(dict(c=c,gen=gen(c),m=m,miss=miss,fr=fr,sig=sig(c),on=c in ondisk))
covered=set(r["sig"] for r in recs if r["fr"])

def cat(r):
    if r["gen"] in DROP: return "DROP"
    if r["c"] in COLLAPSED: return "DROP"   # collapsed rlc1xgamma cells: stop auto-continuing
    if r["fr"]: return "DONE"
    if r["sig"] in covered and r["m"]<3000: return "COVERED"        # config reached fruition elsewhere
    if r["m"]==0:
        if is_norl(r["c"]): return "NORL_NEVER"                     # needs a no-RL fresh-start path
        rlc=fresh_env(r["c"])["RLC"]
        if rlc in ("0","0.0"): return "NEEDS_MANUAL"                # RL cell but rlc/aux not in name
        return "NEVER"
    if r["m"]<3000: return "CONT"
    return "EVAL"

B=defaultdict(list)
for r in recs: B[cat(r)].append(r)

# ---------- emit ----------
def _prio(c):  # campaign 2026-07-01: P1 frontier-critical first, then grids, then legacy
    if c.startswith("235bv15klcg_rlc0.5"): return 0
    if c.startswith("235bv16cgr"): return 1
    if "seed202" in c: return 0
    if c.startswith("235bv15klcg"): return 2
    if c.startswith("235bv14cg"): return 3
    return 9
with open(os.path.join(OUT,"gap_cont.txt"),"w") as f:
    for r in sorted(B["CONT"],key=lambda r:(_prio(r["c"]),r["c"])): f.write(r["c"]+"\n")
with open(os.path.join(OUT,"gap_fresh.txt"),"w") as f:
    for r in sorted(B["NEVER"],key=lambda r:(_prio(r["c"]),r["c"])):
        e=fresh_env(r["c"]); e["RUN_NAME"]=r["c"]
        kv=" ".join("%s=%s"%(k,e[k]) for k in ("RUN_NAME","BASELINE","GAMMA","RLC","AUX","KL","LM","REWARD_TYPE"))
        f.write(r["c"]+"|"+kv+"\n")
with open(os.path.join(OUT,"gap_eval.txt"),"w") as f:
    for r in recs:
        if r["gen"] in DROP: continue
        for it in r["miss"]: f.write("%s %d\n"%(r["c"],it))
with open(os.path.join(OUT,"gap_norl.txt"),"w") as f:
    for r in sorted(B["NORL_NEVER"],key=lambda r:r["c"]): f.write(r["c"]+"\n")
with open(os.path.join(OUT,"gap_manual.txt"),"w") as f:
    for r in sorted(B["NEEDS_MANUAL"],key=lambda r:r["c"]): f.write(r["c"]+"\n")

# ---------- summary ----------
print("=== gap_manifest: universe=%d (on-disk %d + planned-only %d) FLOOR=%d ==="
      %(len(recs),len(ondisk),len(set(PLANNED)-ondisk),FLOOR))
for k in ("DONE","COVERED","EVAL","CONT","NEVER","NORL_NEVER","NEEDS_MANUAL","DROP"):
    print("  %-12s %d"%(k,len(B[k])))
ndebt=sum(len(r["miss"]) for r in recs if r["gen"] not in DROP)
print("  unevaluated >=FLOOR checkpoints (eval debt): %d"%ndebt)
print("  LIVE GAP (EVAL+CONT+NEVER) = %d"%(len(B["EVAL"])+len(B["CONT"])+len(B["NEVER"])))
def bygen(bucket):
    d=defaultdict(int)
    for r in B[bucket]: d[r["gen"]]+=1
    return dict(sorted(d.items()))
print("\n  CONT  by gen:",bygen("CONT"))
print("  NEVER by gen:",bygen("NEVER"),"  (incl planned v14cg g0.3 + v15klcg)")
print("  NORL_NEVER  :",[r["c"] for r in B["NORL_NEVER"]])
print("  NEEDS_MANUAL:",[r["c"] for r in B["NEEDS_MANUAL"]])
print("\n  v15klcg planned (18): present=%d / fresh-needed=%d"
      %(sum(1 for c in PLANNED if c.startswith('235bv15') and c in ondisk),
        sum(1 for c in PLANNED if c.startswith('235bv15') and c not in ondisk)))
print("\nwrote: %s/{gap_cont,gap_fresh,gap_eval,gap_norl}.txt"%OUT)

# ---------- live machine snapshot (auto-refreshed every supervisor cycle) ----------
def _sq(args):
    try: return subprocess.run("squeue -u jonathanp -h "+args,shell=True,capture_output=True,text=True).stdout
    except Exception: return ""
qall=[l for l in _sq("-o '%j %t'").splitlines() if l.strip()]
qtot=len(qall); qrun=sum(1 for l in qall if l.split()[-1]=="R"); qpd=qtot-qrun
def _c(tag): return sum(1 for l in qall if tag in l)
ndebt=sum(len(r["miss"]) for r in recs if r["gen"] not in DROP)
live=len(B["EVAL"])+len(B["CONT"])+len(B["NEVER"])
AUTO=os.path.join(os.path.dirname(os.path.abspath(__file__)),"GAP_STATUS_AUTO.md")
try:
    with open(AUTO,"w") as f:
        f.write("# Gap-closure — live machine snapshot (auto-generated by gap_manifest.py each cycle)\n\n")
        f.write("**Updated:** %s UTC  ·  do not hand-edit (regenerated every supervisor cycle).\n"
                "See `GAP_CLOSURE_STATUS.md` for the curated plan/narrative.\n\n"
                %datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        f.write("**LIVE GAP = %d**  (EVAL %d + CONT %d + NEVER %d)  ·  eval-debt %d surviving ckpts\n\n"
                %(live,len(B["EVAL"]),len(B["CONT"]),len(B["NEVER"]),ndebt))
        f.write("| bucket | N |\n|---|---|\n")
        for k in ("DONE","COVERED","EVAL","CONT","NEVER","NORL_NEVER","NEEDS_MANUAL","DROP"):
            f.write("| %s | %d |\n"%(k,len(B[k])))
        f.write("\n**Queue:** total %d · running %d · pending %d  ·  fresh_=%d c3000_=%d v15klcg=%d\n\n"
                %(qtot,qrun,qpd,_c("fresh_"),_c("c3000_"),_c("v15klcg")))
        f.write("**CONT by gen:** %s\n\n"%bygen("CONT"))
        f.write("**NEVER by gen:** %s\n"%bygen("NEVER"))
    print("wrote live snapshot: %s"%AUTO)
except Exception as e:
    print("auto-status write failed: %s"%e)
