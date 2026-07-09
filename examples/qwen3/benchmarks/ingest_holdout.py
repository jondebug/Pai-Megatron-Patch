#!/usr/bin/env python3
# Ingest HOLDOUT eval results (mmlu/gsm8k[/gpqa]) from hlmeval stdout tables into benchmark_results.csv.
# Adds columns mmlu, gsm8k, gpqa, holdout_avg if absent. Upserts onto the existing inf row for
# (cell,iter). Log source: ~/hlmeval_ord_hlmeval_*.out with header "HOLDOUT lm-eval  name=<cell>_iter<N>".
import csv, glob, os, re, tempfile
B="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks"
CSV=B+"/benchmark_results.csv"
LOGS="/lustre/fsw/portfolios/nvr/users/jonathanp"
def parse_log(path):
    txt=open(path,errors="ignore").read()
    m=re.search(r"HOLDOUT lm-eval\s+name=(\S+)_iter(\d+)",txt)
    if not m: return None
    cell,it=m.group(1),int(m.group(2))
    scores={}
    # lm-eval markdown table rows: |mmlu | ... |acc |0.8123|... ; gsm8k: exact_match,strict-match
    # lm-eval table has an up-arrow column between metric and value: |metric|arrow|value|
    for task,pat in (("mmlu",r"^\|mmlu\s*\|.*?\|\s*acc\s*\|[^|]*\|\s*([0-9.]+)"),
                     ("mmlu_pro",r"^\|mmlu_pro\s*\|.*?\|\s*(?:exact_match|acc)\s*\|[^|]*\|\s*([0-9.]+)"),
                     ("bbh",r"^\|bbh\s*\|.*?\|\s*(?:exact_match|acc)\s*\|[^|]*\|\s*([0-9.]+)"),
                     ("gpqa",r"^\|gpqa[^|]*\|.*?\|\s*acc(?:_norm)?\s*\|[^|]*\|\s*([0-9.]+)")):
        for mm in re.finditer(pat,txt,re.M):
            scores[task]=float(mm.group(1))*100.0   # store as %
    # gsm8k: two filter rows (strict-match / flexible-extract) — prefer strict
    strict=re.search(r"strict-match\s*\|\s*\d*\s*\|\s*exact_match\s*\|[^|]*\|\s*([0-9.]+)",txt)
    anyem=re.search(r"^\|gsm8k\s*\|.*?exact_match\s*\|[^|]*\|\s*([0-9.]+)",txt,re.M)
    if strict: scores["gsm8k"]=float(strict.group(1))*100.0
    elif anyem: scores["gsm8k"]=float(anyem.group(1))*100.0
    # gsm8k has two exact_match rows (strict, flexible) — prefer strict (first row)
    return (cell,it,scores) if scores else None
rows=list(csv.DictReader(open(CSV)))
cols=list(rows[0].keys())
for c in ("mmlu","gsm8k","mmlu_pro","bbh","gpqa","holdout_avg"):
    if c not in cols:
        cols.append(c)
        for r in rows: r[c]=""
byki={}
for r in rows:
    it=r.get("bench_iteration") or r.get("train_iters")
    try: byki.setdefault((r["run_name"].strip(),int(float(it)),r["limit"].strip()),r)
    except: pass
n=0
for lg in glob.glob(LOGS+"/hlmeval_ord_*.out"):
    p=parse_log(lg)
    if not p: continue
    cell,it,sc=p
    r=byki.get((cell,it,"inf"))
    if r is None:
        # pretrained baseline or row without inf entry: upsert a fresh row
        r={k:"" for k in cols}; r["run_name"]=cell
        r["bench_iteration"]=str(it); r["limit"]="inf"
        rows.append(r); byki[(cell,it,"inf")]=r
    changed=False
    for t in ("mmlu","gsm8k","mmlu_pro","bbh","gpqa"):
        if t in sc and not (r.get(t) or "").strip():
            r[t]="%.2f"%sc[t]; changed=True
    vals=[float(r[t]) for t in ("mmlu","gsm8k","mmlu_pro","bbh","gpqa") if (r.get(t) or "").strip()]
    if vals: r["holdout_avg"]="%.2f"%(sum(vals)/len(vals))
    if changed: n+=1
print("holdout ingest: %d rows updated"%n)
if n:
    fd,tmp=tempfile.mkstemp(dir=B)
    with os.fdopen(fd,"w",newline="") as o:
        w=csv.DictWriter(o,fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in cols})
    os.replace(tmp,CSV); print("CSV written (+holdout cols)")
