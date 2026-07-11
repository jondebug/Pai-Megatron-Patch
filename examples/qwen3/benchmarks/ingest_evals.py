#!/usr/bin/env python3
"""Fast, wandb-free ingester: upsert completed eval results into benchmark_results.csv.
Accuracy is captured from the STDOUT MARKDOWN TABLE in eval logs (results.json is NOT written
reliably -- OSError filename-too-long). Sources:
  (1) benchmark_logs/conv_eval_*.out   (submit_convert_and_eval; header 'CONVERT + EVAL: <cell> iter=<N> limit=<L>')
  (2) lmeval_ord_*.out                 (run_lm_eval_ord; header 'lm-eval ... name=<cell>_iter<N> ... limit=<L>')
  (3) <ckpt>/benchmark_iter<N>_limit<L>/accuracy_summary.json  (legacy, if present)
Preserves existing rows/columns; carries hyperparams + eval_crit_path/eval_lm_loss from any
existing row of the same cell; tags cluster from the checkpoint path root.
"""
import csv, json, re, glob, os, sys, tempfile
from datetime import datetime
from pathlib import Path

# Stamp every row this run touches (added/updated) so CSV recency is trackable.
# Without this, "newest timestamp" silently reflected the last collect_results_unified.py
# run (June 2), making freshly-ingested rows look stale. See OPEN_ITEMS.md.
TS = datetime.now().strftime("%Y-%m-%d %H:%M")

CSV = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
ROOT = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
BLOG = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/benchmark_logs"
OLOG = "/lustre/fsw/portfolios/nvr/users/jonathanp"   # lmeval_ord_*.out live here

# Canonical schema. NEVER derive the schema from the live CSV alone: if a previous
# write was truncated (e.g. OSError: Disk quota exceeded), the live CSV can be empty or
# a 1-column "cluster" stub, and deriving cols from it would permanently destroy the
# registry. We instead always anchor on this canonical header and union in any real
# extra columns found in a valid source.
CANONICAL_COLS = ["run_name","bench_iteration","sweep_id","category","rl_enabled",
    "aux_enabled","train_iters","rl_reward_type","rl_loss_coeff","aux_loss_coeff",
    "kl_loss_coeff","lm_reward_coeff","eval_crit_path","eval_lm_loss","hellaswag",
    "arc_challenge","winogrande","benchmark_avg","benchmark_time_sec","limit","timestamp",
    "comments","checkpoint_path","cluster","wt_decode_tps","wt_ttft_ms","wt_e2e_ms"]

def load_valid_csv(path):
    """Return (rows, cols) from a CSV only if it has the real schema (run_name col).
    Returns (None, None) for a missing / empty / degenerate (truncated) file."""
    try:
        with open(path) as fh:
            rr = list(csv.DictReader(fh))
    except (OSError, IOError):
        return None, None
    if not rr:
        return None, None
    cc = list(rr[0].keys())
    if "run_name" not in cc:            # degenerate stub (e.g. just "cluster") -> reject
        return None, None
    return rr, cc

rows, cols = load_valid_csv(CSV)
if rows is None:
    # Live CSV is empty/degenerate (prior truncated write). Recover from the freshest
    # valid backup so we never start from an empty schema and re-truncate the registry.
    bdir = os.path.dirname(CSV)
    cands = sorted(glob.glob(os.path.join(bdir, "benchmark_results.csv.bak*")),
                   key=lambda p: os.path.getmtime(p), reverse=True)
    for b in cands:
        rows, cols = load_valid_csv(b)
        if rows is not None:
            print(f"WARNING: live CSV was empty/degenerate; recovered schema+rows from {b} "
                  f"({len(rows)} rows)", file=sys.stderr)
            break
    if rows is None:
        rows, cols = [], list(CANONICAL_COLS)
        print("WARNING: no valid CSV or backup found; starting from canonical schema",
              file=sys.stderr)

# Union canonical cols in (preserve existing order, append any missing canonical cols).
for c in CANONICAL_COLS:
    if c not in cols:
        cols.append(c)
idx = {}
for r in rows:
    idx[(r["run_name"], str(r.get("bench_iteration","")), r.get("limit","").strip())] = r

def hyper_template(cell):
    for r in rows:
        if r["run_name"] == cell:
            return {k: r.get(k,"") for k in ("sweep_id","category","rl_enabled","aux_enabled",
                    "train_iters","rl_reward_type","rl_loss_coeff","aux_loss_coeff",
                    "kl_loss_coeff","lm_reward_coeff","eval_crit_path","eval_lm_loss","checkpoint_path")}
    return {}

def carry_cp_lm(cell, it):
    for r in rows:
        if r["run_name"]==cell and str(r.get("bench_iteration",""))==str(it):
            return r.get("eval_crit_path",""), r.get("eval_lm_loss","")
    best=None
    for r in rows:
        if r["run_name"]==cell and r.get("eval_crit_path"):
            try: d=abs(int(float(r["bench_iteration"]))-int(it))
            except: continue
            if best is None or d<best[0]: best=(d,r.get("eval_crit_path",""),r.get("eval_lm_loss",""))
    return (best[1],best[2]) if best else ("","")

RESULTS_ORD = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/lm_eval_results_ord"
def read_cp_json(cell, it):
    """Read auto-measured critical path (+lm_loss) from <cell>_iter<it>_critpath.json."""
    p = os.path.join(RESULTS_ORD, "%s_iter%s_critpath.json" % (cell, it))
    try:
        d = json.load(open(p)); sm = d.get("summary", {})
        cp = sm.get("avg_num_tokens_on_critical_path"); lm = sm.get("avg_lm_loss")
        return ("" if cp is None else round(float(cp), 2), "" if lm is None else round(float(lm), 4))
    except Exception:
        return ("", "")

def _dup_suspect(cell, it, limit, hella, arc, wino):
    # Audit F1 guard (2026-07-11): a NEW eval whose 3-task triple exactly matches a DIFFERENT
    # iteration of the same cell is almost surely a wrong-checkpoint eval (converter latest-file
    # race; P(genuine tie) ~1e-4). Reject + log; operator can force by clearing the sibling row.
    trip=(round(hella*100,2),round(arc*100,2),round(wino*100,2))
    for (c,i,l),r in idx.items():
        if c==cell and l==limit and i!=str(int(it)):
            try:
                if (float(r.get("hellaswag")),float(r.get("arc_challenge")),float(r.get("winogrande")))==trip:
                    return i
            except Exception: pass
    return None

def upsert(cell, it, limit, hella, arc, wino, ckpt_path, src):
    it=str(int(it)); avg=round((hella+arc+wino)/3*100,2)
    _d=_dup_suspect(cell,it,limit,hella,arc,wino)
    if _d is not None:
        print("SUSPECT-DUP: %s@%s triple identical to @%s -- REJECTED (wrong-ckpt eval?)"%(cell,it,_d))
        return "suspect-dup"
    clu="nrt" if "/lustre/fs1/" in (ckpt_path or "") else "ord"
    key=(cell,it,limit)
    if key in idx:
        r=idx[key]
        if not str(r.get("benchmark_avg") or "").strip():
            r.update(hellaswag=round(hella*100,2),arc_challenge=round(arc*100,2),
                     winogrande=round(wino*100,2),benchmark_avg=avg,comments="ingested:"+src,
                     timestamp=TS)
            if not r.get("cluster"): r["cluster"]=clu
            return "updated"
        if not str(r.get("eval_crit_path") or "").strip():
            _cp,_lm=read_cp_json(cell,it)
            if _cp!="":
                r["eval_crit_path"]=_cp
                if _lm!="" and not str(r.get("eval_lm_loss") or "").strip(): r["eval_lm_loss"]=_lm
                return "updated"
        return "have"
    _cpj,_lmj=read_cp_json(cell,it); _cc,_ll=carry_cp_lm(cell,it)
    cp = _cpj if _cpj!="" else _cc; lm = _lmj if _lmj!="" else _ll; base=hyper_template(cell)
    row={k:"" for k in cols}; row.update(base)
    row.update(run_name=cell,bench_iteration=it,limit=limit,
               hellaswag=round(hella*100,2),arc_challenge=round(arc*100,2),winogrande=round(wino*100,2),
               benchmark_avg=avg,eval_crit_path=cp or base.get("eval_crit_path",""),
               eval_lm_loss=lm or base.get("eval_lm_loss",""),
               checkpoint_path=ckpt_path or base.get("checkpoint_path",""),
               cluster=clu,comments="ingested:"+src,timestamp=TS)
    rows.append(row); idx[key]=row
    return "added"

# table-row metric extractor
def metrics(txt):
    def grab(task, metric):
        # row like: |task ... |none  | 0|metric|↑|0.8430|
        m=re.search(r"\|\s*"+re.escape(task)+r"\b.*?\n(?:.*\n)??.*?\|\s*"+metric+r"\s*\|[^\d]*([\d.]+)\s*\|", txt)
        if m: return float(m.group(1))
        # same-line variant
        m=re.search(r"\|\s*"+re.escape(task)+r"\b[^\n]*\|\s*"+metric+r"\s*\|[^\d]*([\d.]+)\s*\|", txt)
        return float(m.group(1)) if m else None
    h=grab("hellaswag","acc_norm"); a=grab("arc_challenge","acc_norm"); w=grab("winogrande","acc")
    return h,a,w

added=upd=have=0
def take(cell,it,lim,h,a,w,ckpt,src):
    global added,upd,have
    if None in (h,a,w): return
    r=upsert(cell,it,lim,h,a,w,ckpt,src)
    added+=r=="added"; upd+=r=="updated"; have+=r=="have"

# Source 1: conv_eval logs
for f in glob.glob(BLOG+"/conv_eval_*.out"):
    txt=open(f,errors="ignore").read()
    m=re.search(r"CONVERT \+ EVAL:\s+(\S+)\s+iter=(\d+)\s+limit=(\S+)", txt)
    if not m: continue
    cell,it,lim=m.group(1),int(m.group(2)),m.group(3).strip()
    h,a,w=metrics(txt); take(cell,it,lim,h,a,w,"","convlog")
# Source 2: lmeval_ord logs
for f in glob.glob(OLOG+"/lmeval_ord_*.out"):
    txt=open(f,errors="ignore").read()
    m=re.search(r"name=(\S+?)(?:_iter(\d+))?\s+hf=(\S+).*?limit=(\S+)", txt)
    if not m: continue
    cell=m.group(1); it=m.group(2); hf=m.group(3); lim=m.group(4).strip()
    # iter precedence: name token (_iterNNNN) > trailing _iterNNNN in cell > HF path iter{N}.
    # Robust to new cell names with extra underscore-fields (basemean/basecritic,
    # rwd<type>, seedN, kl/lm) and a trailing _iterNNNN.
    if it is None:
        mm=re.search(r"_iter(\d+)", cell); it=mm.group(1) if mm else None; cell=re.sub(r"_iter\d+$","",cell)
    if it is None:
        mm=re.search(r"iter(\d+)", hf or ""); it=mm.group(1) if mm else None
    if it is None: continue
    h,a,w=metrics(txt); take(cell,int(it),lim,h,a,w,hf,"ordlog")
# Source 3: accuracy_summary.json
for f in glob.glob(ROOT+"/*/checkpoint/*/benchmark_iter*_limit*/accuracy_summary.json"):
    try: d=json.load(open(f))
    except: continue
    cell=d.get("run_name"); it=d.get("iteration"); lim=str(d.get("limit","")).strip() or "inf"
    h=d.get("hellaswag"); a=d.get("arc_challenge"); w=d.get("winogrande")
    if None in (cell,it,h,a,w): continue
    take(cell,int(it),lim,h/100.0,a/100.0,w/100.0,str(Path(f).parents[1]),"summary")

# Atomic write: build the full file in a temp file in the same dir, fsync, then
# os.replace(). A partial / quota-exceeded write therefore can NEVER truncate the live
# registry -- the live CSV is only ever swapped for a fully-written replacement.
# Pre-write backup so the prior good state is always recoverable.
if rows:
    try:
        import shutil
        if os.path.exists(CSV) and os.path.getsize(CSV) > 0:
            shutil.copy2(CSV, CSV + ".bak_preingest_auto")
    except OSError as e:
        print(f"WARNING: could not write pre-ingest backup: {e}", file=sys.stderr)
    d = os.path.dirname(CSV) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".ingest_tmp_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})
            fo.flush()
            os.fsync(fo.fileno())
        os.replace(tmp, CSV)   # atomic on same filesystem
    except BaseException:
        # Leave the live CSV untouched on any failure (incl. quota) and clean the temp.
        try: os.unlink(tmp)
        except OSError: pass
        raise
else:
    print("ERROR: refusing to write an empty CSV (0 rows) -- live registry left untouched",
          file=sys.stderr)
    sys.exit(1)
print(f"ingest: +{added} added, {upd} updated, {have} already-present, total {len(rows)} rows")
