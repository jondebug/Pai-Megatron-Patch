import csv, os, tempfile, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_metadata import parse_meta
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
APPLY=os.environ.get("APPLY","0")=="1"
FIELDS=["category","rl_enabled","aux_enabled","rl_reward_type","rl_loss_coeff","aux_loss_coeff","kl_loss_coeff","lm_reward_coeff"]
rows=list(csv.DictReader(open(CSV))); hdr=list(rows[0].keys())
filled=collections={}
import collections as _c
filled=_c.Counter(); rows_touched=0
for r in rows:
    # SCOPE: only rows whose category is blank (the cells the metadata bug affects),
    # and only when the run_name parses to a confident category. Never touch rows that
    # already have a category (their old-convention metadata is authoritative).
    if (r.get("category") or "").strip():
        continue
    m=parse_meta(r["run_name"])
    if m["category"]=="other":
        continue
    touched=False
    for k in FIELDS:
        if not (r.get(k) or "").strip() and m.get(k,""):   # fill ONLY if blank
            r[k]=m[k]; filled[k]+=1; touched=True
    if touched: rows_touched+=1
print("rows touched:",rows_touched,"| fills per field:",dict(filled))
# integrity: row count + that no non-blank was changed (we only touched blanks)
if APPLY and rows_touched>0:
    os.system("cp %s %s.bak_meta"%(CSV,CSV))
    fd,tmp=tempfile.mkstemp(dir=os.path.dirname(CSV),suffix=".csv"); os.close(fd)
    with open(tmp,"w",newline="") as fo:
        w=csv.DictWriter(fo,fieldnames=hdr); w.writeheader(); w.writerows(rows); fo.flush(); os.fsync(fo.fileno())
    os.replace(tmp,CSV); print("WROTE CSV (%d rows); backup .bak_meta"%len(rows))
else:
    print("DRY-RUN (APPLY=1 to write)")
