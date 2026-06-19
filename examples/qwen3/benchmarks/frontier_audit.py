# Coverage-complete frontier audit: goes over EVERY complete checkpoint >800 iters,
# checks acc + CP for each, and classifies FRONTIER / DOMINATED / EXCLUDED(missing data).
# EXCLUDED is NEVER counted as dominated -- it means "we cannot classify it yet, go measure it".
import csv, glob, re, sys, importlib.util
B="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks"
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
sys.path.insert(0,B)
from cell_metadata import parse_meta
def f(x):
    try: return float(x)
    except: return 0.0
# 1) universe: every complete (>=32 shard) 235B checkpoint > 800 iters on disk
universe=set()
for itd in glob.glob(ROOT+"/*/checkpoint/*/iter_*"):
    m=re.search(r"iter_0*([0-9]+)$",itd)
    if not m or int(m.group(1))<800: continue
    cell=itd.split("/output_router_finetuning/")[1].split("/")[0]
    if not cell.startswith("235b"): continue
    if len(glob.glob(itd+"/*.distcp"))>=32: universe.add((cell,int(m.group(1))))
# 2) CSV: acc + cp per (cell,iter)
acc={}; cp={}
for r in csv.DictReader(open(B+"/benchmark_results.csv")):
    if r["limit"].strip()!="inf": continue
    try: it=int(float(r.get("bench_iteration") or r.get("train_iters")))
    except: continue
    k=(r["run_name"],it)
    if f(r.get("benchmark_avg"))>0: acc[k]=f(r.get("benchmark_avg"))
    c=f(r.get("cp_critical_eval"))   # CANONICAL CP ONLY (legacy eval_crit_path is stale/unreliable, never used)
    if c and c>1000: cp[k]=c
# 3) classify
eligible=[]; excl_noacc=[]; excl_nocp=[]
for k in universe:
    ha=k in acc; hc=k in cp
    if ha and hc: eligible.append((k,parse_meta(k[0])["category"],cp[k],acc[k]))
    elif not ha: excl_noacc.append(k)      # not evaluated yet -> NOT dominated
    else: excl_nocp.append(k)              # has acc, no CP -> NOT dominated
# 4) frontier per category over ELIGIBLE only; report dominated vs frontier
def is_dom(cat,c,a,pool):
    red=8800-c
    return any((8800-c2)>=red and a2>=a and ((8800-c2)>red or a2>a) for (c2,a2) in pool)
print("=== FRONTIER AUDIT: universe = %d complete checkpoints >800 (235B) ==="%len(universe))
print("eligible (acc+CP): %d | EXCLUDED no-accuracy: %d | EXCLUDED no-CP: %d"%(len(eligible),len(excl_noacc),len(excl_nocp)))
for cat in ("aux_only","rl_only","rl+aux"):
    pool=[(c,a) for (k,ct,c,a) in eligible if ct==cat]
    fr=[(k,c,a) for (k,ct,c,a) in eligible if ct==cat and not is_dom(cat,c,a,[p for p in pool if p!=(c,a)])]
    print("-- %s: %d eligible, %d FRONTIER, %d dominated"%(cat,len(pool),len(fr),len(pool)-len(fr)))
print("=== !!! EXCLUDED (unclassified -- MUST eval/measure before any 'dominated' claim) !!! ===")
for k in sorted(excl_noacc): print("   NO-ACCURACY (pending eval): %s@%d"%(k[0][:48],k[1]))
for k in sorted(excl_nocp): print("   NO-CP: %s@%d"%(k[0][:48],k[1]))
