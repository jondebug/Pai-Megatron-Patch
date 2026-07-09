#!/usr/bin/env python3
# Delete hf_converted_iter{it}_cp dirs whose (cell,iter) HAS an inf eval -- NO frontier exemption
# (HF is re-convertible). Keeps HF for not-yet-evaluated checkpoints (those are eval inputs).
# Never touches distcp. DRY-RUN unless DELETE=1.
import csv,glob,re,os,shutil,subprocess
ROOT="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/output_router_finetuning"
CSV="/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/Pai-Megatron-Patch/examples/qwen3/benchmarks/benchmark_results.csv"
DELETE=os.environ.get("DELETE","0")=="1"
def f(x):
    try: return float(x)
    except: return None
evald=set()
for r in csv.DictReader(open(CSV)):
    if r["limit"].strip()=="inf" and f(r.get("benchmark_avg")) and f(r.get("benchmark_avg"))>=40:
        it=r.get("bench_iteration") or r.get("train_iters")
        try: evald.add((r["run_name"].strip(),int(float(it))))
        except: pass
freed=0; n=0
for hf in glob.glob(ROOT+"/235bv*/checkpoint/*/hf_converted_iter*_cp"):
    m=re.search(r"/(235bv[^/]+)/checkpoint/.*/hf_converted_iter([0-9]+)_cp$",hf)
    if not m: continue
    if (m.group(1),int(m.group(2))) not in evald: continue   # not yet evaluated -> keep (eval input)
    try: sz=int(subprocess.check_output(["du","-sb",hf]).split()[0])
    except: sz=0
    if DELETE:
        try: shutil.rmtree(hf); freed+=sz; n+=1
        except Exception as e: print("rm FAILED",hf,e)
    else: freed+=sz; n+=1
print("%s %d evaluated-HF dirs, %.2f TB (distcp untouched)"%("DELETED" if DELETE else "would delete",n,freed/1e12))
