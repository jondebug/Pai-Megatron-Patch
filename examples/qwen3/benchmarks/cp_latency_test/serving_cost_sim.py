"""
Serving-cost simulation. For each (cell, EP, regime, bs) where we have TTFT and TPOT
measurements, compose total_latency(N_out) = TTFT + (N_out - 1) * TPOT and find the
break-even N* where r15 vs pretrained crosses zero.

Q: For what output-token-count N does router-RL (r15) net win vs pretrained, at each
   serving configuration?

Inputs (all on /lustre/.../cp_latency_results):
  Prefill EP=8 bs sweep:  bs_sweep_29332445.json (bs=1,2,4,8, plen=8192, max_tokens=8)
  Decode EP=8 (mt=128):   vllm_ep8_sweep_{pretrained,r15}_*.json (plen=256, bs={1,8,64,256,512,1024})
  Decode EP=32 (mt=128):  vllm_ep32_sweep_{pretrained,r15}_*.json (plen=256, bs={64,512})
  Decode EP=64 (mt=128):  vllm_ep64_sweep_{pretrained,r15}_*.json (plen=256, bs={64,512})

Outputs:
  /lustre/.../cp_latency_results/serving_cost_simulation.json
  stdout summary
"""
import json, os
import numpy as np

RES = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

def load(name):
    return json.load(open(os.path.join(RES, name + ".json")))

def cells_from(d, max_tokens=None):
    out = []
    mt = d.get("config", {}).get("max_tokens") or max_tokens or 8
    for m in d.get("models", []):
        for ck, cv in m.get("cells", {}).items():
            out.append({
                "name": m["name"],
                "plen": cv.get("prompt_len"),
                "bs":   cv.get("batch_size"),
                "ttft_ms": cv.get("ttft_ms_mean"),
                "e2e_ms":  cv.get("end_to_end_ms_mean"),
                "max_tok": mt,
                "ttft_std": cv.get("ttft_ms_std"),
                "e2e_std": cv.get("end_to_end_ms_std"),
            })
    return out

def tpot_of(c):
    n = max(c["max_tok"] - 1, 1)
    return (c["e2e_ms"] - c["ttft_ms"]) / n

# === Collect data per (regime, EP, bs) ===
records = {}  # (regime, ep, plen, bs) -> {pre: cell, r15: cell}

def add(regime, ep, name_pre, name_r15):
    for j_name, role in [(name_pre, "pre"), (name_r15, "r15")]:
        try:
            d = load(j_name)
        except FileNotFoundError:
            print(f"  missing {j_name}")
            continue
        for c in cells_from(d):
            k = (regime, ep, c["plen"], c["bs"])
            if k not in records:
                records[k] = {}
            records[k][role] = c

# Prefill EP=8 bs sweep (max_tok=8). Single file, both cells.
d = load("bs_sweep_29332445")
for m in d.get("models", []):
    role = "pre" if m["name"].startswith("pre") else "r15"
    for ck, cv in m["cells"].items():
        k = ("prefill", 8, cv["prompt_len"], cv["batch_size"])
        if k not in records: records[k] = {}
        records[k][role] = {"name": m["name"], "plen": cv["prompt_len"], "bs": cv["batch_size"],
                           "ttft_ms": cv["ttft_ms_mean"], "e2e_ms": cv["end_to_end_ms_mean"], "max_tok": 8,
                           "ttft_std": cv.get("ttft_ms_std", 0), "e2e_std": cv.get("end_to_end_ms_std", 0)}

# Decode EP=8/32/64
add("decode", 8,  "vllm_ep8_sweep_pretrained_235b_28803089", "vllm_ep8_sweep_r15_cp4682_28803091")
add("decode", 32, "vllm_ep32_sweep_pretrained_235b_28818149", "vllm_ep32_sweep_r15_cp4682_28818150")
add("decode", 64, "vllm_ep64_sweep_pretrained_235b_28818217", "vllm_ep64_sweep_r15_cp4682_28818218")

# === Compute break-even per (regime, EP, bs) ===
print("="*135)
print("SERVING-COST DECOMPOSITION: TTFT and TPOT per (regime, EP, plen, bs)")
print("="*135)
print(f"{'regime':<8} {'EP':<4} {'plen':<6} {'bs':<6} {'TTFT_pre':<10} {'TTFT_r15':<10} {'ΔTTFT_ms':<10} {'TPOT_pre':<10} {'TPOT_r15':<10} {'ΔTPOT/tok':<10} {'N*':<12} {'note':<20}")

out_rows = []
for k in sorted(records.keys()):
    rec = records[k]
    if "pre" not in rec or "r15" not in rec: continue
    p, r = rec["pre"], rec["r15"]
    tt_p, tt_r = p["ttft_ms"], r["ttft_ms"]
    tp_p, tp_r = tpot_of(p), tpot_of(r)
    d_ttft = tt_r - tt_p
    d_tpot = tp_r - tp_p
    # total_latency_r15(N) - total_latency_pre(N) = d_ttft + (N-1)*d_tpot
    # zero at N* = 1 - d_ttft / d_tpot
    if abs(d_tpot) < 1e-9:
        nstar = float("inf"); note = "d_tpot~0"
    else:
        nstar = 1.0 - d_ttft / d_tpot
        if d_ttft >= 0 and d_tpot >= 0:
            note = "r15 always slower (TTFT+TPOT both worse)"
        elif d_ttft <= 0 and d_tpot <= 0:
            note = "r15 always faster"
        elif d_ttft < 0 and d_tpot > 0:
            note = f"r15 wins for N<{int(nstar)}"
        elif d_ttft > 0 and d_tpot < 0:
            note = f"r15 wins for N>{int(nstar)}"
    nstar_disp = f"{nstar:.0f}" if abs(nstar) < 1e6 else ("∞" if nstar > 0 else "-∞")
    regime, ep, plen, bs = k
    print(f"{regime:<8} {ep:<4} {plen:<6} {bs:<6} {tt_p:<10.2f} {tt_r:<10.2f} {d_ttft:<+10.2f} {tp_p:<10.2f} {tp_r:<10.2f} {d_tpot:<+10.3f} {nstar_disp:<12} {note}")
    out_rows.append({"regime": regime, "ep": ep, "plen": plen, "bs": bs,
                     "ttft_pre": tt_p, "ttft_r15": tt_r, "d_ttft": d_ttft,
                     "tpot_pre": tp_p, "tpot_r15": tp_r, "d_tpot": d_tpot,
                     "nstar": float(nstar) if abs(nstar) < 1e10 else None,
                     "note": note})

# === Plot total-latency curves at canonical N values ===
print()
print("="*135)
print("TOTAL LATENCY at fixed N (ms) — r15 / pretrained ratio, percent diff")
print("="*135)
print(f"{'regime':<8} {'EP':<4} {'plen':<6} {'bs':<6} {'N=1':<12} {'N=10':<12} {'N=50':<12} {'N=100':<12} {'N=500':<12} {'N=1000':<12}")
for k in sorted(records.keys()):
    rec = records[k]
    if "pre" not in rec or "r15" not in rec: continue
    p, r = rec["pre"], rec["r15"]
    regime, ep, plen, bs = k
    row_cells = []
    for N in [1, 10, 50, 100, 500, 1000]:
        lp = p["ttft_ms"] + (N-1) * tpot_of(p)
        lr = r["ttft_ms"] + (N-1) * tpot_of(r)
        pct = (lr - lp) / lp * 100
        row_cells.append(f"{pct:+.2f}%")
    print(f"{regime:<8} {ep:<4} {plen:<6} {bs:<6} " + " ".join(f"{c:<12}" for c in row_cells))

# Save
out = RES + "/serving_cost_simulation.json"
with open(out, "w") as f:
    json.dump({"rows": out_rows}, f, indent=2)
print(f"\nSaved -> {out}")
