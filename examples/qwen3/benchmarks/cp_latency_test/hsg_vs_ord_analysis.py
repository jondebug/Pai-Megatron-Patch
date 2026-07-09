"""Direct HSG-vs-ORD comparison for pretrained Qwen3-235B-A22B on EP=8."""
import json, os

RES_HSG = "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/cp_latency_results"

# HSG pretrained data (from bench outputs)
hsg_prefill = json.load(open(f"{RES_HSG}/hsg_phaseA_pre_A_3808341.json"))
hsg_decode  = json.load(open(f"{RES_HSG}/hsg_dec_pre_3808375.json"))

# ORD pretrained baseline (from Phase A + earlier campaign)
ord_prefill = {
    1: {"ttft_mean": 53.98, "ttft_std": 0.27, "e2e": 98.71, "n": 150},
    2: {"ttft_mean": 89.28, "ttft_std": 15.82, "e2e": 143.20, "n": 150},
    4: {"ttft_mean": 116.91, "ttft_std": 18.91, "e2e": 179.58, "n": 150},
    8: {"ttft_mean": 155.44, "ttft_std": 21.44, "e2e": 227.46, "n": 150},
    16: {"ttft_mean": 196.46, "ttft_std": 13.57, "e2e": 277.74, "n": 150},
}
ord_decode = {
    # Old campaign: plen=256, bs=64,512 was 2 trials from vllm_ep8_sweep_pretrained_235b
    64:   {"ttft": 112.4, "tpot": 112.39},
    512:  {"tpot": 226.59},  # actual mean from earlier
    1024: {"tpot": 440.83},
}

print("="*80)
print("HSG GB200 vs ORD H100 — Pretrained EP=8 Prefill (plen=8192)")
print("="*80)
print(f"{'bs':<4} {'HSG TTFT':<18} {'ORD TTFT':<18} {'GB200 speedup':<15}")
for m in hsg_prefill["models"]:
    for ck, cv in sorted(m["cells"].items(), key=lambda x: x[1]["batch_size"]):
        bs = cv["batch_size"]
        hsg_t = cv["ttft_ms_mean"]
        ord_t = ord_prefill.get(bs, {}).get("ttft_mean", None)
        sp = ord_t / hsg_t if ord_t else 0
        print(f"{bs:<4} {hsg_t:6.2f} ± {cv.get('ttft_ms_std',0):5.2f} ms  {ord_t if ord_t else 'n/a':>8}          {sp:.2f}×" if ord_t else f"{bs:<4} {hsg_t:6.2f} ms")

print()
print("="*80)
print("HSG GB200 vs ORD H100 — Pretrained EP=8 Decode (plen=256)")
print("="*80)
print(f"{'bs':<6} {'HSG TPOT':<15} {'ORD TPOT':<15} {'GB200 TPOT speedup':<15}")
for m in hsg_decode["models"]:
    for ck, cv in sorted(m["cells"].items(), key=lambda x: x[1]["batch_size"]):
        bs = cv["batch_size"]
        # HSG uses max_tokens=32, so TPOT = (e2e-ttft)/31
        hsg_ttft = cv["ttft_ms_mean"]; hsg_e2e = cv["end_to_end_ms_mean"]
        hsg_tpot = (hsg_e2e - hsg_ttft) / 31 if hsg_e2e > hsg_ttft else 0
        ord_tpot = ord_decode.get(bs, {}).get("tpot", None)
        sp = ord_tpot / hsg_tpot if (ord_tpot and hsg_tpot > 0) else 0
        note = ""
        if bs == 1024: note = " (memory pressure)"
        line = f"{bs:<6} {hsg_tpot:6.2f} ms/tok" + (f"  {ord_tpot:7.2f} ms/tok    {sp:.2f}×" if ord_tpot else "")
        print(line + note)

print()
print("Throughput at bs=512 EP=8 decode:")
for m in hsg_decode["models"]:
    for ck, cv in m["cells"].items():
        if cv["batch_size"] in [512, 1024, 256, 64]:
            print(f"  bs={cv['batch_size']} decode_tps={cv.get('decode_tps',0):.1f} tok/s")
