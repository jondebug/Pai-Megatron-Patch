"""Collect serving-time expert-routing distribution via vLLM
--enable-return-routed-experts (mechanism from O. Ullman Argov's snippet 15741,
dataset replaced with the campaign's seeded number-salad prompts).
Drop-in CLI replacement for http_bench.py in the serve A/B launcher."""
import argparse, base64, io, json, os, time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True); p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--prompts", type=int, default=128)
    p.add_argument("--plen", type=int, default=256)
    p.add_argument("--gen", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    with urllib.request.urlopen(a.url + "/v1/models", timeout=30) as r:
        model_id = json.load(r)["data"][0]["id"]

    def onem(i):
        import random
        rnd = random.Random((a.seed, i).__hash__())
        prompt = " ".join(str(rnd.randint(0, 9999)) for _ in range(max(1, a.plen // 2)))
        body = json.dumps({"model": model_id,
                           "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": a.gen, "temperature": 0.0,
                           "ignore_eos": True}).encode()
        req = urllib.request.Request(a.url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
        blob = data["choices"][0].get("routed_experts")
        if blob is None:
            return None
        return np.load(io.BytesIO(base64.b64decode(blob)))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        arrs = [x for x in ex.map(onem, range(a.prompts)) if x is not None]
    if not arrs:
        print(f"{a.name}: NO routed_experts in responses — server flag missing?")
        return
    shape = arrs[0].shape
    # expected [tokens, layers, topk]; accumulate per-layer expert counts
    L = shape[1]; E = 128
    counts = np.zeros((L, E), dtype=np.int64)
    for arr in arrs:
        for l in range(L):
            counts[l] += np.bincount(arr[:, l, :].ravel(), minlength=E)[:E]
    npy_path = a.out.replace(".json", f"_{a.name}_heatmap.npy")
    np.save(npy_path, counts)

    summary = {"name": a.name, "n_requests": len(arrs), "blob_shape": list(shape),
               "total_tokens": int(sum(x.shape[0] for x in arrs)),
               "wall_s": time.perf_counter() - t0, "heatmap": npy_path}
    for ep in (8, 16, 32, 64):
        per_rank = counts.reshape(L, ep, E // ep).sum(axis=2)  # contiguous mapping
        imb = per_rank.max(axis=1) / np.maximum(per_rank.mean(axis=1), 1)
        summary[f"ep{ep}_imbalance_mean"] = float(imb.mean())
        summary[f"ep{ep}_imbalance_max"] = float(imb.max())
    data = []
    if os.path.exists(a.out):
        data = json.load(open(a.out))
    data.append(summary)
    json.dump(data, open(a.out, "w"), indent=1)
    print(f"{a.name}: reqs={len(arrs)} shape={shape} "
          + " ".join(f"ep{e}={summary[f'ep{e}_imbalance_mean']:.2f}x" for e in (8, 16, 32, 64)))

if __name__ == "__main__":
    main()
