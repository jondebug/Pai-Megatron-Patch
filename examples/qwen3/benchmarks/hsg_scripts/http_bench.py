"""HTTP completion bench with streaming: TTFT + ITL + wall per request."""
import argparse, json, time, os
from concurrent.futures import ThreadPoolExecutor
import urllib.request

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True); p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--prompts", type=int, default=128)
    p.add_argument("--plen", type=int, default=256)
    p.add_argument("--gen", type=int, default=32)
    a = p.parse_args()
    with urllib.request.urlopen(a.url + "/v1/models", timeout=30) as r:
        model_id = json.load(r)["data"][0]["id"]

    def onem(_):
        body = json.dumps({"model": model_id, "prompt": "hello " * a.plen,
                           "max_tokens": a.gen, "temperature": 0.0,
                           "ignore_eos": True, "stream": True}).encode()
        req = urllib.request.Request(a.url + "/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        stamps = []
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    stamps.append(time.perf_counter())
        tend = time.perf_counter()
        ttft = (stamps[0] - t0) * 1000 if stamps else None
        itls = [(b - c) * 1000 for b, c in zip(stamps[1:], stamps[:-1])]
        return {"total": (tend - t0) * 1000, "ttft": ttft, "itls": itls}

    with ThreadPoolExecutor(a.concurrency) as ex:   # warmup
        list(ex.map(onem, range(a.concurrency)))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(onem, range(a.prompts)))
    wall = (time.perf_counter() - t0) * 1000

    lats = sorted(r["total"] for r in res)
    ttfts = sorted(r["ttft"] for r in res if r["ttft"] is not None)
    itls = sorted(x for r in res for x in r["itls"])
    def pct(v, q): return v[min(int(len(v) * q), len(v) - 1)] if v else None
    rec = {"name": a.name, "wall_ms": wall, "n": len(lats),
           "lat_p50": pct(lats, .5), "lat_p95": pct(lats, .95),
           "lat_mean": sum(lats) / len(lats),
           "ttft_p50": pct(ttfts, .5), "ttft_p95": pct(ttfts, .95),
           "itl_p50": pct(itls, .5), "itl_p95": pct(itls, .95),
           "itl_mean": (sum(itls) / len(itls)) if itls else None,
           "concurrency": a.concurrency, "plen": a.plen, "gen": a.gen,
           "tokps": a.prompts * a.gen / (wall / 1000)}
    data = []
    if os.path.exists(a.out):
        data = json.load(open(a.out))
    data.append(rec)
    json.dump(data, open(a.out, "w"), indent=1)
    print(f"{a.name}: wall={wall:.0f}ms p50={rec['lat_p50']:.0f} "
          f"ttft_p50={rec['ttft_p50']:.0f} itl_p50={rec['itl_p50']:.1f} "
          f"tok/s={rec['tokps']:.1f}")

if __name__ == "__main__":
    main()
