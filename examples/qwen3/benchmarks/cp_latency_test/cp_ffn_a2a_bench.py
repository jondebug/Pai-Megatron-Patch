#!/usr/bin/env python3
"""cp_ffn_a2a_bench.py -- standalone measurement of the two parameters the
CP->latency model needs, with CORRECT Qwen3-235B-A22B dims:

  (1) Expert-FFN compute curve: median GEMM time vs tokens-on-one-expert.
      This is the function that turns CP (tokens on busiest expert) into time.
      Per-expert SwiGLU: gate(4096->1536), up(4096->1536), down(1536->4096), bf16.

  (2) All-to-all dispatch/combine BW + latency at the model's REAL message
      sizes, on whatever interconnect this job's GPUs span:
        - single node (8 GPUs)  -> NVLink
        - multi node  (N*8 GPUs, ray/torchrun) -> IB
      Decode dispatch payload per step ~ (tokens * top_k) hidden vectors of
      hidden_size*2 bytes, scattered across EP ranks. We sweep realistic
      per-rank message sizes and also the exact decode-step sizes for
      bs in {1,8,32,64,128}.

Run via torchrun (1 proc/GPU). For the FFN curve only, rank-0's numbers are
used (every rank measures the same; we report rank-0 + cross-rank spread).

Outputs JSON: ffn_curve_ms_by_tokens, a2a_bw_by_msgbytes (algbw/busbw + lat),
plus metadata (world_size, device, single_node bool).
"""
import argparse, json, os, statistics, socket, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# --- TRUE 235B-A22B dims (verified config.json 2026-06-06) ---
HIDDEN = 4096
MOE_INTER = 1536
DTYPE = torch.bfloat16
FFN_GRID = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
            1024, 1536, 2048, 3072, 4096]


class ExpertFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, MOE_INTER, bias=False)
        self.up = nn.Linear(HIDDEN, MOE_INTER, bias=False)
        self.down = nn.Linear(MOE_INTER, HIDDEN, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def time_ffn_curve(device, repeats, warmup):
    expert = ExpertFFN().to(device=device, dtype=DTYPE).eval()
    out = {}
    with torch.no_grad():
        for b in FFN_GRID:
            x = torch.randn(b, HIDDEN, device=device, dtype=DTYPE)
            for _ in range(warmup):
                _ = expert(x)
            torch.cuda.synchronize(device)
            samples = []
            for _ in range(repeats):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); _ = expert(x); e.record(); e.synchronize()
                samples.append(s.elapsed_time(e))
            out[b] = {
                "median_ms": float(np.median(samples)),
                "mean_ms": float(np.mean(samples)),
                "std_ms": float(np.std(samples)),
                "p95_ms": float(np.percentile(samples, 95)),
                "n": repeats,
            }
            if device.index in (0, None):
                print(f"  FFN B={b:>5d}  median={out[b]['median_ms']:8.4f}ms "
                      f"std={out[b]['std_ms']:.4f}", flush=True)
    del expert
    torch.cuda.empty_cache()
    return out


def a2a_bench(device, world, rank, repeats, warmup):
    """all_to_all_single over `world` ranks. Sweep per-rank chunk in elements.
    msg sizes chosen to bracket the decode-step dispatch payload. Each rank
    sends `chunk` elements to every rank; total bytes moved per rank =
    chunk*world*2 (bf16). We report algbw (per-rank out) and busbw.
    """
    if world < 2:
        return {"note": "world<2, a2a skipped"}, []
    # per-rank-per-peer element counts to sweep (bf16). hidden=4096 vectors.
    # e.g. 4096 elems = 1 hidden vector = 8KB.
    elem_grid = [4096, 4096*4, 4096*16, 4096*64, 4096*256, 4096*1024]
    res = {}
    for chunk in elem_grid:
        send = torch.randn(chunk * world, device=device, dtype=DTYPE)
        recv = torch.empty_like(send)
        for _ in range(warmup):
            dist.all_to_all_single(recv, send)
        torch.cuda.synchronize(device)
        dist.barrier()
        samples = []
        for _ in range(repeats):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            s.record(); dist.all_to_all_single(recv, send); e.record()
            e.synchronize()
            samples.append(s.elapsed_time(e))
        # reduce max across ranks (slowest rank governs the collective)
        t = torch.tensor([np.median(samples)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        med_ms = float(t.item())
        bytes_out = chunk * world * 2  # bf16, each rank sends chunk to each of world
        algbw = bytes_out / (med_ms / 1e3) / 1e9  # GB/s
        busbw = algbw * (world - 1) / world
        res[chunk] = {
            "msg_bytes_per_rank": bytes_out,
            "median_ms": med_ms,
            "algbw_GBps": algbw,
            "busbw_GBps": busbw,
            "rank0_std_ms": float(np.std(samples)),
            "n": repeats,
        }
        if rank == 0:
            print(f"  A2A chunk={chunk:>9d}elem  bytes/rank={bytes_out/1e6:7.2f}MB  "
                  f"med={med_ms:8.4f}ms  algbw={algbw:7.2f} busbw={busbw:7.2f} GB/s",
                  flush=True)
    # latency floor: tiny message
    tiny = torch.randn(world, device=device, dtype=DTYPE)
    trecv = torch.empty_like(tiny)
    for _ in range(warmup):
        dist.all_to_all_single(trecv, tiny)
    torch.cuda.synchronize(device); dist.barrier()
    lat = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        s.record(); dist.all_to_all_single(trecv, tiny); e.record(); e.synchronize()
        lat.append(s.elapsed_time(e))
    tl = torch.tensor([np.median(lat)], device=device)
    dist.all_reduce(tl, op=dist.ReduceOp.MAX)
    return res, float(tl.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--ffn-repeats", type=int, default=80)
    ap.add_argument("--ffn-warmup", type=int, default=20)
    ap.add_argument("--a2a-repeats", type=int, default=50)
    ap.add_argument("--a2a-warmup", type=int, default=15)
    ap.add_argument("--skip-ffn", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world > 1:
        dist.init_process_group(backend="nccl")

    host = socket.gethostname()
    # how many distinct hosts? gather hostnames
    single_node = True
    if world > 1:
        names = [None] * world
        dist.all_gather_object(names, host)
        single_node = len(set(names)) == 1
        if rank == 0:
            print(f"world={world} hosts={sorted(set(names))} single_node={single_node}", flush=True)

    ffn = None
    if not args.skip_ffn and rank == 0:
        print(f"[rank0 {host}] timing FFN curve on {torch.cuda.get_device_name(local_rank)}", flush=True)
        ffn = time_ffn_curve(device, args.ffn_repeats, args.ffn_warmup)

    a2a, lat = ({}, None)
    if world > 1:
        if rank == 0:
            print(f"[rank0] a2a over world={world} single_node={single_node}", flush=True)
        a2a, lat = a2a_bench(device, world, rank, args.a2a_repeats, args.a2a_warmup)

    if rank == 0:
        out = {
            "meta": {
                "world_size": world,
                "single_node": single_node,
                "device_name": torch.cuda.get_device_name(local_rank),
                "hidden": HIDDEN, "moe_inter": MOE_INTER, "dtype": "bfloat16",
                "host": host,
            },
            "ffn_curve": ffn,
            "a2a_bw": a2a,
            "a2a_latency_floor_ms": lat,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved {args.output}", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
