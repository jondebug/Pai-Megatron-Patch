#!/usr/bin/env python3
"""
build_r15_router_swap.py

Create an r15-equivalent Qwen3-235B-A22B checkpoint by patching the router
(mlp.gate.weight) tensors of a pretrained HF safetensors directory with
the 94 routers extracted from r15.

Usage:
    python build_r15_router_swap.py [--dry-run]

Assumes standard HF layout:
    <pretrained>/model.safetensors.index.json  ("weight_map": {name: shard_file})
    <pretrained>/model-NNNNN-of-00118.safetensors
    <pretrained>/config.json, tokenizer*, generation_config.json, ...
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Config (paths hard-coded per spec; override via CLI if desired)
# ---------------------------------------------------------------------------
PRETRAINED_DIR = Path(
    "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/"
    "qwen-ckpts/Qwen3-235B-A22B"
)
ROUTER_FILE = Path(
    "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/"
    "r15_routers_only.safetensors"
)
OUTPUT_DIR = Path(
    "/lustre/fsw/portfolios/nvr/users/jonathanp/rl_token_routing/r15_router_swap"
)

INDEX_CANDIDATES = ("model.safetensors.index.json", "safetensors_index.json")

# Router keys are model.layers.N.mlp.gate.weight for N in 0..93
ROUTER_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$")
EXPECTED_ROUTER_COUNT = 94  # layers 0..93


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_index_file(root: Path) -> Path:
    for name in INDEX_CANDIDATES:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No safetensors index found in {root} "
        f"(looked for {INDEX_CANDIDATES})"
    )


def group_shards_by_router_keys(weight_map: dict[str, str]) -> dict[str, list[str]]:
    """Return {shard_filename: [router_keys_in_that_shard]}."""
    shards: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_file in weight_map.items():
        if ROUTER_KEY_RE.match(tensor_name):
            shards[shard_file].append(tensor_name)
    return shards


def load_router_tensors(router_file: Path) -> dict[str, "torch.Tensor"]:
    """Load all router tensors into a dict; small file (~99 MB)."""
    import torch  # noqa: F401  (used by safetensors.torch backend)

    tensors: dict[str, "torch.Tensor"] = {}
    with safe_open(str(router_file), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def copy_aux_files(src: Path, dst: Path) -> list[str]:
    """
    Copy config, tokenizer, index, and any other non-shard aux files
    (anything that is NOT model-*.safetensors).
    """
    copied: list[str] = []
    for item in src.iterdir():
        if item.is_dir():
            continue
        # Skip the shard files themselves (we rewrite these)
        if item.name.startswith("model-") and item.name.endswith(".safetensors"):
            continue
        target = dst / item.name
        shutil.copy2(item, target)
        copied.append(item.name)
    return copied


def patch_shard(
    shard_path: Path,
    out_path: Path,
    router_keys_in_shard: list[str],
    router_tensors: dict[str, "torch.Tensor"],
) -> tuple[int, int]:
    """
    Read shard, replace listed router keys with r15 tensors, write to out_path.
    Preserves the on-disk key ordering (safetensors save_file orders by dict
    insertion; we insert in the same order the source shard exposes them).

    Returns (patched_count, kept_count).
    """
    import torch  # noqa: F401

    patched = 0
    kept = 0
    new_tensors: dict[str, "torch.Tensor"] = {}
    metadata: dict[str, str] | None = None

    with safe_open(str(shard_path), framework="pt") as f:
        md = f.metadata()
        if md:
            metadata = dict(md)
        for key in f.keys():
            if key in router_keys_in_shard:
                if key not in router_tensors:
                    raise KeyError(
                        f"Router key {key!r} present in pretrained shard but "
                        f"missing from {ROUTER_FILE}"
                    )
                src_t = f.get_tensor(key)
                new_t = router_tensors[key]
                # Sanity: shape + dtype must match
                if tuple(src_t.shape) != tuple(new_t.shape):
                    raise ValueError(
                        f"Shape mismatch on {key}: "
                        f"{tuple(src_t.shape)} vs {tuple(new_t.shape)}"
                    )
                if src_t.dtype != new_t.dtype:
                    raise ValueError(
                        f"Dtype mismatch on {key}: {src_t.dtype} vs {new_t.dtype}"
                    )
                # Ensure the tensor we save is contiguous
                new_tensors[key] = new_t.contiguous().clone()
                patched += 1
            else:
                # Load only via get_tensor so we do not hold whole shard twice.
                new_tensors[key] = f.get_tensor(key)
                kept += 1

    save_file(new_tensors, str(out_path), metadata=metadata)
    return patched, kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report which shards would be modified.")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED_DIR)
    parser.add_argument("--routers", type=Path, default=ROUTER_FILE)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    pretrained: Path = args.pretrained
    routers: Path = args.routers
    out: Path = args.out

    if not pretrained.is_dir():
        print(f"ERROR: pretrained dir not found: {pretrained}", file=sys.stderr)
        return 2
    if not routers.is_file():
        print(f"ERROR: router file not found: {routers}", file=sys.stderr)
        return 2

    index_path = find_index_file(pretrained)
    with open(index_path) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    shard_to_routers = group_shards_by_router_keys(weight_map)
    total_router_keys = sum(len(v) for v in shard_to_routers.values())

    print(f"Index:            {index_path}")
    print(f"Total tensors:    {len(weight_map)}")
    print(f"Router tensors:   {total_router_keys} "
          f"(expected {EXPECTED_ROUTER_COUNT})")
    print(f"Shards w/ router: {len(shard_to_routers)}")
    print()

    if total_router_keys != EXPECTED_ROUTER_COUNT:
        print(
            f"WARNING: expected {EXPECTED_ROUTER_COUNT} routers but index "
            f"lists {total_router_keys}. Continuing anyway.",
            file=sys.stderr,
        )

    if args.dry_run:
        print("--- DRY RUN: shards that would be modified ---")
        for shard, keys in sorted(shard_to_routers.items()):
            print(f"  {shard}: {len(keys)} routers")
            for k in keys:
                print(f"      {k}")
        # Also list shards that would just be copied? No: we always rewrite
        # every shard listed in the index, but only the ones in
        # shard_to_routers get any patching. Non-router shards would be
        # rewritten byte-identically; to save time we short-circuit:
        all_shards = set(weight_map.values())
        untouched = sorted(all_shards - set(shard_to_routers.keys()))
        print(f"\n--- {len(untouched)} shards without routers (would be "
              f"hard-linked / copied unchanged) ---")
        for s in untouched[:5]:
            print(f"  {s}")
        if len(untouched) > 5:
            print(f"  ... ({len(untouched) - 5} more)")
        return 0

    out.mkdir(parents=True, exist_ok=True)

    # Load router tensors once (~99 MB, cheap)
    print(f"Loading router tensors from {routers} ...")
    router_tensors = load_router_tensors(routers)
    print(f"  loaded {len(router_tensors)} router tensors")
    print()

    # Copy aux files (config, tokenizer, index, generation_config, ...)
    print("Copying aux files ...")
    copied = copy_aux_files(pretrained, out)
    for name in copied:
        print(f"  copied {name}")
    print()

    # Process shards
    all_shards = sorted(set(weight_map.values()))
    total_patched = 0
    total_kept = 0
    per_shard_report: list[tuple[str, int, int]] = []

    for i, shard_name in enumerate(all_shards, 1):
        src_shard = pretrained / shard_name
        dst_shard = out / shard_name
        router_keys = shard_to_routers.get(shard_name, [])

        if not router_keys:
            # No routers in this shard — hard-link (or copy) unchanged to
            # avoid the cost of re-serialising a ~4 GB shard.
            if dst_shard.exists():
                dst_shard.unlink()
            try:
                dst_shard.symlink_to(src_shard.resolve())
                mode = "symlink"
            except (OSError, AttributeError):
                shutil.copy2(src_shard, dst_shard)
                mode = "copy"
            # Count kept tensors for report
            with safe_open(str(src_shard), framework="pt") as f:
                kept = len(f.keys())
            total_kept += kept
            per_shard_report.append((shard_name, 0, kept))
            print(f"[{i:3d}/{len(all_shards)}] {shard_name}: "
                  f"0 patched, {kept} kept ({mode})")
            continue

        patched, kept = patch_shard(
            src_shard, dst_shard, router_keys, router_tensors,
        )
        total_patched += patched
        total_kept += kept
        per_shard_report.append((shard_name, patched, kept))
        print(f"[{i:3d}/{len(all_shards)}] {shard_name}: "
              f"{patched} patched, {kept} kept")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Shards processed:      {len(all_shards)}")
    print(f"Total tensors patched: {total_patched} "
          f"(expected {EXPECTED_ROUTER_COUNT})")
    print(f"Total tensors kept:    {total_kept}")
    print(f"Total tensors:         {total_patched + total_kept} "
          f"(index says {len(weight_map)})")
    print()

    ok = True
    if total_patched != EXPECTED_ROUTER_COUNT:
        print(f"FAIL: expected {EXPECTED_ROUTER_COUNT} patched, "
              f"got {total_patched}", file=sys.stderr)
        ok = False
    if total_patched + total_kept != len(weight_map):
        print(
            f"FAIL: patched+kept ({total_patched + total_kept}) != "
            f"index total ({len(weight_map)})",
            file=sys.stderr,
        )
        ok = False

    if ok:
        print("OK: router swap complete.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
