"""Build an FP4 router-swap variant: copy of the NVFP4 checkpoint with all
mlp.gate weights replaced by a CP-RL cell's routers (gates are excluded from
quantization in the official checkpoint, so they are plain high-precision
tensors with identical names/shapes)."""
import json, os, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file

FP4_SRC, CELL_SRC, DEST = sys.argv[1], sys.argv[2], sys.argv[3]

if not os.path.exists(DEST):
    print("copying FP4 checkpoint ->", DEST, flush=True)
    shutil.copytree(FP4_SRC, DEST)

def load_index(d):
    with open(os.path.join(d, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]

dest_map = load_index(DEST)
cell_map = load_index(CELL_SRC)
gate_names = [n for n in dest_map if ".mlp.gate." in n]
print(f"{len(gate_names)} gate tensors to swap", flush=True)
assert gate_names, "no gate tensors found"
missing = [n for n in gate_names if n not in cell_map]
assert not missing, f"missing in cell ckpt: {missing[:3]}"

# group by destination shard
by_shard = {}
for n in gate_names:
    by_shard.setdefault(dest_map[n], []).append(n)

for shard, names in sorted(by_shard.items()):
    path = os.path.join(DEST, shard)
    tensors, meta = {}, None
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    for n in names:
        cpath = os.path.join(CELL_SRC, cell_map[n])
        with safe_open(cpath, framework="pt") as cf:
            t = cf.get_tensor(n)
        assert t.shape == tensors[n].shape, f"{n}: {t.shape} vs {tensors[n].shape}"
        tensors[n] = t.to(tensors[n].dtype)
    save_file(tensors, path, metadata=meta)
    print(f"patched {shard}: {len(names)} gates", flush=True)

# verify one gate matches the cell
n = gate_names[0]
with safe_open(os.path.join(DEST, dest_map[n]), framework="pt") as f:
    a = f.get_tensor(n)
with safe_open(os.path.join(CELL_SRC, cell_map[n]), framework="pt") as f:
    b = f.get_tensor(n).to(a.dtype)
assert (a == b).all(), "verification failed"
print("VERIFY_OK:", n)
