"""Extract mlp.gate.weight tensors from a full HF safetensors dir into a compact file."""
import argparse, glob, os
from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument("--hf", required=True, help="Full HF safetensors dir")
p.add_argument("--out", required=True, help="Output compact routers file")
args = p.parse_args()

routers = {}
tot = 0
for f in sorted(glob.glob(os.path.join(args.hf, "model-*.safetensors"))):
    with safe_open(f, framework="pt") as st:
        for k in st.keys():
            if "mlp.gate.weight" in k and "experts" not in k:
                t = st.get_tensor(k)
                routers[k] = t.contiguous()
                tot += t.numel() * t.element_size()

print(f"Extracted {len(routers)} routers ({tot/1e6:.1f} MB) from {args.hf}")
save_file(routers, args.out)
print(f"Saved -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB on disk)")
