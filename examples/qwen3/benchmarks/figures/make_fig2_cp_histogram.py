#!/usr/bin/env python3
"""Figure 2 — "what CP measures": per-expert token histograms for a few example MoE layers,
with the busiest expert highlighted in each, illustrating CP = Σ_layers max_e(load).

Illustrative/schematic (synthetic skewed loads with a fixed seed); the per-layer token total
is held constant so the figure also shows that rebalancing redistributes load without changing
the total. Output: fig2_cp_histogram.png next to this script."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_cp_histogram.png")

rng = np.random.default_rng(1)
N_EXPERTS = 16
TOTAL = 192                      # tokens routed per layer (held constant across layers)
MEAN = TOTAL / N_EXPERTS        # balanced load = 12 tokens/expert
N_LAYERS = 3

MUTED = "#90a4ae"
HOT = "#e53935"

def skewed_layer():
    """A moderately skewed allocation of TOTAL tokens over N_EXPERTS experts (a clear hot
    expert at ~2.5-3x the balanced mean, with the rest near the mean). Dirichlet concentration
    controls the skew."""
    w = rng.dirichlet(np.full(N_EXPERTS, 4.0))
    h = w / w.sum() * TOTAL
    h = np.round(h).astype(int)
    h[h.argmax()] += TOTAL - h.sum()  # fix rounding so it sums exactly to TOTAL
    return h

layers = [skewed_layer() for _ in range(N_LAYERS)]
maxes = [int(h.max()) for h in layers]
cp = sum(maxes)
balanced_cp = int(round(MEAN)) * N_LAYERS

fig, axes = plt.subplots(1, N_LAYERS, figsize=(15, 4.8), sharey=True)
for i, (ax, h) in enumerate(zip(axes, layers), start=1):
    colors = [MUTED] * N_EXPERTS
    hot_idx = int(h.argmax())
    colors[hot_idx] = HOT
    ax.bar(np.arange(1, N_EXPERTS + 1), h, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(MEAN, ls="--", color="#555", lw=1.6)
    ax.annotate("max = {}".format(int(h[hot_idx])),
                xy=(hot_idx + 1, h[hot_idx]), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=14, fontweight="bold", color=HOT)
    ax.set_title("Layer {}".format(i), fontsize=17, fontweight="bold")
    ax.set_xlabel("expert", fontsize=15)
    ax.tick_params(labelsize=12)
    if i == 1:
        ax.set_ylabel("tokens routed to expert", fontsize=15, fontweight="bold")
        ax.annotate("balanced load = {:.0f}".format(MEAN),
                    xy=(N_EXPERTS, MEAN), xytext=(0, 4), textcoords="offset points",
                    ha="right", fontsize=12, color="#555")

axes[0].set_ylim(0, max(maxes) * 1.28)  # headroom so the "max =" labels don't hit the titles
fig.suptitle("What the critical path measures:  CP = \u03a3 over layers of the busiest expert's load",
             fontsize=18, fontweight="bold", y=1.02)
fig.text(0.5, -0.04,
         "CP (busiest per layer)  =  {}  =  {}      vs.  perfectly balanced  =  {}\u00d7{:.0f}  =  {}"
         .format("  +  ".join(str(m) for m in maxes), cp, N_LAYERS, MEAN, balanced_cp),
         ha="center", fontsize=15, fontweight="bold")
fig.text(0.5, -0.11,
         "Each layer routes the same total tokens; only the busiest expert (red) gates that layer's compute. "
         "Rebalancing lowers each red bar toward the dashed mean \u2014 shrinking CP \u2014 without changing the per-layer total.",
         ha="center", fontsize=12, color="#444")

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print("maxes={} CP={} balanced={}".format(maxes, cp, balanced_cp))
print("written:", OUT)
