#!/usr/bin/env python3
"""Compare the accuracy-vs-CP Pareto frontier of RL with a MEAN baseline vs a learned CRITIC.

Reuses generate_pareto.load_data (which already tags each point's baseline_type from the run
name: 'basecritic'/'c256'/'crit*' -> critic, 'basemean'/default -> mean) and
compute_pareto_frontier. Splits the RL points (rl_only + rl+aux) by baseline_type, draws a
frontier per group, and writes a PNG.

Usage:
    python compare_baseline_pareto.py [--min-accuracy 72] [--baseline-cp 8800]
                                      [--run-name-prefix 235b] [--output pareto_mean_vs_critic.png]
"""
import argparse
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def _load_gp():
    spec = importlib.util.spec_from_file_location("gp", str(SCRIPT_DIR / "generate_pareto.py"))
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    return gp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(SCRIPT_DIR / "benchmark_results.csv"))
    ap.add_argument("--baseline-cp", type=float, default=8800.0)
    ap.add_argument("--run-name-prefix", default="235b")
    ap.add_argument("--limit-filter", default="inf")
    ap.add_argument("--min-accuracy", type=float, default=72.0,
                    help="Focus the comparison on the meaningful trade-off regime.")
    ap.add_argument("--max-train-iters", type=int, default=8000)
    ap.add_argument("--output", default=str(SCRIPT_DIR / "pareto_mean_vs_critic.png"))
    args = ap.parse_args()

    gp = _load_gp()
    gp.BASELINE_CP = args.baseline_cp

    pts = gp.load_data(args.csv, min_accuracy=args.min_accuracy, limit_filter=args.limit_filter,
                       max_train_iters=args.max_train_iters, run_name_prefix=args.run_name_prefix)
    rl = [p for p in pts if p["cat"] in ("rl_only", "rl+aux")]
    groups = {
        "mean":   [p for p in rl if p["baseline_type"] == "mean"],
        "critic": [p for p in rl if p["baseline_type"] == "critic"],
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    STYLE = {
        "mean":   dict(color="#1565c0", marker="o", label="RL — mean baseline"),
        "critic": dict(color="#e65100", marker="s", label="RL — learned critic"),
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    for name, g in groups.items():
        if not g:
            continue
        st = STYLE[name]
        # faint scatter of all points in the group
        ax.scatter([p["x"] for p in g], [p["y"] for p in g], c=st["color"],
                   marker=st["marker"], s=42, alpha=0.28, edgecolors="none", zorder=2)
        # bold frontier line + markers
        fr = gp.compute_pareto_frontier(g)
        if len(fr) >= 2:
            ax.plot([p["x"] for p in fr], [p["y"] for p in fr], "-", color=st["color"],
                    lw=2.6, zorder=4)
        ax.scatter([p["x"] for p in fr], [p["y"] for p in fr], c=st["color"],
                   marker=st["marker"], s=95, edgecolors="white", linewidths=0.9,
                   zorder=5, label="{} (n={})".format(st["label"], len(g)))

    # pretrained baseline
    base = [p for p in pts if p["cat"] == "pretrained"]
    if base:
        b = base[0]
        ax.scatter([b["x"]], [b["y"]], c="black", marker="*", s=420, zorder=6)
        ax.annotate("Pretrained", (b["x"], b["y"]), textcoords="offset points",
                    xytext=(12, 8), fontsize=17, fontweight="bold", zorder=7)

    xs = [p["x"] for p in rl] or [0]
    ax.set_xlim(round(min(xs + [0]) - 3, 1), round(max(xs) + 4, 1))
    ax.set_ylim(args.min_accuracy, 78)
    ax.set_xlabel("Critical Path Reduction (%)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Benchmark Accuracy (%)", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:g}%".format(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:g}%".format(v)))
    ax.grid(True, color="#f0f0f0")
    ax.annotate("Better", xy=(0.96, 0.96), xytext=(0.80, 0.80),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=16, fontweight="bold", color="#444", ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=2.6))
    ax.set_title("Mean baseline vs. learned critic — RL Pareto frontier (Qwen3-235B, acc \u2265 {:g}%)".format(args.min_accuracy),
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=15, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print("mean pts: {}  critic pts: {}".format(len(groups["mean"]), len(groups["critic"])))
    print("PNG written to: {}".format(args.output))


if __name__ == "__main__":
    main()
