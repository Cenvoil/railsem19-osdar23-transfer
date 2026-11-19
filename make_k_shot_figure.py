"""Regenerate P2 Figure 2: k-shot curve with 1.2 holdout panel."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "数据表")
OUT_FIG = os.path.join(HERE, "..", "figures")
OUT_TEX = os.path.join(HERE, "..", "LaTeX_official", "figures")

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#7F7F7F"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
})

ks = [0, 1, 5, 10, 30, 50, 100]
x = np.arange(len(ks))

# --- load multi-seed data ---
ms = {}
with open(os.path.join(DATA, "multi_seed.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (row["target"], int(row["k"]))
        seeds = [float(row[s]) for s in ("seed0", "seed1", "seed2") if row[s]]
        ms[key] = seeds

# zero-shot anchors
zs = {"5.1 rail-track": 7.43, "5.1 on-rails": 55.92,
      "3.1 rail-track": 25.94, "3.1 traffic-light": 14.22}

# k=100 from Table 9 (mean +/- std)
k100 = {"5.1 rail-track": (71.30, 4.78), "5.1 on-rails": (0.01, 0.01),
        "3.1 rail-track": (95.21, 2.70), "3.1 traffic-light": (48.51, 4.89)}


def get_series(target, cls):
    means, lo, hi = [], [], []
    for k in ks:
        if k == 0:
            v = zs[f"{target} {cls}"]
            means.append(v); lo.append(v); hi.append(v)
        elif k == 100:
            m, s = k100[f"{target} {cls}"]
            means.append(m); lo.append(m - s); hi.append(m + s)
        else:
            seeds = ms.get((f"{target} {cls}", k), [])
            if seeds:
                means.append(np.mean(seeds)); lo.append(min(seeds)); hi.append(max(seeds))
            else:
                means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
    return np.array(means), np.array(lo), np.array(hi)


fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)

# Panel (a): 5.1
ax = axes[0]
for cls, color, marker, label in [("rail-track", BLUE, "o", "5.1 rail-track"),
                                   ("on-rails", GRAY, "s", "5.1 on-rails")]:
    m, lo, hi = get_series("5.1", cls)
    ax.errorbar(x, m, yerr=[m - lo, hi - m], color=color, marker=marker,
                markersize=4, linewidth=1.2, capsize=2.5, label=label)
ax.axvspan(3.5, 4.5, color="#56B4E9", alpha=0.12)
ax.axvspan(4.5, 5.5, color="#E69F00", alpha=0.12)
ax.set_xticks(x); ax.set_xticklabels(ks)
ax.set_xlabel("target frames k (from 1.2)")
ax.set_ylabel("region-restricted IoU (%)")
ax.set_title("(a) 5.1 station", fontsize=8, loc="left")
ax.legend(loc="lower left", framealpha=0.9)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, linestyle="--")

# Panel (b): 3.1
ax = axes[1]
for cls, color, marker, label in [("rail-track", ORANGE, "^", "3.1 rail-track"),
                                   ("traffic-light", GREEN, "D", "3.1 traffic-light")]:
    m, lo, hi = get_series("3.1", cls)
    ax.errorbar(x, m, yerr=[m - lo, hi - m], color=color, marker=marker,
                markersize=4, linewidth=1.2, capsize=2.5, label=label)
ax.axvspan(3.5, 4.5, color="#56B4E9", alpha=0.12)
ax.axvspan(4.5, 5.5, color="#E69F00", alpha=0.12)
ax.set_xticks(x); ax.set_xticklabels(ks)
ax.set_xlabel("target frames k (from 1.2)")
ax.set_title("(b) 3.1 fire site", fontsize=8, loc="left")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle="--")

# Panel (c): 1.2 holdout
ax = axes[2]
holdout = {}
with open(os.path.join(DATA, "holdout_12.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        k = int(row["k"])
        holdout.setdefault(k, []).append(row)

hks = [0, 10, 30, 50]
hx = np.arange(len(hks))
for cls, color, marker, label in [("rail_track", BLUE, "o", "1.2 rail-track"),
                                   ("rail_raised", PURPLE, "s", "1.2 rail-raised"),
                                   ("traffic_light", GREEN, "D", "1.2 traffic-light")]:
    means, lo, hi = [], [], []
    for k in hks:
        vals = [float(r[cls]) for r in holdout[k]]
        means.append(np.mean(vals)); lo.append(min(vals)); hi.append(max(vals))
    means, lo, hi = map(np.array, (means, lo, hi))
    ax.errorbar(hx, means, yerr=[means - lo, hi - means], color=color, marker=marker,
                markersize=4, linewidth=1.2, capsize=2.5, label=label)
ax.axvspan(1.5, 2.5, color="#E69F00", alpha=0.12)
ax.set_xticks(hx); ax.set_xticklabels(hks)
ax.set_xlabel("training frames k (holdout = 100−k)")
ax.set_title("(c) 1.2 same-seq holdout", fontsize=8, loc="left")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle="--")

fig.tight_layout(pad=0.8)
for out in (OUT_FIG, OUT_TEX):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "P2_k_shot_curve.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "P2_k_shot_curve.png"), dpi=600, bbox_inches="tight")
print("saved P2_k_shot_curve.pdf/png to", OUT_FIG, "and", OUT_TEX)