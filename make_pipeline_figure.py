import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = str(Path(__file__).resolve().parent.parent / "figures" / "P2_pipeline.png")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "figure.dpi": 160, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(6.6, 2.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

boxes = [
    (0.015, "RailSem19 source\n19 classes", "#1f4e79"),
    (0.215, "YOLO26m-sem\nreal-time seg.", "#1f4e79"),
    (0.415, "Zero-shot\n1.2 / 3.1 / 5.1", "#548235"),
    (0.615, "k-shot adaptation\nk frames from 1.2", "#c55a11"),
    (0.815, "Target eval\n5.1 / 3.1", "#548235"),
]
w, h, y = 0.17, 0.56, 0.36
for x, text, color in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.006,rounding_size=0.012",
                 linewidth=0.9, edgecolor=color, facecolor=color + "16"))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=8.8, linespacing=1.25, color="#1a1a1a")
for i in range(4):
    x0 = boxes[i][0] + w + 0.006
    x1 = boxes[i+1][0] - 0.006
    ax.add_patch(FancyArrowPatch((x0, y + h/2), (x1, y + h/2),
                 arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color="#555555"))

ax.add_patch(FancyBboxPatch((0.415, 0.03), 0.57, 0.18,
             boxstyle="round,pad=0.006,rounding_size=0.012",
             linewidth=0.8, edgecolor="#7f7f7f", facecolor="#f2f2f2",
             linestyle=(0, (3, 2))))
ax.text(0.70, 0.12,
        "Diagnostics: mix30  ·  resolution 1024/1536/2048\nm2 annealing  ·  SAM refinement",
        ha="center", va="center", fontsize=7.0, color="#333333", linespacing=1.3)

fig.tight_layout(pad=0.2)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", OUT, os.path.getsize(OUT))
