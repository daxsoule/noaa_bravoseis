#!/usr/bin/env python3
"""Bar chart of event location quality distribution."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "outputs" / "data"
FIG_DIR = REPO / "outputs" / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

locs = pd.read_parquet(DATA_DIR / "event_locations_full.parquet")
pub = locs[locs["quality_tier"].isin(["A", "B", "C"])].copy()


def quality_bin(row):
    n = row["n_moorings"]
    r = row["residual_s"]
    t = row["quality_tier"]
    if t == "A" and n >= 5 and r < 1.0:
        return "Excellent\n(A, 5+ moor, res<1s)"
    elif t == "A":
        return "Very good\n(A, 4+ moor)"
    elif n >= 4 and r < 2.0:
        return "Good\n(4+ moor, res<2s)"
    elif n >= 4:
        return "Fair\n(4+ moor, res\u22652s)"
    elif n == 3 and r < 0.5:
        return "Moderate\n(3 moor, res<0.5s)"
    elif n == 3 and r < 2.0:
        return "Uncertain\n(3 moor, res<2s)"
    else:
        return "Poor\n(3 moor, res\u22652s)"


pub["quality"] = pub.apply(quality_bin, axis=1)

order = [
    "Excellent\n(A, 5+ moor, res<1s)",
    "Very good\n(A, 4+ moor)",
    "Good\n(4+ moor, res<2s)",
    "Fair\n(4+ moor, res\u22652s)",
    "Moderate\n(3 moor, res<0.5s)",
    "Uncertain\n(3 moor, res<2s)",
    "Poor\n(3 moor, res\u22652s)",
]
colors = ["#1a9641", "#66bd63", "#a6d96a", "#fee08b", "#fdae61", "#f46d43", "#d73027"]

counts = [pub[pub["quality"] == q].shape[0] for q in order]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(range(len(order)), counts, color=colors, edgecolor="black", linewidth=0.5)

# Labels on bars
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3000,
            f"{count:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, fontsize=9)
ax.set_ylabel("Number of events", fontsize=12)
ax.set_title("Location Quality Distribution — Full Dataset (548,311 events)",
             fontsize=14, fontweight="bold")

# Annotate expected accuracy from cross-validation
ax.annotate("~8 km accuracy\n(cross-validated)", xy=(0.5, counts[0] * 0.7),
            fontsize=9, ha="center", style="italic", color="#1a6630")
ax.annotate("~25-40 km", xy=(2.5, counts[2] * 0.7),
            fontsize=9, ha="center", style="italic", color="#666")
ax.annotate("~50-70 km", xy=(5, counts[5] * 0.7),
            fontsize=9, ha="center", style="italic", color="#993020")

# Add cumulative line on secondary axis
ax2 = ax.twinx()
cumulative = np.cumsum(counts)
ax2.plot(range(len(order)), cumulative, "k--o", markersize=5, linewidth=1.5,
         label="Cumulative", zorder=10)
ax2.set_ylabel("Cumulative events", fontsize=12)
ax2.set_ylim(0, cumulative[-1] * 1.12)
ax.set_ylim(0, max(counts) * 1.2)

fig.tight_layout()
outpath = FIG_DIR / "location_quality_distribution.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {outpath}")
