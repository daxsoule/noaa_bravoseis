#!/usr/bin/env python
"""Generate monthly bar chart of T-phase and icequake detections."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from pathlib import Path

# --- Paths ---
DATA = Path("outputs/data/event_locations.parquet")
OUT = Path("outputs/figures/paper/monthly_detections.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# --- Load and filter ---
df = pd.read_parquet(DATA)
df = df[df["quality_tier"].isin(["A", "B", "C"])].copy()
df = df[df["event_class"].isin(["tphase", "icequake"])].copy()
df["month"] = df["earliest_utc"].dt.to_period("M")

# --- Build full month range ---
all_months = pd.period_range("2019-01", "2020-02", freq="M")

# --- Pivot counts ---
counts = (
    df.groupby(["month", "event_class"])
    .size()
    .unstack(fill_value=0)
    .reindex(all_months, fill_value=0)
)
for col in ["tphase", "icequake"]:
    if col not in counts.columns:
        counts[col] = 0

# --- Colors ---
COLOR_T = "#E69F00"
COLOR_I = "#56B4E9"

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

month_labels = [p.strftime("%b\n%Y") if p.month in (1, 7) else p.strftime("%b")
                for p in all_months]
x = range(len(all_months))

# Left: T-phases
ax = axes[0]
ax.bar(x, counts["tphase"], color=COLOR_T, edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(month_labels, fontsize=9)
ax.set_ylabel("Number of located events", fontsize=10)
ax.set_title("(a) T-phase Events by Month", fontsize=14)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: Icequakes
ax = axes[1]
ax.bar(x, counts["icequake"], color=COLOR_I, edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(month_labels, fontsize=9)
ax.set_ylabel("Number of located events", fontsize=10)
ax.set_title("(b) Icequake Events by Month", fontsize=14)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Figure saved to {OUT}")

# --- Summary statistics ---
print("\n=== Monthly Event Counts (Tiers A+B+C) ===\n")
print(f"{'Month':<12} {'T-phase':>8} {'Icequake':>9}")
print("-" * 32)
for p in all_months:
    t = counts.loc[p, "tphase"]
    i = counts.loc[p, "icequake"]
    print(f"{p.strftime('%Y-%m'):<12} {t:>8,} {i:>9,}")
print("-" * 32)
print(f"{'Total':<12} {counts['tphase'].sum():>8,} {counts['icequake'].sum():>9,}")
