"""
Fin whale contamination check for lowband_6 high-SNR events.

Fin whale 20 Hz calls have sub-14 Hz spectral leakage that may survive
the whale filter (which only removes events with catalogue peak_freq > 14 Hz).
Known fin whale IPI for this population is ~14.7s.

Checks:
1. Inter-event interval histogram (10-20s range)
2. Sequences of 3+ consecutive events with IPI in 12-18s range (calling bouts)
3. Catalogue peak_freq near 20 Hz in these events
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# ── Load data ──────────────────────────────────────────────────────────
features = pd.read_parquet("outputs/data/event_features_lowband.parquet")
umap = pd.read_parquet("outputs/data/umap_coordinates_lowband.parquet")
catalogue = pd.read_parquet("outputs/data/event_catalogue.parquet")

# Merge cluster labels onto features
features = features.merge(
    umap[["event_id", "mooring", "cluster_id"]],
    on=["event_id", "mooring"],
    how="inner",
)

# Filter: lowband_6, SNR >= 6
lb6 = features[(features["cluster_id"] == "lowband_6") & (features["snr"] >= 6)].copy()
print(f"lowband_6 SNR>=6 events: {len(lb6)}")

# Ensure onset_utc is datetime
lb6["onset_utc"] = pd.to_datetime(lb6["onset_utc"])

# ── 1. Inter-event intervals per mooring ───────────────────────────────
print("\n" + "=" * 70)
print("1. INTER-EVENT INTERVAL ANALYSIS")
print("=" * 70)

all_intervals = []
intervals_by_mooring = {}

for mooring, grp in lb6.groupby("mooring"):
    grp_sorted = grp.sort_values("onset_utc")
    dt = grp_sorted["onset_utc"].diff().dt.total_seconds().dropna()
    intervals_by_mooring[mooring] = dt
    all_intervals.extend(dt.values)

all_intervals = np.array(all_intervals)

# Focus on 10-20s range (fin whale IPI territory)
ipi_range = all_intervals[(all_intervals >= 10) & (all_intervals <= 20)]
print(f"\nTotal inter-event intervals: {len(all_intervals)}")
print(f"Intervals in 10-20s range: {len(ipi_range)} ({100*len(ipi_range)/len(all_intervals):.2f}%)")

# Histogram in 1s bins from 10-20s
print("\nHistogram of intervals in 10-20s range (1s bins):")
bins = np.arange(10, 21, 1)
counts, edges = np.histogram(ipi_range, bins=bins)
for i in range(len(counts)):
    bar = "#" * (counts[i] // max(1, max(counts) // 40))
    print(f"  {edges[i]:5.0f}-{edges[i+1]:5.0f}s: {counts[i]:5d}  {bar}")

# Finer bins around 14.7s
print("\nFine histogram around expected IPI 14.7s (0.5s bins, 12-18s):")
bins_fine = np.arange(12, 18.5, 0.5)
ipi_fine = all_intervals[(all_intervals >= 12) & (all_intervals <= 18)]
counts_fine, edges_fine = np.histogram(ipi_fine, bins=bins_fine)
for i in range(len(counts_fine)):
    bar = "#" * (counts_fine[i] // max(1, max(counts_fine) // 40))
    print(f"  {edges_fine[i]:5.1f}-{edges_fine[i+1]:5.1f}s: {counts_fine[i]:4d}  {bar}")

# ── 2. Find calling bout sequences ────────────────────────────────────
print("\n" + "=" * 70)
print("2. FIN WHALE CALLING BOUT DETECTION (3+ consecutive, 12-18s IPI)")
print("=" * 70)

IPI_LO, IPI_HI = 12.0, 18.0
MIN_BOUT_LEN = 3

flagged_events = set()
bout_count = 0
bout_lengths = []

for mooring, grp in lb6.groupby("mooring"):
    grp_sorted = grp.sort_values("onset_utc").reset_index(drop=True)
    intervals = grp_sorted["onset_utc"].diff().dt.total_seconds()

    # Find runs of consecutive IPI-range intervals
    in_range = (intervals >= IPI_LO) & (intervals <= IPI_HI)
    current_run = []

    for idx in range(len(grp_sorted)):
        if idx > 0 and in_range.iloc[idx]:
            if not current_run:
                current_run = [idx - 1, idx]  # start includes previous event
            else:
                current_run.append(idx)
        else:
            if len(current_run) >= MIN_BOUT_LEN:
                bout_count += 1
                bout_lengths.append(len(current_run))
                for j in current_run:
                    eid = grp_sorted.iloc[j]["event_id"]
                    flagged_events.add((eid, mooring))
            current_run = []

    # Handle run at end
    if len(current_run) >= MIN_BOUT_LEN:
        bout_count += 1
        bout_lengths.append(len(current_run))
        for j in current_run:
            eid = grp_sorted.iloc[j]["event_id"]
            flagged_events.add((eid, mooring))

print(f"\nBouts found (>={MIN_BOUT_LEN} consecutive events, IPI {IPI_LO}-{IPI_HI}s): {bout_count}")
if bout_lengths:
    print(f"  Bout lengths: min={min(bout_lengths)}, max={max(bout_lengths)}, "
          f"mean={np.mean(bout_lengths):.1f}")
print(f"  Total flagged events: {len(flagged_events)}")

# Show some example bouts
if flagged_events:
    print("\nExample flagged bouts (first 5):")
    shown = 0
    for mooring, grp in lb6.groupby("mooring"):
        if shown >= 5:
            break
        grp_sorted = grp.sort_values("onset_utc").reset_index(drop=True)
        intervals = grp_sorted["onset_utc"].diff().dt.total_seconds()
        in_range = (intervals >= IPI_LO) & (intervals <= IPI_HI)
        current_run = []

        for idx in range(len(grp_sorted)):
            if idx > 0 and in_range.iloc[idx]:
                if not current_run:
                    current_run = [idx - 1, idx]
                else:
                    current_run.append(idx)
            else:
                if len(current_run) >= MIN_BOUT_LEN and shown < 5:
                    shown += 1
                    print(f"\n  Bout #{shown} on {mooring} ({len(current_run)} events):")
                    for j in current_run[:6]:  # show up to 6
                        row = grp_sorted.iloc[j]
                        ipi = intervals.iloc[j] if j > 0 else float("nan")
                        print(f"    {row['event_id']}  {row['onset_utc']}  "
                              f"SNR={row['snr']:.1f}  peak_freq={row['peak_freq_hz']:.1f} Hz  "
                              f"IPI={ipi:.1f}s" if not np.isnan(ipi) else
                              f"    {row['event_id']}  {row['onset_utc']}  "
                              f"SNR={row['snr']:.1f}  peak_freq={row['peak_freq_hz']:.1f} Hz  "
                              f"IPI=---")
                    if len(current_run) > 6:
                        print(f"    ... +{len(current_run)-6} more")
                current_run = []

        if len(current_run) >= MIN_BOUT_LEN and shown < 5:
            shown += 1
            print(f"\n  Bout #{shown} on {mooring} ({len(current_run)} events):")
            for j in current_run[:6]:
                row = grp_sorted.iloc[j]
                ipi = intervals.iloc[j] if j > 0 else float("nan")
                print(f"    {row['event_id']}  {row['onset_utc']}  "
                      f"SNR={row['snr']:.1f}  peak_freq={row['peak_freq_hz']:.1f} Hz  "
                      f"IPI={ipi:.1f}s" if not np.isnan(ipi) else
                      f"    {row['event_id']}  {row['onset_utc']}  "
                      f"SNR={row['snr']:.1f}  peak_freq={row['peak_freq_hz']:.1f} Hz  "
                      f"IPI=---")
            if len(current_run) > 6:
                print(f"    ... +{len(current_run)-6} more")

# ── 3. Tighter IPI check around 14.7s ─────────────────────────────────
print("\n" + "=" * 70)
print("3. TIGHT IPI CHECK (14.0-15.5s, centered on 14.7s)")
print("=" * 70)

IPI_TIGHT_LO, IPI_TIGHT_HI = 14.0, 15.5
flagged_tight = set()
bout_count_tight = 0

for mooring, grp in lb6.groupby("mooring"):
    grp_sorted = grp.sort_values("onset_utc").reset_index(drop=True)
    intervals = grp_sorted["onset_utc"].diff().dt.total_seconds()
    in_range = (intervals >= IPI_TIGHT_LO) & (intervals <= IPI_TIGHT_HI)
    current_run = []

    for idx in range(len(grp_sorted)):
        if idx > 0 and in_range.iloc[idx]:
            if not current_run:
                current_run = [idx - 1, idx]
            else:
                current_run.append(idx)
        else:
            if len(current_run) >= MIN_BOUT_LEN:
                bout_count_tight += 1
                for j in current_run:
                    eid = grp_sorted.iloc[j]["event_id"]
                    flagged_tight.add((eid, mooring))
            current_run = []
    if len(current_run) >= MIN_BOUT_LEN:
        bout_count_tight += 1
        for j in current_run:
            eid = grp_sorted.iloc[j]["event_id"]
            flagged_tight.add((eid, mooring))

print(f"Bouts with tight IPI (14.0-15.5s): {bout_count_tight}")
print(f"Flagged events (tight): {len(flagged_tight)}")

# ── 4. Catalogue peak_freq check ──────────────────────────────────────
print("\n" + "=" * 70)
print("4. CATALOGUE PEAK FREQUENCY CHECK")
print("=" * 70)

# Get catalogue peak_freq for lowband_6 SNR>=6 events
lb6_ids = lb6[["event_id", "mooring"]].drop_duplicates()
cat_merged = lb6_ids.merge(catalogue[["event_id", "mooring", "peak_freq_hz", "snr"]],
                           on=["event_id", "mooring"], how="left",
                           suffixes=("", "_cat"))

print(f"\nCatalogue peak_freq distribution for lowband_6 SNR>=6:")
print(cat_merged["peak_freq_hz"].describe())

# Events with catalogue peak_freq near 20 Hz
near_20 = cat_merged[(cat_merged["peak_freq_hz"] >= 18) & (cat_merged["peak_freq_hz"] <= 22)]
print(f"\nEvents with catalogue peak_freq 18-22 Hz: {len(near_20)}")
if len(near_20) > 0:
    print(f"  Peak freq distribution:")
    print(near_20["peak_freq_hz"].value_counts().head(10))

# Events with catalogue peak_freq > 14 Hz
above_14 = cat_merged[cat_merged["peak_freq_hz"] > 14]
print(f"\nEvents with catalogue peak_freq > 14 Hz: {len(above_14)} "
      f"({100*len(above_14)/len(cat_merged):.2f}%)")
if len(above_14) > 0:
    print(f"  Peak freq distribution (top 10):")
    print(above_14["peak_freq_hz"].value_counts().head(10))

# ── 5. Cross-reference: flagged bout events with high peak_freq ───────
print("\n" + "=" * 70)
print("5. CROSS-REFERENCE: BOUT-FLAGGED EVENTS WITH HIGH CATALOGUE PEAK_FREQ")
print("=" * 70)

if flagged_events:
    flagged_df = pd.DataFrame(list(flagged_events), columns=["event_id", "mooring"])
    flagged_with_cat = flagged_df.merge(
        catalogue[["event_id", "mooring", "peak_freq_hz"]],
        on=["event_id", "mooring"], how="left"
    )
    high_pf_flagged = flagged_with_cat[flagged_with_cat["peak_freq_hz"] > 14]
    print(f"Bout-flagged events with catalogue peak_freq > 14 Hz: {len(high_pf_flagged)}")
    if len(high_pf_flagged) > 0:
        print(high_pf_flagged[["event_id", "mooring", "peak_freq_hz"]].to_string(index=False))

# ── 6. Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total lowband_6 SNR>=6 events:              {len(lb6)}")
print(f"Events in IPI bouts (12-18s, >=3 consec):   {len(flagged_events)} "
      f"({100*len(flagged_events)/len(lb6):.2f}%)")
print(f"Events in tight IPI bouts (14-15.5s):       {len(flagged_tight)} "
      f"({100*len(flagged_tight)/len(lb6):.2f}%)")
print(f"Events with catalogue peak_freq > 14 Hz:    {len(above_14)} "
      f"({100*len(above_14)/len(cat_merged):.2f}%)")
print(f"Events with catalogue peak_freq 18-22 Hz:   {len(near_20)} "
      f"({100*len(near_20)/len(cat_merged):.2f}%)")

# Assess risk level
if len(flagged_tight) > 50 or len(near_20) > 100:
    print("\n⚠ ELEVATED RISK of fin whale contamination — manual review recommended")
elif len(flagged_tight) > 10 or len(near_20) > 20:
    print("\nMODERATE RISK — spot-check flagged events")
else:
    print("\nLOW RISK — minimal evidence of fin whale contamination")
