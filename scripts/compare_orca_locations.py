"""
Compare our Phase 3 seismic event locations with the independent Orca OBS
earthquake catalogue for co-identified events.

Outputs:
  - Detailed match statistics to stdout
  - outputs/data/orca_our_matched_details.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from pyproj import Geod

# ── 1. Load Orca OBS catalogue ──────────────────────────────────────────────

orca = pd.read_csv("/home/jovyan/my_data/bravoseis/earthquakes/Orca_EQ_data.csv")
orca = orca.drop(columns=["Unnamed: 11"], errors="ignore")

# Convert MATLAB datenums to UTC
orca["datetime_utc"] = orca["date"].apply(
    lambda d: datetime(1, 1, 1) + timedelta(days=d - 367)
)
orca["datetime_utc"] = pd.to_datetime(orca["datetime_utc"])

print(f"Orca OBS catalogue: {len(orca)} events")
print(f"  Date range: {orca['datetime_utc'].min()} to {orca['datetime_utc'].max()}")
print(f"  Complete flag: {orca['complete'].sum()} complete, {(~orca['complete'].astype(bool)).sum()} incomplete")
print(f"  erh median: {orca['erh'].median():.2f} km, mean: {orca['erh'].mean():.2f} km")
print()

# ── 2. Build recording windows and filter Orca ──────────────────────────────

cat = pd.read_parquet("outputs/data/event_catalogue.parquet")
# Group by mooring + file_number to get recording segments
segments = cat.groupby(["mooring", "file_number"])["onset_utc"].agg(["min", "max"]).reset_index()
segments["win_start"] = segments["min"] - pd.Timedelta(minutes=5)
segments["win_end"] = segments["max"] + pd.Timedelta(minutes=5)

# Merge overlapping windows across moorings
all_starts = segments["win_start"].sort_values().values
all_ends = segments["win_end"].sort_values().values

# Build merged windows
windows = []
for _, row in segments.iterrows():
    windows.append((row["win_start"], row["win_end"]))
windows.sort()

merged = []
for start, end in windows:
    if merged and start <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
        merged.append((start, end))

print(f"Recording windows: {len(merged)} merged segments")

# Filter Orca events to those within our recording windows
orca_times = orca["datetime_utc"].values
in_window = np.zeros(len(orca), dtype=bool)
for wstart, wend in merged:
    ws = np.datetime64(wstart)
    we = np.datetime64(wend)
    in_window |= (orca_times >= ws) & (orca_times <= we)

orca_in_window = orca[in_window].copy().reset_index(drop=True)
print(f"Orca events within our recording windows: {len(orca_in_window)} / {len(orca)}")
print()

# ── 3. Load our locations, filter to seismic/both, tiers A+B+C ─────────────

loc = pd.read_parquet("outputs/data/event_locations_phase3.parquet")
loc_seis = loc[
    (loc["phase3_class"].isin(["seismic", "both"]))
    & (loc["quality_tier"].isin(["A", "B", "C"]))
].copy().reset_index(drop=True)

print(f"Our seismic/both locations (A+B+C): {len(loc_seis)}")
print(f"  Tier A: {(loc_seis['quality_tier'] == 'A').sum()}")
print(f"  Tier B: {(loc_seis['quality_tier'] == 'B').sum()}")
print(f"  Tier C: {(loc_seis['quality_tier'] == 'C').sum()}")
print(f"  Date range: {loc_seis['earliest_utc'].min()} to {loc_seis['earliest_utc'].max()}")
print()

# ── 4. Temporal matching within ±30s using binary search ────────────────────

our_times = loc_seis["earliest_utc"].sort_values()
our_unix = our_times.values.astype("datetime64[s]").astype("int64")
our_idx = our_times.index.values  # original indices into loc_seis

orca_unix = orca_in_window["datetime_utc"].values.astype("datetime64[s]").astype("int64")

WINDOW_S = 30
matches = []  # (orca_idx, our_idx, time_offset_s)

for oi, orca_t in enumerate(orca_unix):
    # Binary search for closest our event
    pos = np.searchsorted(our_unix, orca_t)
    best_dt = None
    best_idx = None
    for candidate in [pos - 1, pos, pos + 1]:
        if 0 <= candidate < len(our_unix):
            dt = our_unix[candidate] - orca_t
            if abs(dt) <= WINDOW_S:
                if best_dt is None or abs(dt) < abs(best_dt):
                    best_dt = dt
                    best_idx = candidate
    if best_idx is not None:
        matches.append((oi, our_idx[best_idx], best_dt))

print(f"Temporal matches (±{WINDOW_S}s): {len(matches)}")
print()

# ── 5. Compute spatial offsets and build match table ────────────────────────

geod = Geod(ellps="WGS84")

rows = []
for orca_i, our_i, dt_s in matches:
    o = orca_in_window.iloc[orca_i]
    u = loc_seis.loc[our_i]

    _, _, dist_m = geod.inv(o["lon"], o["lat"], u["lon"], u["lat"])
    dist_km = dist_m / 1000.0

    dlat = u["lat"] - o["lat"]
    dlon = u["lon"] - o["lon"]

    rows.append({
        "orca_datetime": o["datetime_utc"],
        "our_datetime": u["earliest_utc"],
        "time_offset_s": dt_s,
        "orca_lat": o["lat"],
        "orca_lon": o["lon"],
        "our_lat": u["lat"],
        "our_lon": u["lon"],
        "dlat": dlat,
        "dlon": dlon,
        "offset_km": dist_km,
        "our_assoc_id": u["assoc_id"],
        "our_quality_tier": u["quality_tier"],
        "our_n_moorings": u["n_moorings"],
        "our_residual_s": u["residual_s"],
        "our_uncertainty_km": u["uncertainty_km"],
        "our_phase3_class": u["phase3_class"],
        "orca_erh": o["erh"],
        "orca_erz": o["erz"],
        "orca_complete": o["complete"],
        "orca_x": o["x"],
        "orca_y": o["y"],
        "orca_z": o["z"],
    })

matched = pd.DataFrame(rows)

# Save
outdir = Path("outputs/data")
outdir.mkdir(parents=True, exist_ok=True)
matched.to_csv(outdir / "orca_our_matched_details.csv", index=False)
print(f"Saved {len(matched)} matched pairs to outputs/data/orca_our_matched_details.csv")
print()

# ── 6. Summary statistics ───────────────────────────────────────────────────

print("=" * 72)
print("LOCATION COMPARISON: Our Phase 3 vs Orca OBS (independent ground truth)")
print("=" * 72)
print()

# Match rates
n_orca_window = len(orca_in_window)
n_matched = len(matched)
print(f"Match rate: {n_matched}/{n_orca_window} Orca events matched ({100*n_matched/n_orca_window:.1f}%)")
print(f"  (Orca events within our recording windows)")
print()

# Overall offset statistics
offsets = matched["offset_km"]
print("── OVERALL SPATIAL OFFSET ──")
print(f"  N matched:   {len(offsets)}")
print(f"  Median:      {offsets.median():.1f} km")
print(f"  Mean:        {offsets.mean():.1f} km")
print(f"  Std:         {offsets.std():.1f} km")
print(f"  Min:         {offsets.min():.1f} km")
print(f"  Max:         {offsets.max():.1f} km")
print(f"  25th pctile: {offsets.quantile(0.25):.1f} km")
print(f"  75th pctile: {offsets.quantile(0.75):.1f} km")
print(f"  90th pctile: {offsets.quantile(0.90):.1f} km")
print()
print(f"  Within  5 km: {(offsets <= 5).sum()} ({100*(offsets <= 5).mean():.1f}%)")
print(f"  Within 10 km: {(offsets <= 10).sum()} ({100*(offsets <= 10).mean():.1f}%)")
print(f"  Within 20 km: {(offsets <= 20).sum()} ({100*(offsets <= 20).mean():.1f}%)")
print(f"  Within 50 km: {(offsets <= 50).sum()} ({100*(offsets <= 50).mean():.1f}%)")
print()

# Time offset stats
toff = matched["time_offset_s"]
print("── TIME OFFSET (our - orca) ──")
print(f"  Median: {toff.median():.1f} s")
print(f"  Mean:   {toff.mean():.1f} s")
print(f"  Std:    {toff.std():.1f} s")
print()

# By quality tier
print("── BY OUR QUALITY TIER ──")
for tier in ["A", "B", "C"]:
    sub = matched[matched["our_quality_tier"] == tier]
    if len(sub) == 0:
        print(f"  Tier {tier}: 0 matches")
        continue
    o = sub["offset_km"]
    print(f"  Tier {tier}: N={len(sub)}, median={o.median():.1f} km, mean={o.mean():.1f} km, "
          f"<=5km: {100*(o<=5).mean():.0f}%, <=10km: {100*(o<=10).mean():.0f}%, <=20km: {100*(o<=20).mean():.0f}%")
print()

# By n_moorings
print("── BY OUR N_MOORINGS ──")
for nm in sorted(matched["our_n_moorings"].unique()):
    sub = matched[matched["our_n_moorings"] == nm]
    o = sub["offset_km"]
    print(f"  {nm} moorings: N={len(sub)}, median={o.median():.1f} km, mean={o.mean():.1f} km, "
          f"<=10km: {100*(o<=10).mean():.0f}%")
print()

# By Orca complete flag
print("── BY ORCA COMPLETE FLAG ──")
for cf in [1, 0]:
    sub = matched[matched["orca_complete"] == cf]
    if len(sub) == 0:
        continue
    o = sub["offset_km"]
    label = "Complete" if cf == 1 else "Incomplete"
    print(f"  {label}: N={len(sub)}, median={o.median():.1f} km, mean={o.mean():.1f} km, "
          f"<=10km: {100*(o<=10).mean():.0f}%")
print()

# By Orca erh bins
print("── BY ORCA ERH (horizontal error) ──")
erh_bins = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 1000)]
for lo, hi in erh_bins:
    sub = matched[(matched["orca_erh"] >= lo) & (matched["orca_erh"] < hi)]
    if len(sub) == 0:
        continue
    o = sub["offset_km"]
    print(f"  erh [{lo}-{hi}) km: N={len(sub)}, median offset={o.median():.1f} km, mean={o.mean():.1f} km")
print()

# Directional bias
print("── DIRECTIONAL BIAS (our location minus Orca) ──")
print(f"  Mean dlat: {matched['dlat'].mean():.4f}° ({matched['dlat'].mean()*111:.1f} km)")
print(f"  Mean dlon: {matched['dlon'].mean():.4f}° ({matched['dlon'].mean()*111*np.cos(np.radians(-62)):.1f} km)")
print(f"  Median dlat: {matched['dlat'].median():.4f}°")
print(f"  Median dlon: {matched['dlon'].median():.4f}°")
pct_north = (matched["dlat"] > 0).mean()
pct_east = (matched["dlon"] > 0).mean()
print(f"  % of our locations north of Orca: {100*pct_north:.0f}%")
print(f"  % of our locations east of Orca:  {100*pct_east:.0f}%")
print()

# Correlations
print("── CORRELATIONS WITH OFFSET ──")
from scipy import stats

for col, label in [
    ("our_residual_s", "our residual_s"),
    ("our_n_moorings", "our n_moorings"),
    ("orca_erh", "Orca erh"),
    ("our_uncertainty_km", "our uncertainty_km"),
]:
    valid = matched[[col, "offset_km"]].dropna()
    if len(valid) > 3:
        r, p = stats.pearsonr(valid[col], valid["offset_km"])
        rs, ps = stats.spearmanr(valid[col], valid["offset_km"])
        print(f"  offset vs {label}: Pearson r={r:.3f} (p={p:.3e}), Spearman rho={rs:.3f} (p={ps:.3e})")
print()

# ── 7. Unmatched Orca events — check other classes ─────────────────────────

print("=" * 72)
print("UNMATCHED ORCA EVENTS — checking other classes in our catalogue")
print("=" * 72)
print()

matched_orca_idx = set(m[0] for m in matches)
unmatched_orca = orca_in_window[~orca_in_window.index.isin(matched_orca_idx)].copy()
print(f"Orca events in our windows but NOT matched to seismic/both A+B+C: {len(unmatched_orca)}")

# Try matching against ALL our located events (any class, any tier)
loc_all = loc.copy()
all_times = loc_all["earliest_utc"].sort_values()
all_unix = all_times.values.astype("datetime64[s]").astype("int64")
all_idx = all_times.index.values

unmatched_orca_unix = unmatched_orca["datetime_utc"].values.astype("datetime64[s]").astype("int64")

other_matches = []
for oi_local, (oi_global, orca_t) in enumerate(zip(unmatched_orca.index, unmatched_orca_unix)):
    pos = np.searchsorted(all_unix, orca_t)
    for candidate in [pos - 1, pos, pos + 1]:
        if 0 <= candidate < len(all_unix):
            dt = all_unix[candidate] - orca_t
            if abs(dt) <= WINDOW_S:
                idx = all_idx[candidate]
                other_matches.append({
                    "orca_idx": oi_global,
                    "our_idx": idx,
                    "time_offset_s": dt,
                    "our_phase3_class": loc_all.loc[idx, "phase3_class"],
                    "our_quality_tier": loc_all.loc[idx, "quality_tier"],
                    "our_event_class": loc_all.loc[idx, "event_class"],
                })
                break  # take first match

other_df = pd.DataFrame(other_matches)
if len(other_df) > 0:
    print(f"\nOf those {len(unmatched_orca)}, {len(other_df)} match ANY of our located events (±{WINDOW_S}s):")
    print("\n  By phase3_class:")
    print(other_df["our_phase3_class"].value_counts().to_string(header=False))
    print("\n  By quality_tier:")
    print(other_df["our_quality_tier"].value_counts().to_string(header=False))
    print("\n  By event_class (original):")
    print(other_df["our_event_class"].value_counts().to_string(header=False))
else:
    print("  None matched any of our located events.")

# Also check against the full detection catalogue (not just located)
phase3_cat = pd.read_parquet("outputs/data/phase3_catalogue.parquet")
p3_times = phase3_cat["onset_utc"].sort_values()
p3_unix = p3_times.values.astype("datetime64[s]").astype("int64")
p3_idx = p3_times.index.values

# Check ALL Orca in-window against phase3 catalogue
still_unmatched_unix = unmatched_orca_unix
still_unmatched_global = unmatched_orca.index.values

# First subtract the ones matched to located events
if len(other_df) > 0:
    matched_to_located = set(other_df["orca_idx"])
    mask = ~np.isin(still_unmatched_global, list(matched_to_located))
    still_unmatched_unix = still_unmatched_unix[mask]
    still_unmatched_global = still_unmatched_global[mask]

print(f"\nOrca events not matched to ANY located event: {len(still_unmatched_unix)}")

cat_matches = 0
for orca_t in still_unmatched_unix:
    pos = np.searchsorted(p3_unix, orca_t)
    for candidate in [pos - 1, pos, pos + 1]:
        if 0 <= candidate < len(p3_unix):
            dt = p3_unix[candidate] - orca_t
            if abs(dt) <= WINDOW_S:
                cat_matches += 1
                break

print(f"  Of those, {cat_matches} match Phase 3 catalogue detections (not located)")
print(f"  Remaining completely unmatched: {len(still_unmatched_unix) - cat_matches}")
print()

# ── Final summary ───────────────────────────────────────────────────────────

print("=" * 72)
print("FINAL SUMMARY")
print("=" * 72)
print(f"""
Orca OBS catalogue:        {len(orca)} total events
  In our recording windows:  {len(orca_in_window)}
  Matched to seismic/both:   {n_matched} ({100*n_matched/n_orca_window:.1f}%)

Our seismic/both A+B+C:    {len(loc_seis)} events
  Matched to Orca:           {n_matched} ({100*n_matched/len(loc_seis):.1f}%)

Spatial accuracy (all matched pairs):
  Median offset:  {offsets.median():.1f} km
  Mean offset:    {offsets.mean():.1f} km
  Within 10 km:   {100*(offsets <= 10).mean():.0f}%
  Within 20 km:   {100*(offsets <= 20).mean():.0f}%

Directional bias:
  Mean lat shift: {matched['dlat'].mean()*111:.1f} km ({'north' if matched['dlat'].mean() > 0 else 'south'})
  Mean lon shift: {matched['dlon'].mean()*111*np.cos(np.radians(-62)):.1f} km ({'east' if matched['dlon'].mean() > 0 else 'west'})
""")
