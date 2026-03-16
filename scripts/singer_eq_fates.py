#!/usr/bin/env python
"""Trace fates of Singer's in-window EQ events through our pipeline."""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── 1. Parse Singer catalogue ──────────────────────────────────────────────
singer_path = Path("/home/jovyan/my_data/bravoseis/NOAA/merged_data_amended.txt")
rows = []
with open(singer_path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        ts_str = parts[0]
        # Class token is in fields 10+ — find EQ/IQ/IDK
        class_token = None
        for p in parts[9:]:
            if p in ("EQ", "IQ", "IDK"):
                class_token = p
                break
        if class_token is None:
            continue
        # Parse YYYYDOYHHMMSF → datetime
        # Format: YYYYDOYHHMMSF where S is seconds (2 digits), F is tenths
        year = int(ts_str[:4])
        doy = int(ts_str[4:7])
        hh = int(ts_str[7:9])
        mm = int(ts_str[9:11])
        ss = int(ts_str[11:13])
        frac = int(ts_str[13]) / 10.0
        dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss + frac)
        rows.append({"singer_time": dt, "singer_class": class_token,
                      "lat": float(parts[3]), "lon": float(parts[4])})

singer = pd.DataFrame(rows)
singer["singer_time"] = pd.to_datetime(singer["singer_time"])
print(f"Singer catalogue: {len(singer)} total events")
print(f"  EQ: {(singer.singer_class == 'EQ').sum()}, IQ: {(singer.singer_class == 'IQ').sum()}, IDK: {(singer.singer_class == 'IDK').sum()}")

singer_eq = singer[singer.singer_class == "EQ"].copy().reset_index(drop=True)
print(f"Singer EQ events: {len(singer_eq)}")

# ── 2. Build recording windows ─────────────────────────────────────────────
cat = pd.read_parquet("outputs/data/event_catalogue.parquet")
# Group by mooring+file_number to get file windows
windows = (
    cat.groupby(["mooring", "file_number"])["onset_utc"]
    .agg(["min", "max"])
    .reset_index()
)
windows["win_start"] = windows["min"] - pd.Timedelta(minutes=5)
windows["win_end"] = windows["max"] + pd.Timedelta(minutes=5)
# Merge overlapping windows across moorings
all_starts = windows["win_start"].sort_values().values
all_ends = windows["win_end"].sort_values().values

# Build merged intervals
intervals = []
for _, row in windows.iterrows():
    intervals.append((row["win_start"], row["win_end"]))
intervals.sort()

merged = []
for start, end in intervals:
    if merged and start <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
        merged.append((start, end))

print(f"Recording windows: {len(merged)} merged intervals")

# ── 3. Filter Singer EQ to in-window ───────────────────────────────────────
singer_eq_times = singer_eq["singer_time"].values.astype("datetime64[ns]")
in_window = np.zeros(len(singer_eq), dtype=bool)
for ws, we in merged:
    ws_np = np.datetime64(ws)
    we_np = np.datetime64(we)
    in_window |= (singer_eq_times >= ws_np) & (singer_eq_times <= we_np)

singer_inw = singer_eq[in_window].copy().reset_index(drop=True)
print(f"Singer EQ in recording windows: {len(singer_inw)}")

# ── 4. Load all catalogues ─────────────────────────────────────────────────
loc = pd.read_parquet("outputs/data/event_locations_phase3.parquet")
p3 = pd.read_parquet("outputs/data/phase3_catalogue.parquet")
# cat already loaded

# Convert times to seconds for fast matching
singer_secs = singer_inw["singer_time"].values.astype("datetime64[s]").astype("int64")

loc_secs = loc["earliest_utc"].values.astype("datetime64[s]").astype("int64")
p3_secs = p3["onset_utc"].values.astype("datetime64[s]").astype("int64")
cat_secs = cat["onset_utc"].values.astype("datetime64[s]").astype("int64")

WINDOW = 30  # seconds

# ── 5. Classify fates ──────────────────────────────────────────────────────
fates = []
for i, row in singer_inw.iterrows():
    t = singer_secs[i]
    result = {"singer_time": row["singer_time"], "singer_lat": row["lat"], "singer_lon": row["lon"]}

    # (a) Match in locations: seismic/both, tier A/B/C
    dt_loc = np.abs(loc_secs - t)
    close_loc = dt_loc <= WINDOW
    if close_loc.any():
        seismic_mask = close_loc & loc["phase3_class"].isin(["seismic", "both"]).values & loc["quality_tier"].isin(["A", "B", "C"]).values
        if seismic_mask.any():
            best = np.argmin(np.where(seismic_mask, dt_loc, 1e9))
            result["fate"] = "our_seismic_ABC"
            result["match_id"] = loc.iloc[best]["assoc_id"]
            result["match_class"] = loc.iloc[best]["phase3_class"]
            result["match_tier"] = loc.iloc[best]["quality_tier"]
            result["match_dt_s"] = dt_loc[best]
            result["detail"] = f"{loc.iloc[best]['phase3_class']}, tier {loc.iloc[best]['quality_tier']}"
            fates.append(result)
            continue

        # (b) Match in locations: any class, any tier
        best = np.argmin(np.where(close_loc, dt_loc, 1e9))
        result["fate"] = "located_other"
        result["match_id"] = loc.iloc[best]["assoc_id"]
        result["match_class"] = loc.iloc[best]["phase3_class"]
        result["match_tier"] = loc.iloc[best]["quality_tier"]
        result["match_dt_s"] = dt_loc[best]
        result["detail"] = f"{loc.iloc[best]['phase3_class']}, tier {loc.iloc[best]['quality_tier']}"
        fates.append(result)
        continue

    # (c) Match in phase3 catalogue
    dt_p3 = np.abs(p3_secs - t)
    close_p3 = dt_p3 <= WINDOW
    if close_p3.any():
        best = np.argmin(np.where(close_p3, dt_p3, 1e9))
        result["fate"] = "in_phase3_not_located"
        result["match_id"] = p3.iloc[best]["event_id"]
        result["match_class"] = p3.iloc[best]["phase3_class"]
        result["match_tier"] = ""
        result["match_dt_s"] = dt_p3[best]
        band = p3.iloc[best]["detection_band"]
        result["detail"] = f"detection_band={band}"
        fates.append(result)
        continue

    # (d) Match in full event catalogue
    dt_cat = np.abs(cat_secs - t)
    close_cat = dt_cat <= WINDOW
    if close_cat.any():
        best = np.argmin(np.where(close_cat, dt_cat, 1e9))
        result["fate"] = "detected_not_in_phase3"
        result["match_id"] = cat.iloc[best]["event_id"]
        result["match_class"] = ""
        result["match_tier"] = ""
        result["match_dt_s"] = dt_cat[best]
        band = cat.iloc[best]["detection_band"]
        pf = cat.iloc[best]["peak_freq_hz"]
        result["detail"] = f"detection_band={band}, peak_freq={pf:.1f} Hz"
        fates.append(result)
        continue

    # (e) No match
    result["fate"] = "not_detected"
    result["match_id"] = ""
    result["match_class"] = ""
    result["match_tier"] = ""
    result["match_dt_s"] = np.nan
    result["detail"] = ""
    fates.append(result)

fates_df = pd.DataFrame(fates)

# ── 6. Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FATE SUMMARY: Singer in-window EQ events")
print("=" * 70)

fate_counts = fates_df["fate"].value_counts()
for fate, count in fate_counts.items():
    pct = 100.0 * count / len(fates_df)
    print(f"\n  {fate}: {count} ({pct:.1f}%)")
    sub = fates_df[fates_df.fate == fate]

    # Sub-breakdown for located_other
    if fate == "located_other":
        print("    Breakdown by phase3_class / tier:")
        for (cls, tier), g in sub.groupby(["match_class", "match_tier"]):
            print(f"      {cls}, tier {tier}: {len(g)}")

    # Sub-breakdown for detected_not_in_phase3
    if fate == "detected_not_in_phase3":
        print("    Breakdown by detection_band:")
        for detail, g in sub.groupby("detail"):
            print(f"      {detail}: {len(g)}")

    # Show a few examples
    examples = sub.head(3)
    for _, ex in examples.iterrows():
        ts = ex["singer_time"]
        det = ex["detail"]
        mid = ex.get("match_id", "")
        dt = ex.get("match_dt_s", "")
        dt_str = f", dt={dt:.0f}s" if pd.notna(dt) and dt != "" else ""
        print(f"    - {ts}  ({ex['singer_lat']:.3f}, {ex['singer_lon']:.3f})  → {mid}{dt_str}  {det}")

# Extra: for located_other, show class distribution
if "located_other" in fates_df.fate.values:
    print("\n  located_other class breakdown:")
    sub = fates_df[fates_df.fate == "located_other"]
    print(sub.groupby(["match_class", "match_tier"]).size().to_string())

# ── 7. Save ─────────────────────────────────────────────────────────────────
out_path = Path("outputs/data/singer_inwindow_eq_fates.csv")
fates_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path} ({len(fates_df)} rows)")
