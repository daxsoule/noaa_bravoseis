"""
Investigate WHY our locations differ from Singer's for co-identified seismic events.

Compares Singer EQ catalogue locations against our Phase 3 seismic locations
for temporally matched events within our recording windows.
"""

import pandas as pd
import numpy as np
from pyproj import Geod
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

geod = Geod(ellps="WGS84")

# ─── 1. Parse Singer catalogue ───────────────────────────────────────────────

print("=" * 90)
print("1. PARSING SINGER CATALOGUE")
print("=" * 90)

singer_rows = []
with open("/home/jovyan/my_data/bravoseis/NOAA/merged_data_amended.txt") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        ts_str = parts[0]  # YYYYDOYHHMMSF (14 chars)
        n_moor = int(parts[1])
        arrival_order = parts[2]
        lat = float(parts[3])
        lon = float(parts[4])
        err1 = float(parts[5])
        err2 = float(parts[6])
        residual = float(parts[7])
        # Class tokens from field 10 onward (fields 8,9 are azimuth values)
        tokens = parts[8:]
        # Find class label
        event_class = "UNK"
        for tok in tokens:
            if tok in ("EQ", "IQ", "IDK", "SS"):
                event_class = tok
                break

        # Parse timestamp: YYYYDDDHHMMSF
        # YYYY=year, DDD=day-of-year, HH=hour, MM=minute, S=tens-of-seconds(?), F=tenths
        # Actually looking at: 20190140137360 -> 2019 014 01 37 36 0
        # Wait, 14 chars: 2019 014 0137 36 0? Let me re-examine.
        # 20190140137360: Y=2019, DOY=014, HH=01, MM=37, SS=36, F=0
        # But that's 2019-014 01:37:36.0 -> 15 chars? No, it's 14: 20190140137360
        # 2019 014 01 37 36 0 -> that's only 13 chars if single-digit F
        # Actually: 20190140137360 = 2019 014 0137 360 ? No...
        # Let me count: 2-0-1-9-0-1-4-0-1-3-7-3-6-0 = 14 chars
        # YYYY=2019, DOY=014, HH=01, MM=37, S=36, F=0
        # Hmm but HH=01 leaves: 2019 014 01 37 36 0 = 4+3+2+2+2+1 = 14. Yes!
        # But wait: HHMMSF = 013736 + 0? That's 7 chars for time.
        # 2019 + 014 + 01 + 37 + 36 + 0 = 4+3+2+2+2+1 = 14
        # So: year=0:4, doy=4:7, hh=7:9, mm=9:11, ss=11:13, frac=13:14

        year = int(ts_str[0:4])
        doy = int(ts_str[4:7])
        hh = int(ts_str[7:9])
        mm = int(ts_str[9:11])
        ss = int(ts_str[11:13])
        frac = int(ts_str[13:14])

        base = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm,
                                                  seconds=ss, milliseconds=frac * 100)

        singer_rows.append({
            "singer_time": pd.Timestamp(base),
            "singer_n_moorings": n_moor,
            "singer_arrival_order": arrival_order,
            "singer_lat": lat,
            "singer_lon": lon,
            "singer_err1": err1,
            "singer_err2": err2,
            "singer_residual": residual,
            "singer_class": event_class,
            "singer_notes": " ".join(tokens),
        })

singer = pd.DataFrame(singer_rows)
print(f"Total Singer events: {len(singer)}")
print(f"Singer class counts:\n{singer['singer_class'].value_counts().to_string()}")

singer_eq = singer[singer["singer_class"] == "EQ"].copy().reset_index(drop=True)
print(f"\nSinger EQ events: {len(singer_eq)}")
print(f"  Time range: {singer_eq['singer_time'].min()} to {singer_eq['singer_time'].max()}")

# ─── 2. Build recording windows ──────────────────────────────────────────────

print("\n" + "=" * 90)
print("2. BUILDING RECORDING WINDOWS")
print("=" * 90)

cat = pd.read_parquet("outputs/data/event_catalogue.parquet",
                       columns=["onset_utc", "mooring", "file_number"])

# Group by mooring + file_number to get file time ranges
file_windows = cat.groupby(["mooring", "file_number"])["onset_utc"].agg(["min", "max"])
file_windows = file_windows.reset_index()
file_windows["win_start"] = file_windows["min"] - pd.Timedelta(minutes=5)
file_windows["win_end"] = file_windows["max"] + pd.Timedelta(minutes=5)

# Merge overlapping windows across moorings — just use union of all file windows
# Convert to simple intervals
intervals = list(zip(file_windows["win_start"].values, file_windows["win_end"].values))
# Sort by start
intervals.sort(key=lambda x: x[0])

# Merge overlapping
merged = []
for start, end in intervals:
    if merged and start <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
        merged.append((start, end))

print(f"Merged recording windows: {len(merged)}")
total_hours = sum((pd.Timestamp(e) - pd.Timestamp(s)).total_seconds() / 3600 for s, e in merged)
print(f"Total coverage: {total_hours:.0f} hours")

# ─── 3. Filter Singer EQ to recording windows ────────────────────────────────

print("\n" + "=" * 90)
print("3. FILTERING SINGER EQ TO RECORDING WINDOWS")
print("=" * 90)

singer_times = singer_eq["singer_time"].values.astype("datetime64[s]").astype("int64")
in_window = np.zeros(len(singer_eq), dtype=bool)

for ws, we in merged:
    ws_s = np.int64(pd.Timestamp(ws).timestamp())
    we_s = np.int64(pd.Timestamp(we).timestamp())
    in_window |= (singer_times >= ws_s) & (singer_times <= we_s)

singer_eq_inwindow = singer_eq[in_window].copy().reset_index(drop=True)
print(f"Singer EQ in our recording windows: {len(singer_eq_inwindow)} / {len(singer_eq)}")

# ─── 4. Load our Phase 3 locations ───────────────────────────────────────────

print("\n" + "=" * 90)
print("4. LOADING OUR PHASE 3 LOCATIONS")
print("=" * 90)

locs = pd.read_parquet("outputs/data/event_locations_phase3.parquet")
# Filter to seismic/both, tier A/B/C
our_seis = locs[
    (locs["phase3_class"].isin(["seismic", "both"])) &
    (locs["quality_tier"].isin(["A", "B", "C"]))
].copy().reset_index(drop=True)

print(f"Our seismic/both locations (A/B/C): {len(our_seis)}")
print(f"  By tier: {our_seis['quality_tier'].value_counts().to_string()}")
print(f"  By class: {our_seis['phase3_class'].value_counts().to_string()}")

# ─── 5. Match Singer EQ to our seismic locations within 30s ──────────────────

print("\n" + "=" * 90)
print("5. MATCHING SINGER EQ TO OUR LOCATIONS (within 30s)")
print("=" * 90)

our_times = our_seis["earliest_utc"].values.astype("datetime64[s]").astype("int64")
singer_match_times = singer_eq_inwindow["singer_time"].values.astype("datetime64[s]").astype("int64")

matches = []
for i, st in enumerate(singer_match_times):
    dt = our_times - st
    abs_dt = np.abs(dt)
    closest_idx = np.argmin(abs_dt)
    if abs_dt[closest_idx] <= 30:
        matches.append((i, closest_idx, dt[closest_idx]))

print(f"Matched pairs (within 30s): {len(matches)}")

# ─── 6. Build comparison table ───────────────────────────────────────────────

print("\n" + "=" * 90)
print("6. DETAILED COMPARISON OF MATCHED PAIRS")
print("=" * 90)

rows = []
for si, oi, time_offset_s in matches:
    sr = singer_eq_inwindow.iloc[si]
    our = our_seis.iloc[oi]

    # Geodesic distance
    _, _, dist_m = geod.inv(sr["singer_lon"], sr["singer_lat"],
                             our["lon"], our["lat"])
    dist_km = dist_m / 1000.0

    # Azimuth from Singer to ours
    az, _, _ = geod.inv(sr["singer_lon"], sr["singer_lat"],
                         our["lon"], our["lat"])

    # Lat/lon components
    dlat = our["lat"] - sr["singer_lat"]
    dlon = our["lon"] - sr["singer_lon"]

    # Arrival order comparison
    # Singer: "654321" means M6 first, M5 second, etc.
    # Ours: "m1,m3,m5" — this is the set of moorings used, not arrival order
    singer_moorings_set = set(sr["singer_arrival_order"])
    our_moorings_set = set(our["moorings"].replace("m", "").split(","))
    moorings_overlap = singer_moorings_set & our_moorings_set

    # Direction label
    if az >= -22.5 and az < 22.5:
        direction = "N"
    elif az >= 22.5 and az < 67.5:
        direction = "NE"
    elif az >= 67.5 and az < 112.5:
        direction = "E"
    elif az >= 112.5 and az < 157.5:
        direction = "SE"
    elif az >= 157.5 or az < -157.5:
        direction = "S"
    elif az >= -157.5 and az < -112.5:
        direction = "SW"
    elif az >= -112.5 and az < -67.5:
        direction = "W"
    else:
        direction = "NW"

    rows.append({
        "singer_time": sr["singer_time"],
        "our_time": our["earliest_utc"],
        "time_offset_s": time_offset_s,
        "singer_lat": sr["singer_lat"],
        "singer_lon": sr["singer_lon"],
        "our_lat": our["lat"],
        "our_lon": our["lon"],
        "dlat_deg": dlat,
        "dlon_deg": dlon,
        "offset_km": dist_km,
        "azimuth_singer_to_ours": az,
        "direction": direction,
        "singer_n_moorings": sr["singer_n_moorings"],
        "our_n_moorings": our["n_moorings"],
        "singer_arrival_order": sr["singer_arrival_order"],
        "our_moorings": our["moorings"],
        "moorings_in_common": len(moorings_overlap),
        "singer_residual": sr["singer_residual"],
        "our_residual_s": our["residual_s"],
        "our_tier": our["quality_tier"],
        "our_phase3_class": our["phase3_class"],
        "our_assoc_id": our["assoc_id"],
        "singer_notes": sr["singer_notes"],
        "our_uncertainty_km": our.get("uncertainty_km", np.nan),
    })

df = pd.DataFrame(rows)

# Print the table
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 30)

print(f"\n{'─' * 180}")
print("MATCHED EVENT DETAILS")
print(f"{'─' * 180}")

for i, r in df.iterrows():
    print(f"\n── Match {i+1} ──")
    print(f"  Singer time:    {r['singer_time']}   Our time: {r['our_time']}   Δt = {r['time_offset_s']:+.0f}s")
    print(f"  Singer loc:     ({r['singer_lat']:.3f}, {r['singer_lon']:.3f})   "
          f"Our loc: ({r['our_lat']:.3f}, {r['our_lon']:.3f})")
    print(f"  Offset:         {r['offset_km']:.1f} km   direction: {r['direction']} (az={r['azimuth_singer_to_ours']:.0f}°)")
    print(f"  Δlat: {r['dlat_deg']:+.3f}°   Δlon: {r['dlon_deg']:+.3f}°")
    print(f"  Singer: {r['singer_n_moorings']} moorings (order: {r['singer_arrival_order']})   "
          f"Ours: {r['our_n_moorings']} moorings ({r['our_moorings']})")
    print(f"  Moorings in common: {r['moorings_in_common']}   "
          f"Singer resid: {r['singer_residual']:.3f}   Our resid: {r['our_residual_s']:.3f}s")
    print(f"  Our tier: {r['our_tier']}   Our class: {r['our_phase3_class']}   "
          f"Our uncertainty: {r['our_uncertainty_km']:.1f} km")
    print(f"  Singer notes: {r['singer_notes']}")

# ─── 7. Summary statistics ───────────────────────────────────────────────────

print("\n" + "=" * 90)
print("7. SUMMARY STATISTICS")
print("=" * 90)

print(f"\nTotal matched pairs: {len(df)}")
print(f"\nSpatial offset (km):")
print(f"  Mean:   {df['offset_km'].mean():.1f}")
print(f"  Median: {df['offset_km'].median():.1f}")
print(f"  Std:    {df['offset_km'].std():.1f}")
print(f"  Min:    {df['offset_km'].min():.1f}")
print(f"  Max:    {df['offset_km'].max():.1f}")

print(f"\nTime offset (s):")
print(f"  Mean:   {df['time_offset_s'].mean():.1f}")
print(f"  Median: {df['time_offset_s'].median():.1f}")
print(f"  Std:    {df['time_offset_s'].std():.1f}")

# ─── 7a. Do events with more moorings have smaller offsets? ───────────────────

print(f"\n{'─' * 60}")
print("7a. OFFSET vs NUMBER OF MOORINGS")
print(f"{'─' * 60}")

# Use the minimum of Singer/our moorings as a measure of constraint
df["min_n_moorings"] = df[["singer_n_moorings", "our_n_moorings"]].min(axis=1)
df["max_n_moorings"] = df[["singer_n_moorings", "our_n_moorings"]].max(axis=1)

for nm in sorted(df["singer_n_moorings"].unique()):
    sub = df[df["singer_n_moorings"] == nm]
    print(f"  Singer {nm} moorings: n={len(sub):2d}, "
          f"median offset={sub['offset_km'].median():.1f} km, "
          f"mean={sub['offset_km'].mean():.1f} km")

print()
for nm in sorted(df["our_n_moorings"].unique()):
    sub = df[df["our_n_moorings"] == nm]
    print(f"  Our {nm} moorings: n={len(sub):2d}, "
          f"median offset={sub['offset_km'].median():.1f} km, "
          f"mean={sub['offset_km'].mean():.1f} km")

print()
for nm in sorted(df["min_n_moorings"].unique()):
    sub = df[df["min_n_moorings"] == nm]
    print(f"  Min(both) {nm} moorings: n={len(sub):2d}, "
          f"median offset={sub['offset_km'].median():.1f} km, "
          f"mean={sub['offset_km'].mean():.1f} km")

# ─── 7b. Systematic directional bias ─────────────────────────────────────────

print(f"\n{'─' * 60}")
print("7b. DIRECTIONAL BIAS (our location relative to Singer's)")
print(f"{'─' * 60}")

print(f"\n  Mean Δlat: {df['dlat_deg'].mean():+.4f}° ({'N' if df['dlat_deg'].mean() > 0 else 'S'})")
print(f"  Mean Δlon: {df['dlon_deg'].mean():+.4f}° ({'E' if df['dlon_deg'].mean() > 0 else 'W'})")
print(f"  Median Δlat: {df['dlat_deg'].median():+.4f}°")
print(f"  Median Δlon: {df['dlon_deg'].median():+.4f}°")

# Convert mean lat/lon offset to approx km
mean_lat_km = df['dlat_deg'].mean() * 111.0
mean_lon_km = df['dlon_deg'].mean() * 111.0 * np.cos(np.radians(-62.5))
print(f"\n  Mean offset in km: Δlat ≈ {mean_lat_km:+.1f} km, Δlon ≈ {mean_lon_km:+.1f} km")

print(f"\n  Direction distribution:")
dir_counts = df["direction"].value_counts()
for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
    n = dir_counts.get(d, 0)
    print(f"    {d:2s}: {n:2d} {'█' * n}")

# Circular mean of azimuth
az_rad = np.radians(df["azimuth_singer_to_ours"].values)
mean_az = np.degrees(np.arctan2(np.mean(np.sin(az_rad)), np.mean(np.cos(az_rad))))
r_strength = np.sqrt(np.mean(np.sin(az_rad))**2 + np.mean(np.cos(az_rad))**2)
print(f"\n  Circular mean azimuth: {mean_az:.0f}° (R={r_strength:.2f}, 1=all same direction)")

# ─── 7c. Residual vs offset correlation ──────────────────────────────────────

print(f"\n{'─' * 60}")
print("7c. RESIDUAL vs SPATIAL OFFSET")
print(f"{'─' * 60}")

from scipy import stats

# Singer residual vs offset
mask = np.isfinite(df["singer_residual"]) & np.isfinite(df["offset_km"])
if mask.sum() > 2:
    r_singer, p_singer = stats.pearsonr(df.loc[mask, "singer_residual"], df.loc[mask, "offset_km"])
    print(f"  Singer residual vs offset: r={r_singer:.3f}, p={p_singer:.4f}")

# Our residual vs offset
mask2 = np.isfinite(df["our_residual_s"]) & np.isfinite(df["offset_km"])
if mask2.sum() > 2:
    r_ours, p_ours = stats.pearsonr(df.loc[mask2, "our_residual_s"], df.loc[mask2, "offset_km"])
    print(f"  Our residual vs offset:    r={r_ours:.3f}, p={p_ours:.4f}")

# Sum of residuals
df["residual_sum"] = df["singer_residual"] + df["our_residual_s"]
mask3 = np.isfinite(df["residual_sum"]) & np.isfinite(df["offset_km"])
if mask3.sum() > 2:
    r_sum, p_sum = stats.pearsonr(df.loc[mask3, "residual_sum"], df.loc[mask3, "offset_km"])
    print(f"  Sum of residuals vs offset: r={r_sum:.3f}, p={p_sum:.4f}")

print(f"\n  Singer residual stats: mean={df['singer_residual'].mean():.2f}, median={df['singer_residual'].median():.2f}")
print(f"  Our residual stats:    mean={df['our_residual_s'].mean():.2f}, median={df['our_residual_s'].median():.2f}")

# ─── 7d. Matching arrival order ──────────────────────────────────────────────

print(f"\n{'─' * 60}")
print("7d. MOORING OVERLAP vs OFFSET")
print(f"{'─' * 60}")

for n in sorted(df["moorings_in_common"].unique()):
    sub = df[df["moorings_in_common"] == n]
    print(f"  {n} moorings in common: n={len(sub):2d}, "
          f"median offset={sub['offset_km'].median():.1f} km, "
          f"mean={sub['offset_km'].mean():.1f} km")

# ─── 7e. Additional diagnostics ──────────────────────────────────────────────

print(f"\n{'─' * 60}")
print("7e. ADDITIONAL DIAGNOSTICS")
print(f"{'─' * 60}")

# Cases where Singer uses more moorings
more_singer = df[df["singer_n_moorings"] > df["our_n_moorings"]]
more_ours = df[df["our_n_moorings"] > df["singer_n_moorings"]]
same = df[df["singer_n_moorings"] == df["our_n_moorings"]]

print(f"\n  Singer uses MORE moorings: {len(more_singer)} events, "
      f"median offset={more_singer['offset_km'].median():.1f} km" if len(more_singer) > 0 else "")
print(f"  We use MORE moorings:     {len(more_ours)} events, "
      f"median offset={more_ours['offset_km'].median():.1f} km" if len(more_ours) > 0 else "")
print(f"  Same # moorings:          {len(same)} events, "
      f"median offset={same['offset_km'].median():.1f} km" if len(same) > 0 else "")

# Events with very small offset (< 10 km) — what makes them agree?
good = df[df["offset_km"] < 10]
bad = df[df["offset_km"] >= 50]
print(f"\n  Well-matched (< 10 km): {len(good)} events")
if len(good) > 0:
    print(f"    Mean Singer moorings: {good['singer_n_moorings'].mean():.1f}, "
          f"Mean our moorings: {good['our_n_moorings'].mean():.1f}")
    print(f"    Mean Singer residual: {good['singer_residual'].mean():.2f}, "
          f"Mean our residual: {good['our_residual_s'].mean():.2f}")

print(f"\n  Poorly-matched (>= 50 km): {len(bad)} events")
if len(bad) > 0:
    print(f"    Mean Singer moorings: {bad['singer_n_moorings'].mean():.1f}, "
          f"Mean our moorings: {bad['our_n_moorings'].mean():.1f}")
    print(f"    Mean Singer residual: {bad['singer_residual'].mean():.2f}, "
          f"Mean our residual: {bad['our_residual_s'].mean():.2f}")

# Check: are the large-offset events out-of-network?
print(f"\n  Singer notes for poorly-matched events:")
if len(bad) > 0:
    for _, r in bad.iterrows():
        print(f"    {r['singer_time']} offset={r['offset_km']:.0f}km: {r['singer_notes']}")

# ─── 7f. Tier breakdown ─────────────────────────────────────────────────────

print(f"\n{'─' * 60}")
print("7f. OFFSET BY OUR QUALITY TIER")
print(f"{'─' * 60}")

for t in ["A", "B", "C"]:
    sub = df[df["our_tier"] == t]
    if len(sub) > 0:
        print(f"  Tier {t}: n={len(sub):2d}, "
              f"median offset={sub['offset_km'].median():.1f} km, "
              f"mean={sub['offset_km'].mean():.1f} km")

# ─── 7g. Mooring-count mismatch details ─────────────────────────────────────

print(f"\n{'─' * 60}")
print("7g. MOORING COUNT COMPARISON")
print(f"{'─' * 60}")

print(f"\n  Singer moorings → Our moorings cross-tab:")
ct = pd.crosstab(df["singer_n_moorings"], df["our_n_moorings"],
                  margins=True, margins_name="Total")
print(ct.to_string())

# ─── 7h. Which specific moorings differ? ─────────────────────────────────────

print(f"\n{'─' * 60}")
print("7h. SPECIFIC MOORING DIFFERENCES")
print(f"{'─' * 60}")

for _, r in df.iterrows():
    singer_set = set(r["singer_arrival_order"])
    our_set = set(r["our_moorings"].replace("m", "").split(","))
    only_singer = singer_set - our_set
    only_ours = our_set - singer_set
    if only_singer or only_ours:
        print(f"  {r['singer_time']} offset={r['offset_km']:.0f}km: "
              f"Singer-only: M{',M'.join(sorted(only_singer)) if only_singer else 'none'}, "
              f"Ours-only: M{',M'.join(sorted(only_ours)) if only_ours else 'none'}")

# ─── Save ─────────────────────────────────────────────────────────────────────

out_path = "outputs/data/singer_our_matched_eq_details.csv"
df.to_csv(out_path, index=False)
print(f"\n\nSaved matched pair details to: {out_path}")
print(f"Total rows: {len(df)}")
