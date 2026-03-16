#!/usr/bin/env python3
"""
recover_secondary_picks.py — Recover picks for 3-mooring events via re-association.

For each 3-mooring located event (seismic, tiers A/B/C), re-searches the full
detection catalogue (ignoring the greedy assignment) for additional mooring picks
within pair-specific travel-time windows. Uses ALL existing picks as anchors
(not the location), so this is association-based, not location-based.

Then re-locates enhanced events and compares against Singer/Orca.

Usage:
    uv run python scripts/recover_secondary_picks.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from pyproj import Geod

import sys
sys.path.insert(0, str(Path(__file__).parent))
from read_dat import MOORINGS
from locate_events import refine_location

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "outputs"
DATA_DIR = OUTPUT_DIR / "data"

SINGER_PATH = Path("/home/jovyan/my_data/bravoseis/NOAA/merged_data_amended.txt")
ORCA_PATH = Path("/home/jovyan/my_data/bravoseis/earthquakes/Orca_EQ_data.csv")

GEOD_WGS84 = Geod(ellps="WGS84")
MOORING_KEYS = sorted(MOORINGS.keys())
ALL_MOORINGS = set(MOORING_KEYS)


def load_pair_windows():
    """Load pair-specific max travel times and speeds."""
    with open(DATA_DIR / "travel_times.json") as f:
        data = json.load(f)
    pair_max_dt = {}
    pair_speeds = {}
    for pk, info in data["pairs"].items():
        k1, k2 = pk.split("-")
        pair_max_dt[(k1, k2)] = info["max_travel_time_s"]
        pair_max_dt[(k2, k1)] = info["max_travel_time_s"]
        pair_speeds[(k1, k2)] = info["c_eff_ms"]
        pair_speeds[(k2, k1)] = info["c_eff_ms"]
    c_eff = data["effective_speed_mean_ms"]
    global_max = data["global_max_travel_time_s"]
    return pair_max_dt, pair_speeds, c_eff, global_max


def parse_singer_eq():
    """Parse Singer EQ events."""
    records = []
    with open(SINGER_PATH) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 11:
                continue
            ts = parts[0]
            if len(ts) < 14:
                continue
            try:
                y, doy = int(ts[:4]), int(ts[4:7])
                hh, mm, ss = int(ts[7:9]), int(ts[9:11]), int(ts[11:13])
                dt = datetime(y, 1, 1) + timedelta(days=doy-1, hours=hh, minutes=mm, seconds=ss)
            except (ValueError, IndexError):
                continue
            if not any(t.upper() == "EQ" for t in parts[10:]):
                continue
            try:
                records.append({
                    "datetime": pd.Timestamp(dt),
                    "lat": float(parts[3]), "lon": float(parts[4]),
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(records)


def load_orca():
    """Load Orca OBS catalogue."""
    df = pd.read_csv(ORCA_PATH)
    epoch = datetime(1, 1, 1)
    df["datetime"] = df["date"].apply(lambda d: epoch + timedelta(days=d - 367))
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def match_to_reference(our_df, ref_df, tol_s=30.0):
    """Match and return offset_km for each of our events (NaN if no match)."""
    our_t = our_df["datetime"].values.astype("datetime64[s]").astype("int64")
    ref_t = ref_df["datetime"].values.astype("datetime64[s]").astype("int64")
    ref_sort = np.argsort(ref_t)
    ref_sorted = ref_t[ref_sort]

    offsets = []
    for i, qt in enumerate(our_t):
        pos = np.searchsorted(ref_sorted, qt)
        best_dt = np.inf
        best_j = -1
        for c in [pos - 1, pos]:
            if 0 <= c < len(ref_sorted):
                dt = abs(qt - ref_sorted[c])
                if dt < best_dt:
                    best_dt = dt
                    best_j = ref_sort[c]
        if best_dt <= tol_s and best_j >= 0:
            _, _, dist = GEOD_WGS84.inv(
                our_df.iloc[i]["lon"], our_df.iloc[i]["lat"],
                ref_df.iloc[best_j]["lon"], ref_df.iloc[best_j]["lat"]
            )
            offsets.append(dist / 1000.0)
        else:
            offsets.append(np.nan)
    return np.array(offsets)


def main():
    print("=" * 60)
    print("Secondary Pick Recovery via Re-Association")
    print("=" * 60)

    # --- Load ---
    pair_max_dt, pair_speeds, c_eff, global_max = load_pair_windows()
    print(f"Global max travel time: {global_max:.1f}s")

    locs = pd.read_parquet(DATA_DIR / "event_locations_phase3.parquet")
    locs["datetime"] = pd.to_datetime(locs["earliest_utc"])
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["datetime"] = pd.to_datetime(cat["onset_utc"])
    assoc = pd.read_parquet(DATA_DIR / "cross_mooring_associations.parquet")

    # Pre-index full catalogue by time for fast search
    cat_unix = cat["datetime"].values.astype("datetime64[s]").astype("int64")
    cat_sort = np.argsort(cat_unix)
    cat_sorted_times = cat_unix[cat_sort]
    cat_moorings = cat["mooring"].values
    cat_snr = cat["snr"].values

    # Build onset lookup: use refined onset where available
    cat_onset_s = np.empty(len(cat))
    for i in range(len(cat)):
        t = cat.iloc[i]["onset_utc_refined"]
        if pd.isna(t):
            t = cat.iloc[i]["datetime"]
        cat_onset_s[i] = pd.Timestamp(t).to_datetime64().astype("datetime64[s]").astype("int64")

    # Target: 3-mooring seismic events
    target = locs[
        (locs["quality_tier"].isin(["A", "B", "C"])) &
        (locs["phase3_class"].isin(["seismic", "both"])) &
        (locs["n_moorings"] == 3)
    ].copy()
    print(f"3-mooring seismic events: {len(target):,}")

    # --- Re-associate each 3-mooring event ---
    results = []
    n_enhanced = 0

    for idx, row in target.iterrows():
        present = set(row["moorings"].split(","))
        missing = ALL_MOORINGS - present

        # Get original associated events
        assoc_row = assoc[assoc["assoc_id"] == row["assoc_id"]]
        if len(assoc_row) == 0:
            continue
        eids = assoc_row.iloc[0]["event_ids"].split(",")
        aevts = cat[cat["event_id"].isin(eids)]
        if len(aevts) == 0:
            continue

        # Original onset times per mooring
        orig_onsets = {}
        for _, evt in aevts.iterrows():
            t = evt["onset_utc_refined"]
            if pd.isna(t):
                t = evt["datetime"]
            orig_onsets[evt["mooring"]] = float(
                pd.Timestamp(t).to_datetime64().astype("datetime64[s]").astype("int64")
            )

        # For each missing mooring, search for picks consistent with ALL existing picks
        recovered = {}
        for tm in missing:
            # For each existing pick, compute the allowed time window at the target mooring
            # The pick on tm must be within pair_max_dt of each existing pick
            windows = []
            for em, et in orig_onsets.items():
                max_dt = pair_max_dt.get((em, tm), global_max)
                windows.append((et - max_dt, et + max_dt))

            # Intersection of all windows
            win_lo = max(w[0] for w in windows)
            win_hi = min(w[1] for w in windows)

            if win_lo > win_hi:
                continue  # no valid window

            # Search catalogue in this window
            lo_idx = np.searchsorted(cat_sorted_times, win_lo)
            hi_idx = np.searchsorted(cat_sorted_times, win_hi)

            best_snr = -1
            best_orig_idx = None
            for k in range(lo_idx, hi_idx):
                oi = cat_sort[k]
                if cat_moorings[oi] != tm:
                    continue
                if cat_snr[oi] > best_snr:
                    best_snr = cat_snr[oi]
                    best_orig_idx = oi

            if best_orig_idx is not None:
                recovered[tm] = cat_onset_s[best_orig_idx]

        # Combine and re-locate if we gained moorings
        all_onsets = {**orig_onsets, **recovered}
        new_n = len(all_onsets)

        if new_n > 3 and not pd.isna(row["lat"]):
            # Re-locate with wider search pad since location may shift
            refined = refine_location(
                row["lat"], row["lon"], all_onsets, c_eff,
                pair_speeds=pair_speeds, fine_spacing=0.001, pad_deg=0.5
            )
            if refined is not None:
                n_enhanced += 1
                results.append({
                    "assoc_id": row["assoc_id"],
                    "orig_lat": row["lat"], "orig_lon": row["lon"],
                    "orig_residual": row["residual_s"],
                    "orig_n_moorings": 3,
                    "new_lat": refined["lat"], "new_lon": refined["lon"],
                    "new_residual": refined["residual_s"],
                    "new_n_moorings": new_n,
                    "n_recovered": new_n - 3,
                    "recovered_moorings": ",".join(sorted(recovered.keys())),
                    "datetime": row["datetime"],
                    "phase3_class": row["phase3_class"],
                })
                continue

        # No recovery — keep original
        results.append({
            "assoc_id": row["assoc_id"],
            "orig_lat": row["lat"], "orig_lon": row["lon"],
            "orig_residual": row["residual_s"],
            "orig_n_moorings": 3,
            "new_lat": row["lat"], "new_lon": row["lon"],
            "new_residual": row["residual_s"],
            "new_n_moorings": 3,
            "n_recovered": 0,
            "datetime": row["datetime"],
            "phase3_class": row["phase3_class"],
        })

    res_df = pd.DataFrame(results)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("RECOVERY SUMMARY")
    print(f"{'=' * 60}")
    print(f"Events examined: {len(res_df):,}")
    print(f"Events enhanced: {n_enhanced:,} ({100*n_enhanced/max(len(res_df),1):.1f}%)")

    enhanced = res_df[res_df["n_recovered"] > 0]
    unchanged = res_df[res_df["n_recovered"] == 0]
    if len(enhanced) > 0:
        print(f"\nEnhanced mooring counts:")
        print(enhanced["new_n_moorings"].value_counts().sort_index().to_string())
        print(f"\nResidual change (enhanced only):")
        print(f"  Before: median {enhanced['orig_residual'].median():.3f}s, "
              f"mean {enhanced['orig_residual'].mean():.3f}s")
        print(f"  After:  median {enhanced['new_residual'].median():.3f}s, "
              f"mean {enhanced['new_residual'].mean():.3f}s")

        # Location shift
        from pyproj import Geod
        g = Geod(ellps="WGS84")
        shifts = []
        for _, r in enhanced.iterrows():
            if pd.notna(r["orig_lat"]) and pd.notna(r["new_lat"]):
                _, _, d = g.inv(r["orig_lon"], r["orig_lat"], r["new_lon"], r["new_lat"])
                shifts.append(d / 1000)
        shifts = np.array(shifts)
        print(f"\nLocation shift (enhanced only):")
        print(f"  Median: {np.median(shifts):.1f} km")
        print(f"  Mean:   {np.mean(shifts):.1f} km")
        print(f"  <10km:  {(shifts<10).sum()} ({100*(shifts<10).sum()/len(shifts):.0f}%)")
        print(f"  <50km:  {(shifts<50).sum()} ({100*(shifts<50).sum()/len(shifts):.0f}%)")

    # --- Compare against Singer/Orca ---
    print(f"\n{'=' * 60}")
    print("ACCURACY COMPARISON vs Singer EQ & Orca OBS")
    print(f"{'=' * 60}")

    # Filter refs to our recording windows
    windows = cat.groupby(["mooring", "file_number"])["datetime"].agg(["min", "max"]).reset_index()
    windows["min"] = windows["min"] - pd.Timedelta("5min")
    windows["max"] = windows["max"] + pd.Timedelta("5min")

    def in_windows(df):
        mask = np.zeros(len(df), dtype=bool)
        for _, w in windows.iterrows():
            mask |= (df["datetime"].values >= w["min"]) & (df["datetime"].values <= w["max"])
        return df[mask].copy()

    singer = in_windows(parse_singer_eq())
    orca = in_windows(load_orca())
    print(f"Singer EQ in-window: {len(singer)}, Orca in-window: {len(orca)}")

    # Compare: all events
    for label, lat_col, lon_col in [("Original 3-moor", "orig_lat", "orig_lon"),
                                      ("Re-associated", "new_lat", "new_lon")]:
        df = res_df[["assoc_id", lat_col, lon_col, "datetime"]].copy()
        df.rename(columns={lat_col: "lat", lon_col: "lon"}, inplace=True)
        s_off = match_to_reference(df, singer)
        o_off = match_to_reference(df, orca)
        sv, ov = s_off[~np.isnan(s_off)], o_off[~np.isnan(o_off)]
        print(f"\n  {label} (all {len(res_df):,}):")
        if len(sv):
            print(f"    Singer: {len(sv)} matches, median {np.median(sv):.1f} km, "
                  f"<20km: {(sv<20).sum()}/{len(sv)}")
        if len(ov):
            print(f"    Orca:   {len(ov)} matches, median {np.median(ov):.1f} km, "
                  f"<20km: {(ov<20).sum()}/{len(ov)}")

    # Compare: enhanced events only
    if len(enhanced) > 0:
        print(f"\n  --- Enhanced events only ({len(enhanced)}) ---")
        for label, lat_col, lon_col in [("Before", "orig_lat", "orig_lon"),
                                          ("After", "new_lat", "new_lon")]:
            df = enhanced[["assoc_id", lat_col, lon_col, "datetime"]].copy()
            df.rename(columns={lat_col: "lat", lon_col: "lon"}, inplace=True)
            s_off = match_to_reference(df, singer)
            o_off = match_to_reference(df, orca)
            sv, ov = s_off[~np.isnan(s_off)], o_off[~np.isnan(o_off)]
            parts = [f"  {label}:"]
            if len(sv):
                parts.append(f"Singer {len(sv)} matches, median {np.median(sv):.1f} km, <20km {(sv<20).sum()}")
            if len(ov):
                parts.append(f"Orca {len(ov)} matches, median {np.median(ov):.1f} km, <20km {(ov<20).sum()}")
            print("    " + " | ".join(parts))

    # Save
    outpath = DATA_DIR / "recovered_picks_relocations.parquet"
    res_df.to_parquet(outpath)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
