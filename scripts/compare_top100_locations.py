#!/usr/bin/env python3
"""
compare_top100_locations.py — Bidirectional top-100 location comparison.

Compares the 100 highest-confidence located events from our Phase 3 catalogue
against Singer's manual EQ catalogue and the Orca OBS catalogue, and vice versa.

Approach:
  Direction A: Our top 100 → do Singer/Orca have matching events nearby?
  Direction B: Their top 100 → did we detect and locate the same events?

Matching: temporal (±30 s), then spatial offset for matched pairs.

Usage:
    uv run python scripts/compare_top100_locations.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from pyproj import Geod

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SINGER_PATH = Path("/home/jovyan/my_data/bravoseis/NOAA/merged_data_amended.txt")
ORCA_PATH = Path("/home/jovyan/my_data/bravoseis/earthquakes/Orca_EQ_data.csv")
LOCS_PATH = DATA_DIR / "event_locations_phase3.parquet"

MATCH_TOL_S = 30.0
GEOD = Geod(ellps="WGS84")


# ============================================================
# Catalogue loaders (reused from crossvalidate script)
# ============================================================
def parse_singer_catalogue(filepath):
    """Parse Singer's fixed-width catalogue, return ALL events with locations."""
    records = []
    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 11:
                continue

            ts_str = parts[0]
            if len(ts_str) < 14:
                continue

            try:
                year = int(ts_str[0:4])
                doy = int(ts_str[4:7])
                hh = int(ts_str[7:9])
                mm = int(ts_str[9:11])
                ss = int(ts_str[11:13])
                ff = int(ts_str[13:])
                dt = datetime(year, 1, 1) + timedelta(
                    days=doy - 1, hours=hh, minutes=mm,
                    seconds=ss, microseconds=ff * 10000
                )
            except (ValueError, IndexError):
                continue

            try:
                n_moorings = int(parts[1])
                arrival_order = parts[2]
                lat = float(parts[3])
                lon = float(parts[4])
                err1 = float(parts[5])
                err2 = float(parts[6])
                residual = float(parts[7])
            except (ValueError, IndexError):
                continue

            notes_text = " ".join(parts[10:])
            singer_class = "other"
            for token in parts[10:]:
                tok_upper = token.upper()
                if tok_upper in ("EQ", "IQ", "IDK", "SS"):
                    singer_class = tok_upper
                    break

            records.append({
                "datetime": pd.Timestamp(dt),
                "n_moorings": n_moorings,
                "arrival_order": arrival_order,
                "lat": lat,
                "lon": lon,
                "err1": err1,
                "err2": err2,
                "residual": residual,
                "singer_class": singer_class,
                "notes": notes_text,
            })

    df = pd.DataFrame(records)
    print(f"Singer catalogue: {len(df):,} total events")
    for cls, cnt in df["singer_class"].value_counts().items():
        print(f"  {cls}: {cnt:,}")
    return df


def load_orca_catalogue(filepath):
    """Load Orca OBS earthquake catalogue, convert MATLAB datenums."""
    df = pd.read_csv(filepath)
    matlab_epoch = datetime(1, 1, 1)
    df["datetime"] = df["date"].apply(
        lambda d: matlab_epoch + timedelta(days=d - 367)
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"\nOrca OBS catalogue: {len(df):,} events")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df


# ============================================================
# Matching functions
# ============================================================
def temporal_match(query_df, ref_df, tol_s=30.0):
    """For each query event, find the closest reference event within ±tol_s.

    Returns a DataFrame with columns from query plus:
      matched: bool
      ref_idx: index into ref_df (or NaN)
      dt_s: time offset in seconds (query - ref)
    """
    ref_times = ref_df["datetime"].values.astype("datetime64[s]").astype("int64")
    query_times = query_df["datetime"].values.astype("datetime64[s]").astype("int64")
    ref_sorted_idx = np.argsort(ref_times)
    ref_sorted = ref_times[ref_sorted_idx]

    results = []
    for i, qt in enumerate(query_times):
        # Binary search for closest
        pos = np.searchsorted(ref_sorted, qt)
        best_dt = np.inf
        best_j = -1
        for candidate in [pos - 1, pos]:
            if 0 <= candidate < len(ref_sorted):
                dt = abs(qt - ref_sorted[candidate])
                if dt < best_dt:
                    best_dt = dt
                    best_j = ref_sorted_idx[candidate]

        if best_dt <= tol_s:
            results.append({"matched": True, "ref_idx": best_j, "dt_s": qt - ref_times[best_j]})
        else:
            results.append({"matched": False, "ref_idx": np.nan, "dt_s": np.nan})

    return pd.DataFrame(results, index=query_df.index)


def compute_spatial_offsets(query_df, ref_df, match_info):
    """Compute geodesic distance (km) between matched query and ref locations."""
    matched = match_info[match_info["matched"]].copy()
    if len(matched) == 0:
        return matched

    q_lons = query_df.loc[matched.index, "lon"].values
    q_lats = query_df.loc[matched.index, "lat"].values
    r_idx = matched["ref_idx"].astype(int).values
    r_lons = ref_df.iloc[r_idx]["lon"].values
    r_lats = ref_df.iloc[r_idx]["lat"].values

    _, _, dists = GEOD.inv(q_lons, q_lats, r_lons, r_lats)
    matched["offset_km"] = dists / 1000.0
    matched["ref_lon"] = r_lons
    matched["ref_lat"] = r_lats
    return matched


# ============================================================
# Confidence ranking
# ============================================================
def rank_our_events(locs):
    """Rank our Phase 3 located events by confidence.

    Scoring: tier A > B > C, then by n_moorings (desc), residual (asc),
    uncertainty_km (asc).
    """
    df = locs.copy()
    tier_rank = {"A": 0, "B": 1, "C": 2}
    df["tier_rank"] = df["quality_tier"].map(tier_rank)
    df = df.sort_values(
        ["tier_rank", "n_moorings", "residual_s", "uncertainty_km"],
        ascending=[True, False, True, True],
    )
    return df


def rank_singer_events(singer):
    """Rank Singer events by confidence: more moorings, lower residual, lower err1."""
    df = singer.copy()
    # Exclude sentinel residuals
    df = df[df["residual"] < 90.0]
    df = df.sort_values(
        ["n_moorings", "residual", "err1"],
        ascending=[False, True, True],
    )
    return df


def rank_orca_events(orca):
    """Rank Orca events by confidence: complete=1, lower erh, lower erz."""
    df = orca.copy()
    df = df.sort_values(
        ["complete", "erh", "erz"],
        ascending=[False, True, True],
    )
    return df


# ============================================================
# Reporting
# ============================================================
def print_match_report(name, query_label, ref_label, query_df, match_info, spatial):
    """Print a comparison report."""
    n_query = len(query_df)
    n_matched = match_info["matched"].sum()
    n_unmatched = n_query - n_matched

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"  {query_label}: {n_query}")
    print(f"  Temporal matches (±{MATCH_TOL_S:.0f}s) in {ref_label}: {n_matched} ({100*n_matched/n_query:.1f}%)")
    print(f"  No match: {n_unmatched} ({100*n_unmatched/n_query:.1f}%)")

    if len(spatial) > 0:
        offsets = spatial["offset_km"]
        print(f"\n  Spatial offsets for {len(spatial)} matched events:")
        print(f"    Median: {offsets.median():.1f} km")
        print(f"    Mean:   {offsets.mean():.1f} km")
        print(f"    Std:    {offsets.std():.1f} km")
        print(f"    Min:    {offsets.min():.1f} km")
        print(f"    Max:    {offsets.max():.1f} km")
        print(f"    <5 km:  {(offsets < 5).sum()} ({100*(offsets < 5).sum()/len(offsets):.0f}%)")
        print(f"    <10 km: {(offsets < 10).sum()} ({100*(offsets < 10).sum()/len(offsets):.0f}%)")
        print(f"    <20 km: {(offsets < 20).sum()} ({100*(offsets < 20).sum()/len(offsets):.0f}%)")
        print(f"    <50 km: {(offsets < 50).sum()} ({100*(offsets < 50).sum()/len(offsets):.0f}%)")

        dt = spatial["dt_s"]
        print(f"\n  Temporal offsets:")
        print(f"    Median: {dt.median():.1f} s")
        print(f"    Mean:   {dt.mean():.1f} s")
        print(f"    Std:    {dt.std():.1f} s")


# ============================================================
# Figure
# ============================================================
def make_comparison_figure(our_top, singer_top, orca_top,
                           our_vs_singer_spatial, our_vs_orca_spatial,
                           singer_vs_us_spatial, orca_vs_us_spatial):
    """4-panel figure: offset histograms + location scatter for each direction."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        (axes[0, 0], our_vs_singer_spatial, "Our top 100 → Singer", "#1f77b4"),
        (axes[0, 1], our_vs_orca_spatial, "Our top 100 → Orca OBS", "#2ca02c"),
        (axes[1, 0], singer_vs_us_spatial, "Singer top 100 → Us", "#d62728"),
        (axes[1, 1], orca_vs_us_spatial, "Orca top 100 → Us", "#ff7f0e"),
    ]

    for ax, spatial, title, color in panels:
        if len(spatial) == 0:
            ax.text(0.5, 0.5, "No matches", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=12, fontweight="bold")
            continue

        offsets = spatial["offset_km"].values
        bins = np.arange(0, max(offsets.max() + 5, 55), 5)
        ax.hist(offsets, bins=bins, color=color, edgecolor="black",
                linewidth=0.5, alpha=0.8)
        ax.axvline(np.median(offsets), color="black", linestyle="--",
                   linewidth=1.5, label=f"Median: {np.median(offsets):.1f} km")
        n_match = len(spatial)
        n_total = 100
        ax.set_title(f"{title}\n{n_match}/100 matched, median offset {np.median(offsets):.1f} km",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Location offset (km)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle("Bidirectional Top-100 Location Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = FIG_DIR / "top100_location_comparison.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")
    return outpath


# ============================================================
# Main
# ============================================================
def build_recording_windows(cat_path):
    """Build recording time windows from our detection catalogue."""
    cat = pd.read_parquet(cat_path)
    cat["datetime"] = pd.to_datetime(cat["onset_utc"])
    windows = cat.groupby(["mooring", "file_number"])["datetime"].agg(["min", "max"]).reset_index()
    windows["min"] = windows["min"] - pd.Timedelta("5min")
    windows["max"] = windows["max"] + pd.Timedelta("5min")
    return windows


def filter_to_our_windows(df, windows, time_col="datetime"):
    """Keep only events that fall within our recording windows."""
    mask = np.zeros(len(df), dtype=bool)
    for _, w in windows.iterrows():
        mask |= (df[time_col].values >= w["min"]) & (df[time_col].values <= w["max"])
    return df[mask].copy()


def main():
    print("=" * 60)
    print("Bidirectional Top-100 Location Comparison")
    print("(filtered to overlapping recording windows)")
    print("=" * 60)

    # --- Build recording windows from our detection catalogue ---
    CAT_PATH = DATA_DIR / "event_catalogue.parquet"
    windows = build_recording_windows(CAT_PATH)
    print(f"\nRecording windows: {len(windows)} file-mooring segments")

    # --- Load our Phase 3 locations (seismic only) ---
    locs = pd.read_parquet(LOCS_PATH)
    pub = locs[locs["quality_tier"].isin(["A", "B", "C"])].copy()
    pub["datetime"] = pd.to_datetime(pub["earliest_utc"])
    pub_seis = pub[pub["phase3_class"].isin(["seismic", "both"])].copy()
    print(f"\nOur publishable seismic locations (A+B+C): {len(pub_seis):,}")

    # --- Load reference catalogues ---
    singer_all = parse_singer_catalogue(SINGER_PATH)
    orca_all = load_orca_catalogue(ORCA_PATH)

    # Singer: EQ only, filtered to our recording windows
    singer_eq_all = singer_all[singer_all["singer_class"] == "EQ"].copy()
    singer_eq = filter_to_our_windows(singer_eq_all, windows)
    print(f"\nSinger EQ events (total): {len(singer_eq_all):,}")
    print(f"Singer EQ events in our recording windows: {len(singer_eq):,}")

    # Orca OBS: filtered to our recording windows
    orca = filter_to_our_windows(orca_all, windows)
    print(f"Orca OBS events (total): {len(orca_all):,}")
    print(f"Orca OBS events in our recording windows: {len(orca):,}")

    # --- Rank and select top N ---
    our_ranked = rank_our_events(pub_seis)
    our_top = our_ranked.head(100).copy()
    print(f"\nOur top 100 seismic events:")
    print(f"  Tiers: {our_top['quality_tier'].value_counts().to_dict()}")
    print(f"  Moorings: {our_top['n_moorings'].min()}–{our_top['n_moorings'].max()}")
    print(f"  Residual: {our_top['residual_s'].min():.3f}–{our_top['residual_s'].max():.3f} s")
    print(f"  Classes: {our_top['phase3_class'].value_counts().to_dict()}")

    singer_ranked = rank_singer_events(singer_eq)
    n_singer_available = len(singer_ranked)
    n_singer_top = min(100, n_singer_available)
    singer_top = singer_ranked.head(n_singer_top).copy()
    print(f"\nSinger top {n_singer_top} EQ events (of {n_singer_available} in-window with residual < 90):")
    if n_singer_top > 0:
        print(f"  Moorings: {singer_top['n_moorings'].min()}–{singer_top['n_moorings'].max()}")
        print(f"  Residual: {singer_top['residual'].min():.3f}–{singer_top['residual'].max():.3f} s")

    orca_ranked = rank_orca_events(orca)
    n_orca_top = min(100, len(orca_ranked))
    orca_top = orca_ranked.head(n_orca_top).copy()
    print(f"\nOrca top {n_orca_top} events (in our recording windows):")
    if n_orca_top > 0:
        print(f"  erh: {orca_top['erh'].min():.2f}–{orca_top['erh'].max():.2f} km")
        print(f"  complete: {orca_top['complete'].value_counts().to_dict()}")

    # === Direction A: Our top 100 seismic → Singer EQ (in-window) ===
    match_singer = temporal_match(our_top, singer_eq, MATCH_TOL_S)
    spatial_singer = compute_spatial_offsets(our_top, singer_eq, match_singer)
    print_match_report(
        "Direction A: Our top 100 seismic → Singer EQ (in-window)",
        "Our seismic events", "Singer EQ catalogue",
        our_top, match_singer, spatial_singer,
    )

    # === Direction A: Our top 100 seismic → Orca OBS (in-window) ===
    match_orca = temporal_match(our_top, orca, MATCH_TOL_S)
    spatial_orca = compute_spatial_offsets(our_top, orca, match_orca)
    print_match_report(
        "Direction A: Our top 100 seismic → Orca OBS (in-window)",
        "Our seismic events", "Orca catalogue",
        our_top, match_orca, spatial_orca,
    )

    # === Direction B: Singer top EQ (in-window) → Our seismic ===
    match_us_singer = temporal_match(singer_top, pub_seis, MATCH_TOL_S)
    spatial_us_singer = compute_spatial_offsets(singer_top, pub_seis, match_us_singer)
    print_match_report(
        f"Direction B: Singer top {n_singer_top} EQ (in-window) → Our seismic",
        "Singer EQ events", "Our seismic catalogue",
        singer_top, match_us_singer, spatial_us_singer,
    )

    # === Direction B: Orca top (in-window) → Our seismic ===
    match_us_orca = temporal_match(orca_top, pub_seis, MATCH_TOL_S)
    spatial_us_orca = compute_spatial_offsets(orca_top, pub_seis, match_us_orca)
    print_match_report(
        f"Direction B: Orca top {n_orca_top} (in-window) → Our seismic",
        "Orca events", "Our seismic catalogue",
        orca_top, match_us_orca, spatial_us_orca,
    )

    # === Figure ===
    make_comparison_figure(
        our_top, singer_top, orca_top,
        spatial_singer, spatial_orca,
        spatial_us_singer, spatial_us_orca,
    )

    # === Detailed matched event table ===
    print(f"\n{'=' * 60}")
    print("Sample matched events (Our top 100 seismic → Singer EQ)")
    print(f"{'=' * 60}")
    if len(spatial_singer) > 0:
        sample = spatial_singer.head(20).copy()
        for idx, row in sample.iterrows():
            our_evt = our_top.loc[idx]
            singer_evt = singer_eq.iloc[int(row["ref_idx"])]
            print(f"  {our_evt['datetime']} | tier={our_evt['quality_tier']} "
                  f"moor={our_evt['n_moorings']} res={our_evt['residual_s']:.3f}s "
                  f"class={our_evt['phase3_class']}")
            print(f"    → Singer {singer_evt['datetime']} | "
                  f"moor={singer_evt['n_moorings']} res={singer_evt['residual']:.1f}s "
                  f"err1={singer_evt['err1']:.3f}")
            print(f"    Δt={row['dt_s']:.1f}s  Δx={row['offset_km']:.1f}km  "
                  f"({our_evt['lat']:.3f},{our_evt['lon']:.3f}) vs "
                  f"({singer_evt['lat']:.3f},{singer_evt['lon']:.3f})")

    print(f"\n{'=' * 60}")
    print(f"Sample matched events (Singer top EQ → Our seismic)")
    print(f"{'=' * 60}")
    if len(spatial_us_singer) > 0:
        sample = spatial_us_singer.head(20).copy()
        for idx, row in sample.iterrows():
            singer_evt = singer_top.loc[idx]
            our_evt = pub_seis.iloc[int(row["ref_idx"])]
            print(f"  Singer {singer_evt['datetime']} | "
                  f"moor={singer_evt['n_moorings']} res={singer_evt['residual']:.1f}s "
                  f"err1={singer_evt['err1']:.3f}")
            print(f"    → Ours {our_evt['datetime']} | tier={our_evt['quality_tier']} "
                  f"moor={our_evt['n_moorings']} res={our_evt['residual_s']:.3f}s")
            print(f"    Δt={row['dt_s']:.1f}s  Δx={row['offset_km']:.1f}km  "
                  f"({singer_evt['lat']:.3f},{singer_evt['lon']:.3f}) vs "
                  f"({our_evt['lat']:.3f},{our_evt['lon']:.3f})")


if __name__ == "__main__":
    main()
