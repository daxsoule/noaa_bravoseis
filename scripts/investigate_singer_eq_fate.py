#!/usr/bin/env python3
"""
investigate_singer_eq_fate.py — Where did Singer's top EQ events go in our pipeline?

For each of Singer's top 100 EQ events, trace through our full catalogue
(all classes, all tiers) to understand why only 12% appear as seismic locations.

Possible fates:
  1. Matched in our seismic locations (A+B+C)
  2. Matched in our located events but different class (cryogenic, unclassified, both)
  3. Matched in our located events but tier D (rejected)
  4. Detected but not located (in catalogue but no association/location)
  5. Not detected at all (not in recording window, or missed by STA/LTA)
  6. In recording window but not detected

Usage:
    uv run python scripts/investigate_singer_eq_fate.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SINGER_PATH = Path("/home/jovyan/my_data/bravoseis/NOAA/merged_data_amended.txt")

MATCH_TOL_S = 30.0


# ============================================================
# Singer parser (EQ only)
# ============================================================
def parse_singer_eq(filepath):
    """Parse Singer catalogue, return all events (with class label)."""
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
                lat = float(parts[3])
                lon = float(parts[4])
                err1 = float(parts[5])
                err2 = float(parts[6])
                residual = float(parts[7])
            except (ValueError, IndexError):
                continue

            singer_class = "other"
            for token in parts[10:]:
                tok_upper = token.upper()
                if tok_upper in ("EQ", "IQ", "IDK", "SS"):
                    singer_class = tok_upper
                    break

            records.append({
                "datetime": pd.Timestamp(dt),
                "n_moorings": n_moorings,
                "lat": lat,
                "lon": lon,
                "err1": err1,
                "err2": err2,
                "residual": residual,
                "singer_class": singer_class,
            })

    df = pd.DataFrame(records)
    return df


# ============================================================
# Temporal matching
# ============================================================
def find_closest_match(query_time_s, ref_times_s, ref_indices, tol_s):
    """Find closest reference event within tolerance. Returns (ref_idx, dt_s) or (None, None)."""
    pos = np.searchsorted(ref_times_s, query_time_s)
    best_dt = np.inf
    best_idx = None
    for candidate in [pos - 1, pos]:
        if 0 <= candidate < len(ref_times_s):
            dt = query_time_s - ref_times_s[candidate]
            if abs(dt) < abs(best_dt):
                best_dt = dt
                best_idx = ref_indices[candidate]
    if abs(best_dt) <= tol_s:
        return best_idx, best_dt
    return None, None


# ============================================================
# Main investigation
# ============================================================
def main():
    print("=" * 70)
    print("Investigation: Where did Singer's top EQ events go?")
    print("=" * 70)

    # --- Parse Singer EQ events ---
    singer_all = parse_singer_eq(SINGER_PATH)
    singer_eq = singer_all[singer_all["singer_class"] == "EQ"].copy()
    # Rank by confidence: most moorings, lowest residual
    singer_eq = singer_eq[singer_eq["residual"] < 90.0]
    singer_eq = singer_eq.sort_values(
        ["n_moorings", "residual", "err1"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    singer_top = singer_eq.head(100).copy()
    print(f"\nSinger top 100 EQ events:")
    print(f"  Moorings: {singer_top['n_moorings'].min()}–{singer_top['n_moorings'].max()}")
    print(f"  Residual: {singer_top['residual'].min():.3f}–{singer_top['residual'].max():.3f} s")

    # --- Load our data at multiple levels ---

    # Level 1: Phase 3 locations (all tiers, all classes)
    locs_all = pd.read_parquet(DATA_DIR / "event_locations_phase3.parquet")
    locs_all["datetime"] = pd.to_datetime(locs_all["earliest_utc"])
    print(f"\nOur located events (all tiers): {len(locs_all):,}")
    print(f"  Tiers: {locs_all['quality_tier'].value_counts().to_dict()}")
    print(f"  Classes: {locs_all['phase3_class'].value_counts().to_dict()}")

    # Level 2: Full detection catalogue (before location)
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["datetime"] = pd.to_datetime(cat["onset_utc"])
    print(f"\nFull detection catalogue: {len(cat):,} events")

    # Level 3: Phase 3 catalogue (classified events before location)
    phase3_cat = pd.read_parquet(DATA_DIR / "phase3_catalogue.parquet")
    phase3_cat["datetime"] = pd.to_datetime(phase3_cat["onset_utc"])
    print(f"Phase 3 catalogue: {len(phase3_cat):,} events")

    # Pre-sort for binary search
    for df_name, df in [("locs_all", locs_all), ("cat", cat), ("phase3_cat", phase3_cat)]:
        df.sort_values("datetime", inplace=True)

    def make_search_arrays(df):
        times = df["datetime"].values.astype("datetime64[s]").astype("int64")
        sort_idx = np.argsort(times)
        return times[sort_idx], sort_idx

    locs_times, locs_sort = make_search_arrays(locs_all)
    cat_times, cat_sort = make_search_arrays(cat)
    p3_times, p3_sort = make_search_arrays(phase3_cat)

    # Also prepare seismic-only locations for quick check
    locs_seis = locs_all[locs_all["phase3_class"].isin(["seismic", "both"])].copy()
    locs_seis_times, locs_seis_sort = make_search_arrays(locs_seis)

    # Also prepare A+B+C seismic
    locs_pub_seis = locs_seis[locs_seis["quality_tier"].isin(["A", "B", "C"])].copy()
    locs_pub_seis_times, locs_pub_seis_sort = make_search_arrays(locs_pub_seis)

    # --- Trace each Singer EQ ---
    fates = []
    for i, row in singer_top.iterrows():
        qt = int(row["datetime"].to_datetime64().astype("datetime64[s]").astype("int64"))
        fate = {"singer_idx": i, "singer_time": row["datetime"],
                "singer_moor": row["n_moorings"], "singer_res": row["residual"],
                "singer_err1": row["err1"],
                "singer_lat": row["lat"], "singer_lon": row["lon"]}

        # Check 1: In our publishable seismic locations?
        idx, dt = find_closest_match(qt, locs_pub_seis_times, locs_pub_seis_sort, MATCH_TOL_S)
        if idx is not None:
            evt = locs_pub_seis.iloc[idx]
            fate["fate"] = "seismic_located_ABC"
            fate["our_time"] = evt["datetime"]
            fate["our_class"] = evt["phase3_class"]
            fate["our_tier"] = evt["quality_tier"]
            fate["dt_s"] = dt
            fates.append(fate)
            continue

        # Check 2: In our located events (any class, any tier)?
        idx, dt = find_closest_match(qt, locs_times, locs_sort, MATCH_TOL_S)
        if idx is not None:
            evt = locs_all.iloc[idx]
            fate["fate"] = f"located_{evt['phase3_class']}_{evt['quality_tier']}"
            fate["our_time"] = evt["datetime"]
            fate["our_class"] = evt["phase3_class"]
            fate["our_tier"] = evt["quality_tier"]
            fate["dt_s"] = dt
            fates.append(fate)
            continue

        # Check 3: In Phase 3 catalogue (classified but not located)?
        idx, dt = find_closest_match(qt, p3_times, p3_sort, MATCH_TOL_S)
        if idx is not None:
            evt = phase3_cat.iloc[idx]
            band = evt.get("detection_band", "unknown")
            fate["fate"] = f"phase3_not_located_{band}"
            fate["our_time"] = evt["datetime"]
            fate["dt_s"] = dt
            fates.append(fate)
            continue

        # Check 4: In full detection catalogue (detected but not classified)?
        idx, dt = find_closest_match(qt, cat_times, cat_sort, MATCH_TOL_S)
        if idx is not None:
            evt = cat.iloc[idx]
            fate["fate"] = "detected_not_classified"
            fate["our_time"] = evt["datetime"]
            fate["dt_s"] = dt
            # Add mooring info if available
            if "mooring" in evt.index:
                fate["our_mooring"] = evt["mooring"]
            fates.append(fate)
            continue

        # Check 5: Not detected — but was it in a recording window?
        # Use a wider window to check for near-misses
        idx_wide, dt_wide = find_closest_match(qt, cat_times, cat_sort, 300.0)
        if idx_wide is not None:
            fate["fate"] = "not_detected_but_near_activity"
            fate["nearest_detection_dt_s"] = dt_wide
        else:
            fate["fate"] = "not_detected_no_nearby"

        fates.append(fate)

    fates_df = pd.DataFrame(fates)

    # --- Summarize fates ---
    print(f"\n{'=' * 70}")
    print("FATE SUMMARY: Singer's top 100 EQ events")
    print(f"{'=' * 70}")
    fate_counts = fates_df["fate"].value_counts()
    for fate, count in fate_counts.items():
        print(f"  {fate}: {count}")

    # Group into broader categories
    fate_categories = {
        "Our seismic (A+B+C)": fates_df["fate"].str.startswith("seismic_located").sum(),
        "Located, different class": fates_df["fate"].str.startswith("located_").sum(),
        "Phase 3 cat, not located": fates_df["fate"].str.startswith("phase3_not_located").sum(),
        "Detected, not classified": (fates_df["fate"] == "detected_not_classified").sum(),
        "Not detected (near activity)": (fates_df["fate"] == "not_detected_but_near_activity").sum(),
        "Not detected (no nearby)": (fates_df["fate"] == "not_detected_no_nearby").sum(),
    }
    print(f"\n  --- Grouped ---")
    for cat, count in fate_categories.items():
        print(f"  {cat}: {count} ({100*count/len(fates_df):.0f}%)")

    # --- Details for each category ---
    print(f"\n{'=' * 70}")
    print("DETAILS: Events that ended up in different classes")
    print(f"{'=' * 70}")
    diff_class = fates_df[fates_df["fate"].str.startswith("located_")]
    if len(diff_class) > 0:
        for _, row in diff_class.iterrows():
            print(f"  Singer {row['singer_time']} (moor={row['singer_moor']}, "
                  f"res={row['singer_res']:.1f}s, err1={row['singer_err1']:.3f})")
            print(f"    → Our class={row.get('our_class','?')}, tier={row.get('our_tier','?')}, "
                  f"Δt={row.get('dt_s', '?')}s")

    print(f"\n{'=' * 70}")
    print("DETAILS: Events not detected at all")
    print(f"{'=' * 70}")
    not_detected = fates_df[fates_df["fate"].str.startswith("not_detected")]
    if len(not_detected) > 0:
        for _, row in not_detected.iterrows():
            extra = ""
            if "nearest_detection_dt_s" in row and pd.notna(row.get("nearest_detection_dt_s")):
                extra = f" (nearest detection: {row['nearest_detection_dt_s']:.0f}s away)"
            print(f"  Singer {row['singer_time']} (moor={row['singer_moor']}, "
                  f"res={row['singer_res']:.1f}s, err1={row['singer_err1']:.3f}){extra}")

    print(f"\n{'=' * 70}")
    print("DETAILS: Detected but not classified (not in Phase 3)")
    print(f"{'=' * 70}")
    det_not_class = fates_df[fates_df["fate"] == "detected_not_classified"]
    if len(det_not_class) > 0:
        for _, row in det_not_class.iterrows():
            print(f"  Singer {row['singer_time']} (moor={row['singer_moor']}, "
                  f"res={row['singer_res']:.1f}s) → detected Δt={row.get('dt_s', '?')}s")

    print(f"\n{'=' * 70}")
    print("DETAILS: In Phase 3 catalogue but not located")
    print(f"{'=' * 70}")
    p3_not_loc = fates_df[fates_df["fate"].str.startswith("phase3_not_located")]
    if len(p3_not_loc) > 0:
        for _, row in p3_not_loc.iterrows():
            print(f"  Singer {row['singer_time']} (moor={row['singer_moor']}, "
                  f"res={row['singer_res']:.1f}s) → {row['fate']}, Δt={row.get('dt_s', '?')}s")

    # --- Figure: fate breakdown ---
    fig, ax = plt.subplots(figsize=(10, 6))
    cats = list(fate_categories.keys())
    vals = list(fate_categories.values())
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#d62728", "#7f7f7f"]
    bars = ax.barh(cats, vals, color=colors[:len(cats)], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of events", fontsize=12)
    ax.set_title("Fate of Singer's Top 100 EQ Events in Our Pipeline",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    outpath = FIG_DIR / "singer_eq_fate_investigation.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")

    # Save fate table
    fates_df.to_csv(DATA_DIR / "singer_top100_eq_fates.csv", index=False)
    print(f"Fate table saved: {DATA_DIR / 'singer_top100_eq_fates.csv'}")


if __name__ == "__main__":
    main()
