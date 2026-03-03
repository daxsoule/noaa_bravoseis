#!/usr/bin/env python3
"""
associate_events.py — Cross-mooring event association for BRAVOSEIS.

Finds events detected on 2+ moorings within a plausible travel-time window
and groups them into associations using connected components.

Uses pair-specific maximum travel times derived from in-situ XBT sound speed
profiles (see compute_travel_times.py). Falls back to a constant 120 s window
if the travel_times.json file is not found.

Usage:
    uv run python associate_events.py

Spec: specs/001-event-detection/
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

from read_dat import MOORINGS

# === Paths ===
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
TABLE_DIR = OUTPUT_DIR / "tables"

# === Fallback constant (used if travel_times.json is missing) ===
FALLBACK_MAX_TRAVEL_TIME_S = 120

MOORING_KEYS = sorted(MOORINGS.keys())


def load_pair_travel_times():
    """Load pair-specific max travel times from JSON.

    Returns
    -------
    pair_max_dt : dict
        {(k1, k2): max_travel_time_s} for all 15 pairs (sorted keys).
    global_max : float
        Maximum travel time across all pairs.
    source : str
        "xbt" or "fallback".
    """
    json_path = DATA_DIR / "travel_times.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        pair_max_dt = {}
        for pair_key, info in data["pairs"].items():
            k1, k2 = pair_key.split("-")
            pair_max_dt[(k1, k2)] = info["max_travel_time_s"]
            pair_max_dt[(k2, k1)] = info["max_travel_time_s"]
        global_max = data["global_max_travel_time_s"]
        return pair_max_dt, global_max, "xbt"
    else:
        print(f"  WARNING: {json_path} not found — using constant "
              f"{FALLBACK_MAX_TRAVEL_TIME_S} s for all pairs")
        print(f"  Run 'uv run python compute_travel_times.py' first")
        pair_max_dt = {}
        for i, k1 in enumerate(MOORING_KEYS):
            for k2 in MOORING_KEYS[i + 1:]:
                pair_max_dt[(k1, k2)] = FALLBACK_MAX_TRAVEL_TIME_S
                pair_max_dt[(k2, k1)] = FALLBACK_MAX_TRAVEL_TIME_S
        return pair_max_dt, FALLBACK_MAX_TRAVEL_TIME_S, "fallback"


def compute_mooring_distances():
    """Compute pairwise distances between moorings (km)."""
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    distances = {}
    for i, k1 in enumerate(MOORING_KEYS):
        for k2 in MOORING_KEYS[i+1:]:
            _, _, dist_m = geod.inv(
                MOORINGS[k1]["lon"], MOORINGS[k1]["lat"],
                MOORINGS[k2]["lon"], MOORINGS[k2]["lat"],
            )
            distances[(k1, k2)] = dist_m / 1000
    return distances


def find_associations(df, pair_max_dt, global_max):
    """Find events on different moorings within pair-specific travel-time windows.

    Uses greedy windowed clustering: for each unassigned event, scan forward up
    to global_max seconds. For each candidate on another mooring, check the
    pair-specific travel time limit before including it.

    Parameters
    ----------
    df : pd.DataFrame
        Event catalogue with onset_utc, mooring, snr, event_id columns.
    pair_max_dt : dict
        {(k1, k2): max_s} for all mooring pairs.
    global_max : float
        Maximum travel time across all pairs (outer scan window).
    """
    # Sort by onset time
    df = df.sort_values("onset_utc").reset_index(drop=True)

    onset = df["onset_utc"].values.astype("datetime64[ms]").astype(np.int64) / 1000
    moorings = df["mooring"].values

    n = len(df)
    assigned = np.zeros(n, dtype=bool)
    associations = []

    for i in range(n):
        if assigned[i]:
            continue

        anchor_mooring = moorings[i]
        members = [i]
        member_moorings = {anchor_mooring}

        # Scan forward up to global_max (outer window)
        j = i + 1
        while j < n and (onset[j] - onset[i]) <= global_max:
            if not assigned[j] and moorings[j] != anchor_mooring:
                # Check pair-specific travel time
                dt = onset[j] - onset[i]
                pair_key = (anchor_mooring, moorings[j])
                max_dt = pair_max_dt.get(pair_key, global_max)
                if dt <= max_dt:
                    members.append(j)
                    member_moorings.add(moorings[j])
            j += 1

        if len(member_moorings) < 2:
            continue

        # Keep only the best event (highest SNR) per mooring
        best_per_mooring = {}
        for m_idx in members:
            mk = moorings[m_idx]
            if mk not in best_per_mooring or df.iloc[m_idx]["snr"] > df.iloc[best_per_mooring[mk]]["snr"]:
                best_per_mooring[mk] = m_idx

        final_members = list(best_per_mooring.values())
        final_moorings = set(best_per_mooring.keys())

        if len(final_moorings) < 2:
            continue

        # Mark as assigned
        for m_idx in final_members:
            assigned[m_idx] = True

        event_ids = [df.iloc[m]["event_id"] for m in final_members]
        onsets_utc = [df.iloc[m]["onset_utc"] for m in final_members]
        earliest = min(onsets_utc)
        latest = max(onsets_utc)
        dt_s = (latest - earliest).total_seconds()

        associations.append({
            "assoc_id": f"A{len(associations):06d}",
            "n_moorings": len(final_moorings),
            "moorings": ",".join(sorted(final_moorings)),
            "n_events": len(final_members),
            "event_ids": ",".join(event_ids),
            "earliest_utc": earliest,
            "latest_utc": latest,
            "dt_s": round(dt_s, 3),
            "detection_band": df.iloc[final_members[0]]["detection_band"],
        })

        if len(associations) % 1000 == 0:
            print(f"    {len(associations):,} associations found...")

    return pd.DataFrame(associations)


def main():
    print("=" * 60)
    print("Cross-Mooring Event Association")
    print("=" * 60)

    # Load pair-specific travel times
    pair_max_dt, global_max, source = load_pair_travel_times()
    if source == "xbt":
        print(f"  Travel times: XBT-derived (global max {global_max:.1f} s)")
    else:
        print(f"  Travel times: FALLBACK constant ({global_max} s)")

    # Load catalogue
    cat_path = DATA_DIR / "event_catalogue.parquet"
    df = pd.read_parquet(cat_path)
    print(f"\nLoaded {len(df):,} events from {cat_path.name}")

    # Mooring distances
    distances = compute_mooring_distances()
    print("\nMooring pair distances and max travel times:")
    for (k1, k2), dist in sorted(distances.items()):
        max_t = pair_max_dt.get((k1, k2), global_max)
        print(f"  {k1}-{k2}: {dist:.0f} km (max {max_t:.1f} s)")

    # Find associations
    print("\nFinding associations...")
    assoc_df = find_associations(df, pair_max_dt, global_max)

    if len(assoc_df) == 0:
        print("No multi-mooring associations found!")
        return

    print(f"\nTotal associations: {len(assoc_df):,}")

    # Summary: events by number of moorings
    print("\nAssociations by number of moorings:")
    mooring_counts = assoc_df["n_moorings"].value_counts().sort_index()
    for n_m, count in mooring_counts.items():
        print(f"  {n_m} moorings: {count:,} associations")

    # How many events are in associations vs. isolated?
    all_assoc_events = set()
    for eids in assoc_df["event_ids"]:
        all_assoc_events.update(eids.split(","))
    n_associated = len(all_assoc_events)
    print(f"\nEvents in multi-mooring associations: {n_associated:,} "
          f"({100*n_associated/len(df):.1f}%)")
    print(f"Isolated (single-mooring) events: {len(df) - n_associated:,} "
          f"({100*(len(df)-n_associated)/len(df):.1f}%)")

    # Travel time statistics for 2-mooring associations
    two_mooring = assoc_df[assoc_df["n_moorings"] >= 2]
    print(f"\nTravel time (dt_s) for multi-mooring associations:")
    print(f"  Median: {two_mooring['dt_s'].median():.1f} s")
    print(f"  IQR: [{two_mooring['dt_s'].quantile(0.25):.1f}, "
          f"{two_mooring['dt_s'].quantile(0.75):.1f}] s")
    print(f"  Max: {two_mooring['dt_s'].max():.1f} s")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    assoc_path = DATA_DIR / "cross_mooring_associations.parquet"
    assoc_df.to_parquet(assoc_path, index=False)
    print(f"\nSaved: {assoc_path} ({len(assoc_df):,} associations)")

    # Summary tables
    # 1. Cross-mooring counts by band
    cross_tab = pd.crosstab(assoc_df["n_moorings"],
                            assoc_df["detection_band"],
                            margins=True)
    cross_path = TABLE_DIR / "cross_mooring_counts.csv"
    cross_tab.to_csv(cross_path)
    print(f"Saved: {cross_path}")

    # 2. Catalogue summary
    summary_rows = []
    for mkey in MOORING_KEYS:
        m_events = df[df["mooring"] == mkey]
        m_assoc = m_events[m_events["event_id"].isin(all_assoc_events)]
        for band in sorted(df["detection_band"].unique()):
            b_events = m_events[m_events["detection_band"] == band]
            b_assoc = m_assoc[m_assoc["detection_band"] == band]
            summary_rows.append({
                "mooring": mkey,
                "band": band,
                "n_events": len(b_events),
                "n_associated": len(b_assoc),
                "pct_associated": round(100 * len(b_assoc) / max(len(b_events), 1), 1),
                "median_duration_s": round(b_events["duration_s"].median(), 2)
                    if len(b_events) > 0 else None,
                "median_snr": round(b_events["snr"].median(), 2)
                    if len(b_events) > 0 else None,
                "median_peak_freq_hz": round(b_events["peak_freq_hz"].median(), 1)
                    if len(b_events) > 0 else None,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLE_DIR / "catalogue_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
