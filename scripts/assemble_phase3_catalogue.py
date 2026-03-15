#!/usr/bin/env python3
"""
assemble_phase3_catalogue.py — Build the Phase 3 combined event catalogue.

Combines accepted lowband (seismic, 1–14 Hz) and highband (cryogenic, >30 Hz)
events into a single classified catalogue. Discards highband_1 (mixed
icequake/humpback cluster).

Output:
    outputs/data/phase3_catalogue.parquet

Usage:
    uv run python scripts/assemble_phase3_catalogue.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"

# === Acceptance criteria ===
# Lowband: from re-clustering with 17 Hz whale filter
LB_ACCEPT_ALL = {"lowband_1"}           # accept unconditionally
LB_ACCEPT_SNR = {"lowband_2", "lowband_0"}  # accept if SNR >= 6
SNR_THRESHOLD = 6.0

# Highband: from gold standard review
HB_ACCEPT = {"highband_0", "highband_2", "highband_3"}
# highband_1 discarded: 33% icequake, 27% humpback, rest noise


def main():
    print("=" * 60)
    print("Phase 3 Catalogue Assembly")
    print("=" * 60)

    # Load base catalogue
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])
    print(f"Base catalogue: {len(cat):,} events")

    # --- Lowband ---
    lb_feat = pd.read_parquet(DATA_DIR / "event_features_lowband.parquet")
    lb_umap = pd.read_parquet(DATA_DIR / "umap_coordinates_lowband.parquet")
    lb = lb_feat[["event_id", "snr"]].merge(
        lb_umap[["event_id", "cluster_id"]], on="event_id", how="inner"
    )

    lb_mask = (
        lb["cluster_id"].isin(LB_ACCEPT_ALL) |
        (lb["cluster_id"].isin(LB_ACCEPT_SNR) & (lb["snr"] >= SNR_THRESHOLD))
    )
    lb_accepted = lb[lb_mask].copy()
    lb_accepted["phase3_class"] = "seismic"
    lb_accepted["phase3_band"] = "lowband"

    print(f"\nLowband accepted: {len(lb_accepted):,}")
    for cid, cnt in lb_accepted["cluster_id"].value_counts().items():
        print(f"  {cid}: {cnt:,}")

    # --- Highband ---
    hb_feat = pd.read_parquet(DATA_DIR / "event_features_highband.parquet")
    hb_umap = pd.read_parquet(DATA_DIR / "umap_coordinates_highband.parquet")
    hb = hb_feat[["event_id", "snr"]].merge(
        hb_umap[["event_id", "cluster_id"]], on="event_id", how="inner"
    )

    hb_accepted = hb[hb["cluster_id"].isin(HB_ACCEPT)].copy()
    hb_accepted["phase3_class"] = "cryogenic"
    hb_accepted["phase3_band"] = "highband"

    # Discarded
    hb_discarded = hb[~hb["cluster_id"].isin(HB_ACCEPT)]
    print(f"\nHighband accepted: {len(hb_accepted):,}")
    for cid, cnt in hb_accepted["cluster_id"].value_counts().items():
        print(f"  {cid}: {cnt:,}")
    print(f"Highband discarded: {len(hb_discarded):,}")
    for cid, cnt in hb_discarded["cluster_id"].value_counts().items():
        print(f"  {cid}: {cnt:,}")

    # --- Combine ---
    combined = pd.concat([
        lb_accepted[["event_id", "cluster_id", "phase3_class", "phase3_band"]],
        hb_accepted[["event_id", "cluster_id", "phase3_class", "phase3_band"]],
    ], ignore_index=True)

    # Merge with base catalogue
    result = cat.merge(combined, on="event_id", how="inner")
    result = result.sort_values("onset_utc").reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print(f"Combined Phase 3 catalogue: {len(result):,} events")
    print(f"{'=' * 60}")
    print(f"\nBy class:")
    for cls, cnt in result["phase3_class"].value_counts().items():
        print(f"  {cls}: {cnt:,}")
    print(f"\nBy band:")
    for band, cnt in result["phase3_band"].value_counts().items():
        print(f"  {band}: {cnt:,}")
    print(f"\nBy mooring:")
    for m, cnt in result["mooring"].value_counts().sort_index().items():
        print(f"  {m}: {cnt:,}")
    print(f"\nDate range: {result['onset_utc'].min()} to {result['onset_utc'].max()}")

    # Monthly breakdown
    result["month"] = result["onset_utc"].dt.to_period("M")
    print(f"\nMonthly counts by class:")
    monthly = result.groupby(["month", "phase3_class"]).size().unstack(fill_value=0)
    print(monthly.to_string())

    # Events not in Phase 3 catalogue
    n_uncategorized = len(cat) - len(result)
    print(f"\nEvents not in Phase 3 catalogue: {n_uncategorized:,}")
    print(f"  ({100 * n_uncategorized / len(cat):.1f}% of original catalogue)")
    print(f"  These include: mid-band (14-30 Hz whale), low-SNR, noise clusters,")
    print(f"  highband_1 (mixed), and events not in lowband/highband features")

    # Save
    out_cols = [
        "event_id", "onset_utc", "end_utc", "duration_s", "mooring",
        "file_number", "instrument_id", "detection_band",
        "peak_freq_hz", "bandwidth_hz", "peak_db", "snr",
        "onset_method", "onset_quality",
        "phase3_class", "phase3_band", "cluster_id",
    ]
    # Only keep columns that exist
    out_cols = [c for c in out_cols if c in result.columns]
    result = result[out_cols]

    outpath = DATA_DIR / "phase3_catalogue.parquet"
    result.to_parquet(outpath, index=False)
    print(f"\nSaved: {outpath} ({len(result):,} events)")


if __name__ == "__main__":
    main()
