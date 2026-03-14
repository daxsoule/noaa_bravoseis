#!/usr/bin/env python3
"""
gold_consistency_check.py — Automated consistency checks on gold standard verdicts.

Two checks:
1. IPI neighbor filter: Flag any event with peak_freq 14–25 Hz that has
   repeating ~15s-spaced neighbors → probable fin whale, regardless of verdict.
2. Spectral profile check: Build feature profiles from later (more experienced)
   reviews, then flag earlier verdicts that don't match their label's profile.

Usage:
    uv run python scripts/gold_consistency_check.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# === Paths ===
PROJECT = Path(__file__).parent.parent
NOTEBOOK = PROJECT / "notebooks" / "gold_standard_review.ipynb"
DATA_DIR = PROJECT / "outputs" / "data"
OUTPUT_DIR = PROJECT / "outputs" / "data"

# === Label normalization ===
# Map raw verdict labels to canonical categories
LABEL_MAP = {
    # Seismic
    "eq": "seismic",
    "t": "seismic",
    "tphase": "seismic",
    "t-phase": "seismic",
    # Fin whale
    "fw": "fin_whale",
    # Icequake
    "iq": "icequake",
    "iq?": "icequake",
    "ice quake": "icequake",
    "icequake": "icequake",
    "ic": "icequake",
    # Humpback
    "hb": "humpback",
    "hb2": "humpback",
    # Whale (generic)
    "wc": "whale_unknown",
    "wc50": "whale_unknown",
    "wc97": "whale_unknown",
    # Unknown
    "?": "unknown",
    "lf": "unknown_lf",
    "not sure": "unknown",
    "unsure": "unknown",
    # Other
    "lshaped": "l_shaped",
    "ls": "l_shaped",
    "ship noise": "vessel",
}

# Review order — later clusters are more experienced reviews
CLUSTER_ORDER = ["high_3", "mid_1", "high_2", "high_0", "low_0"]

# === IPI parameters ===
IPI_FREQ_MIN = 14.0   # Hz — lower bound for fin whale band
IPI_FREQ_MAX = 25.0   # Hz — upper bound
IPI_TARGET = 15.0      # seconds — expected fin whale IPI
IPI_TOLERANCE = 3.0    # seconds — tolerance around target
IPI_WINDOW = 120.0     # seconds — look for neighbors within ±2 minutes
IPI_MIN_NEIGHBORS = 3  # need at least this many regular pulses to flag


def parse_verdicts():
    """Parse all verdict cells from gold standard notebook."""
    with open(NOTEBOOK) as f:
        nb = json.load(f)

    current_cluster = None
    current_sample = None
    verdicts = []

    for cell in nb["cells"]:
        src = "".join(cell["source"])

        m = re.search(r"## Cluster (\w+)", src)
        if m:
            current_cluster = m.group(1)
            current_sample = None
            continue

        if "STRATIFIED" in src or "Stratified" in src:
            current_sample = "stratified"
        elif "RANDOM" in src or "Random sample" in src:
            current_sample = "random"

        if cell["cell_type"] == "markdown" and "**Panel" in src:
            m2 = re.search(
                r"\*\*Panel\s+(\d+)\*\*.*Verdict:\s*([ar])\s*\|.*Identified As:\s*([^|]*?)(?:\|.*Comments:\s*(.*))?$",
                src.strip(),
                re.IGNORECASE,
            )
            if m2:
                raw_label = m2.group(3).strip().lower()
                canonical = LABEL_MAP.get(raw_label, raw_label)
                verdicts.append(
                    {
                        "cluster": current_cluster,
                        "sample": current_sample,
                        "panel": int(m2.group(1)),
                        "verdict": m2.group(2).strip().lower(),
                        "raw_label": raw_label,
                        "canonical_label": canonical,
                        "comments": m2.group(4).strip() if m2.group(4) else "",
                    }
                )

    return pd.DataFrame(verdicts)


def load_catalogue_and_features():
    """Load event catalogue and features."""
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])

    features = pd.read_parquet(DATA_DIR / "event_features.parquet")
    umap = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")

    merged = umap.merge(cat, on="event_id", suffixes=("", "_cat"))
    merged = merged.merge(
        features[
            [
                "event_id",
                "peak_freq_hz",
                "peak_power_db",
                "spectral_slope",
                "bandwidth_hz",
                "spectral_centroid_hz",
            ]
        ],
        on="event_id",
        suffixes=("", "_feat"),
        how="left",
    )
    return merged


def get_sampled_event_ids(merged_df, cluster_id, strategy, n=30):
    """Reproduce the same sampling used by make_gold_single_panels.py."""
    cluster_df = merged_df[merged_df["cluster_id"] == cluster_id].copy()

    cx = cluster_df["umap_1"].mean()
    cy = cluster_df["umap_2"].mean()
    cluster_df["_centroid_dist"] = np.sqrt(
        (cluster_df["umap_1"] - cx) ** 2 + (cluster_df["umap_2"] - cy) ** 2
    )

    n = min(n, len(cluster_df))

    if strategy == "stratified":
        n_bins = min(5, n)
        samples_per_bin = n // n_bins
        remainder = n % n_bins
        cluster_df["_dist_quantile"] = pd.qcut(
            cluster_df["_centroid_dist"], q=n_bins, labels=False, duplicates="drop"
        )
        actual_bins = sorted(cluster_df["_dist_quantile"].unique())
        rng = np.random.default_rng(42)
        parts = []
        for i, q in enumerate(actual_bins):
            bin_df = cluster_df[cluster_df["_dist_quantile"] == q]
            k = samples_per_bin + (1 if i < remainder else 0)
            k = min(k, len(bin_df))
            idx = rng.choice(len(bin_df), size=k, replace=False)
            parts.append(bin_df.iloc[idx])
        selected = pd.concat(parts).sort_values("_centroid_dist")
    else:
        rng = np.random.default_rng(43)
        idx = rng.choice(len(cluster_df), size=n, replace=False)
        selected = cluster_df.iloc[idx].copy()

    selected = selected.reset_index(drop=True)
    selected["panel"] = range(1, len(selected) + 1)
    return selected[["event_id", "panel"]]


def ipi_check(cat_df, verdicts_with_events):
    """Flag events that have fin-whale-like IPI neighbors.

    For each reviewed event in the 14–25 Hz band, look at the full catalogue
    for events on the same mooring within ±IPI_WINDOW seconds. Count how many
    neighbors fall at ~15s intervals. If enough regular neighbors exist,
    flag as probable fin whale.
    """
    flags = []

    # Get all mid-freq events from full catalogue for neighbor lookup
    mid_freq_mask = (
        (cat_df["peak_freq_hz"] >= IPI_FREQ_MIN)
        & (cat_df["peak_freq_hz"] <= IPI_FREQ_MAX)
    )
    mid_freq = cat_df[mid_freq_mask].copy()
    mid_freq = mid_freq.sort_values(["mooring", "onset_utc"])

    for _, row in verdicts_with_events.iterrows():
        event_id = row["event_id"]
        pf = row.get("peak_freq_hz", 0)
        if pd.isna(pf) or pf < IPI_FREQ_MIN or pf > IPI_FREQ_MAX:
            flags.append({"event_id": event_id, "ipi_flag": False, "n_ipi_neighbors": 0})
            continue

        onset = row["onset_utc"]
        mooring = row["mooring"]

        # Find neighbors on same mooring
        neighbors = mid_freq[
            (mid_freq["mooring"] == mooring)
            & (mid_freq["onset_utc"] >= onset - pd.Timedelta(seconds=IPI_WINDOW))
            & (mid_freq["onset_utc"] <= onset + pd.Timedelta(seconds=IPI_WINDOW))
            & (mid_freq["event_id"] != event_id)
        ]

        if len(neighbors) < IPI_MIN_NEIGHBORS:
            flags.append({"event_id": event_id, "ipi_flag": False, "n_ipi_neighbors": 0})
            continue

        # Check for regular ~15s spacing
        all_times = sorted(
            list(neighbors["onset_utc"].values) + [np.datetime64(onset)]
        )
        intervals = np.diff(all_times) / np.timedelta64(1, "s")

        # Count intervals near the target IPI
        near_target = np.sum(
            np.abs(intervals - IPI_TARGET) <= IPI_TOLERANCE
        )

        is_flagged = near_target >= IPI_MIN_NEIGHBORS
        flags.append(
            {
                "event_id": event_id,
                "ipi_flag": is_flagged,
                "n_ipi_neighbors": int(near_target),
            }
        )

    return pd.DataFrame(flags)


def spectral_profile_check(verdicts_with_events):
    """Build feature profiles from later reviews, flag earlier mismatches.

    Uses the last 2 clusters reviewed (most experienced) as the reference
    standard for each canonical label. Then checks earlier clusters for
    events whose features fall outside that label's expected range.
    """
    feature_cols = ["peak_freq_hz", "peak_power_db", "spectral_slope", "duration_s"]

    # Split into reference (later) and test (earlier) sets
    reference_clusters = CLUSTER_ORDER[-2:]  # high_0, low_0
    test_clusters = CLUSTER_ORDER[:-2]  # high_3, mid_1, high_2

    ref_mask = verdicts_with_events["cluster"].isin(reference_clusters)
    ref_data = verdicts_with_events[ref_mask].copy()
    test_data = verdicts_with_events[~verdicts_with_events["cluster"].isin(reference_clusters)].copy()

    # Build profiles per canonical label from reference data
    profiles = {}
    for label in ref_data["canonical_label"].unique():
        if label in ("unknown", "unknown_lf"):
            continue
        label_data = ref_data[ref_data["canonical_label"] == label]
        if len(label_data) < 3:
            continue
        profile = {}
        for col in feature_cols:
            vals = label_data[col].dropna()
            if len(vals) >= 3:
                profile[col] = {
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "p10": vals.quantile(0.1),
                    "p90": vals.quantile(0.9),
                }
        if profile:
            profiles[label] = profile

    # Check test events against their assigned label's profile
    flags = []
    for _, row in test_data.iterrows():
        label = row["canonical_label"]
        if label not in profiles:
            flags.append(
                {
                    "event_id": row["event_id"],
                    "spectral_flag": False,
                    "flag_reason": "",
                }
            )
            continue

        reasons = []
        profile = profiles[label]
        for col in feature_cols:
            if col not in profile:
                continue
            val = row.get(col, np.nan)
            if pd.isna(val):
                continue
            p = profile[col]
            # Flag if outside the reference p10–p90 range
            if val < p["p10"] or val > p["p90"]:
                reasons.append(
                    f"{col}={val:.1f} outside [{p['p10']:.1f}, {p['p90']:.1f}]"
                )

        flags.append(
            {
                "event_id": row["event_id"],
                "spectral_flag": len(reasons) >= 2,  # flag if ≥2 features outside range
                "flag_reason": "; ".join(reasons),
            }
        )

    return pd.DataFrame(flags)


def main():
    print("=" * 70)
    print("GOLD STANDARD CONSISTENCY CHECK")
    print("=" * 70)

    # 1. Parse verdicts
    print("\n1. Parsing verdicts from notebook...")
    verdicts = parse_verdicts()
    print(f"   {len(verdicts)} verdicts across {verdicts['cluster'].nunique()} clusters")
    print(f"   Label distribution:")
    for label, count in verdicts["canonical_label"].value_counts().items():
        print(f"     {label}: {count}")

    # 2. Load catalogue and match events to verdicts
    print("\n2. Loading catalogue and matching events to panels...")
    merged = load_catalogue_and_features()

    # Reconstruct which event_id corresponds to which panel
    verdict_events = []
    for cluster in verdicts["cluster"].unique():
        for sample in ["stratified", "random"]:
            cluster_verdicts = verdicts[
                (verdicts["cluster"] == cluster) & (verdicts["sample"] == sample)
            ]
            if len(cluster_verdicts) == 0:
                continue
            sampled = get_sampled_event_ids(merged, cluster, sample)
            matched = cluster_verdicts.merge(sampled, on="panel", how="left")
            verdict_events.append(matched)

    verdict_events = pd.concat(verdict_events, ignore_index=True)

    # Merge with event features
    verdict_events = verdict_events.merge(
        merged[
            [
                "event_id",
                "onset_utc",
                "mooring",
                "duration_s",
                "snr",
                "peak_freq_hz",
                "peak_power_db",
                "spectral_slope",
                "bandwidth_hz",
                "cluster_id",
                "detection_band",
            ]
        ],
        on="event_id",
        how="left",
    )
    matched_count = verdict_events["event_id"].notna().sum()
    print(f"   Matched {matched_count}/{len(verdict_events)} verdicts to events")

    # 3. IPI check
    print("\n3. Running IPI neighbor check (fin whale detection)...")
    ipi_flags = ipi_check(merged, verdict_events)
    verdict_events = verdict_events.merge(ipi_flags, on="event_id", how="left")

    ipi_flagged = verdict_events[verdict_events["ipi_flag"] == True]
    print(f"   {len(ipi_flagged)} events flagged as probable fin whale by IPI")

    # Show IPI flags that conflict with verdict
    ipi_conflicts = ipi_flagged[
        ~ipi_flagged["canonical_label"].isin(["fin_whale", "unknown", "unknown_lf"])
    ]
    if len(ipi_conflicts) > 0:
        print(f"\n   ⚠ IPI CONFLICTS — {len(ipi_conflicts)} events labeled non-whale but have fin whale IPI pattern:")
        for _, row in ipi_conflicts.iterrows():
            print(
                f"     {row['cluster']} {row['sample']} panel {row['panel']}: "
                f"labeled '{row['raw_label']}' ({row['canonical_label']}), "
                f"peak_freq={row['peak_freq_hz']:.1f} Hz, "
                f"{row['n_ipi_neighbors']} IPI neighbors"
            )
    else:
        print("   No IPI conflicts found.")

    # 4. Spectral profile check
    print("\n4. Running spectral profile check (later reviews as reference)...")
    print(f"   Reference clusters (experienced): {CLUSTER_ORDER[-2:]}")
    print(f"   Test clusters (earlier): {CLUSTER_ORDER[:-2]}")

    spectral_flags = spectral_profile_check(verdict_events)
    verdict_events = verdict_events.merge(spectral_flags, on="event_id", how="left")

    spectral_flagged = verdict_events[verdict_events["spectral_flag"] == True]
    print(f"   {len(spectral_flagged)} events flagged for spectral inconsistency")

    if len(spectral_flagged) > 0:
        print(f"\n   ⚠ SPECTRAL CONFLICTS:")
        for _, row in spectral_flagged.iterrows():
            print(
                f"     {row['cluster']} {row['sample']} panel {row['panel']}: "
                f"labeled '{row['raw_label']}' ({row['canonical_label']})"
            )
            if row.get("flag_reason"):
                print(f"       Reason: {row['flag_reason']}")

    # 5. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    any_flag = verdict_events[
        (verdict_events["ipi_flag"] == True) | (verdict_events["spectral_flag"] == True)
    ]
    print(f"Total verdicts analyzed: {len(verdict_events)}")
    print(f"Events flagged by IPI check: {len(ipi_flagged)}")
    print(f"  - Of which conflict with label: {len(ipi_conflicts)}")
    print(f"Events flagged by spectral check: {len(spectral_flagged)}")
    print(f"Events flagged by either check: {len(any_flag)}")

    # Save results
    out_path = OUTPUT_DIR / "gold_consistency_flags.parquet"
    verdict_events.to_parquet(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
