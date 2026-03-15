#!/usr/bin/env python3
"""
crossvalidate_seismic_catalogues.py -- Cross-validate our lowband seismic
events against Singer's manual EQ catalogue and the Orca OBS catalogue.

Key questions:
  1. Do Singer's EQs appear in our accepted lowband events?
  2. Do Orca OBS earthquakes appear in our accepted lowband events?
  3. If not, where did they go? (whale filter, discarded cluster, low SNR, etc.)

Produces:
  - Console report with match statistics and fate analysis
  - outputs/figures/exploratory/crossval_seismic_singer.png
  - outputs/figures/exploratory/crossval_seismic_orca.png

Usage:
    uv run python scripts/crossvalidate_seismic_catalogues.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime, timedelta

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SINGER_PATH = Path("/home/jovyan/my_data/bravoseis/NOAA/merged_data_amended.txt")
ORCA_PATH = Path("/home/jovyan/my_data/bravoseis/earthquakes/Orca_EQ_data.csv")

# Match tolerance in seconds
MATCH_TOL_S = 30.0

# Lowband accepted clusters and SNR threshold (from review)
# After re-clustering with 17 Hz whale filter threshold:
#   lowband_1 = strong T-phases (was lowband_7 + lowband_0)
#   lowband_2 = mega-cluster bulk T-phases (was lowband_6 subset + lowband_5)
#   lowband_0 = recovered 12-14 Hz borderline events (new)
# For now, accept lowband_1 unconditionally, lowband_2 with SNR >= 6,
# and lowband_0 with SNR >= 6 (needs review)
ACCEPTED_CLUSTERS = {"lowband_1"}
ACCEPTED_WITH_SNR = {"lowband_2", "lowband_0"}  # only SNR >= 6
SNR_THRESHOLD = 6.0

# Whale filter: catalogue peak_freq threshold (raised from 14 to 17 Hz)
WHALE_FREQ_THRESHOLD = 17.0


# ============================================================
# 1. Parse Singer's EQ catalogue
# ============================================================
def parse_singer_catalogue(filepath):
    """Parse Singer's fixed-width catalogue, return only EQ events."""
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

            first_mooring = f"m{arrival_order[0]}" if arrival_order else None
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
                "first_mooring": first_mooring,
                "lat": lat,
                "lon": lon,
                "residual": residual,
                "singer_class": singer_class,
                "notes": notes_text,
            })

    df = pd.DataFrame(records)
    print(f"Singer catalogue: {len(df):,} total events")
    for cls, cnt in df["singer_class"].value_counts().items():
        print(f"  {cls}: {cnt:,}")
    return df


# ============================================================
# 2. Parse Orca OBS catalogue
# ============================================================
def load_orca_catalogue(filepath):
    """Load Orca OBS earthquake catalogue, convert MATLAB datenums."""
    df = pd.read_csv(filepath)
    # MATLAB datenum: days since 0000-01-00
    # Python: datetime(1, 1, 1) + timedelta(days=datenum - 367)
    # More precisely: MATLAB datenum 1 = Jan 1, 0000
    # Python can't handle year 0, use offset from a known date
    # MATLAB datenum 737791 = 2020-01-01
    matlab_epoch = datetime(1, 1, 1)
    df["datetime"] = df["date"].apply(
        lambda d: matlab_epoch + timedelta(days=d - 367)
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"\nOrca OBS catalogue: {len(df):,} events")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df


# ============================================================
# 3. Load our lowband pipeline data
# ============================================================
def load_lowband_data():
    """Load catalogue, lowband features, and cluster assignments.

    Returns:
        cat: full event catalogue
        lb_features: lowband feature extraction results (84K events)
        lb_merged: features with cluster + catalogue peak_freq info
        accepted_ids: set of accepted event IDs
    """
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])

    lb_features = pd.read_parquet(DATA_DIR / "event_features_lowband.parquet")
    lb_features["onset_utc"] = pd.to_datetime(lb_features["onset_utc"])

    lb_umap = pd.read_parquet(DATA_DIR / "umap_coordinates_lowband.parquet")

    # Build merged dataset with cluster info and catalogue peak_freq
    lb_merged = lb_features.merge(
        lb_umap[["event_id", "cluster_id"]],
        on="event_id", how="left"
    )
    # Whale filter uses CATALOGUE peak_freq, not lowband peak_freq
    # (lowband peak_freq is always <= 14 Hz by construction)
    cat_freq = cat[["event_id", "peak_freq_hz"]].rename(
        columns={"peak_freq_hz": "cat_peak_freq_hz"}
    )
    lb_merged = lb_merged.merge(cat_freq, on="event_id", how="left")

    # Count whale-filtered
    whale_mask = lb_merged["cat_peak_freq_hz"] > WHALE_FREQ_THRESHOLD
    n_whale = whale_mask.sum()

    # Accepted: lowband_7, lowband_0, or lowband_6 with SNR >= 6
    # (all already passed whale filter since they have cluster assignments)
    accepted_mask = (
        lb_merged["cluster_id"].isin(ACCEPTED_CLUSTERS) |
        (lb_merged["cluster_id"].isin(ACCEPTED_WITH_SNR) &
         (lb_merged["snr"] >= SNR_THRESHOLD))
    )

    accepted_ids = set(lb_merged.loc[accepted_mask, "event_id"])

    print(f"\nOur lowband pipeline:")
    print(f"  Total catalogue events: {len(cat):,}")
    print(f"  Lowband features extracted: {len(lb_features):,}")
    print(f"  Whale-filtered (cat peak_freq > 14 Hz): {n_whale:,}")
    print(f"  Lowband clustered: {len(lb_umap):,}")
    print(f"  Accepted lowband events: {len(accepted_ids):,}")

    return cat, lb_features, lb_merged, accepted_ids


# ============================================================
# 4. Match and trace fate
# ============================================================
def match_and_trace(ref_df, ref_time_col, ref_label,
                    cat, lb_merged, accepted_ids,
                    tol_s=MATCH_TOL_S):
    """Match reference catalogue events to our pipeline and trace fates.

    For each reference event, determine:
    - Did it match any detection in our catalogue? (within tol_s)
    - If matched, was the detection in the lowband feature set?
    - If in lowband, which cluster? Accepted or discarded?
    - If discarded, why? (whale filter, bad cluster, low SNR, noise)

    Returns DataFrame with one row per reference event.
    """
    ref_sorted = ref_df.sort_values(ref_time_col).reset_index(drop=True)

    # Prepare catalogue sorted by onset_utc
    cat_sorted = cat.sort_values("onset_utc").reset_index(drop=True)
    cat_times = cat_sorted["onset_utc"].values.astype("datetime64[ns]").astype(np.int64) / 1e9

    # Sets for fast lookup
    lb_feature_ids = set(lb_merged["event_id"])
    lb_cluster_map = dict(zip(lb_merged["event_id"], lb_merged["cluster_id"]))
    lb_snr_map = dict(zip(lb_merged["event_id"], lb_merged["snr"]))
    lb_peakfreq_map = dict(zip(lb_merged["event_id"], lb_merged["peak_freq_hz"]))
    # Whale filter uses catalogue peak_freq (broadband), not lowband peak_freq
    lb_cat_peakfreq_map = dict(zip(lb_merged["event_id"], lb_merged["cat_peak_freq_hz"]))

    results = []

    for i in range(len(ref_sorted)):
        row = ref_sorted.iloc[i]
        ref_time = row[ref_time_col].timestamp()

        # Find closest catalogue match within tolerance
        lo = np.searchsorted(cat_times, ref_time - tol_s, side="left")
        hi = np.searchsorted(cat_times, ref_time + tol_s, side="right")

        fate = "no_detection"
        match_dt = np.nan
        matched_event_id = None
        matched_mooring = None
        matched_band = None
        cluster = None
        snr = np.nan
        peak_freq = np.nan
        n_matches = 0

        if lo < hi:
            dts = np.abs(cat_times[lo:hi] - ref_time)
            best_local = np.argmin(dts)
            best_dt = dts[best_local]
            best_row = cat_sorted.iloc[lo + best_local]
            match_dt = best_dt
            matched_event_id = best_row["event_id"]
            matched_mooring = best_row["mooring"]
            matched_band = best_row["detection_band"]
            n_matches = hi - lo

            # Check if ANY match in window is in accepted set
            window_eids = cat_sorted.iloc[lo:hi]["event_id"].values
            accepted_in_window = [e for e in window_eids if e in accepted_ids]

            if accepted_in_window:
                # At least one accepted match
                fate = "accepted"
                matched_event_id = accepted_in_window[0]
                cluster = lb_cluster_map.get(matched_event_id)
                snr = lb_snr_map.get(matched_event_id, np.nan)
                peak_freq = lb_peakfreq_map.get(matched_event_id, np.nan)
            else:
                # Check if any match is in lowband features
                lb_in_window = [e for e in window_eids if e in lb_feature_ids]
                if lb_in_window:
                    eid = lb_in_window[0]
                    cluster = lb_cluster_map.get(eid)
                    snr = lb_snr_map.get(eid, np.nan)
                    peak_freq = lb_peakfreq_map.get(eid, np.nan)
                    cat_peak_freq = lb_cat_peakfreq_map.get(eid, np.nan)

                    if cat_peak_freq > WHALE_FREQ_THRESHOLD:
                        # Removed by whale filter (catalogue peak_freq > 17 Hz)
                        fate = "whale_filtered"
                    elif cluster in ("lowband_3", "lowband_4", "lowband_5"):
                        fate = "discarded_cluster"
                    elif cluster == "lowband_noise":
                        fate = "noise_cluster"
                    elif cluster in ACCEPTED_WITH_SNR and snr < SNR_THRESHOLD:
                        fate = "low_snr"
                    elif cluster is None:
                        # In lowband features but not clustered — whale-filtered
                        fate = "whale_filtered"
                    else:
                        fate = f"other_lb_{cluster}"
                else:
                    # Detected but not in lowband feature set
                    # (e.g., only detected in high or mid band)
                    fate = "not_in_lowband"

        results.append({
            "ref_idx": i,
            "ref_datetime": row[ref_time_col],
            "ref_lat": row.get("lat", np.nan),
            "ref_lon": row.get("lon", np.nan),
            "ref_class": row.get("singer_class", "EQ"),
            "fate": fate,
            "match_dt_s": match_dt,
            "matched_event_id": matched_event_id,
            "matched_mooring": matched_mooring,
            "matched_band": matched_band,
            "cluster": cluster,
            "snr": snr,
            "peak_freq": peak_freq,
            "n_catalogue_matches": n_matches,
        })

    result_df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"Cross-validation: {ref_label}")
    print(f"{'='*60}")
    print(f"Reference events: {len(result_df):,}")
    print(f"\nFate distribution:")
    for fate, cnt in result_df["fate"].value_counts().items():
        pct = 100 * cnt / len(result_df)
        print(f"  {fate:25s}: {cnt:6,} ({pct:5.1f}%)")

    # Detection rate
    detected = result_df["fate"] != "no_detection"
    n_det = detected.sum()
    print(f"\nOverall detection rate: {n_det:,}/{len(result_df):,} ({100*n_det/len(result_df):.1f}%)")

    accepted = result_df["fate"] == "accepted"
    n_acc = accepted.sum()
    print(f"Accepted in lowband: {n_acc:,}/{len(result_df):,} ({100*n_acc/len(result_df):.1f}%)")

    if n_det > 0:
        print(f"Accepted / detected: {n_acc:,}/{n_det:,} ({100*n_acc/n_det:.1f}%)")

    # Time offset stats for matches
    matched_mask = result_df["match_dt_s"].notna()
    if matched_mask.any():
        offsets = result_df.loc[matched_mask, "match_dt_s"]
        print(f"\nTime offset (matched): median={offsets.median():.1f}s, "
              f"mean={offsets.mean():.1f}s, max={offsets.max():.1f}s")

    return result_df


# ============================================================
# 5. Detailed investigation of missed events
# ============================================================
def investigate_missed(result_df, ref_label, cat):
    """Print details about missed reference events for investigation."""
    missed = result_df[result_df["fate"] == "no_detection"]
    if len(missed) == 0:
        print(f"\n  No missed events for {ref_label}!")
        return

    print(f"\n--- Missed {ref_label} events ({len(missed):,}) ---")

    # Check temporal distribution
    missed_dt = pd.to_datetime(missed["ref_datetime"])
    cat_min = cat["onset_utc"].min()
    cat_max = cat["onset_utc"].max()

    outside_range = ((missed_dt < cat_min) | (missed_dt > cat_max)).sum()
    print(f"  Outside our recording period: {outside_range:,}")
    print(f"  Our recording: {cat_min} to {cat_max}")

    inside = missed[(missed_dt >= cat_min) & (missed_dt <= cat_max)]
    if len(inside) > 0:
        print(f"  Within recording period but undetected: {len(inside):,}")
        print(f"  Monthly distribution of in-range misses:")
        monthly = inside["ref_datetime"].dt.to_period("M").value_counts().sort_index()
        for m, cnt in monthly.items():
            print(f"    {m}: {cnt}")

    # Filtered events breakdown
    filtered = result_df[~result_df["fate"].isin(["no_detection", "accepted"])]
    if len(filtered) > 0:
        print(f"\n--- Filtered {ref_label} events ({len(filtered):,}) ---")
        print(f"  These were detected but not accepted:")
        for fate, sub in filtered.groupby("fate"):
            print(f"\n  {fate} ({len(sub):,}):")
            if "cluster" in sub.columns:
                clusters = sub["cluster"].value_counts()
                for c, n in clusters.head(5).items():
                    print(f"    cluster {c}: {n}")
            if len(sub) <= 5:
                for _, r in sub.iterrows():
                    print(f"    {r['ref_datetime']}  "
                          f"SNR={r['snr']:.1f}  "
                          f"peak_freq={r['peak_freq']:.1f} Hz  "
                          f"cluster={r['cluster']}")


# ============================================================
# 6. Figures
# ============================================================
def make_crossval_figure(result_df, ref_label, out_path):
    """Create a 3-panel cross-validation figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel (a): Fate pie chart ---
    ax = axes[0]
    fate_counts = result_df["fate"].value_counts()
    # Group small categories
    fate_colors = {
        "accepted": "#2ca02c",
        "no_detection": "#d62728",
        "not_in_lowband": "#ff7f0e",
        "whale_filtered": "#9467bd",
        "discarded_cluster": "#8c564b",
        "noise_cluster": "#7f7f7f",
        "low_snr": "#e377c2",
        "deferred_cluster": "#17becf",
        "not_clustered": "#bcbd22",
    }
    colors = [fate_colors.get(f, "#999999") for f in fate_counts.index]
    labels = [f"{f}\n({c:,})" for f, c in zip(fate_counts.index, fate_counts.values)]

    wedges, texts, autotexts = ax.pie(
        fate_counts.values, labels=labels, colors=colors,
        autopct="%1.0f%%", pctdistance=0.75, startangle=90,
        textprops={"fontsize": 8}
    )
    for t in autotexts:
        t.set_fontsize(7)
    ax.set_title(f"(a) Fate of {ref_label} events", fontsize=14, fontweight="bold")

    # --- Panel (b): Monthly timeline ---
    ax = axes[1]
    result_df = result_df.copy()
    result_df["month"] = pd.to_datetime(result_df["ref_datetime"]).dt.to_period("M")

    fate_groups = {
        "accepted": "Accepted",
        "no_detection": "No detection",
    }
    other_fates = [f for f in result_df["fate"].unique()
                   if f not in fate_groups]

    months = sorted(result_df["month"].unique())
    x = np.arange(len(months))

    acc_counts = np.array([
        ((result_df["month"] == m) & (result_df["fate"] == "accepted")).sum()
        for m in months
    ])
    miss_counts = np.array([
        ((result_df["month"] == m) & (result_df["fate"] == "no_detection")).sum()
        for m in months
    ])
    filt_counts = np.array([
        ((result_df["month"] == m) & (result_df["fate"].isin(other_fates))).sum()
        for m in months
    ])

    bar_w = 0.25
    ax.bar(x - bar_w, acc_counts, bar_w, color="#2ca02c", label="Accepted")
    ax.bar(x, filt_counts, bar_w, color="#ff7f0e", label="Filtered")
    ax.bar(x + bar_w, miss_counts, bar_w, color="#d62728", label="No detection")

    month_labels = [str(m) for m in months]
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Events / month")
    ax.legend(fontsize=8)
    ax.set_title(f"(b) Monthly fate of {ref_label}", fontsize=14, fontweight="bold")

    # --- Panel (c): Time offset distribution for matches ---
    ax = axes[2]
    matched = result_df[result_df["match_dt_s"].notna()].copy()
    if len(matched) > 0:
        for fate_val, color, lbl in [
            ("accepted", "#2ca02c", "Accepted"),
            ("not_in_lowband", "#ff7f0e", "Not in lowband"),
            ("whale_filtered", "#9467bd", "Whale filtered"),
            ("discarded_cluster", "#8c564b", "Discarded cluster"),
            ("low_snr", "#e377c2", "Low SNR"),
        ]:
            sub = matched[matched["fate"] == fate_val]
            if len(sub) > 0:
                ax.hist(sub["match_dt_s"], bins=30, alpha=0.6, color=color,
                        label=f"{lbl} ({len(sub):,})", range=(0, MATCH_TOL_S))

        ax.set_xlabel("Time offset (s)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    ax.set_title(f"(c) Match time offsets", fontsize=14, fontweight="bold")

    fig.suptitle(f"Cross-validation: {ref_label} vs lowband seismic pipeline",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# ============================================================
# 7. Singer class breakdown
# ============================================================
def singer_class_breakdown(result_df):
    """Show fate breakdown by Singer class (EQ, IQ, IDK, SS)."""
    print(f"\n{'='*60}")
    print("FATE BY SINGER CLASS")
    print(f"{'='*60}")
    for cls in ["EQ", "IQ", "IDK", "SS", "other"]:
        sub = result_df[result_df["ref_class"] == cls]
        if len(sub) == 0:
            continue
        print(f"\n  Singer {cls} ({len(sub):,} events):")
        for fate, cnt in sub["fate"].value_counts().items():
            pct = 100 * cnt / len(sub)
            print(f"    {fate:25s}: {cnt:6,} ({pct:5.1f}%)")


# ============================================================
# 8. Build coverage intervals
# ============================================================
def build_coverage_intervals(cat):
    """Build merged time intervals where we have data, per mooring and overall.

    Each DAT file contributes a coverage window from its first to last detection,
    padded by 30 minutes on each side.
    """
    file_spans = cat.groupby(["mooring", "file_number"]).agg(
        t_min=("onset_utc", "min"),
        t_max=("onset_utc", "max")
    ).reset_index()
    file_spans["t_min"] -= pd.Timedelta(minutes=30)
    file_spans["t_max"] += pd.Timedelta(minutes=30)

    def merge_ivs(df):
        intervals = []
        for _, row in df.sort_values("t_min").iterrows():
            if intervals and row["t_min"] <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], row["t_max"]))
            else:
                intervals.append((row["t_min"], row["t_max"]))
        return intervals

    # Any-mooring coverage (union of all moorings)
    all_recs = []
    for _, row in file_spans.iterrows():
        all_recs.append({"t_min": row["t_min"], "t_max": row["t_max"]})
    any_intervals = merge_ivs(pd.DataFrame(all_recs))

    any_hours = sum((t1 - t0).total_seconds() / 3600 for t0, t1 in any_intervals)
    deploy_hours = (cat["onset_utc"].max() - cat["onset_utc"].min()).total_seconds() / 3600
    print(f"\nTemporal coverage: {any_hours:.0f}h / {deploy_hours:.0f}h "
          f"= {100 * any_hours / deploy_hours:.1f}% of deployment")

    return any_intervals


def filter_to_coverage(ref_df, time_col, any_intervals, label):
    """Filter reference catalogue to events within our coverage windows."""
    def in_coverage(dt):
        for t0, t1 in any_intervals:
            if t0 <= dt <= t1:
                return True
        return False

    mask = ref_df[time_col].apply(in_coverage)
    n_total = len(ref_df)
    n_in = mask.sum()
    print(f"\n{label}: {n_in:,}/{n_total:,} events in our coverage windows "
          f"({100 * n_in / n_total:.1f}%)")
    return ref_df[mask].copy()


# ============================================================
# Main
# ============================================================
def main():
    # Load our data
    cat, lb_features, lb_merged, accepted_ids = load_lowband_data()

    # Build coverage intervals
    any_intervals = build_coverage_intervals(cat)

    # --- Singer cross-validation ---
    singer_df = parse_singer_catalogue(SINGER_PATH)
    singer_covered = filter_to_coverage(singer_df, "datetime", any_intervals,
                                         "Singer catalogue")

    singer_results = match_and_trace(
        singer_covered, "datetime", "Singer (in coverage)",
        cat, lb_merged, accepted_ids, tol_s=MATCH_TOL_S
    )
    singer_class_breakdown(singer_results)
    investigate_missed(singer_results, "Singer", cat)

    make_crossval_figure(
        singer_results, "Singer (in coverage)",
        FIG_DIR / "crossval_seismic_singer.png"
    )

    # --- Orca OBS cross-validation ---
    orca_df = load_orca_catalogue(ORCA_PATH)

    # Filter to deployment period first, then coverage
    cat_min = cat["onset_utc"].min()
    cat_max = cat["onset_utc"].max()
    orca_deploy = orca_df[
        (orca_df["datetime"] >= cat_min) &
        (orca_df["datetime"] <= cat_max)
    ].copy()
    print(f"\nOrca in deployment period: {len(orca_deploy):,} / {len(orca_df):,}")

    orca_covered = filter_to_coverage(orca_deploy, "datetime", any_intervals,
                                       "Orca OBS")

    if len(orca_covered) > 0:
        orca_results = match_and_trace(
            orca_covered, "datetime", "Orca OBS (in coverage)",
            cat, lb_merged, accepted_ids, tol_s=MATCH_TOL_S
        )
        investigate_missed(orca_results, "Orca OBS", cat)

        make_crossval_figure(
            orca_results, "Orca OBS (in coverage)",
            FIG_DIR / "crossval_seismic_orca.png"
        )
    else:
        print("No Orca events in our coverage windows.")


if __name__ == "__main__":
    main()
