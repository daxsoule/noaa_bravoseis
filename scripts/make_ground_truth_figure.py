#!/usr/bin/env python3
"""
make_ground_truth_figure.py -- Ground truth validation against Singer's manual catalogue.

Compares our automated BRAVOSEIS catalogue against Jackie Singer's manual picks
from the same hydrophone data (merged_data_amended.txt).

Singer's catalogue uses PMEL format where the timestamp is the arrival time at the
first-arriving hydrophone. The arrival order column encodes which moorings detected
the event and in what order (digits 1-6 = moorings m1-m6).

Produces:
  outputs/figures/paper/ground_truth_singer.png
    Panel (a): Confusion matrix heatmap (Singer vs our labels)
    Panel (b): Monthly counts -- Singer EQ/IQ vs our T-phase/icequake
    Panel (c): Duration vs spectral slope, colored by Singer label

Usage:
    uv run python scripts/make_ground_truth_figure.py
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
FIG_DIR = OUTPUT_DIR / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SINGER_PATH = Path("/home/jovyan/my_data/bravoseis/NOAA/merged_data_amended.txt")

# Match tolerance in seconds — Singer's timestamp is the first arrival,
# which should be close to our onset_utc on the same mooring.  Allow some
# slack for different onset-picking methods.
MATCH_TOL_S = 30.0


# ============================================================
# 1. Parse Singer's catalogue
# ============================================================

def parse_singer_catalogue(filepath):
    """Parse Singer's fixed-width catalogue into a DataFrame.

    Format: YYYYDDDHHMMSSFF  N_moorings  arrival_order  lat  lon  err1  err2
            residual  db1  db2  class + notes

    The arrival_order field contains mooring numbers as digits (1-6).
    The first digit is the first-arriving mooring.
    """
    records = []

    with open(filepath, "r") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 11:
                continue

            # Parse timestamp: YYYYDDDHHMMSSFF
            ts_str = parts[0]
            if len(ts_str) < 14:
                continue

            try:
                year = int(ts_str[0:4])
                doy = int(ts_str[4:7])
                hh = int(ts_str[7:9])
                mm = int(ts_str[9:11])
                ss = int(ts_str[11:13])
                ff = int(ts_str[13:])  # fractional seconds (hundredths)

                dt = datetime(year, 1, 1) + timedelta(
                    days=doy - 1, hours=hh, minutes=mm,
                    seconds=ss, microseconds=ff * 10000
                )
            except (ValueError, IndexError):
                continue

            try:
                n_moorings = int(parts[1])
                arrival_order = parts[2]  # e.g., "231" or "654321"
                lat = float(parts[3])
                lon = float(parts[4])
                err1 = float(parts[5])
                err2 = float(parts[6])
                residual = float(parts[7])
                db1 = float(parts[8])
                db2 = float(parts[9])
            except (ValueError, IndexError):
                continue

            # First-arrival mooring (digit -> "m{digit}")
            first_mooring = f"m{arrival_order[0]}" if arrival_order else None

            # Classification is in the remaining text
            notes_text = " ".join(parts[10:])
            singer_class = "other"
            for token in parts[10:]:
                tok_upper = token.upper()
                if tok_upper == "EQ":
                    singer_class = "EQ"
                    break
                elif tok_upper == "IQ":
                    singer_class = "IQ"
                    break
                elif tok_upper == "IDK":
                    singer_class = "IDK"
                    break
                elif tok_upper == "SS":
                    singer_class = "SS"
                    break

            records.append({
                "datetime": pd.Timestamp(dt),
                "n_moorings": n_moorings,
                "arrival_order": arrival_order,
                "first_mooring": first_mooring,
                "lat": lat,
                "lon": lon,
                "err1": err1,
                "err2": err2,
                "residual": residual,
                "db1": db1,
                "db2": db2,
                "singer_class": singer_class,
                "notes": notes_text,
            })

    df = pd.DataFrame(records)
    print(f"Parsed Singer catalogue: {len(df):,} events")
    print(f"  Class distribution:")
    for cls, cnt in df["singer_class"].value_counts().items():
        print(f"    {cls}: {cnt:,}")
    print(f"  First-mooring distribution:")
    for m, cnt in df["first_mooring"].value_counts().sort_index().items():
        print(f"    {m}: {cnt:,}")
    return df


# ============================================================
# 2. Build our combined classification
# ============================================================

def load_our_labels():
    """Load event catalogue with combined Phase 1 + Phase 2 labels."""
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    features = pd.read_parquet(DATA_DIR / "event_features.parquet")
    umap_df = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")

    # --- Phase 1 labels (replicate assign_labels from train_cnn.py) ---
    tphase_clusters = {"low_0", "low_1", "mid_0"}
    cluster_tphase = set(
        umap_df[umap_df["cluster_id"].isin(tphase_clusters)]["event_id"]
    )
    feat_tphase_mask = (
        (features["spectral_slope"] < -0.5)
        & (features["peak_freq_hz"] < 30)
        & (features["peak_power_db"] > 48)
        & (features["duration_s"] <= 3)
    )
    all_tphase = cluster_tphase | set(features.loc[feat_tphase_mask, "event_id"])

    feat_ice_mask = (
        (features["duration_s"] > 3)
        & (features["peak_power_db"] > 48)
        & (features["peak_freq_hz"] < 30)
        & (features["spectral_slope"] < -0.2)
    )
    ice_cluster = set(umap_df[umap_df["cluster_id"] == "high_2"]["event_id"])
    all_ice = (set(features.loc[feat_ice_mask, "event_id"]) | ice_cluster) - all_tphase

    type_a_mask = (
        (features["spectral_slope"] > 0)
        & (features["peak_freq_hz"] > 100)
        & (features["bandwidth_hz"] > 150)
        & (features["freq_modulation"] > 30)
    )
    all_vessel = set(features.loc[type_a_mask, "event_id"]) - all_tphase - all_ice

    label_map = {}
    for eid in all_tphase:
        label_map[eid] = "tphase"
    for eid in all_ice:
        label_map[eid] = "icequake"
    for eid in all_vessel:
        label_map[eid] = "vessel"

    cnn = pd.read_parquet(DATA_DIR / "cnn_predictions.parquet")
    cnn_map = dict(zip(cnn["event_id"], cnn["cnn_label"]))

    def get_class(eid):
        if eid in label_map:
            return label_map[eid]
        if eid in cnn_map:
            return cnn_map[eid]
        return "unresolved"

    cat["our_class"] = cat["event_id"].apply(get_class)

    # Merge features for scatter plot
    feat_cols = ["event_id", "duration_s", "spectral_slope", "peak_freq_hz",
                 "peak_power_db", "bandwidth_hz"]
    feat_sub = features[feat_cols].copy()
    cat = cat.merge(feat_sub, on="event_id", how="left", suffixes=("", "_feat"))

    print(f"\nOur catalogue: {len(cat):,} events")
    print(f"  Class distribution:")
    for cls, cnt in cat["our_class"].value_counts().items():
        print(f"    {cls}: {cnt:,}")

    return cat


# ============================================================
# 3. Match events by time + mooring
# ============================================================

def match_events(singer_df, our_df, tol_s=MATCH_TOL_S):
    """Match Singer events to our catalogue.

    Strategy: Singer's timestamp is the first-arrival time at the
    first mooring in the arrival sequence. We match against our
    per-mooring detections on that specific mooring first (tight match),
    then fall back to any mooring if no mooring-specific match is found.

    For each Singer event, we find the closest our-detection within the
    tolerance window. We allow the same our-event to match multiple
    Singer events (since these are independent detections).
    """
    singer_sorted = singer_df.sort_values("datetime").reset_index(drop=True)

    # Pre-sort our catalogue by mooring for per-mooring matching
    our_by_mooring = {}
    for mooring in our_df["mooring"].unique():
        m_df = our_df[our_df["mooring"] == mooring].sort_values("onset_utc").reset_index(drop=True)
        m_times = m_df["onset_utc"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
        our_by_mooring[mooring] = (m_df, m_times)

    # Also prepare all-mooring sorted array for fallback
    our_all = our_df.sort_values("onset_utc").reset_index(drop=True)
    our_all_times = our_all["onset_utc"].values.astype("datetime64[ns]").astype(np.int64) / 1e9

    matched_rows = []
    match_method = {"mooring": 0, "any": 0, "none": 0}

    for i in range(len(singer_sorted)):
        s_row = singer_sorted.iloc[i]
        s_time = s_row["datetime"].timestamp()
        first_m = s_row["first_mooring"]

        best_j = None
        best_dt = tol_s + 1
        best_df = None
        method = "none"

        # Try mooring-specific match first
        if first_m in our_by_mooring:
            m_df, m_times = our_by_mooring[first_m]
            lo = np.searchsorted(m_times, s_time - tol_s, side="left")
            hi = np.searchsorted(m_times, s_time + tol_s, side="right")
            if lo < hi:
                dts = np.abs(m_times[lo:hi] - s_time)
                best_local = np.argmin(dts)
                if dts[best_local] <= tol_s:
                    best_j = lo + best_local
                    best_dt = dts[best_local]
                    best_df = m_df
                    method = "mooring"

        # Fallback: try any mooring
        if best_j is None:
            lo = np.searchsorted(our_all_times, s_time - tol_s, side="left")
            hi = np.searchsorted(our_all_times, s_time + tol_s, side="right")
            if lo < hi:
                dts = np.abs(our_all_times[lo:hi] - s_time)
                best_local = np.argmin(dts)
                if dts[best_local] <= tol_s:
                    best_j = lo + best_local
                    best_dt = dts[best_local]
                    best_df = our_all
                    method = "any"

        match_method[method] += 1

        if best_j is not None and best_df is not None:
            r = best_df.iloc[best_j]

            # Get majority class from all detections in window (any mooring)
            lo_all = np.searchsorted(our_all_times, s_time - tol_s, side="left")
            hi_all = np.searchsorted(our_all_times, s_time + tol_s, side="right")
            if lo_all < hi_all:
                classes_in_window = our_all.iloc[lo_all:hi_all]["our_class"].value_counts()
                majority_class = classes_in_window.index[0]
                n_in_window = hi_all - lo_all
                n_moorings_window = our_all.iloc[lo_all:hi_all]["mooring"].nunique()
            else:
                majority_class = r["our_class"]
                n_in_window = 1
                n_moorings_window = 1

            matched_rows.append({
                "singer_idx": i,
                "match_dt_s": best_dt,
                "match_method": method,
                "n_our_in_window": n_in_window,
                "n_moorings_matched": n_moorings_window,
                "majority_class": majority_class,
                "singer_class": s_row["singer_class"],
                "singer_datetime": s_row["datetime"],
                "singer_lat": s_row["lat"],
                "singer_lon": s_row["lon"],
                "singer_db1": s_row["db1"],
                "singer_n_moorings": s_row["n_moorings"],
                "our_class": r["our_class"],
                "our_event_id": r["event_id"],
                "our_onset_utc": r["onset_utc"],
                "our_mooring": r["mooring"],
                "duration_s_feat": r.get("duration_s_feat", r.get("duration_s")),
                "spectral_slope": r.get("spectral_slope", np.nan),
                "peak_freq_hz_feat": r.get("peak_freq_hz_feat", r.get("peak_freq_hz")),
            })

    matched = pd.DataFrame(matched_rows)

    print(f"\nMatching results (tolerance={tol_s:.0f}s):")
    print(f"  Singer events: {len(singer_sorted):,}")
    print(f"  Matched: {len(matched):,} ({100*len(matched)/len(singer_sorted):.1f}%)")
    print(f"  Match method: mooring-specific={match_method['mooring']:,}, "
          f"any-mooring={match_method['any']:,}, none={match_method['none']:,}")
    if len(matched) > 0:
        print(f"  Median time offset: {matched['match_dt_s'].median():.2f} s")
        print(f"  Mean time offset: {matched['match_dt_s'].mean():.2f} s")
        print(f"  Median our-detections per match window: "
              f"{matched['n_our_in_window'].median():.0f}")

    # Diagnostic: check time offset distribution
    if len(matched) > 0:
        print(f"\n  Time offset distribution:")
        for t in [1, 2, 5, 10, 15, 20, 30]:
            n = (matched["match_dt_s"] <= t).sum()
            print(f"    <= {t}s: {n:,} ({100*n/len(matched):.1f}%)")

    return matched


# ============================================================
# 4. Summary statistics
# ============================================================

def check_coverage(singer_df, our_df):
    """Check how many Singer events fall within our data coverage.

    Uses a 2-hour window around each DAT file's detection range as the
    coverage interval. Reports how many Singer events are 'covered'.
    """
    print("\n--- Data coverage analysis ---")

    # Build coverage intervals per mooring+file
    cov = our_df.groupby(["mooring", "file_number"]).agg(
        t_min=("onset_utc", "min"),
        t_max=("onset_utc", "max")
    ).reset_index()

    # Extend by 30 min on each side (DAT files are ~4h, events at edges
    # may have been missed by our detector but not by Singer)
    cov["t_min"] = cov["t_min"] - pd.Timedelta(minutes=30)
    cov["t_max"] = cov["t_max"] + pd.Timedelta(minutes=30)

    # Merge overlapping intervals per mooring
    covered_intervals = []
    for mooring, g in cov.groupby("mooring"):
        g_sorted = g.sort_values("t_min")
        intervals = []
        for _, row in g_sorted.iterrows():
            if intervals and row["t_min"] <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], row["t_max"]))
            else:
                intervals.append((row["t_min"], row["t_max"]))
        for t0, t1 in intervals:
            covered_intervals.append((mooring, t0, t1))

    cov_df = pd.DataFrame(covered_intervals, columns=["mooring", "t_min", "t_max"])

    # For each Singer event, check if first_mooring has coverage at that time
    singer_covered = 0
    singer_any_covered = 0
    for _, s in singer_df.iterrows():
        st = s["datetime"]
        fm = s["first_mooring"]

        # Check specific mooring
        fm_cov = cov_df[cov_df["mooring"] == fm]
        if ((fm_cov["t_min"] <= st) & (fm_cov["t_max"] >= st)).any():
            singer_covered += 1

        # Check any mooring
        if ((cov_df["t_min"] <= st) & (cov_df["t_max"] >= st)).any():
            singer_any_covered += 1

    total = len(singer_df)
    print(f"  Singer events in first-mooring coverage: "
          f"{singer_covered}/{total} ({100*singer_covered/total:.1f}%)")
    print(f"  Singer events in any-mooring coverage: "
          f"{singer_any_covered}/{total} ({100*singer_any_covered/total:.1f}%)")

    return singer_covered, singer_any_covered


def print_summary(matched, singer_df, n_any_covered=None):
    """Print detailed summary statistics."""
    print("\n" + "=" * 60)
    print("GROUND TRUTH VALIDATION SUMMARY")
    print("=" * 60)

    total = len(singer_df)
    n_matched = len(matched)
    print(f"\nTotal Singer events: {total:,}")
    print(f"Matched to our catalogue: {n_matched:,} ({100*n_matched/total:.1f}%)")
    print(f"Unmatched: {total - n_matched:,}")
    if n_any_covered and n_any_covered > 0:
        print(f"Detection rate (within coverage): {n_matched}/{n_any_covered} "
              f"({100*n_matched/n_any_covered:.1f}%)")

    singer_classes = ["EQ", "IQ", "IDK", "SS"]
    our_classes = ["tphase", "icequake", "vessel", "unresolved"]

    # Use majority_class for confusion (more robust than single closest match)
    class_col = "majority_class" if "majority_class" in matched.columns else "our_class"

    print(f"\n--- Confusion summary (matched events, using {class_col}) ---")
    for sc in singer_classes:
        sub = matched[matched["singer_class"] == sc]
        if len(sub) == 0:
            continue
        print(f"\n  Singer {sc} ({len(sub):,} matched):")
        for oc in our_classes:
            n = (sub[class_col] == oc).sum()
            if n > 0:
                print(f"    -> our {oc}: {n:,} ({100*n/len(sub):.1f}%)")

    # Key disagreements
    print(f"\n--- Key disagreements ---")
    eq_to_ice = ((matched["singer_class"] == "EQ") & (matched[class_col] == "icequake")).sum()
    ice_to_tphase = ((matched["singer_class"] == "IQ") & (matched[class_col] == "tphase")).sum()
    eq_to_tphase = ((matched["singer_class"] == "EQ") & (matched[class_col] == "tphase")).sum()
    iq_to_ice = ((matched["singer_class"] == "IQ") & (matched[class_col] == "icequake")).sum()

    print(f"  Singer EQ -> our icequake: {eq_to_ice:,}")
    print(f"  Singer EQ -> our tphase: {eq_to_tphase:,}")
    print(f"  Singer IQ -> our tphase: {ice_to_tphase:,}")
    print(f"  Singer IQ -> our icequake: {iq_to_ice:,}")

    # Agreement rate
    eq_matched = matched[matched["singer_class"] == "EQ"]
    iq_matched = matched[matched["singer_class"] == "IQ"]

    if len(eq_matched) > 0:
        eq_agree = (eq_matched[class_col] == "tphase").sum()
        print(f"\n  Singer EQ -> our tphase agreement: {eq_agree}/{len(eq_matched)} "
              f"({100*eq_agree/len(eq_matched):.1f}%)")

    if len(iq_matched) > 0:
        iq_agree = (iq_matched[class_col] == "icequake").sum()
        print(f"  Singer IQ -> our icequake agreement: {iq_agree}/{len(iq_matched)} "
              f"({100*iq_agree/len(iq_matched):.1f}%)")


# ============================================================
# 5. Publication figure
# ============================================================

def make_figure(matched, singer_df, our_df):
    """Create 3-panel ground truth validation figure."""

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.28,
                  left=0.06, right=0.96, top=0.96, bottom=0.05)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # --- Panel (a): Confusion matrix ---
    singer_labels = ["EQ", "IQ", "IDK", "SS"]
    our_labels = ["tphase", "icequake", "vessel", "unresolved"]
    our_display = ["T-phase", "Icequake", "Vessel", "Unresolved"]
    singer_display = ["Singer EQ", "Singer IQ", "Singer IDK", "Singer SS"]

    class_col = "majority_class" if "majority_class" in matched.columns else "our_class"

    conf = np.zeros((len(singer_labels), len(our_labels)), dtype=int)
    for i, sl in enumerate(singer_labels):
        for j, ol in enumerate(our_labels):
            conf[i, j] = ((matched["singer_class"] == sl) &
                           (matched[class_col] == ol)).sum()

    # Drop rows/cols with no data
    row_mask = conf.sum(axis=1) > 0
    col_mask = conf.sum(axis=0) > 0
    conf_plot = conf[np.ix_(row_mask, col_mask)]
    s_disp = [singer_display[i] for i in range(len(singer_labels)) if row_mask[i]]
    o_disp = [our_display[j] for j in range(len(our_labels)) if col_mask[j]]

    row_sums = conf_plot.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    conf_pct = conf_plot / row_sums_safe * 100

    im = ax_a.imshow(conf_pct, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    for i in range(conf_plot.shape[0]):
        for j in range(conf_plot.shape[1]):
            count = conf_plot[i, j]
            pct = conf_pct[i, j]
            color = "white" if pct > 60 else "black"
            ax_a.text(j, i, f"{count:,}\n({pct:.0f}%)",
                      ha="center", va="center", fontsize=9, color=color,
                      fontweight="bold" if pct > 30 else "normal")

    ax_a.set_xticks(range(len(o_disp)))
    ax_a.set_xticklabels(o_disp, fontsize=10, rotation=20, ha="right")
    ax_a.set_yticks(range(len(s_disp)))
    ax_a.set_yticklabels(s_disp, fontsize=10)
    ax_a.set_xlabel("Our classification", fontsize=11)
    ax_a.set_ylabel("Singer classification", fontsize=11)
    ax_a.set_title("(a) Classification cross-comparison", fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax_a, shrink=0.8, pad=0.02)
    cbar.set_label("Row %", fontsize=10)

    # --- Panel (b): Monthly counts ---
    # Show Singer EQ and IQ alongside our T-phase and icequake.
    # Use dual y-axis: left for Singer counts, right for our counts.
    singer_eq = singer_df[singer_df["singer_class"] == "EQ"].copy()
    singer_iq = singer_df[singer_df["singer_class"] == "IQ"].copy()
    singer_eq["month"] = singer_eq["datetime"].dt.to_period("M")
    singer_iq["month"] = singer_iq["datetime"].dt.to_period("M")

    our_tp = our_df[our_df["our_class"] == "tphase"].copy()
    our_iq = our_df[our_df["our_class"] == "icequake"].copy()
    our_tp["month"] = our_tp["onset_utc"].dt.to_period("M")
    our_iq["month"] = our_iq["onset_utc"].dt.to_period("M")

    all_months_set = set()
    for df_tmp in [singer_eq, singer_iq, our_tp, our_iq]:
        if len(df_tmp) > 0:
            all_months_set.update(df_tmp["month"].unique())
    all_months = sorted(all_months_set) if all_months_set else []
    month_idx = np.arange(len(all_months))

    def counts_by_month(df, col="month"):
        counts = df[col].value_counts()
        return np.array([counts.get(m, 0) for m in all_months])

    s_eq_counts = counts_by_month(singer_eq)
    s_iq_counts = counts_by_month(singer_iq)
    o_tp_counts = counts_by_month(our_tp)
    o_iq_counts = counts_by_month(our_iq)

    bar_w = 0.2

    # Singer bars on primary axis
    ax_b.bar(month_idx - bar_w, s_eq_counts, bar_w * 2, color="#4878CF",
             alpha=0.8, label="Singer EQ")
    ax_b.bar(month_idx + bar_w, s_iq_counts, bar_w * 2, color="#EE854A",
             alpha=0.8, label="Singer IQ")
    ax_b.set_ylabel("Singer events / month", fontsize=11)

    # Our line traces on secondary axis (different scale)
    ax_b2 = ax_b.twinx()
    ax_b2.plot(month_idx, o_tp_counts, "o-", color="#4878CF", markersize=4,
               linewidth=1.5, alpha=0.6, label="Our T-phase")
    ax_b2.plot(month_idx, o_iq_counts, "s-", color="#EE854A", markersize=4,
               linewidth=1.5, alpha=0.6, label="Our icequake")
    ax_b2.set_ylabel("Our detections / month", fontsize=11)

    month_labels = [m.strftime("%b\n%Y") if m.month in (1, 4, 7, 10)
                    else m.strftime("%b") for m in all_months]
    ax_b.set_xticks(month_idx)
    ax_b.set_xticklabels(month_labels, fontsize=8, rotation=0)
    ax_b.set_title("(b) Monthly event counts", fontsize=14, fontweight="bold")

    # Combine legends from both axes
    lines1, labels1 = ax_b.get_legend_handles_labels()
    lines2, labels2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(lines1 + lines2, labels1 + labels2,
                fontsize=8, loc="upper right", ncol=2)
    ax_b.set_xlim(-0.6, len(all_months) - 0.4)

    # --- Panel (c): Duration vs spectral slope colored by Singer label ---
    plot_df = matched.dropna(subset=["duration_s_feat", "spectral_slope"]).copy()

    color_map = {"EQ": "#4878CF", "IQ": "#EE854A", "IDK": "#6ACC65", "SS": "#D65F5F",
                 "other": "#999999"}
    marker_map = {"EQ": "o", "IQ": "s", "IDK": "^", "SS": "D", "other": "P"}
    zorder_map = {"EQ": 4, "IQ": 3, "IDK": 2, "SS": 1, "other": 0}

    for cls in ["other", "SS", "IDK", "IQ", "EQ"]:
        sub = plot_df[plot_df["singer_class"] == cls]
        if len(sub) == 0:
            continue
        ax_c.scatter(
            sub["duration_s_feat"], sub["spectral_slope"],
            c=color_map.get(cls, "#999999"),
            marker=marker_map.get(cls, "o"),
            s=12, alpha=0.4, label=f"Singer {cls} (n={len(sub):,})",
            zorder=zorder_map.get(cls, 0), edgecolors="none"
        )

    # Feature-space region boundaries
    ax_c.axvline(3, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax_c.axhline(-0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax_c.axhline(-0.2, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    ax_c.text(1.5, -0.85, "T-phase\nfeature space",
              fontsize=9, ha="center", va="center", color="gray",
              fontstyle="italic", alpha=0.8)
    ax_c.text(8, -0.35, "Icequake\nfeature space",
              fontsize=9, ha="center", va="center", color="gray",
              fontstyle="italic", alpha=0.8)

    ax_c.set_xlabel("Duration (s)", fontsize=11)
    ax_c.set_ylabel("Spectral slope", fontsize=11)
    ax_c.set_title("(c) Singer labels in feature space", fontsize=14, fontweight="bold")
    ax_c.legend(fontsize=9, loc="lower left", ncol=2, markerscale=2)

    if len(plot_df) > 10:
        ax_c.set_xlim(0, min(30, plot_df["duration_s_feat"].quantile(0.99) * 1.1))
        ax_c.set_ylim(plot_df["spectral_slope"].quantile(0.01) * 1.1,
                      max(0.5, plot_df["spectral_slope"].quantile(0.99) * 1.1))

    out_path = FIG_DIR / "ground_truth_singer.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Ground Truth Validation: Singer vs Automated Catalogue")
    print("=" * 60)

    # 1. Parse Singer
    singer_df = parse_singer_catalogue(SINGER_PATH)

    # 2. Load our labels
    our_df = load_our_labels()

    # 3. Coverage analysis
    n_fm_covered, n_any_covered = check_coverage(singer_df, our_df)

    # 4. Match events
    matched = match_events(singer_df, our_df, tol_s=MATCH_TOL_S)

    # 5. Print summary
    print_summary(matched, singer_df, n_any_covered=n_any_covered)

    # 6. Make figure
    make_figure(matched, singer_df, our_df)


if __name__ == "__main__":
    main()
