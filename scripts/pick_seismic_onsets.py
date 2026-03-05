#!/usr/bin/env python3
"""
pick_seismic_onsets.py — Dual onset picker tuned for seismic/T-phase events.

The AIC picker (refine_onsets.py) works well for impulsive arrivals but picks
late on emergent T-phases. This script runs two independent methods — envelope
STA/LTA and kurtosis onset — and takes the earlier valid pick, targeting the
11,958 events in 6 seismic clusters.

Target clusters:
  low_0   (4,686)  — emergent T-phase      1-15 Hz, 10s pre-window
  low_1   (3,666)  — mixed seismic          1-15 Hz, 8s pre-window
  low_2_2 (1,134)  — impulsive broadband    1-15 Hz, 5s pre-window
  mid_0   (1,056)  — mixed seismic          5-30 Hz, 8s pre-window
  mid_3_1 (1,067)  — emergent broadband     5-30 Hz, 8s pre-window
  mid_3_3   (349)  — impulsive              5-30 Hz, 5s pre-window

Usage:
    uv run python pick_seismic_onsets.py                  # all seismic events
    uv run python pick_seismic_onsets.py --mooring m1     # single mooring
    uv run python pick_seismic_onsets.py --qc-montage     # also generate QC figures
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
from scipy.signal import butter, sosfilt, hilbert

from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "seismic_onsets"

# === Cluster configuration ===
# Each cluster gets a seismic-tuned filter and pre-window.
# Mid-band filter extends down to 5 Hz to capture leading-edge T-phase energy.
SEISMIC_CLUSTERS = {
    "low_0":   {"filter_low": 1,  "filter_high": 15, "pre_window_s": 10.0},
    "low_1":   {"filter_low": 1,  "filter_high": 15, "pre_window_s": 8.0},
    "low_2_2": {"filter_low": 1,  "filter_high": 15, "pre_window_s": 5.0},
    "mid_0":   {"filter_low": 5,  "filter_high": 30, "pre_window_s": 8.0},
    "mid_3_1": {"filter_low": 5,  "filter_high": 30, "pre_window_s": 8.0},
    "mid_3_3": {"filter_low": 5,  "filter_high": 30, "pre_window_s": 5.0},
}

POST_WINDOW_S = 2.0
MIN_PRE_WINDOW_S = 1.0
FILTER_PAD_S = 1.0
MIN_QUALITY = 0.15

# === Picker 1: Envelope STA/LTA parameters ===
ENV_SMOOTH_S = 0.05      # 50 ms boxcar for envelope smoothing
ENV_STA_S = 0.5           # short-term average window
ENV_LTA_S = 5.0           # long-term average window
ENV_TRIGGER_RATIO = 2.0   # STA/LTA trigger threshold
ENV_BACKTRACK_RATIO = 1.2  # backtrack to this ratio for true onset

# === Picker 2: Kurtosis parameters ===
KURT_WINDOW_S = 0.25      # sliding kurtosis window
KURT_NOISE_FRAC = 0.25    # fraction of window used for noise stats
KURT_TRIGGER_Z = 3.0      # z-score trigger threshold
KURT_SUSTAIN_S = 0.1      # must exceed threshold for this duration
KURT_BACKTRACK_Z = 1.0    # backtrack to this z-score


def apply_bandpass(data, low_hz, high_hz, fs=SAMPLE_RATE, order=4):
    """Apply bandpass filter."""
    nyq = fs / 2
    wn = [max(low_hz / nyq, 0.001), min(high_hz / nyq, 0.999)]
    sos = butter(order, wn, btype="bandpass", output="sos")
    return sosfilt(sos, data)


def envelope_stalta_pick(signal, fs=SAMPLE_RATE):
    """Envelope STA/LTA picker with backtracking.

    1. Hilbert transform → instantaneous amplitude envelope
    2. Smooth (50ms boxcar) → square (energy)
    3. STA/LTA via vectorized cumsum
    4. Trigger at ratio 2.0, backtrack to ratio 1.2

    Returns
    -------
    pick_idx : int or None
        Sample index of onset, or None if no valid pick.
    quality : float
        Confidence score 0-1.
    """
    n = len(signal)
    smooth_samp = max(1, int(ENV_SMOOTH_S * fs))
    sta_samp = int(ENV_STA_S * fs)
    lta_samp = int(ENV_LTA_S * fs)

    if n < lta_samp + sta_samp + 1:
        return None, 0.0

    # Hilbert envelope → smooth → energy
    analytic = hilbert(signal)
    envelope = np.abs(analytic)

    # Boxcar smoothing
    kernel = np.ones(smooth_samp) / smooth_samp
    envelope = np.convolve(envelope, kernel, mode="same")

    # Square for energy
    energy = envelope ** 2

    # STA/LTA via cumulative sum (vectorized)
    cs = np.cumsum(energy)
    cs = np.insert(cs, 0, 0.0)  # prepend zero for indexing

    # STA[i] = mean(energy[i-sta_samp+1 : i+1])
    # LTA[i] = mean(energy[i-lta_samp+1 : i+1])
    # Valid range: [lta_samp-1, n-1]
    start = lta_samp
    end = n

    idx = np.arange(start, end)
    sta = (cs[idx + 1] - cs[idx + 1 - sta_samp]) / sta_samp
    lta = (cs[idx + 1] - cs[idx + 1 - lta_samp]) / lta_samp

    # Avoid division by zero
    lta_safe = np.where(lta > 0, lta, 1e-30)
    ratio = sta / lta_safe

    # Find first trigger point
    triggered = np.where(ratio >= ENV_TRIGGER_RATIO)[0]
    if len(triggered) == 0:
        return None, 0.0

    trigger_pos = triggered[0]  # position in ratio array
    trigger_idx = start + trigger_pos  # position in signal

    # Backtrack to ratio 1.2 (creep onset)
    pre_trigger = ratio[:trigger_pos + 1]
    below_backtrack = np.where(pre_trigger < ENV_BACKTRACK_RATIO)[0]
    if len(below_backtrack) > 0:
        pick_pos = below_backtrack[-1] + 1  # first sample above backtrack
    else:
        pick_pos = 0  # can't backtrack further
    pick_idx = start + pick_pos

    # Clamp to valid range
    pick_idx = max(0, min(pick_idx, n - 1))

    # Quality from peak STA/LTA and rise steadiness
    peak_ratio = ratio[trigger_pos]
    # Rise steadiness: monotonicity of ratio from pick to trigger
    rise_segment = ratio[max(0, pick_pos):trigger_pos + 1]
    if len(rise_segment) > 2:
        diffs = np.diff(rise_segment)
        steadiness = np.mean(diffs > 0)  # fraction of positive steps
    else:
        steadiness = 0.5

    # Quality combines peak ratio strength and rise steadiness
    ratio_score = min(1.0, (peak_ratio - ENV_TRIGGER_RATIO) / 5.0)
    quality = 0.6 * min(1.0, max(0.0, ratio_score)) + 0.4 * steadiness

    return int(pick_idx), round(quality, 4)


def kurtosis_onset_pick(signal, fs=SAMPLE_RATE):
    """Kurtosis-based onset picker with z-score triggering.

    1. Sliding-window excess kurtosis (cumsum method)
    2. Noise kurtosis from first 25% of window
    3. Z-score relative to noise stats
    4. Trigger at 3σ with 0.1s sustained exceedance
    5. Backtrack to z > 1.0

    Returns
    -------
    pick_idx : int or None
        Sample index of onset, or None if no valid pick.
    quality : float
        Confidence score 0-1.
    """
    n = len(signal)
    w = max(2, int(KURT_WINDOW_S * fs))
    n_windows = n - w + 1

    if n_windows < 10:
        return None, 0.0

    # Sliding kurtosis via cumulative sums
    x = signal.astype(np.float64)
    cs1 = np.concatenate(([0], np.cumsum(x)))
    cs2 = np.concatenate(([0], np.cumsum(x ** 2)))
    cs3 = np.concatenate(([0], np.cumsum(x ** 3)))
    cs4 = np.concatenate(([0], np.cumsum(x ** 4)))

    win_sum = cs1[w:] - cs1[:n_windows]
    win_sum2 = cs2[w:] - cs2[:n_windows]
    win_sum3 = cs3[w:] - cs3[:n_windows]
    win_sum4 = cs4[w:] - cs4[:n_windows]

    m1 = win_sum / w
    m2 = win_sum2 / w
    m3 = win_sum3 / w
    m4_raw = win_sum4 / w

    # Central 4th moment
    mu4 = m4_raw - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
    # Central 2nd moment (variance)
    win_var = m2 - m1**2

    kurt = np.zeros(n_windows)
    valid = win_var > 0
    kurt[valid] = mu4[valid] / (win_var[valid] ** 2) - 3.0

    # Noise statistics from first 25%
    noise_end = max(2, int(KURT_NOISE_FRAC * n_windows))
    noise_kurt = kurt[:noise_end]
    noise_mean = np.median(noise_kurt)  # median is more robust
    noise_std = np.std(noise_kurt)

    if noise_std < 1e-10:
        noise_std = 1.0

    # Z-score
    z = (kurt - noise_mean) / noise_std

    # Find trigger: z > 3σ sustained for 0.1s
    sustain_samp = max(1, int(KURT_SUSTAIN_S * fs))
    above_trigger = z >= KURT_TRIGGER_Z

    # Check for sustained exceedance
    if sustain_samp <= 1:
        triggered = np.where(above_trigger)[0]
    else:
        # Convolve with sustain window to find runs
        kernel = np.ones(sustain_samp)
        conv = np.convolve(above_trigger.astype(float), kernel, mode="valid")
        sustained = np.where(conv >= sustain_samp)[0]
        if len(sustained) == 0:
            return None, 0.0
        # Map back: the sustained run starts at this index
        triggered = sustained

    if len(triggered) == 0:
        return None, 0.0

    trigger_idx = triggered[0]

    # Backtrack to z > 1.0
    pre_trigger = z[:trigger_idx + 1]
    below_backtrack = np.where(pre_trigger < KURT_BACKTRACK_Z)[0]
    if len(below_backtrack) > 0:
        pick_pos = below_backtrack[-1] + 1
    else:
        pick_pos = 0

    # The kurtosis window is centered, so adjust by half-window
    pick_idx = pick_pos + w // 2
    pick_idx = max(0, min(pick_idx, n - 1))

    # Quality from peak z-score and sustainment
    peak_z = np.max(z[trigger_idx:min(trigger_idx + sustain_samp * 5, n_windows)])
    z_score_quality = min(1.0, (peak_z - KURT_TRIGGER_Z) / 15.0)

    # Sustainment: how long does z stay above trigger after first crossing
    post_trigger = z[trigger_idx:]
    sustained_samples = 0
    for val in post_trigger:
        if val >= KURT_TRIGGER_Z:
            sustained_samples += 1
        else:
            break
    sustain_quality = min(1.0, sustained_samples / (0.5 * fs))  # normalize by 0.5s

    quality = 0.5 * max(0.0, z_score_quality) + 0.5 * sustain_quality

    return int(pick_idx), round(quality, 4)


def pick_seismic_event(data, onset_samp, cluster_cfg, file_nsamples,
                       aic_quality, aic_grade,
                       prev_event_end_samp=None):
    """Run dual picker on a single seismic event.

    Strategy: trust the existing AIC pick when it works (grade A/B). Only
    switch to the seismic dual pick when AIC is struggling (grade C or
    quality < 0.4). This way impulsive events keep their good AIC picks
    and emergent events get rescued by the envelope/kurtosis pickers.

    Parameters
    ----------
    data : np.ndarray
        Full file waveform.
    onset_samp : int
        Original onset sample within file.
    cluster_cfg : dict
        Cluster-specific config (filter_low, filter_high, pre_window_s).
    file_nsamples : int
        Total samples in file.
    aic_quality : float
        Quality score from the existing AIC picker (refine_onsets.py).
    aic_grade : str
        Grade from the existing AIC picker ("A", "B", or "C").
    prev_event_end_samp : int or None
        End sample of previous event.

    Returns
    -------
    dict with pick results.
    """
    fs = SAMPLE_RATE
    pre_samp = int(cluster_cfg["pre_window_s"] * fs)
    post_samp = int(POST_WINDOW_S * fs)
    pad_samp = int(FILTER_PAD_S * fs)
    min_pre_samp = int(MIN_PRE_WINDOW_S * fs)
    pre_window_s = cluster_cfg["pre_window_s"]

    # Window boundaries
    win_start = onset_samp - pre_samp
    win_end = onset_samp + post_samp

    win_start = max(0, win_start)
    win_end = min(file_nsamples, win_end)

    if prev_event_end_samp is not None:
        win_start = max(win_start, prev_event_end_samp)

    actual_pre = onset_samp - win_start
    if actual_pre < min_pre_samp:
        return _no_pick_result(pre_window_s, cluster_cfg, aic_quality)

    # Extract with filter padding
    extract_start = max(0, win_start - pad_samp)
    extract_end = min(file_nsamples, win_end + pad_samp)
    segment = data[extract_start:extract_end].astype(np.float64)

    # Apply seismic-tuned bandpass filter
    filtered = apply_bandpass(segment, cluster_cfg["filter_low"],
                              cluster_cfg["filter_high"])

    # Trim padding
    trim_left = win_start - extract_start
    trim_right = len(filtered) - (extract_end - win_end)
    filtered = filtered[trim_left:trim_right]

    if len(filtered) < 100:
        return _no_pick_result(pre_window_s, cluster_cfg, aic_quality)

    # Onset sample within the extracted window
    onset_in_win = onset_samp - win_start

    # --- Picker 1: Envelope STA/LTA ---
    env_idx, env_quality = envelope_stalta_pick(filtered)
    env_shift_samp = None
    if env_idx is not None:
        env_shift_samp = env_idx - onset_in_win  # negative = earlier

    # --- Picker 2: Kurtosis onset ---
    kurt_idx, kurt_quality = kurtosis_onset_pick(filtered)
    kurt_shift_samp = None
    if kurt_idx is not None:
        kurt_shift_samp = kurt_idx - onset_in_win

    # --- Combine picks: AIC-first with seismic rescue ---
    env_shift_s = env_shift_samp / fs if env_shift_samp is not None else None
    kurt_shift_s = kurt_shift_samp / fs if kurt_shift_samp is not None else None

    # Determine if AIC needs rescue
    aic_needs_rescue = (aic_grade == "C") or (aic_quality < 0.4)

    def _valid_pick(shift_s, quality):
        """Check if a seismic pick passes basic sanity checks."""
        if shift_s is None or quality < MIN_QUALITY:
            return False
        if shift_s > 0:  # later than trigger
            return False
        if abs(shift_s) > 0.95 * pre_window_s:  # picking in noise
            return False
        return True

    if not aic_needs_rescue:
        # AIC is good — keep it. Still record the seismic picks for comparison.
        method, shift_s, quality = "aic_kept", 0.0, aic_quality
    else:
        # AIC struggling — find the best seismic rescue pick
        candidates = []
        if _valid_pick(env_shift_s, env_quality):
            candidates.append(("envelope", env_shift_s, env_quality))
        if _valid_pick(kurt_shift_s, kurt_quality):
            candidates.append(("kurtosis", kurt_shift_s, kurt_quality))

        if candidates:
            # Take the earlier valid pick (more negative = earlier onset)
            candidates.sort(key=lambda c: c[1])
            method, shift_s, quality = candidates[0]
        else:
            # Rescue failed — keep AIC pick as-is (no-op)
            method, shift_s, quality = "aic_kept", 0.0, aic_quality

    return {
        "seis_onset_shift_s": round(shift_s, 4),
        "seis_onset_method": method,
        "seis_onset_quality": round(quality, 4),
        "env_pick_shift_s": round(env_shift_s, 4) if env_shift_s is not None else np.nan,
        "env_pick_quality": round(env_quality, 4),
        "kurt_pick_shift_s": round(kurt_shift_s, 4) if kurt_shift_s is not None else np.nan,
        "kurt_pick_quality": round(kurt_quality, 4),
        "pre_window_s": pre_window_s,
        "filter_low_hz": cluster_cfg["filter_low"],
        "filter_high_hz": cluster_cfg["filter_high"],
    }


def _no_pick_result(pre_window_s, cluster_cfg, aic_quality=0.0):
    """Return a result dict when picking is not possible (preserves AIC pick)."""
    return {
        "seis_onset_shift_s": 0.0,
        "seis_onset_method": "aic_kept",
        "seis_onset_quality": aic_quality,
        "env_pick_shift_s": np.nan,
        "env_pick_quality": 0.0,
        "kurt_pick_shift_s": np.nan,
        "kurt_pick_quality": 0.0,
        "pre_window_s": pre_window_s,
        "filter_low_hz": cluster_cfg["filter_low"],
        "filter_high_hz": cluster_cfg["filter_high"],
    }


def compute_onset_grade(quality):
    """Map quality score to letter grade."""
    if quality >= 0.7:
        return "A"
    elif quality >= 0.4:
        return "B"
    else:
        return "C"


def build_seismic_catalogue():
    """Load event catalogue and identify seismic cluster events.

    Merges top-level cluster IDs from UMAP coordinates with sub-cluster IDs
    from subcluster parquets to resolve all 6 target clusters.

    Returns
    -------
    DataFrame with seismic events and their cluster_id column.
    """
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])

    # Get cluster IDs: top-level from UMAP, sub-clusters from parquets
    umap = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")[
        ["event_id", "cluster_id"]
    ]

    # Sub-cluster files override top-level for mega-clusters
    sub_low = pd.read_parquet(DATA_DIR / "subclusters_low_2.parquet")[
        ["event_id", "subcluster_id"]
    ].rename(columns={"subcluster_id": "cluster_id_sub"})
    sub_mid = pd.read_parquet(DATA_DIR / "subclusters_mid_3.parquet")[
        ["event_id", "subcluster_id"]
    ].rename(columns={"subcluster_id": "cluster_id_sub"})

    subs = pd.concat([sub_low, sub_mid], ignore_index=True)

    # Merge: start with top-level, override with sub-cluster where available
    merged = umap.merge(subs, on="event_id", how="left")
    merged["cluster_id_final"] = merged["cluster_id_sub"].fillna(
        merged["cluster_id"]
    )

    # Join to catalogue
    cat = cat.merge(
        merged[["event_id", "cluster_id_final"]].rename(
            columns={"cluster_id_final": "cluster_id"}
        ),
        on="event_id",
        how="left",
    )

    # Filter to seismic clusters
    seis = cat[cat["cluster_id"].isin(SEISMIC_CLUSTERS.keys())].copy()
    print(f"Seismic events by cluster:")
    for cid, count in seis["cluster_id"].value_counts().sort_index().items():
        print(f"  {cid:10s}: {count:,}")
    print(f"  Total:      {len(seis):,}")

    return seis


def process_seismic_events(seis, mooring_filter=None, file_filter=None):
    """Process all seismic events with the dual picker.

    Groups by (mooring, file_number), reads each DAT file once.
    """
    if mooring_filter:
        seis = seis[seis["mooring"] == mooring_filter].copy()
    if file_filter:
        seis = seis[seis["file_number"] == int(file_filter)].copy()

    if len(seis) == 0:
        print("No events to process after filtering.")
        return seis

    # Prepare result columns
    result_cols = [
        "seis_onset_shift_s", "seis_onset_method", "seis_onset_quality",
        "env_pick_shift_s", "env_pick_quality",
        "kurt_pick_shift_s", "kurt_pick_quality",
        "pre_window_s", "filter_low_hz", "filter_high_hz",
    ]
    for col in result_cols:
        if col in ("seis_onset_method",):
            seis[col] = "original"
        elif col in ("env_pick_shift_s", "kurt_pick_shift_s"):
            seis[col] = np.nan
        else:
            seis[col] = 0.0

    # Group by (mooring, file_number)
    groups = seis.groupby(["mooring", "file_number"])
    n_groups = len(groups)
    n_processed = 0
    n_events_done = 0

    for (mooring, file_num), group_idx in groups.groups.items():
        group = seis.loc[group_idx].sort_values("onset_utc")

        info = MOORINGS[mooring]
        dat_path = DATA_ROOT / info["data_dir"] / f"{file_num:08d}.DAT"

        if not dat_path.exists():
            n_processed += 1
            continue

        file_ts, data, _ = read_dat(dat_path)
        file_nsamples = len(data)

        prev_end_samp = None
        for idx, ev in group.iterrows():
            onset_offset_s = (ev["onset_utc"] - file_ts).total_seconds()
            onset_samp = int(onset_offset_s * SAMPLE_RATE)

            if onset_samp < 0 or onset_samp >= file_nsamples:
                continue

            end_offset_s = (ev["end_utc"] - file_ts).total_seconds()
            end_samp = int(end_offset_s * SAMPLE_RATE)

            cluster_cfg = SEISMIC_CLUSTERS[ev["cluster_id"]]

            result = pick_seismic_event(
                data, onset_samp, cluster_cfg, file_nsamples,
                aic_quality=ev["onset_quality"],
                aic_grade=ev["onset_grade"],
                prev_event_end_samp=prev_end_samp,
            )

            for col, val in result.items():
                seis.at[idx, col] = val

            prev_end_samp = end_samp

        n_processed += 1
        n_events_done += len(group)
        if n_processed % 50 == 0 or n_processed == n_groups:
            print(f"  {n_processed}/{n_groups} file groups, "
                  f"{n_events_done}/{len(seis)} events")

    # Compute derived columns
    seis["seis_onset_utc"] = seis.apply(
        lambda r: r["onset_utc"] + timedelta(seconds=r["seis_onset_shift_s"]),
        axis=1,
    )
    seis["seis_onset_grade"] = seis["seis_onset_quality"].apply(
        compute_onset_grade
    )

    return seis


def save_output(seis):
    """Save seismic onset results to parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    out_cols = [
        "event_id", "cluster_id",
        "seis_onset_utc", "seis_onset_shift_s", "seis_onset_method",
        "seis_onset_quality", "seis_onset_grade",
        "env_pick_shift_s", "env_pick_quality",
        "kurt_pick_shift_s", "kurt_pick_quality",
        "pre_window_s", "filter_low_hz", "filter_high_hz",
    ]
    out = seis[out_cols].copy()
    outpath = DATA_DIR / "seismic_onsets.parquet"
    out.to_parquet(outpath, index=False)
    print(f"\nSaved: {outpath} ({len(out):,} events)")
    return outpath


def plot_shift_histograms(seis):
    """Plot shift histograms: old (refine_onsets) vs new (seismic picker)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    clusters = sorted(seis["cluster_id"].unique())
    n_clusters = len(clusters)
    ncols = 3
    nrows = (n_clusters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows),
                             squeeze=False)

    for i, cluster_id in enumerate(clusters):
        ax = axes[i // ncols][i % ncols]
        subset = seis[seis["cluster_id"] == cluster_id]

        # Old shifts (from refine_onsets)
        old_shifts = subset["onset_shift_s"]
        # New shifts (seismic picker)
        new_shifts = subset["seis_onset_shift_s"]

        bins = np.linspace(
            min(old_shifts.min(), new_shifts.min(), -10),
            max(old_shifts.max(), new_shifts.max(), 1),
            50,
        )

        ax.hist(old_shifts, bins=bins, alpha=0.5, color="#999999",
                edgecolor="white", linewidth=0.3, label="AIC (old)")
        refined = subset[subset["seis_onset_method"] != "original"]
        if len(refined) > 0:
            ax.hist(refined["seis_onset_shift_s"], bins=bins, alpha=0.6,
                    color="#E69F00", edgecolor="white", linewidth=0.3,
                    label="seismic (new)")

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        if len(refined) > 0:
            ax.axvline(refined["seis_onset_shift_s"].median(),
                       color="red", linewidth=1.2, label="new median")
        ax.set_xlabel("Onset shift (s)")
        ax.set_title(f"{cluster_id} (n={len(subset):,})", fontsize=10)
        ax.legend(fontsize=7)

    # Hide unused axes
    for i in range(n_clusters, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    axes[0][0].set_ylabel("Count")
    fig.suptitle("Seismic Onset Shifts: Old AIC vs New Dual Picker",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    outpath = FIG_DIR / "seismic_onset_shifts.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_picker_comparison(seis):
    """Scatter plot of envelope vs kurtosis shift, colored by cluster."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Only events where both pickers produced a result
    both = seis.dropna(subset=["env_pick_shift_s", "kurt_pick_shift_s"])
    if len(both) == 0:
        print("No events with both picker results — skipping comparison plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    clusters = sorted(both["cluster_id"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(clusters)))

    for cluster_id, color in zip(clusters, colors):
        sub = both[both["cluster_id"] == cluster_id]
        ax.scatter(sub["env_pick_shift_s"], sub["kurt_pick_shift_s"],
                   s=8, alpha=0.4, color=color, label=f"{cluster_id} ({len(sub)})")

    # Diagonal
    lim = min(ax.get_xlim()[0], ax.get_ylim()[0])
    ax.plot([lim, 0], [lim, 0], "k--", linewidth=0.8, alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    ax.set_xlabel("Envelope STA/LTA shift (s)")
    ax.set_ylabel("Kurtosis shift (s)")
    ax.set_title("Picker Comparison: Envelope vs Kurtosis Shift",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()

    outpath = FIG_DIR / "seismic_picker_comparison.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_qc_montage(seis, n_events=60):
    """QC montage: waveform + spectrogram with old/new onset lines."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Stratified sample: oversample events where picker changed the onset
    refined = seis[seis["seis_onset_method"] != "original"]
    original = seis[seis["seis_onset_method"] == "original"]

    n_refined = min(len(refined), int(0.8 * n_events))
    n_original = min(len(original), n_events - n_refined)

    sample = pd.concat([
        refined.sample(n=n_refined, random_state=42) if n_refined > 0 else refined.head(0),
        original.sample(n=n_original, random_state=42) if n_original > 0 else original.head(0),
    ]).sort_values("onset_utc")

    if len(sample) == 0:
        print("No events to plot in QC montage.")
        return

    ncols = 6
    nrows = (len(sample) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 3 * nrows),
                             squeeze=False)

    for i, (idx, ev) in enumerate(sample.iterrows()):
        ax = axes[i // ncols][i % ncols]

        info = MOORINGS[ev["mooring"]]
        dat_path = DATA_ROOT / info["data_dir"] / f"{ev['file_number']:08d}.DAT"

        if not dat_path.exists():
            ax.text(0.5, 0.5, "File not found", transform=ax.transAxes,
                    ha="center")
            continue

        file_ts, data, _ = read_dat(dat_path)
        onset_offset_s = (ev["onset_utc"] - file_ts).total_seconds()
        onset_samp = int(onset_offset_s * SAMPLE_RATE)
        cluster_cfg = SEISMIC_CLUSTERS[ev["cluster_id"]]
        pre_samp = int(cluster_cfg["pre_window_s"] * SAMPLE_RATE)
        post_samp = int(POST_WINDOW_S * SAMPLE_RATE)

        win_start = max(0, onset_samp - pre_samp)
        win_end = min(len(data), onset_samp + post_samp)
        segment = data[win_start:win_end].astype(np.float64)

        # Filter
        filtered = apply_bandpass(segment, cluster_cfg["filter_low"],
                                  cluster_cfg["filter_high"])

        # Time axis relative to original onset
        t = np.arange(len(filtered)) / SAMPLE_RATE - (onset_samp - win_start) / SAMPLE_RATE

        ax.plot(t, filtered, "k-", linewidth=0.3, alpha=0.7)

        # Old onset (refine_onsets)
        old_shift = ev["onset_shift_s"]
        ax.axvline(old_shift, color="blue", linewidth=1.0, linestyle="--",
                   alpha=0.7, label="AIC")

        # New onset (seismic picker)
        new_shift = ev["seis_onset_shift_s"]
        ax.axvline(new_shift, color="red", linewidth=1.2,
                   alpha=0.9, label="seis")

        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

        ax.set_title(f"{ev['cluster_id']} {ev['seis_onset_method']}\n"
                     f"q={ev['seis_onset_quality']:.2f}",
                     fontsize=7)
        ax.tick_params(labelsize=6)

        if i == 0:
            ax.legend(fontsize=6)

    # Hide unused
    for i in range(len(sample), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("Seismic Onset QC Montage: Old (blue dashed) vs New (red solid)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    outpath = FIG_DIR / "seismic_onset_qc_montage.png"
    fig.savefig(outpath, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


def print_summary(seis):
    """Print summary statistics."""
    n = len(seis)
    refined = seis[seis["seis_onset_method"] != "original"]

    print(f"\n{'=' * 60}")
    print(f"Seismic Dual Onset Picker Summary")
    print(f"{'=' * 60}")
    print(f"Total seismic events: {n:,}")
    print(f"Refined:              {len(refined):,} ({100*len(refined)/n:.1f}%)")

    print(f"\nMethod distribution:")
    for method, count in seis["seis_onset_method"].value_counts().items():
        print(f"  {method:12s}: {count:6,} ({100*count/n:.1f}%)")

    print(f"\nGrade distribution (seismic picker):")
    for grade in ["A", "B", "C"]:
        count = (seis["seis_onset_grade"] == grade).sum()
        print(f"  {grade}: {count:6,} ({100*count/n:.1f}%)")

    # Compare with old grades
    print(f"\nOld grade distribution (AIC picker):")
    for grade in ["A", "B", "C"]:
        count = (seis["onset_grade"] == grade).sum()
        print(f"  {grade}: {count:6,} ({100*count/n:.1f}%)")

    if len(refined) > 0:
        shifts = refined["seis_onset_shift_s"]
        print(f"\nOnset shifts (refined events):")
        print(f"  Median: {shifts.median():+.3f} s")
        print(f"  IQR:    [{shifts.quantile(0.25):+.3f}, "
              f"{shifts.quantile(0.75):+.3f}] s")

    print(f"\nPer cluster:")
    for cid in sorted(SEISMIC_CLUSTERS.keys()):
        sub = seis[seis["cluster_id"] == cid]
        if len(sub) == 0:
            continue
        n_ref = (sub["seis_onset_method"] != "original").sum()
        n_env = (sub["seis_onset_method"] == "envelope").sum()
        n_kurt = (sub["seis_onset_method"] == "kurtosis").sum()
        print(f"  {cid:10s}: {len(sub):5,} events, {n_ref:5,} refined "
              f"(env={n_env}, kurt={n_kurt})")


def main():
    parser = argparse.ArgumentParser(
        description="Seismic-tuned dual onset picker (envelope STA/LTA + kurtosis)")
    parser.add_argument("--mooring", type=str, default=None,
                        help="Process single mooring (e.g., m1)")
    parser.add_argument("--file", type=str, default=None,
                        help="Process single file number (e.g., 00001282)")
    parser.add_argument("--qc-montage", action="store_true",
                        help="Generate QC montage figure")
    args = parser.parse_args()

    print("=" * 60)
    print("BRAVOSEIS Seismic Dual Onset Picker")
    print(f"  Envelope STA/LTA: {ENV_STA_S}s/{ENV_LTA_S}s, "
          f"trigger={ENV_TRIGGER_RATIO}, backtrack={ENV_BACKTRACK_RATIO}")
    print(f"  Kurtosis: {KURT_WINDOW_S}s window, "
          f"trigger={KURT_TRIGGER_Z}σ, backtrack={KURT_BACKTRACK_Z}σ")
    print(f"  Min quality: {MIN_QUALITY}")
    print("=" * 60)

    # Build seismic catalogue with cluster IDs
    seis = build_seismic_catalogue()

    # Process
    seis = process_seismic_events(seis, mooring_filter=args.mooring,
                                  file_filter=args.file)

    if len(seis) == 0:
        print("No events processed!")
        return

    print_summary(seis)

    # Save
    save_output(seis)

    # Figures
    plot_shift_histograms(seis)
    plot_picker_comparison(seis)

    if args.qc_montage:
        plot_qc_montage(seis)


if __name__ == "__main__":
    main()
