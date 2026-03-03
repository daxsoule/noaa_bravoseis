#!/usr/bin/env python3
"""
refine_onsets.py — Refine STA/LTA onset picks using AIC and kurtosis pickers.

The STA/LTA detector finds events well but picks onsets poorly (68% in coda,
only 11% at true first arrival). This script applies a second-pass onset
refinement to find the true first-arrival time for each event.

Algorithm per event:
  1. Extract 7s window: 5s before STA/LTA trigger + 2s after
  2. Apply band-specific filter (same as detection)
  3. Run AIC picker on squared envelope
  4. If AIC trough is ambiguous, fall back to kurtosis picker (0.5s window)
  5. If both fail, keep original onset with low confidence

Usage:
    uv run python refine_onsets.py                  # full catalogue
    uv run python refine_onsets.py --mooring m1     # single mooring
    uv run python refine_onsets.py --file 00001282  # single file (testing)
    uv run python refine_onsets.py --dry-run        # write to separate file

Spec: specs/001-event-detection/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
from scipy.signal import butter, sosfilt

from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory"

# === Detection filter config (mirrors detect_events.py PASSES) ===
PASSES = {
    1: {"label": "low",  "filter": "lowpass",  "cutoff": 15},
    2: {"label": "mid",  "filter": "bandpass", "cutoff": (15, 30)},
    3: {"label": "high", "filter": "highpass", "cutoff": 30},
}
BAND_TO_PASS = {v["label"]: k for k, v in PASSES.items()}

# === Picker parameters ===
PRE_WINDOW_S = 5.0    # seconds before STA/LTA trigger
POST_WINDOW_S = 2.0   # seconds after STA/LTA trigger
MIN_PRE_WINDOW_S = 1.0  # minimum pre-window required
FILTER_PAD_S = 1.0    # padding for filter edge effects
KURTOSIS_WINDOW_S = 0.5  # kurtosis sliding window


def apply_pass_filter(data, pass_cfg, fs=SAMPLE_RATE, order=4):
    """Apply the pre-filter for a detection pass."""
    nyq = fs / 2
    ftype = pass_cfg["filter"]
    cutoff = pass_cfg["cutoff"]
    if ftype == "lowpass":
        wn = min(cutoff / nyq, 0.999)
        sos = butter(order, wn, btype="lowpass", output="sos")
    elif ftype == "bandpass":
        lo, hi = cutoff
        wn = [max(lo / nyq, 0.001), min(hi / nyq, 0.999)]
        sos = butter(order, wn, btype="bandpass", output="sos")
    elif ftype == "highpass":
        wn = max(cutoff / nyq, 0.001)
        sos = butter(order, wn, btype="highpass", output="sos")
    else:
        raise ValueError(f"Unknown filter type: {ftype}")
    return sosfilt(sos, data)


def aic_pick(signal):
    """AIC picker on a 1-D signal (typically squared envelope).

    Fits a two-segment noise/signal model and finds the minimum AIC value
    as the onset sample. Searches in [10, N-10] to avoid edge artifacts.

    Returns
    -------
    pick_idx : int
        Sample index of the AIC minimum (onset estimate).
    quality : float
        Confidence score 0–1 based on AIC trough sharpness.
    """
    n = len(signal)
    margin = 10
    if n < 2 * margin + 1:
        return n // 2, 0.0

    # Variance-based AIC (Maeda 1985), vectorized using cumulative sums
    # AIC(k) = k * log(var(x[0:k])) + (n-k-1) * log(var(x[k:n]))
    x = signal.astype(np.float64)
    cs = np.cumsum(x)       # cumulative sum
    cs2 = np.cumsum(x ** 2) # cumulative sum of squares

    k = np.arange(margin, n - margin, dtype=np.float64)

    # Left segment [0:k]: var = E[x^2] - E[x]^2
    left_mean = cs[k.astype(int) - 1] / k
    left_mean2 = cs2[k.astype(int) - 1] / k
    var_left = left_mean2 - left_mean ** 2

    # Right segment [k:n]: var from total - left cumsum
    right_n = n - k
    right_sum = cs[-1] - cs[k.astype(int) - 1]
    right_sum2 = cs2[-1] - cs2[k.astype(int) - 1]
    right_mean = right_sum / right_n
    right_mean2 = right_sum2 / right_n
    var_right = right_mean2 - right_mean ** 2

    # Compute AIC where both variances are positive
    aic = np.full(n, np.inf)
    valid_mask = (var_left > 0) & (var_right > 0)
    ki = k[valid_mask].astype(int)
    aic[ki] = (ki * np.log(var_left[valid_mask])
               + (n - ki - 1) * np.log(var_right[valid_mask]))

    valid = aic[margin:n - margin]
    if len(valid) == 0 or np.all(np.isinf(valid)):
        return n // 2, 0.0

    pick_idx = margin + np.argmin(valid)

    # Quality: sharpness of AIC trough (normalized drop from median)
    finite_aic = aic[np.isfinite(aic)]
    if len(finite_aic) > 0:
        aic_range = np.median(finite_aic) - aic[pick_idx]
        aic_spread = np.std(finite_aic)
        if aic_spread > 0:
            quality = min(1.0, max(0.0, aic_range / (3 * aic_spread)))
        else:
            quality = 0.0
    else:
        quality = 0.0

    return int(pick_idx), quality


def kurtosis_pick(signal, window_samples):
    """Kurtosis-based onset picker using a sliding window.

    Detects the point where signal statistics depart from Gaussian noise.

    Returns
    -------
    pick_idx : int
        Sample index of kurtosis onset.
    quality : float
        Confidence score 0–1.
    """
    n = len(signal)
    if n < window_samples + 1:
        return n // 2, 0.0

    # Sliding kurtosis (excess kurtosis: Gaussian = 0), vectorized
    x = signal.astype(np.float64)
    w = window_samples
    n_windows = n - w + 1

    # Sliding statistics via cumulative sums
    cs1 = np.concatenate(([0], np.cumsum(x)))
    cs2 = np.concatenate(([0], np.cumsum(x ** 2)))
    cs4 = np.concatenate(([0], np.cumsum(x ** 4)))

    win_sum = cs1[w:] - cs1[:n_windows]
    win_sum2 = cs2[w:] - cs2[:n_windows]

    win_mean = win_sum / w
    win_var = win_sum2 / w - win_mean ** 2

    # For kurtosis we need E[(x-mu)^4] = E[x^4] - 4*mu*E[x^3] + 6*mu^2*E[x^2] - 3*mu^4
    # Simpler: use (x-mu)^4 expansion, but cumsum of x^3 is needed too
    # Instead, use the bias-corrected approach with centered moments
    # For speed, approximate: kurt = E[x^4]/var^2 - 3 (only valid when mean ~ 0)
    # Better: compute exactly via the shifted formula
    cs3 = np.concatenate(([0], np.cumsum(x ** 3)))
    win_sum3 = cs3[w:] - cs3[:n_windows]
    win_sum4 = cs4[w:] - cs4[:n_windows]

    # Central moments: mu4 = E[(x-m)^4] = E[x^4] - 4m*E[x^3] + 6m^2*E[x^2] - 3m^4
    m1 = win_mean
    m2 = win_sum2 / w
    m3 = win_sum3 / w
    m4_raw = win_sum4 / w
    mu4 = m4_raw - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4

    kurt = np.zeros(n_windows)
    valid = win_var > 0
    kurt[valid] = mu4[valid] / (win_var[valid] ** 2) - 3.0

    # Onset = first sustained departure from noise kurtosis
    # Use the max gradient as the onset indicator
    if len(kurt) < 3:
        return n // 2, 0.0

    gradient = np.gradient(kurt)
    # Smooth gradient slightly
    kernel = np.ones(max(1, window_samples // 4)) / max(1, window_samples // 4)
    if len(gradient) > len(kernel):
        smooth_grad = np.convolve(gradient, kernel, mode="same")
    else:
        smooth_grad = gradient

    pick_idx = int(np.argmax(smooth_grad))

    # Quality based on peak gradient relative to noise
    noise_std = np.std(smooth_grad[:len(smooth_grad) // 4]) if len(smooth_grad) > 4 else 1.0
    if noise_std > 0:
        snr = smooth_grad[pick_idx] / noise_std
        quality = min(1.0, max(0.0, (snr - 1) / 10))
    else:
        quality = 0.0

    return pick_idx, quality


def refine_single_event(data, onset_samp, pass_num, file_nsamples,
                        prev_event_end_samp=None):
    """Refine onset for a single event.

    Parameters
    ----------
    data : np.ndarray
        Full file waveform (raw).
    onset_samp : int
        Original STA/LTA onset sample within the file.
    pass_num : int
        Detection pass number (1, 2, or 3).
    file_nsamples : int
        Total samples in the file.
    prev_event_end_samp : int or None
        End sample of previous event (to avoid overlap).

    Returns
    -------
    dict with onset_shift_samp, onset_method, onset_quality.
    """
    fs = SAMPLE_RATE
    pass_cfg = PASSES[pass_num]
    pre_samp = int(PRE_WINDOW_S * fs)
    post_samp = int(POST_WINDOW_S * fs)
    pad_samp = int(FILTER_PAD_S * fs)
    min_pre_samp = int(MIN_PRE_WINDOW_S * fs)
    kurt_win = int(KURTOSIS_WINDOW_S * fs)

    # Determine search window boundaries
    win_start = onset_samp - pre_samp
    win_end = onset_samp + post_samp

    # Clip to file boundaries
    win_start = max(0, win_start)
    win_end = min(file_nsamples, win_end)

    # Clip to not overlap with previous event
    if prev_event_end_samp is not None:
        win_start = max(win_start, prev_event_end_samp)

    # Check minimum pre-window
    actual_pre = onset_samp - win_start
    if actual_pre < min_pre_samp:
        return {
            "onset_shift_samp": 0,
            "onset_method": "original",
            "onset_quality": 0.0,
        }

    # Extract with filter padding
    extract_start = max(0, win_start - pad_samp)
    extract_end = min(file_nsamples, win_end + pad_samp)
    segment = data[extract_start:extract_end].astype(np.float64)

    # Apply band filter
    filtered = apply_pass_filter(segment, pass_cfg)

    # Trim padding
    trim_left = win_start - extract_start
    trim_right = len(filtered) - (extract_end - win_end)
    filtered = filtered[trim_left:trim_right]

    if len(filtered) < 2 * 10 + 1:
        return {
            "onset_shift_samp": 0,
            "onset_method": "original",
            "onset_quality": 0.0,
        }

    # Squared envelope
    envelope = filtered ** 2

    # AIC picker
    aic_idx, aic_quality = aic_pick(envelope)

    if aic_quality >= 0.4:
        refined_samp = win_start + aic_idx
        shift = refined_samp - onset_samp
        # Reject positive shifts — STA/LTA already triggers late, not early
        if shift > 0:
            return {
                "onset_shift_samp": 0,
                "onset_method": "original",
                "onset_quality": round(float(aic_quality) * 0.3, 3),
            }
        return {
            "onset_shift_samp": int(shift),
            "onset_method": "aic",
            "onset_quality": round(float(aic_quality), 3),
        }

    # Kurtosis fallback
    kurt_idx, kurt_quality = kurtosis_pick(np.abs(filtered), kurt_win)

    if kurt_quality >= 0.2:
        refined_samp = win_start + kurt_idx
        shift = refined_samp - onset_samp
        # Reject positive shifts
        if shift > 0:
            return {
                "onset_shift_samp": 0,
                "onset_method": "original",
                "onset_quality": round(float(kurt_quality) * 0.3, 3),
            }
        return {
            "onset_shift_samp": int(shift),
            "onset_method": "kurtosis",
            "onset_quality": round(float(kurt_quality), 3),
        }

    # Both pickers failed — keep original
    return {
        "onset_shift_samp": 0,
        "onset_method": "original",
        "onset_quality": round(float(max(aic_quality, kurt_quality)), 3),
    }


def compute_onset_grade(quality):
    """Map quality score to letter grade."""
    if quality >= 0.7:
        return "A"
    elif quality >= 0.4:
        return "B"
    else:
        return "C"


def load_catalogue():
    """Load event catalogue."""
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])
    return cat


def process_catalogue(cat, mooring_filter=None, file_filter=None):
    """Refine onsets for all events, grouped by (mooring, file_number).

    Reads each DAT file once, filters per band, then processes all events.
    """
    # Filter catalogue if requested
    if mooring_filter:
        cat = cat[cat["mooring"] == mooring_filter].copy()
    if file_filter:
        cat = cat[cat["file_number"] == int(file_filter)].copy()

    if len(cat) == 0:
        print("No events to process after filtering.")
        return cat

    # New columns
    cat["onset_shift_samp"] = 0
    cat["onset_method"] = "original"
    cat["onset_quality"] = 0.0

    # Group by (mooring, file_number) for I/O efficiency
    groups = cat.groupby(["mooring", "file_number"])
    n_groups = len(groups)
    n_processed = 0
    n_events_done = 0

    for (mooring, file_num), group_idx in groups.groups.items():
        group = cat.loc[group_idx].sort_values("onset_utc")

        # Resolve file path
        info = MOORINGS[mooring]
        mooring_dir = DATA_ROOT / info["data_dir"]
        dat_path = mooring_dir / f"{file_num:08d}.DAT"

        if not dat_path.exists():
            n_processed += 1
            continue

        # Read file once
        file_ts, data, _ = read_dat(dat_path)
        file_nsamples = len(data)

        # Group events by detection pass to avoid re-filtering
        pass_groups = group.groupby("detection_pass")

        for pass_num, pass_group in pass_groups:
            events = pass_group.sort_values("onset_utc")

            prev_end_samp = None
            for i, (idx, ev) in enumerate(events.iterrows()):
                # Compute onset sample within file
                onset_offset_s = (ev["onset_utc"] - file_ts).total_seconds()
                onset_samp = int(onset_offset_s * SAMPLE_RATE)

                if onset_samp < 0 or onset_samp >= file_nsamples:
                    continue

                # Previous event end (to clip search window)
                end_offset_s = (ev["end_utc"] - file_ts).total_seconds()
                end_samp = int(end_offset_s * SAMPLE_RATE)

                result = refine_single_event(
                    data, onset_samp, pass_num, file_nsamples,
                    prev_event_end_samp=prev_end_samp,
                )

                cat.at[idx, "onset_shift_samp"] = result["onset_shift_samp"]
                cat.at[idx, "onset_method"] = result["onset_method"]
                cat.at[idx, "onset_quality"] = result["onset_quality"]

                prev_end_samp = end_samp

        n_processed += 1
        n_events_done += len(group)
        if n_processed % 50 == 0 or n_processed == n_groups:
            print(f"  {n_processed}/{n_groups} file groups, "
                  f"{n_events_done}/{len(cat)} events")

    # Compute derived columns
    cat["onset_shift_s"] = cat["onset_shift_samp"] / SAMPLE_RATE
    cat["onset_utc_refined"] = cat.apply(
        lambda r: r["onset_utc"] + timedelta(seconds=r["onset_shift_s"]),
        axis=1,
    )
    cat["onset_grade"] = cat["onset_quality"].apply(compute_onset_grade)

    # Drop internal column
    cat = cat.drop(columns=["onset_shift_samp"])

    return cat


def plot_shift_histogram(cat):
    """Plot onset shift histogram by detection band."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    band_colors = {"low": "#E69F00", "mid": "#56B4E9", "high": "#009E73"}

    for ax, band in zip(axes, ["low", "mid", "high"]):
        subset = cat[cat["detection_band"] == band]
        if len(subset) == 0:
            ax.set_title(f"{band} (n=0)")
            continue

        shifts = subset["onset_shift_s"]
        refined = subset[subset["onset_method"] != "original"]

        ax.hist(shifts, bins=50, color=band_colors[band], alpha=0.7,
                edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        if len(refined) > 0:
            ax.axvline(refined["onset_shift_s"].median(),
                       color="red", linewidth=1.5, label="median (refined)")
        ax.set_xlabel("Onset shift (s)")
        ax.set_title(f"{band} band (n={len(subset):,})")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle("Onset Refinement Shifts by Detection Band", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()

    outpath = FIG_DIR / "onset_shift_histogram.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


def print_summary(cat):
    """Print summary statistics to stdout."""
    n = len(cat)
    refined = cat[cat["onset_method"] != "original"]

    print(f"\n{'=' * 60}")
    print(f"Onset Refinement Summary")
    print(f"{'=' * 60}")
    print(f"Total events:    {n:,}")
    print(f"Refined:         {len(refined):,} ({100*len(refined)/n:.1f}%)")

    print(f"\nMethod distribution:")
    for method, count in cat["onset_method"].value_counts().items():
        print(f"  {method:12s}: {count:6,} ({100*count/n:.1f}%)")

    print(f"\nGrade distribution:")
    for grade in ["A", "B", "C"]:
        count = (cat["onset_grade"] == grade).sum()
        print(f"  {grade}: {count:6,} ({100*count/n:.1f}%)")

    if len(refined) > 0:
        shifts = refined["onset_shift_s"]
        print(f"\nOnset shifts (refined events only):")
        print(f"  Median: {shifts.median():+.3f} s")
        print(f"  IQR:    [{shifts.quantile(0.25):+.3f}, "
              f"{shifts.quantile(0.75):+.3f}] s")
        print(f"  Range:  [{shifts.min():+.3f}, {shifts.max():+.3f}] s")

    print(f"\nPer band:")
    for band in ["low", "mid", "high"]:
        subset = cat[cat["detection_band"] == band]
        if len(subset) == 0:
            continue
        n_ref = (subset["onset_method"] != "original").sum()
        print(f"  {band:5s}: {len(subset):,} events, "
              f"{n_ref:,} refined ({100*n_ref/len(subset):.0f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Refine STA/LTA onset picks using AIC/kurtosis pickers")
    parser.add_argument("--mooring", type=str, default=None,
                        help="Process single mooring (e.g., m1)")
    parser.add_argument("--file", type=str, default=None,
                        help="Process single file number (e.g., 00001282)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write to separate file instead of overwriting")
    args = parser.parse_args()

    print("=" * 60)
    print("BRAVOSEIS Onset Refinement — AIC + Kurtosis Pickers")
    print(f"  Pre-window: {PRE_WINDOW_S}s, Post-window: {POST_WINDOW_S}s")
    print(f"  Kurtosis window: {KURTOSIS_WINDOW_S}s")
    print("=" * 60)

    cat = load_catalogue()
    print(f"Loaded {len(cat):,} events")

    if args.mooring:
        print(f"Filtering to mooring: {args.mooring}")
    if args.file:
        print(f"Filtering to file: {args.file}")

    cat = process_catalogue(cat, mooring_filter=args.mooring,
                            file_filter=args.file)

    if len(cat) == 0:
        print("No events processed!")
        return

    print_summary(cat)

    # Save
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    plot_shift_histogram(cat)

    if args.dry_run:
        suffix = "_refined_dryrun"
        if args.mooring:
            suffix += f"_{args.mooring}"
        if args.file:
            suffix += f"_{args.file}"
        outpath = DATA_DIR / f"event_catalogue{suffix}.parquet"
    else:
        outpath = DATA_DIR / "event_catalogue.parquet"

    cat.to_parquet(outpath, index=False)
    print(f"\nSaved: {outpath} ({len(cat):,} events)")


if __name__ == "__main__":
    main()
