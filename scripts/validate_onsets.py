#!/usr/bin/env python3
"""
validate_onsets.py — QC montage for refined onset picks.

Plots a montage of events showing both the original STA/LTA onset (dashed)
and the refined AIC/kurtosis onset (solid) for visual validation. Sampling
is stratified by detection band and overweights grade C events.

Usage:
    uv run python validate_onsets.py
    uv run python validate_onsets.py --n 30

Spec: specs/001-event-detection/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from datetime import timedelta
from scipy.signal import spectrogram, butter, sosfilt

from read_dat import read_dat, list_mooring_files, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory"

# === Parameters ===
N_SAMPLES = 50
WINDOW_SEC = 10
SEED = 123
NPERSEG = 256
NOVERLAP = 192
FREQ_MAX = 250

NCOLS = 5
NROWS = 10

BAND_COLORS = {
    "low": "#E69F00",
    "mid": "#56B4E9",
    "high": "#009E73",
}

BAND_FILTERS = {
    "low":  {"btype": "lowpass",  "cutoff": 15},
    "mid":  {"btype": "bandpass", "cutoff": (15, 30)},
    "high": {"btype": "highpass", "cutoff": 30},
}


def band_filter(data, band, fs=SAMPLE_RATE, order=4):
    """Apply the detection-pass filter for a given band."""
    cfg = BAND_FILTERS[band]
    nyq = fs / 2
    if cfg["btype"] == "lowpass":
        wn = min(cfg["cutoff"] / nyq, 0.999)
    elif cfg["btype"] == "bandpass":
        lo, hi = cfg["cutoff"]
        wn = [max(lo / nyq, 0.001), min(hi / nyq, 0.999)]
    elif cfg["btype"] == "highpass":
        wn = max(cfg["cutoff"] / nyq, 0.001)
    sos = butter(order, wn, btype=cfg["btype"], output="sos")
    return sosfilt(sos, data)


def load_catalogue():
    """Load refined event catalogue."""
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])
    if "onset_utc_refined" in cat.columns:
        cat["onset_utc_refined"] = pd.to_datetime(cat["onset_utc_refined"])
    return cat


def sample_events(cat, n=N_SAMPLES, seed=SEED):
    """Stratified sample overweighting grade C events.

    Allocation: 50% grade C, 25% grade B, 25% grade A (within each band).
    """
    rng = np.random.default_rng(seed)

    if "onset_grade" not in cat.columns:
        # No refinement columns — fall back to uniform sampling
        idx = rng.choice(len(cat), size=min(n, len(cat)), replace=False)
        return cat.iloc[idx].sort_values("onset_utc").reset_index(drop=True)

    bands = sorted(cat["detection_band"].unique())
    per_band = max(1, n // len(bands))

    samples = []
    for band in bands:
        band_cat = cat[cat["detection_band"] == band]
        if len(band_cat) == 0:
            continue

        # Grade allocation targets
        targets = {"C": int(per_band * 0.5), "B": int(per_band * 0.25)}
        targets["A"] = per_band - targets["C"] - targets["B"]

        band_samples = []
        for grade, target in targets.items():
            subset = band_cat[band_cat["onset_grade"] == grade]
            k = min(target, len(subset))
            if k > 0:
                idx = rng.choice(len(subset), size=k, replace=False)
                band_samples.append(subset.iloc[idx])

        if band_samples:
            combined = pd.concat(band_samples)
            # Fill if we didn't get enough
            if len(combined) < per_band:
                remaining = band_cat.drop(combined.index, errors="ignore")
                extra_n = min(per_band - len(combined), len(remaining))
                if extra_n > 0:
                    idx = rng.choice(len(remaining), size=extra_n, replace=False)
                    combined = pd.concat([combined, remaining.iloc[idx]])
            samples.append(combined)

    if not samples:
        return pd.DataFrame()

    result = pd.concat(samples)
    if len(result) < n:
        remaining = cat.drop(result.index, errors="ignore")
        extra_n = min(n - len(result), len(remaining))
        if extra_n > 0:
            idx = rng.choice(len(remaining), size=extra_n, replace=False)
            result = pd.concat([result, remaining.iloc[idx]])

    return result.head(n).sort_values("onset_utc").reset_index(drop=True)


# === Data caching (same pattern as validate_false_positives.py) ===
_file_cache = {}
_data_cache = {}
MAX_CACHE = 3


def get_mooring_catalog(mooring_key):
    if mooring_key not in _file_cache:
        info = MOORINGS[mooring_key]
        mooring_dir = DATA_ROOT / info["data_dir"]
        catalog = list_mooring_files(mooring_dir, sort_by="timestamp")
        _file_cache[mooring_key] = [
            (entry["timestamp"], entry["path"]) for entry in catalog
        ]
    return _file_cache[mooring_key]


def get_data(filepath):
    key = str(filepath)
    if key not in _data_cache:
        if len(_data_cache) >= MAX_CACHE:
            oldest = next(iter(_data_cache))
            del _data_cache[oldest]
        ts, data, _ = read_dat(filepath)
        _data_cache[key] = (ts, data)
    return _data_cache[key]


def extract_event_snippet(event_row):
    """Extract waveform and spectrogram for one event, with both onsets."""
    mooring = event_row["mooring"]
    onset = event_row["onset_utc"]
    duration = event_row["duration_s"]
    band = event_row["detection_band"]

    has_refined = (
        "onset_utc_refined" in event_row.index
        and pd.notna(event_row.get("onset_utc_refined"))
    )

    # Anchor window so onset is at 30% from left (ensures pick is visible)
    pre_context = WINDOW_SEC * 0.3
    t_start = onset - timedelta(seconds=pre_context)
    t_end = t_start + timedelta(seconds=WINDOW_SEC)

    catalog = get_mooring_catalog(mooring)
    for file_ts, filepath in catalog:
        file_end = file_ts + timedelta(seconds=14400)
        if file_ts <= t_start and file_end >= t_end:
            ts, data = get_data(filepath)

            offset_s = (t_start - ts).total_seconds()
            start_samp = int(offset_s * SAMPLE_RATE)
            end_samp = start_samp + int(WINDOW_SEC * SAMPLE_RATE)
            if start_samp < 0 or end_samp > len(data):
                return None
            segment = data[start_samp:end_samp]

            filtered = band_filter(segment.astype(np.float64), band)

            freqs, times, Sxx = spectrogram(
                segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
            )
            freq_mask = freqs <= FREQ_MAX
            freqs = freqs[freq_mask]
            Sxx_dB = 10 * np.log10(Sxx[freq_mask, :] + 1e-20)

            ev_start = (onset - t_start).total_seconds()
            ev_end = ev_start + duration

            result = {
                "times": times,
                "freqs": freqs,
                "Sxx_dB": Sxx_dB,
                "waveform": segment,
                "filtered": filtered,
                "time_s": np.arange(len(segment)) / SAMPLE_RATE,
                "ev_start": ev_start,
                "ev_end": ev_end,
                "refined_start": None,
            }

            if has_refined:
                refined_onset = event_row["onset_utc_refined"]
                result["refined_start"] = (
                    refined_onset - t_start
                ).total_seconds()

            return result
    return None


def plot_montage(events, snippets):
    """Plot montage with original and refined onset picks."""
    n = len(events)
    fig = plt.figure(figsize=(22, 38))
    outer_gs = GridSpec(NROWS, NCOLS, figure=fig,
                        hspace=0.55, wspace=0.25,
                        top=0.97, bottom=0.01, left=0.04, right=0.98)
    fig.suptitle(
        f"Onset Refinement QC — {n} Events (overweighted grade C, seed={SEED})",
        fontsize=16, fontweight="bold"
    )

    for idx in range(NROWS * NCOLS):
        row, col = idx // NCOLS, idx % NCOLS

        if idx >= n or snippets[idx] is None:
            ax_tmp = fig.add_subplot(outer_gs[row, col])
            ax_tmp.set_visible(False)
            continue

        ev = events.iloc[idx]
        snip = snippets[idx]
        color = BAND_COLORS.get(ev["detection_band"], "red")

        inner_gs = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[row, col],
            height_ratios=[3, 2], hspace=0.08
        )
        ax_spec = fig.add_subplot(inner_gs[0])
        ax_wave = fig.add_subplot(inner_gs[1], sharex=ax_spec)

        # --- Spectrogram ---
        vmin = np.percentile(snip["Sxx_dB"], 5)
        vmax = np.percentile(snip["Sxx_dB"], 95)
        ax_spec.pcolormesh(snip["times"], snip["freqs"], snip["Sxx_dB"],
                           vmin=vmin, vmax=vmax, cmap="viridis",
                           shading="auto", rasterized=True)

        # Original onset (dashed)
        ax_spec.axvline(snip["ev_start"], color=color, linewidth=1.0,
                        linestyle="--", alpha=0.7, label="original")
        ax_wave.axvline(snip["ev_start"], color=color, linewidth=1.0,
                        linestyle="--", alpha=0.7)

        # Refined onset (solid)
        if snip["refined_start"] is not None:
            ax_spec.axvline(snip["refined_start"], color=color, linewidth=1.8,
                            alpha=0.95, label="refined")
            ax_wave.axvline(snip["refined_start"], color=color, linewidth=1.8,
                            alpha=0.95)

        # Event end
        ax_spec.axvline(snip["ev_end"], color="gray", linewidth=0.8,
                        linestyle=":", alpha=0.5)

        ax_spec.set_ylim(0, FREQ_MAX)
        ax_spec.tick_params(labelsize=5)
        plt.setp(ax_spec.get_xticklabels(), visible=False)

        # Title
        band = ev["detection_band"]
        mooring = ev["mooring"].upper()
        time_str = ev["onset_utc"].strftime("%m-%d %H:%M:%S")
        method = ev.get("onset_method", "?")
        quality = ev.get("onset_quality", 0)
        grade = ev.get("onset_grade", "?")
        shift = ev.get("onset_shift_s", 0)
        ax_spec.set_title(
            f"#{idx+1} {mooring} {band} [{method}/{grade}] q={quality:.2f}\n"
            f"{time_str} shift={shift:+.3f}s",
            fontsize=6.5, fontweight="bold", pad=2
        )

        if col == 0:
            ax_spec.set_ylabel("Hz", fontsize=6)
        else:
            ax_spec.set_yticklabels([])

        # --- Filtered waveform ---
        ax_wave.plot(snip["time_s"], snip["filtered"],
                     color="0.3", linewidth=0.3, rasterized=True)
        ax_wave.tick_params(labelsize=5)

        if row == NROWS - 1:
            ax_wave.set_xlabel("s", fontsize=6)
        else:
            ax_wave.set_xticklabels([])
        if col == 0:
            ax_wave.set_ylabel("amp", fontsize=6)
        else:
            ax_wave.set_yticklabels([])

    outpath = FIG_DIR / "onset_refinement_montage.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")
    return outpath


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QC montage for refined onset picks")
    parser.add_argument("--n", type=int, default=N_SAMPLES,
                        help=f"Number of events to sample (default: {N_SAMPLES})")
    args = parser.parse_args()

    print("=" * 60)
    print("Onset Refinement QC Montage")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cat = load_catalogue()
    print(f"Loaded {len(cat):,} events")

    has_refined = "onset_utc_refined" in cat.columns
    if not has_refined:
        print("WARNING: No refined onset columns found. Run refine_onsets.py first.")
        print("Showing original onsets only.")

    events = sample_events(cat, n=args.n)
    print(f"Sampled {len(events)} events:")
    if has_refined and "onset_grade" in events.columns:
        print(events["onset_grade"].value_counts().to_string())
    print(events["detection_band"].value_counts().to_string())
    print()

    # Extract snippets
    snippets = []
    for idx, (_, ev) in enumerate(events.iterrows()):
        snip = extract_event_snippet(ev)
        if snip is None:
            print(f"  #{idx+1}: SKIP — no data for {ev['mooring']} at "
                  f"{ev['onset_utc']}")
        snippets.append(snip)
        if (idx + 1) % 10 == 0:
            print(f"  Extracted {idx+1}/{len(events)} snippets")

    n_ok = sum(1 for s in snippets if s is not None)
    print(f"\n{n_ok}/{len(events)} snippets extracted successfully")

    print("\nGenerating montage...")
    plot_montage(events, snippets)

    ref_path = FIG_DIR / "onset_refinement_events.csv"
    events.to_csv(ref_path, index=False)
    print(f"Saved event list: {ref_path}")

    print("\nDone. Inspect the montage to verify refined picks.")


if __name__ == "__main__":
    main()
