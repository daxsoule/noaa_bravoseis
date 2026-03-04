#!/usr/bin/env python3
"""
validate_false_positives.py — Visual FP validation for detection catalogue.

Randomly samples 50 events (stratified by band) from the event catalogue,
plots a 10-second spectrogram + waveform snippet centered on each event,
and saves a montage sheet for manual inspection.

Usage:
    uv run python validate_false_positives.py

Spec: specs/001-event-detection/ (Validation Approach)
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
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "validation"

# === Parameters ===
N_SAMPLES = 50
WINDOW_SEC = 10  # seconds of context around each event
SEED = 42
NPERSEG = 256
NOVERLAP = 192
FREQ_MAX = 250

# Layout: 10 rows × 5 columns per sheet
NCOLS = 5
NROWS = 10


def load_catalogue():
    """Load event catalogue."""
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])
    if "onset_utc_refined" in cat.columns:
        cat["onset_utc_refined"] = pd.to_datetime(cat["onset_utc_refined"])
    return cat


def sample_events(cat, n=N_SAMPLES, seed=SEED):
    """Stratified random sample by detection band."""
    rng = np.random.default_rng(seed)
    bands = cat["detection_band"].unique()
    per_band = max(1, n // len(bands))

    samples = []
    for band in sorted(bands):
        subset = cat[cat["detection_band"] == band]
        k = min(per_band, len(subset))
        idx = rng.choice(len(subset), size=k, replace=False)
        samples.append(subset.iloc[idx])

    # Fill remainder if needed
    combined = pd.concat(samples)
    if len(combined) < n:
        remaining = cat.drop(combined.index)
        extra = remaining.sample(n - len(combined), random_state=seed)
        combined = pd.concat([combined, extra])

    return combined.head(n).sort_values("onset_utc").reset_index(drop=True)


# Cache: mooring_key -> list of (file_ts, filepath)
_file_cache = {}


def get_mooring_catalog(mooring_key):
    """Get sorted file catalog for a mooring (cached)."""
    if mooring_key not in _file_cache:
        info = MOORINGS[mooring_key]
        mooring_dir = DATA_ROOT / info["data_dir"]
        catalog = list_mooring_files(mooring_dir, sort_by="timestamp")
        _file_cache[mooring_key] = [
            (entry["timestamp"], entry["path"]) for entry in catalog
        ]
    return _file_cache[mooring_key]


# Cache: filepath -> (file_ts, data)
_data_cache = {}
MAX_CACHE = 3  # keep at most 3 files in memory


def get_data(filepath):
    """Read DAT file with simple LRU cache."""
    key = str(filepath)
    if key not in _data_cache:
        if len(_data_cache) >= MAX_CACHE:
            # Remove oldest
            oldest = next(iter(_data_cache))
            del _data_cache[oldest]
        ts, data, _ = read_dat(filepath)
        _data_cache[key] = (ts, data)
    return _data_cache[key]


def extract_event_snippet(event_row):
    """Extract waveform, filtered waveform, and spectrogram for one event."""
    mooring = event_row["mooring"]
    onset = event_row["onset_utc"]
    duration = event_row["duration_s"]
    band = event_row["detection_band"]

    has_refined = (
        "onset_utc_refined" in event_row.index
        and pd.notna(event_row.get("onset_utc_refined"))
    )

    # Center the window on the event
    center = onset + timedelta(seconds=duration / 2)
    t_start = center - timedelta(seconds=WINDOW_SEC / 2)
    t_end = center + timedelta(seconds=WINDOW_SEC / 2)

    # Find the DAT file
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

            # Band-filtered waveform (same filter used for detection)
            filtered = band_filter(segment.astype(np.float64), band)

            # Spectrogram
            freqs, times, Sxx = spectrogram(
                segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
            )
            freq_mask = freqs <= FREQ_MAX
            freqs = freqs[freq_mask]
            Sxx_dB = 10 * np.log10(Sxx[freq_mask, :] + 1e-20)

            # Event onset/end relative to window
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


BAND_COLORS = {
    "low": "#E69F00",
    "mid": "#56B4E9",
    "high": "#009E73",
}

# Filter config per band (matches detect_events.py PASSES)
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
    sos = butter(order, wn, btype=cfg["btype"], output='sos')
    return sosfilt(sos, data)


def plot_montage(events, snippets):
    """Plot a montage sheet: spectrogram + filtered waveform per event."""
    n = len(events)
    # Each cell has 2 sub-rows (spectrogram + waveform), so total height grows
    fig = plt.figure(figsize=(22, 38))
    outer_gs = GridSpec(NROWS, NCOLS, figure=fig,
                        hspace=0.55, wspace=0.25,
                        top=0.97, bottom=0.01, left=0.04, right=0.98)
    fig.suptitle(
        f"False Positive Validation — {n} Random Events (seed={SEED})",
        fontsize=16, fontweight="bold"
    )

    for idx in range(NROWS * NCOLS):
        row, col = idx // NCOLS, idx % NCOLS

        if idx >= n or snippets[idx] is None:
            # Create and hide placeholder axes
            ax_tmp = fig.add_subplot(outer_gs[row, col])
            ax_tmp.set_visible(False)
            continue

        ev = events.iloc[idx]
        snip = snippets[idx]
        color = BAND_COLORS.get(ev["detection_band"], "red")

        # Split cell into spectrogram (top, 60%) and waveform (bottom, 40%)
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
        ax_spec.axvline(snip["ev_start"], color=color, linewidth=1.0,
                        linestyle="--", alpha=0.7)
        if snip.get("refined_start") is not None:
            ax_spec.axvline(snip["refined_start"], color=color,
                            linewidth=1.8, alpha=0.95)
        ax_spec.axvline(snip["ev_end"], color=color, linewidth=1.0,
                        linestyle="--", alpha=0.7)
        ax_spec.set_ylim(0, FREQ_MAX)
        ax_spec.tick_params(labelsize=5)
        plt.setp(ax_spec.get_xticklabels(), visible=False)

        # Title on spectrogram
        band = ev["detection_band"]
        snr = ev["snr"]
        pf = ev["peak_freq_hz"]
        dur = ev["duration_s"]
        mooring = ev["mooring"].upper()
        time_str = ev["onset_utc"].strftime("%m-%d %H:%M:%S")
        ax_spec.set_title(
            f"#{idx+1} {mooring} {band} SNR={snr:.1f}\n"
            f"{time_str} {pf:.0f}Hz {dur:.1f}s",
            fontsize=6.5, fontweight="bold", pad=2
        )

        if col == 0:
            ax_spec.set_ylabel("Hz", fontsize=6)
        else:
            ax_spec.set_yticklabels([])

        # --- Filtered waveform ---
        ax_wave.plot(snip["time_s"], snip["filtered"],
                     color="0.3", linewidth=0.3, rasterized=True)
        ax_wave.axvline(snip["ev_start"], color=color, linewidth=1.0,
                        linestyle="--", alpha=0.7)
        if snip.get("refined_start") is not None:
            ax_wave.axvline(snip["refined_start"], color=color,
                            linewidth=1.8, alpha=0.95)
        ax_wave.axvline(snip["ev_end"], color=color, linewidth=1.0,
                        linestyle="--", alpha=0.7)
        ax_wave.tick_params(labelsize=5)

        if row == NROWS - 1:
            ax_wave.set_xlabel("s", fontsize=6)
        else:
            ax_wave.set_xticklabels([])
        if col == 0:
            ax_wave.set_ylabel("amp", fontsize=6)
        else:
            ax_wave.set_yticklabels([])

    outpath = FIG_DIR / "fp_validation_montage.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")
    return outpath


def main():
    print("=" * 60)
    print("False Positive Validation")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cat = load_catalogue()
    print(f"Loaded {len(cat):,} events")

    events = sample_events(cat)
    print(f"Sampled {len(events)} events (stratified by band):")
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

    # Plot
    print("\nGenerating montage...")
    plot_montage(events, snippets)

    # Save event list for reference
    ref_path = FIG_DIR / "fp_validation_events.csv"
    events.to_csv(ref_path, index=False)
    print(f"Saved event list: {ref_path}")

    print("\nDone. Inspect the montage and mark false positives.")
    print("Target: < 20% false positive rate (< 10 of 50).")


if __name__ == "__main__":
    main()
