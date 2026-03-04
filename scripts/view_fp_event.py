#!/usr/bin/env python3
"""
view_fp_event.py — Zoom in on a single FP validation event for detailed review.

Usage:
    uv run python scripts/view_fp_event.py <event_index>  # 1-based index from montage

Example:
    uv run python scripts/view_fp_event.py 3
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
WINDOW_SEC = 10
NPERSEG = 256
NOVERLAP = 192
FREQ_MAX = 250

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

# Cache
_file_cache = {}
_data_cache = {}


def band_filter(data, band, fs=SAMPLE_RATE, order=4):
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
        if len(_data_cache) >= 3:
            oldest = next(iter(_data_cache))
            del _data_cache[oldest]
        ts, data, _ = read_dat(filepath)
        _data_cache[key] = (ts, data)
    return _data_cache[key]


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/view_fp_event.py <event_index>")
        print("  event_index: 1-based index from the montage (1-50)")
        sys.exit(1)

    idx = int(sys.argv[1]) - 1  # convert to 0-based

    # Load the sampled events
    events = pd.read_csv(FIG_DIR / "fp_validation_events.csv")
    events["onset_utc"] = pd.to_datetime(events["onset_utc"])
    events["end_utc"] = pd.to_datetime(events["end_utc"])

    if idx < 0 or idx >= len(events):
        print(f"Index must be 1-{len(events)}")
        sys.exit(1)

    ev = events.iloc[idx]
    print(f"Event #{idx+1}: {ev['event_id']}")
    print(f"  Mooring: {ev['mooring']}, Band: {ev['detection_band']}")
    print(f"  Onset: {ev['onset_utc']}, Duration: {ev['duration_s']:.2f}s")
    print(f"  SNR: {ev['snr']:.2f}, Peak freq: {ev['peak_freq_hz']:.1f} Hz")

    mooring = ev["mooring"]
    onset = ev["onset_utc"]
    duration = ev["duration_s"]
    band = ev["detection_band"]

    center = onset + timedelta(seconds=duration / 2)
    t_start = center - timedelta(seconds=WINDOW_SEC / 2)
    t_end = center + timedelta(seconds=WINDOW_SEC / 2)

    catalog = get_mooring_catalog(mooring)
    segment = None
    for file_ts, filepath in catalog:
        file_end = file_ts + timedelta(seconds=14400)
        if file_ts <= t_start and file_end >= t_end:
            ts, data = get_data(filepath)
            offset_s = (t_start - ts).total_seconds()
            start_samp = int(offset_s * SAMPLE_RATE)
            end_samp = start_samp + int(WINDOW_SEC * SAMPLE_RATE)
            if start_samp < 0 or end_samp > len(data):
                print("Event falls outside file bounds")
                sys.exit(1)
            segment = data[start_samp:end_samp]
            break

    if segment is None:
        print("Could not find DAT file for this event")
        sys.exit(1)

    ev_start = (onset - t_start).total_seconds()
    ev_end = ev_start + duration
    time_s = np.arange(len(segment)) / SAMPLE_RATE
    filtered = band_filter(segment.astype(np.float64), band)

    freqs, times, Sxx = spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    freq_mask = freqs <= FREQ_MAX
    freqs = freqs[freq_mask]
    Sxx_dB = 10 * np.log10(Sxx[freq_mask, :] + 1e-20)

    color = BAND_COLORS.get(band, "red")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[3, 2, 2],
                              sharex=True)

    # Spectrogram
    ax = axes[0]
    vmin, vmax = np.percentile(Sxx_dB, [5, 95])
    ax.pcolormesh(times, freqs, Sxx_dB, vmin=vmin, vmax=vmax,
                  cmap="viridis", shading="auto", rasterized=True)
    ax.axvline(ev_start, color=color, linewidth=2, alpha=0.9, label="STA/LTA onset")
    ax.axvline(ev_end, color=color, linewidth=1.5, linestyle="--", alpha=0.7,
               label="STA/LTA end")
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    ax.set_ylim(0, FREQ_MAX)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        f"#{idx+1} — {ev['event_id']}  |  {ev['mooring'].upper()} {band} band  |  "
        f"SNR={ev['snr']:.1f}  peak={ev['peak_freq_hz']:.0f} Hz  dur={duration:.1f}s\n"
        f"{ev['onset_utc']}",
        fontsize=12, fontweight="bold"
    )

    # Raw waveform
    ax = axes[1]
    ax.plot(time_s, segment, color="0.4", linewidth=0.4, rasterized=True)
    ax.axvline(ev_start, color=color, linewidth=2, alpha=0.9)
    ax.axvline(ev_end, color=color, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_ylabel("Raw amplitude", fontsize=10)

    # Band-filtered waveform
    ax = axes[2]
    ax.plot(time_s, filtered, color="0.3", linewidth=0.5, rasterized=True)
    ax.axvline(ev_start, color=color, linewidth=2, alpha=0.9)
    ax.axvline(ev_end, color=color, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_ylabel(f"Filtered ({band})", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)

    plt.tight_layout()
    outpath = FIG_DIR / f"fp_detail_{idx+1:02d}.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
