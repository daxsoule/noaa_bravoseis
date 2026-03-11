#!/usr/bin/env python3
"""
make_12s_signal_figures.py — Generate multi-scale figures of the recurring
~12s signal discovered in mid_0 clustering, for colleague review.

The signal is concentrated in file 943 (2019-06-19), present on all 6 moorings.

Generates:
  - 60-min overview (waveform + spectrogram) for 3 moorings
  - 5-min zoom windows showing signal structure
  - 1-min detail windows showing individual pulses

Usage:
    uv run python make_12s_signal_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
from scipy.signal import spectrogram, butter, sosfilt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "12s_signal"

# === Signal source ===
FILE_NUM = 943
# Moorings to show (pick 3 for multi-mooring comparison)
MOORINGS_TO_PLOT = ["m3", "m5", "m1"]

# === Spectrogram parameters ===
FREQ_MAX_OVERVIEW = 60    # Hz, for overview and 5-min plots
FREQ_MAX_DETAIL = 60      # Hz, for 1-min detail plots
BANDPASS = (10, 40)        # Wider than mid detection band to show full signal


def load_file(mooring, file_num):
    info = MOORINGS[mooring]
    dat_path = DATA_ROOT / info["data_dir"] / f"{file_num:08d}.DAT"
    ts, data, _ = read_dat(dat_path)
    return ts, data


def bandpass_filter(data, low, high, fs, order=4):
    nyq = fs / 2
    high = min(high, nyq * 0.99)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data)


def make_figure(mooring, file_num, file_ts, data, start_min, duration_min,
                freq_max, nperseg, noverlap, tag, bandpass_range=BANDPASS):
    """Generate a waveform + spectrogram figure at given time scale."""

    start_samp = int(start_min * 60 * SAMPLE_RATE)
    end_samp = start_samp + int(duration_min * 60 * SAMPLE_RATE)
    end_samp = min(end_samp, len(data))
    segment = data[start_samp:end_samp].astype(np.float64)

    t_seconds = np.arange(len(segment)) / SAMPLE_RATE

    # Choose x-axis units
    if duration_min >= 5:
        t_plot = t_seconds / 60
        xlabel = "Time (minutes from start)"
    else:
        t_plot = t_seconds
        xlabel = "Time (seconds from start)"

    # Filtered waveform
    filt = bandpass_filter(segment, bandpass_range[0], bandpass_range[1],
                           SAMPLE_RATE)

    # Spectrogram
    freqs, times, Sxx = spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap
    )
    freq_mask = freqs <= freq_max
    freqs = freqs[freq_mask]
    Sxx_dB = 10 * np.log10(Sxx[freq_mask, :] + 1e-20)

    if duration_min >= 5:
        times_plot = times / 60
    else:
        times_plot = times

    # Timestamp for title
    window_start = file_ts + timedelta(minutes=start_min)
    window_end = file_ts + timedelta(minutes=start_min + duration_min)

    fig, (ax_wave, ax_spec) = plt.subplots(
        2, 1, figsize=(16, 8), height_ratios=[1, 1.3],
        gridspec_kw={"hspace": 0.3}
    )

    # Waveform
    ax_wave.plot(t_plot, filt, color="black", linewidth=0.4)
    ax_wave.set_xlim(t_plot[0], t_plot[-1])
    ax_wave.set_ylabel("Amplitude", fontsize=12)
    ax_wave.set_title(
        f"{mooring.upper()} — File {file_num} — "
        f"Bandpass {bandpass_range[0]}–{bandpass_range[1]} Hz\n"
        f"{window_start.strftime('%Y-%m-%d %H:%M:%S')} to "
        f"{window_end.strftime('%H:%M:%S')} UTC "
        f"({duration_min:.0f} min)" if duration_min >= 1 else
        f"{mooring.upper()} — File {file_num} — "
        f"Bandpass {bandpass_range[0]}–{bandpass_range[1]} Hz\n"
        f"{window_start.strftime('%Y-%m-%d %H:%M:%S')} to "
        f"{window_end.strftime('%H:%M:%S')} UTC "
        f"({duration_min*60:.0f} s)",
        fontsize=13, fontweight="bold"
    )
    ax_wave.tick_params(labelsize=10)

    # Spectrogram
    vmin = np.percentile(Sxx_dB, 5)
    vmax = np.percentile(Sxx_dB, 95)
    im = ax_spec.pcolormesh(times_plot, freqs, Sxx_dB,
                             vmin=vmin, vmax=vmax, cmap="viridis",
                             shading="auto", rasterized=True)
    ax_spec.set_ylim(0, freq_max)
    ax_spec.axhline(bandpass_range[0], color="cyan", linewidth=0.7,
                     linestyle=":", alpha=0.6)
    ax_spec.axhline(bandpass_range[1], color="cyan", linewidth=0.7,
                     linestyle=":", alpha=0.6)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=12)
    ax_spec.set_xlabel(xlabel, fontsize=12)
    ax_spec.set_title(f"Spectrogram — 0–{freq_max} Hz",
                       fontsize=12, fontweight="bold", loc="left")
    ax_spec.tick_params(labelsize=10)

    cbar = fig.colorbar(im, ax=ax_spec, pad=0.02, aspect=30)
    cbar.set_label("Power (dB)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    outpath = FIG_DIR / f"12s_{mooring}_{tag}.png"
    fig.savefig(outpath, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {outpath.name}")
    return outpath


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Recurring ~12s Signal — Multi-scale Figures")
    print(f"  Source: file {FILE_NUM} (2019-06-19)")
    print(f"  Moorings: {', '.join(m.upper() for m in MOORINGS_TO_PLOT)}")
    print("=" * 60)

    for mooring in MOORINGS_TO_PLOT:
        print(f"\n--- {mooring.upper()} ---")
        file_ts, data = load_file(mooring, FILE_NUM)
        file_duration_min = len(data) / SAMPLE_RATE / 60
        print(f"  File start: {file_ts}, duration: {file_duration_min:.1f} min")

        # 1) 60-min overview — first hour
        make_figure(mooring, FILE_NUM, file_ts, data,
                    start_min=0, duration_min=60,
                    freq_max=FREQ_MAX_OVERVIEW,
                    nperseg=512, noverlap=448,
                    tag="60min_hour1")

        # 2) 60-min overview — second hour
        if file_duration_min > 60:
            make_figure(mooring, FILE_NUM, file_ts, data,
                        start_min=60, duration_min=min(60, file_duration_min - 60),
                        freq_max=FREQ_MAX_OVERVIEW,
                        nperseg=512, noverlap=448,
                        tag="60min_hour2")

        # 3) 5-min zoom windows — pick 3 windows showing the signal
        #    Signal is concentrated in first ~100 min
        for i, start in enumerate([10, 40, 70]):
            if start + 5 <= file_duration_min:
                make_figure(mooring, FILE_NUM, file_ts, data,
                            start_min=start, duration_min=5,
                            freq_max=FREQ_MAX_OVERVIEW,
                            nperseg=256, noverlap=224,
                            tag=f"5min_window{i+1}")

        # 4) 1-min detail windows — pick 3 windows for individual pulse detail
        #    Use times when signal is active (within first hour)
        for i, start in enumerate([12, 42, 72]):
            if start + 1 <= file_duration_min:
                make_figure(mooring, FILE_NUM, file_ts, data,
                            start_min=start, duration_min=1,
                            freq_max=FREQ_MAX_DETAIL,
                            nperseg=256, noverlap=224,
                            tag=f"1min_detail{i+1}")

    print(f"\nDone. Figures saved to {FIG_DIR}")
    print(f"Total: {len(list(FIG_DIR.glob('12s_*.png')))} figures")


if __name__ == "__main__":
    main()
