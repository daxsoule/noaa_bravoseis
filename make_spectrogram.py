#!/usr/bin/env python3
"""
make_spectrogram.py — Array spectrogram figures for BRAVOSEIS hydrophone data.

Generates 6-row stacked spectrograms (one per mooring) for three 10-minute
time windows, showing signal propagation across the array. Shared UTC time
axis and colorbar enable visual comparison of arrival times between moorings.

Usage:
    uv run python make_spectrogram.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.signal import spectrogram
from datetime import datetime, timedelta
from pathlib import Path

from read_dat import read_dat, read_header, MOORINGS, SAMPLE_RATE
from make_bathy_map import add_caption_justified

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent / "outputs" / "figures" / "exploratory"

# === Font sizes (Paper tier) ===
FS_TITLE = 14
FS_AXIS_LABEL = 10
FS_TICK = 9
FS_COLORBAR = 9
FS_PANEL_LABEL = 10
FS_CAPTION = 10
FS_NODATA = 12

# === Spectrogram parameters ===
NPERSEG = 1024       # 1.024 s window at 1 kHz
NOVERLAP = 512       # 50% overlap
FREQ_MAX = 250       # Hz, display limit
WINDOW_SECONDS = 600 # 10 minutes
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE  # 600,000

# === Mooring order (top to bottom) ===
MOORING_KEYS = ["m1", "m2", "m3", "m4", "m5", "m6"]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

# === Time windows ===
WINDOWS = [
    {
        "start": datetime(2019, 8, 14, 17, 38),
        "dat_file": "00001282.DAT",
        "missing": {"m3"},
        "label": "2019-08-14 17:38 UTC",
        "suffix": "20190814_1738",
    },
    {
        "start": datetime(2019, 8, 14, 21, 16),
        "dat_file": "00001283.DAT",
        "missing": {"m3"},
        "label": "2019-08-14 21:16 UTC",
        "suffix": "20190814_2116",
    },
    {
        "start": datetime(2020, 1, 9, 6, 24),
        "dat_file": "00002166.DAT",
        "missing": set(),
        "label": "2020-01-09 06:24 UTC",
        "suffix": "20200109_0624",
    },
]


def load_window_data(window):
    """Load 10-minute segments for all moorings in a time window.

    Returns dict mapping mooring key -> (times_sec, freqs, Sxx_dB) or None.
    Also returns combined vmin/vmax from 2nd-98th percentile.
    """
    segments = {}
    all_db_values = []

    for mkey in MOORING_KEYS:
        if mkey in window["missing"]:
            segments[mkey] = None
            continue

        info = MOORINGS[mkey]
        dat_path = DATA_ROOT / info["data_dir"] / window["dat_file"]

        if not dat_path.exists():
            print(f"  WARNING: {dat_path} not found, skipping {mkey}")
            segments[mkey] = None
            continue

        # Read file and compute offset
        file_ts, data, meta = read_dat(dat_path)
        offset_sec = (window["start"] - file_ts).total_seconds()
        offset_samples = int(offset_sec * SAMPLE_RATE)

        if offset_samples < 0 or offset_samples + WINDOW_SAMPLES > len(data):
            print(f"  WARNING: {mkey} window out of range "
                  f"(offset={offset_samples}, len={len(data)}), skipping")
            segments[mkey] = None
            continue

        segment = data[offset_samples:offset_samples + WINDOW_SAMPLES]

        # Compute spectrogram
        freqs, times, Sxx = spectrogram(
            segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
        )

        # Clip to frequency range
        freq_mask = freqs <= FREQ_MAX
        freqs = freqs[freq_mask]
        Sxx = Sxx[freq_mask, :]

        # Convert to dB
        Sxx_dB = 10 * np.log10(Sxx + 1e-20)

        segments[mkey] = (times, freqs, Sxx_dB)
        all_db_values.append(Sxx_dB)

        print(f"  {mkey} ({info['name']}): offset={offset_sec:.1f}s, "
              f"segment shape={segment.shape}, spectrogram={Sxx_dB.shape}")

    # Shared color limits from 2nd-98th percentile across all panels
    if all_db_values:
        combined = np.concatenate([v.ravel() for v in all_db_values])
        vmin = np.percentile(combined, 2)
        vmax = np.percentile(combined, 98)
    else:
        vmin, vmax = -100, 0

    return segments, vmin, vmax


def make_spectrogram_figure(window):
    """Create a 6-row spectrogram figure for one time window."""
    print(f"\n--- Window: {window['label']} (file {window['dat_file']}) ---")

    segments, vmin, vmax = load_window_data(window)

    fig = plt.figure(figsize=(7, 9))

    # Layout: 6 spectrogram rows in top ~62% of figure, caption below
    n_rows = len(MOORING_KEYS)
    plot_top = 0.92
    plot_bottom = 0.30
    plot_left = 0.16
    plot_right = 0.85
    cbar_left = 0.87
    cbar_width = 0.015
    row_height = (plot_top - plot_bottom) / n_rows
    row_gap = row_height * 0.08  # small gap between rows

    axes = []
    images = []
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i, mkey in enumerate(MOORING_KEYS):
        info = MOORINGS[mkey]
        row_top = plot_top - i * row_height
        row_bot = row_top - row_height + row_gap

        ax = fig.add_axes([plot_left, row_bot, plot_right - plot_left,
                           row_top - row_bot])
        axes.append(ax)

        if segments[mkey] is None:
            # No data — gray panel
            ax.set_facecolor('#e0e0e0')
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                    ha='center', va='center', fontsize=FS_NODATA,
                    color='#888888', fontweight='bold')
            ax.set_xlim(0, WINDOW_SECONDS / 60)
            ax.set_ylim(0, FREQ_MAX)
        else:
            times, freqs, Sxx_dB = segments[mkey]
            # Convert times from seconds to minutes for display
            im = ax.pcolormesh(times / 60, freqs, Sxx_dB,
                               cmap='viridis', norm=norm,
                               shading='auto', rasterized=True)
            images.append(im)
            ax.set_xlim(0, WINDOW_SECONDS / 60)
            ax.set_ylim(0, FREQ_MAX)

        # Y-axis: mooring name (horizontal, short form)
        ax.set_ylabel(f"{info['name']}\n{mkey.upper()}",
                       fontsize=FS_AXIS_LABEL, fontweight='bold',
                       labelpad=4, rotation=0, ha='right', va='center')
        ax.tick_params(labelsize=FS_TICK)
        ax.set_yticks([0, 50, 100, 150, 200, 250])

        # Light grid for value reading (rubric #12)
        ax.grid(True, alpha=0.25, color='white', linewidth=0.5)

        # Panel label in upper-left
        ax.text(0.01, 0.95, PANEL_LABELS[i], transform=ax.transAxes,
                fontsize=FS_PANEL_LABEL, fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.8, edgecolor='none'))

        # Only bottom panel gets x tick labels
        if i < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (minutes)", fontsize=FS_AXIS_LABEL,
                          fontweight='bold')

        # Spine width
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Shared "Frequency (Hz)" label on left margin
    fig.text(0.005, (plot_top + plot_bottom) / 2, "Frequency (Hz)",
             fontsize=FS_AXIS_LABEL, fontweight='bold',
             rotation=90, ha='left', va='center')

    # Shared colorbar
    cax = fig.add_axes([cbar_left, plot_bottom, cbar_width,
                        plot_top - plot_bottom])
    if images:
        cbar = fig.colorbar(images[0], cax=cax)
    else:
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Power (dB)", fontsize=FS_COLORBAR)
    cbar.ax.tick_params(labelsize=FS_COLORBAR)

    # Suptitle
    fig.suptitle(f"Array Spectrograms — {window['label']}",
                 fontsize=FS_TITLE, fontweight='bold', y=0.96)

    # Determine which moorings have data for caption
    active = [f"{MOORINGS[k]['name']}/{k.upper()}"
              for k in MOORING_KEYS if k not in window["missing"]]
    missing = [f"{MOORINGS[k]['name']}/{k.upper()}"
               for k in MOORING_KEYS if k in window["missing"]]
    data_str = ", ".join(active)
    miss_str = (f" {', '.join(missing)} not recording during this window."
                if missing else " All moorings recording.")

    caption = (
        f"Temporary Caption: Ten-minute spectrogram array for the BRAVOSEIS "
        f"hydrophone network in the Bransfield Strait, starting "
        f"{window['label']}. Each row shows one mooring (M1 at top, M6 at "
        f"bottom). Spectrograms computed via scipy.signal.spectrogram with "
        f"nperseg={NPERSEG} ({NPERSEG/SAMPLE_RATE:.3f} s), 50% overlap, "
        f"fs={SAMPLE_RATE} Hz. Frequency range 0\u2013{FREQ_MAX} Hz. Power "
        f"in dB (re arbitrary). Shared colorscale across all panels "
        f"(2nd\u201398th percentile). Moorings with data: {data_str}.{miss_str}"
    )
    add_caption_justified(fig, caption, caption_width=0.82, fontsize=FS_CAPTION,
                          caption_left=0.08, bold_prefix="Temporary Caption:")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / f"spectrogram_array_{window['suffix']}.png"
    fig.savefig(outpath, dpi=300, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


def main():
    print("=" * 60)
    print("Array Spectrogram Figures")
    print("=" * 60)

    outputs = []
    for window in WINDOWS:
        outputs.append(make_spectrogram_figure(window))

    print("\n" + "=" * 60)
    print("All figures generated:")
    for p in outputs:
        print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
