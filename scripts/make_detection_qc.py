#!/usr/bin/env python3
"""
make_detection_qc.py — Detection QC figure: waveform + spectrogram with picks.

For each selected 10-minute segment, produces a side-by-side figure:
  Left:  Time-domain waveform with vertical lines at event onset times
  Right: Spectrogram (frequency domain)

Usage:
    uv run python make_detection_qc.py

Spec: specs/001-event-detection/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime, timedelta
from scipy.signal import spectrogram, butter, sosfilt

from read_dat import read_dat, list_mooring_files, MOORINGS, SAMPLE_RATE
from make_bathy_map import add_caption_justified

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory"

# === Paper-tier font sizes ===
FS_TITLE = 14
FS_AXIS_LABEL = 10
FS_TICK = 9
FS_LEGEND = 8
FS_CAPTION = 10

# === Spectrogram parameters ===
NPERSEG = 1024
NOVERLAP = 512
FREQ_MAX = 250

# === Band colors (Okabe-Ito) ===
BAND_COLORS = {
    "low":       "#E69F00",   # orange
    "mid":       "#56B4E9",   # sky blue
    "high":      "#009E73",   # bluish green
}

BAND_LABELS = {
    "low":       "Low (1–15 Hz)",
    "mid":       "Mid (15–30 Hz)",
    "high":      "High (30–250 Hz)",
}

# === Segments to plot ===
# Two 10-minute windows: one busy, one moderate
SEGMENTS = [
    {
        "mooring": "m1",
        "start": datetime(2019, 7, 14, 13, 0, 0),
        "label": "Busy window — M1, 14 Jul 2019",
    },
    {
        "mooring": "m4",
        "start": datetime(2019, 8, 21, 6, 20, 0),
        "label": "Moderate window — M4, 21 Aug 2019",
    },
]
WINDOW_SEC = 600  # 10 minutes


def load_catalogue():
    """Load event catalogue (prefer three-pass, fall back to merged/original)."""
    for name in ["event_catalogue.parquet", "event_catalogue_merged.parquet"]:
        path = DATA_DIR / name
        if path.exists():
            cat = pd.read_parquet(path)
            print(f"  Using catalogue: {name}")
            break
    else:
        raise FileNotFoundError("No event catalogue found in outputs/data/")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])
    return cat


def find_dat_file(mooring_key, target_time):
    """Find the DAT file containing the target time for a mooring."""
    info = MOORINGS[mooring_key]
    mooring_dir = DATA_ROOT / info["data_dir"]
    catalog = list_mooring_files(mooring_dir, sort_by="timestamp")

    for entry in catalog:
        file_ts = entry["timestamp"]
        # Each file is 4 hours
        file_end = file_ts + timedelta(seconds=14400)
        if file_ts <= target_time < file_end:
            return entry["path"], file_ts
    return None, None


def extract_segment(dat_path, file_ts, seg_start, seg_duration_s):
    """Extract a time segment from a DAT file."""
    _, data, _ = read_dat(dat_path)
    offset_s = (seg_start - file_ts).total_seconds()
    start_samp = int(offset_s * SAMPLE_RATE)
    end_samp = start_samp + int(seg_duration_s * SAMPLE_RATE)
    end_samp = min(end_samp, len(data))
    return data[start_samp:end_samp]


def plot_segment(seg_info, cat, fig_idx):
    """Plot one 10-minute segment: waveform (left) + spectrogram (right)."""
    mooring = seg_info["mooring"]
    t_start = seg_info["start"]
    t_end = t_start + timedelta(seconds=WINDOW_SEC)

    # Find and read data
    dat_path, file_ts = find_dat_file(mooring, t_start)
    if dat_path is None:
        print(f"  No DAT file found for {mooring} at {t_start}")
        return
    print(f"  Reading {dat_path.name} ...")
    waveform = extract_segment(dat_path, file_ts, t_start, WINDOW_SEC)
    n_samples = len(waveform)
    time_s = np.arange(n_samples) / SAMPLE_RATE

    # Get events in this window for this mooring
    mask = (
        (cat["mooring"] == mooring)
        & (cat["onset_utc"] >= t_start)
        & (cat["onset_utc"] < t_end)
    )
    events = cat[mask].copy()
    events["t_rel"] = (events["onset_utc"] - t_start).dt.total_seconds()
    print(f"  {len(events)} events in window")

    # Compute spectrogram
    freqs, times, Sxx = spectrogram(
        waveform, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    freq_mask = freqs <= FREQ_MAX
    freqs = freqs[freq_mask]
    Sxx = Sxx[freq_mask, :]
    Sxx_dB = 10 * np.log10(Sxx + 1e-20)
    vmin = np.percentile(Sxx_dB, 2)
    vmax = np.percentile(Sxx_dB, 98)

    # --- Build figure: spectrogram on top, waveform below, shared x-axis ---
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.02], height_ratios=[1, 1],
                          hspace=0.25, wspace=0.03,
                          left=0.10, right=0.90, bottom=0.28, top=0.90)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_wave = fig.add_subplot(gs[1, 0], sharex=ax_spec)

    # --- Upper panel: spectrogram ---
    pcm = ax_spec.pcolormesh(
        times, freqs, Sxx_dB, vmin=vmin, vmax=vmax,
        cmap="viridis", shading="auto", rasterized=True
    )
    ax_spec.set_ylim(0, FREQ_MAX)
    ax_spec.set_xlim(0, WINDOW_SEC)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=FS_AXIS_LABEL,
                       fontweight="bold")
    ax_spec.tick_params(labelsize=FS_TICK)
    plt.setp(ax_spec.get_xticklabels(), visible=False)
    ax_spec.text(0.0, 1.06, "(a)  Spectrogram", transform=ax_spec.transAxes,
                 fontsize=FS_AXIS_LABEL + 1, fontweight="bold", va="bottom")

    # Colorbar in its own gridspec column — same height as spectrogram
    cbar = fig.colorbar(pcm, cax=ax_cbar)
    cbar.set_label("Power (dB)", fontsize=FS_AXIS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)

    for spine in ax_spec.spines.values():
        spine.set_linewidth(1.2)

    # --- Lower panel: waveform ---
    ax_wave.plot(time_s, waveform, color="0.3", linewidth=0.15, rasterized=True)
    ax_wave.set_xlim(0, WINDOW_SEC)
    ax_wave.set_xlabel("Time (s)", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax_wave.set_ylabel("Amplitude (counts)", fontsize=FS_AXIS_LABEL,
                       fontweight="bold")
    ax_wave.tick_params(labelsize=FS_TICK)
    ax_wave.text(0.0, 1.06, "(b)  Waveform", transform=ax_wave.transAxes,
                 fontsize=FS_AXIS_LABEL + 1, fontweight="bold", va="bottom")

    # Plot event onset lines on BOTH panels, colored by band
    bands_present = set()
    for _, ev in events.iterrows():
        band = ev["detection_band"]
        color = BAND_COLORS.get(band, "red")
        ax_wave.axvline(ev["t_rel"], color=color, linewidth=1.0, alpha=0.8)
        ax_spec.axvline(ev["t_rel"], color=color, linewidth=1.0, alpha=0.6)
        bands_present.add(band)

    # Legend for bands present
    legend_handles = [
        Line2D([0], [0], color=BAND_COLORS[b], linewidth=1.5,
               label=BAND_LABELS[b])
        for b in ["low", "mid", "high"] if b in bands_present
    ]
    if legend_handles:
        ax_wave.legend(handles=legend_handles, fontsize=FS_LEGEND,
                       loc="upper right", framealpha=0.85)

    for spine in ax_wave.spines.values():
        spine.set_linewidth(1.2)

    # --- Suptitle ---
    info = MOORINGS[mooring]
    fig.suptitle(
        f"{info['name']} / {mooring.upper()}  —  "
        f"{t_start.strftime('%Y-%m-%d %H:%M')}–"
        f"{t_end.strftime('%H:%M UTC')}  "
        f"({len(events)} events)",
        fontsize=FS_TITLE, fontweight="bold"
    )

    # --- Caption ---
    band_counts = events["detection_band"].value_counts()
    band_str = ", ".join(
        f"{BAND_LABELS[b]}: {band_counts.get(b, 0)}"
        for b in ["low", "mid", "high"]
        if b in band_counts.index
    )
    caption = (
        f"Ten-minute detection QC window for mooring {mooring.upper()} "
        f"({info['name']}). (a) Spectrogram (nperseg=1024, 50% overlap, "
        f"0–250 Hz). (b) Raw waveform with colored vertical lines marking "
        f"STA/LTA event onset times by frequency band. Three-pass strategy: "
        f"LP 15 Hz / BP 15–30 Hz / HP 30 Hz. "
        f"Event breakdown: {band_str}. "
        f"STA/LTA parameters: STA=2 s, LTA=60 s, trigger=3.0, detrigger=1.5."
    )
    add_caption_justified(fig, caption, caption_width=0.88, fontsize=FS_CAPTION,
                          caption_left=0.06, bold_prefix="Temporary Caption:")

    ts_str = t_start.strftime("%Y%m%d_%H%M")
    outpath = FIG_DIR / f"detection_qc_{mooring}_{ts_str}.png"
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    print("=" * 60)
    print("Detection QC Figures — Waveform + Spectrogram with Picks")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cat = load_catalogue()
    print(f"Loaded {len(cat):,} events\n")

    for i, seg in enumerate(SEGMENTS):
        print(f"--- Segment {i + 1}: {seg['label']} ---")
        plot_segment(seg, cat, i)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
