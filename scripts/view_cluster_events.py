#!/usr/bin/env python3
"""
view_cluster_events.py — Waveform + spectrogram pairs for cluster inspection.

For a given cluster, plots N events showing both the time-domain waveform
and spectrogram side by side. Helps identify signal type during labeling.

Usage:
    uv run python view_cluster_events.py mid_1
    uv run python view_cluster_events.py low_0 --n 10
    uv run python view_cluster_events.py mid_2 --random

Spec: specs/002-event-discrimination/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import timedelta
from scipy.signal import spectrogram, butter, sosfilt

from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "clustering"

# === Parameters ===
NPERSEG = 256
NOVERLAP = 192
FREQ_MAX = 250
WINDOW_SEC = 10
PRE_CONTEXT = 0.3  # fraction of window before onset

# === Data cache ===
_data_cache = {}
MAX_CACHE = 3


def get_data(filepath):
    """Read DAT file with LRU cache."""
    key = str(filepath)
    if key not in _data_cache:
        if len(_data_cache) >= MAX_CACHE:
            oldest = next(iter(_data_cache))
            del _data_cache[oldest]
        ts, data, _ = read_dat(filepath)
        _data_cache[key] = (ts, data)
    return _data_cache[key]


def load_data():
    """Load UMAP coordinates and event catalogue."""
    umap_df = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])
    merged = umap_df.merge(cat, on="event_id", suffixes=("", "_cat"))
    return merged


def select_nearest_centroid(cluster_df, n=8):
    """Select n events nearest to the UMAP centroid."""
    cx = cluster_df["umap_1"].mean()
    cy = cluster_df["umap_2"].mean()
    dist = np.sqrt((cluster_df["umap_1"] - cx)**2 +
                   (cluster_df["umap_2"] - cy)**2)
    cluster_df = cluster_df.copy()
    cluster_df["_centroid_dist"] = dist.values
    return cluster_df.nsmallest(min(n, len(cluster_df)), "_centroid_dist")


def extract_snippet(event_row):
    """Extract waveform and spectrogram for one event."""
    mooring = event_row["mooring"]
    file_num = event_row["file_number"]
    onset = event_row["onset_utc"]
    duration = event_row["duration_s"]

    info = MOORINGS[mooring]
    mooring_dir = DATA_ROOT / info["data_dir"]
    dat_path = mooring_dir / f"{file_num:08d}.DAT"

    if not dat_path.exists():
        return None

    file_ts, data = get_data(dat_path)
    file_nsamples = len(data)

    pre_context = WINDOW_SEC * PRE_CONTEXT
    t_start = onset - timedelta(seconds=pre_context)
    t_end = t_start + timedelta(seconds=WINDOW_SEC)

    offset_s = (t_start - file_ts).total_seconds()
    start_samp = int(offset_s * SAMPLE_RATE)
    end_samp = start_samp + int(WINDOW_SEC * SAMPLE_RATE)

    if start_samp < 0 or end_samp > file_nsamples:
        return None

    segment = data[start_samp:end_samp].astype(np.float64)
    time_axis = np.arange(len(segment)) / SAMPLE_RATE

    freqs, times, Sxx = spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    freq_mask = freqs <= FREQ_MAX
    freqs = freqs[freq_mask]
    Sxx_dB = 10 * np.log10(Sxx[freq_mask, :] + 1e-20)

    ev_start = pre_context
    ev_end = ev_start + duration

    return {
        "waveform": segment,
        "time_axis": time_axis,
        "times": times,
        "freqs": freqs,
        "Sxx_dB": Sxx_dB,
        "ev_start": ev_start,
        "ev_end": ev_end,
    }


def plot_cluster_detail(cluster_id, events, snippets, cluster_size):
    """Plot waveform + spectrogram pairs for a cluster."""
    n = sum(1 for s in snippets if s is not None)
    if n == 0:
        print(f"  No valid snippets for {cluster_id}")
        return None

    fig = plt.figure(figsize=(16, 3.2 * n))
    gs = GridSpec(n, 2, figure=fig, width_ratios=[1, 1.2],
                  hspace=0.45, wspace=0.25,
                  top=0.94, bottom=0.04, left=0.06, right=0.96)

    fig.suptitle(
        f"Cluster {cluster_id} — {cluster_size:,} events "
        f"(showing {n} nearest centroid)",
        fontsize=14, fontweight="bold"
    )

    row = 0
    for idx in range(len(events)):
        if snippets[idx] is None:
            continue

        ev = events.iloc[idx]
        snip = snippets[idx]

        # Left: waveform
        ax_wave = fig.add_subplot(gs[row, 0])
        ax_wave.plot(snip["time_axis"], snip["waveform"],
                     linewidth=0.3, color="black", rasterized=True)
        ax_wave.axvline(snip["ev_start"], color="red", linewidth=1.0,
                        linestyle="--", alpha=0.8)
        ax_wave.axvline(snip["ev_end"], color="red", linewidth=0.8,
                        linestyle=":", alpha=0.6)
        ax_wave.set_xlim(0, WINDOW_SEC)
        ax_wave.set_ylabel("Amplitude", fontsize=8)
        if row == n - 1:
            ax_wave.set_xlabel("Time (s)", fontsize=9)

        # Title
        mooring = ev["mooring"].upper()
        band = ev.get("detection_band", ev.get("detection_band_cat", "?"))
        pf = ev.get("peak_freq_hz", ev.get("peak_freq_hz_cat", 0))
        dur = ev["duration_s"]
        snr = ev.get("snr", ev.get("snr_cat", 0))
        time_str = ev["onset_utc"].strftime("%Y-%m-%d %H:%M:%S")
        ax_wave.set_title(
            f"{mooring} | {band} | {pf:.0f} Hz | {dur:.1f}s | "
            f"SNR={snr:.1f} | {time_str}",
            fontsize=8, fontweight="bold"
        )

        # Right: spectrogram
        ax_spec = fig.add_subplot(gs[row, 1])
        vmin = np.percentile(snip["Sxx_dB"], 5)
        vmax = np.percentile(snip["Sxx_dB"], 95)
        ax_spec.pcolormesh(snip["times"], snip["freqs"], snip["Sxx_dB"],
                           vmin=vmin, vmax=vmax, cmap="viridis",
                           shading="auto", rasterized=True)
        ax_spec.axvline(snip["ev_start"], color="white", linewidth=1.0,
                        linestyle="--", alpha=0.8)
        ax_spec.axvline(snip["ev_end"], color="white", linewidth=0.8,
                        linestyle=":", alpha=0.6)
        ax_spec.set_ylim(0, FREQ_MAX)
        ax_spec.set_ylabel("Freq (Hz)", fontsize=8)
        if row == n - 1:
            ax_spec.set_xlabel("Time (s)", fontsize=9)

        row += 1

    outpath = FIG_DIR / f"cluster_detail_{cluster_id}.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    return outpath


def main():
    parser = argparse.ArgumentParser(
        description="Waveform + spectrogram pairs for cluster inspection")
    parser.add_argument("cluster_id", type=str,
                        help="Cluster ID (e.g., mid_1, low_0)")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of events to show (default: 8)")
    parser.add_argument("--random", action="store_true",
                        help="Random sample instead of nearest centroid")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    cluster_df = df[df["cluster_id"] == args.cluster_id]
    if len(cluster_df) == 0:
        print(f"No events found for cluster '{args.cluster_id}'")
        print(f"Available: {sorted(df['cluster_id'].unique())}")
        return

    print(f"Cluster {args.cluster_id}: {len(cluster_df):,} events")

    if args.random:
        rng = np.random.default_rng(42)
        n = min(args.n, len(cluster_df))
        idx = rng.choice(len(cluster_df), size=n, replace=False)
        selected = cluster_df.iloc[idx]
    else:
        selected = select_nearest_centroid(cluster_df, n=args.n)

    snippets = []
    for _, ev in selected.iterrows():
        snip = extract_snippet(ev)
        snippets.append(snip)

    n_ok = sum(1 for s in snippets if s is not None)
    print(f"Extracted {n_ok}/{len(selected)} snippets")

    outpath = plot_cluster_detail(
        args.cluster_id,
        selected.reset_index(drop=True),
        snippets,
        len(cluster_df),
    )
    if outpath:
        print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
