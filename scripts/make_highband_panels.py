#!/usr/bin/env python3
"""
make_highband_panels.py — Generate review panels for highband (30–450 Hz) clusters.

Uses umap_coordinates_highband.parquet clusters. Shows spectrograms
focused on the 0–500 Hz range.

Usage:
    uv run python scripts/make_highband_panels.py
    uv run python scripts/make_highband_panels.py --cluster highband_1
    uv run python scripts/make_highband_panels.py --strategy random
    uv run python scripts/make_highband_panels.py --n 15
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
from scipy.signal import spectrogram, butter, sosfilt

from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "gold_standard"

# === Display parameters ===
NPERSEG_DISPLAY = 1024   # ~1 Hz resolution
NOVERLAP_DISPLAY = 768   # 75% overlap
SPEC_FMAX = 250          # Instrument response tops out at 250 Hz
WINDOW_SEC = 10          # Shorter window — icequakes are brief
PAD_SEC = 2.0

# Bandpass for waveform display
BP_LOW = 30.0
BP_HIGH = 450.0

# === Data cache ===
_data_cache = {}
MAX_CACHE = 5


def get_data(filepath):
    key = str(filepath)
    if key not in _data_cache:
        if len(_data_cache) >= MAX_CACHE:
            oldest = next(iter(_data_cache))
            del _data_cache[oldest]
        ts, data, _ = read_dat(filepath)
        _data_cache[key] = (ts, data)
    return _data_cache[key]


def bandpass(data, low, high, fs, order=4):
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data)


def load_data():
    """Load highband UMAP clusters + catalogue + features."""
    umap_df = pd.read_parquet(DATA_DIR / "umap_coordinates_highband.parquet")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    features = pd.read_parquet(DATA_DIR / "event_features_highband.parquet")

    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])

    merged = umap_df.merge(cat, on="event_id", suffixes=("", "_cat"))
    feat_cols = ["event_id", "peak_freq_hz", "peak_power_db",
                 "spectral_slope", "bandwidth_hz", "spectral_centroid_hz"]
    merged = merged.merge(
        features[feat_cols],
        on="event_id", suffixes=("_cat", ""), how="left"
    )
    if "peak_freq_hz_cat" in merged.columns:
        merged = merged.drop(columns=["peak_freq_hz_cat"])
    return merged


def compute_centroid_distances(cluster_df):
    cx = cluster_df["umap_1"].mean()
    cy = cluster_df["umap_2"].mean()
    dist = np.sqrt((cluster_df["umap_1"] - cx)**2 +
                   (cluster_df["umap_2"] - cy)**2)
    cluster_df = cluster_df.copy()
    cluster_df["_centroid_dist"] = dist.values
    return cluster_df


def select_stratified(cluster_df, n=30, seed=42):
    cluster_df = compute_centroid_distances(cluster_df)
    n = min(n, len(cluster_df))
    n_bins = min(5, n)
    samples_per_bin = n // n_bins
    remainder = n % n_bins

    cluster_df["_dist_quantile"] = pd.qcut(
        cluster_df["_centroid_dist"], q=n_bins, labels=False,
        duplicates="drop"
    )
    actual_bins = sorted(cluster_df["_dist_quantile"].unique())

    rng = np.random.default_rng(seed)
    selected_parts = []
    for i, q in enumerate(actual_bins):
        bin_df = cluster_df[cluster_df["_dist_quantile"] == q]
        k = samples_per_bin + (1 if i < remainder else 0)
        k = min(k, len(bin_df))
        idx = rng.choice(len(bin_df), size=k, replace=False)
        selected_parts.append(bin_df.iloc[idx])

    selected = pd.concat(selected_parts).sort_values("_centroid_dist")
    return selected.drop(columns=["_dist_quantile"])


def select_random(cluster_df, n=30, seed=43):
    cluster_df = compute_centroid_distances(cluster_df)
    n = min(n, len(cluster_df))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cluster_df), size=n, replace=False)
    return cluster_df.iloc[idx].copy()


def extract_snippet(event_row):
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

    t_start = onset - timedelta(seconds=PAD_SEC)
    offset_s = (t_start - file_ts).total_seconds()
    start_samp = int(offset_s * SAMPLE_RATE)
    end_samp = start_samp + int(WINDOW_SEC * SAMPLE_RATE)

    if start_samp < 0 or end_samp > file_nsamples:
        return None

    segment = data[start_samp:end_samp].astype(np.float64)

    # Filtered waveform for display
    waveform_filt = bandpass(segment, BP_LOW, BP_HIGH, SAMPLE_RATE)
    t_wave = np.arange(len(segment)) / SAMPLE_RATE

    # Spectrogram — UNFILTERED (raw signal, full bandwidth)
    freqs, times, Sxx = spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=NPERSEG_DISPLAY,
        noverlap=NOVERLAP_DISPLAY
    )
    freq_mask = freqs <= SPEC_FMAX
    freqs = freqs[freq_mask]
    Sxx_dB = 10 * np.log10(Sxx[freq_mask, :] + 1e-20)

    return {
        "t_wave": t_wave,
        "waveform": waveform_filt,
        "times": times,
        "freqs": freqs,
        "Sxx_dB": Sxx_dB,
        "ev_start": PAD_SEC,
        "ev_end": PAD_SEC + duration,
        "total_window": WINDOW_SEC,
    }


def plot_single_event(cluster_id, panel_num, ev, snip, cluster_size,
                      strategy="stratified"):
    fig, (ax_wave, ax_spec) = plt.subplots(
        2, 1, figsize=(14, 7), height_ratios=[1, 1.2],
        gridspec_kw={"hspace": 0.25}
    )

    mooring = ev["mooring"].upper()
    time_str = ev["onset_utc"].strftime("%Y-%m-%d %H:%M:%S")
    eid = ev["event_id"]
    pf = ev.get("peak_freq_hz", 0)
    if pd.isna(pf):
        pf = 0
    dur = ev["duration_s"]
    snr = ev.get("snr", ev.get("snr_cat", 0))
    if pd.isna(snr):
        snr = 0
    pp = ev.get("peak_power_db", 0)
    if pd.isna(pp):
        pp = 0
    slope = ev.get("spectral_slope", 0)
    if pd.isna(slope):
        slope = 0
    cdist = ev.get("_centroid_dist", 0)

    strat_label = strategy.upper()
    fig.suptitle(
        f"Cluster {cluster_id} [{strat_label}] — Panel #{panel_num}  |  {eid}\n"
        f"{mooring}  |  {time_str}  |  "
        f"peak_freq={pf:.1f} Hz   dur={dur:.1f} s   SNR={snr:.1f}   "
        f"power={pp:.1f} dB   slope={slope:.2f}   "
        f"d_centroid={cdist:.3f}",
        fontsize=11, fontweight="bold"
    )

    # --- Waveform ---
    ax_wave.plot(snip["t_wave"], snip["waveform"],
                 color="black", linewidth=0.5)
    ax_wave.axvspan(snip["ev_start"], snip["ev_end"],
                    color="red", alpha=0.12)
    ax_wave.axvline(snip["ev_start"], color="red", linewidth=1.5,
                    linestyle="--", alpha=0.8, label="onset")
    ax_wave.axvline(snip["ev_end"], color="red", linewidth=1.0,
                    linestyle=":", alpha=0.6, label="end")
    ax_wave.set_xlim(0, snip["total_window"])
    ax_wave.set_ylabel("Amplitude", fontsize=10)
    ax_wave.set_title(
        f"Waveform — Bandpass {BP_LOW:.0f}–{BP_HIGH:.0f} Hz",
        fontsize=10, loc="left"
    )
    ax_wave.tick_params(labelsize=9)
    ax_wave.legend(fontsize=8, loc="upper right")

    # --- Spectrogram ---
    vmin = np.percentile(snip["Sxx_dB"], 5)
    vmax = np.percentile(snip["Sxx_dB"], 95)
    im = ax_spec.pcolormesh(snip["times"], snip["freqs"], snip["Sxx_dB"],
                            vmin=vmin, vmax=vmax, cmap="viridis",
                            shading="auto", rasterized=True)
    ax_spec.axvline(snip["ev_start"], color="white", linewidth=1.5,
                    linestyle="--", alpha=0.8)
    ax_spec.axvline(snip["ev_end"], color="white", linewidth=1.0,
                    linestyle=":", alpha=0.6)
    ax_spec.set_ylim(0, SPEC_FMAX)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=10)
    ax_spec.set_xlabel("Time (s)", fontsize=10)
    ax_spec.set_title(f"Spectrogram — 0–{SPEC_FMAX} Hz  UNFILTERED (nperseg={NPERSEG_DISPLAY})",
                      fontsize=10, loc="left")
    ax_spec.tick_params(labelsize=9)

    cbar = fig.colorbar(im, ax=ax_spec, pad=0.02, aspect=30)
    cbar.set_label("Power (dB)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    outpath = FIG_DIR / f"gold_{cluster_id}_{strategy}_panel{panel_num:02d}.png"
    fig.savefig(outpath, dpi=150, facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return outpath


def main():
    parser = argparse.ArgumentParser(
        description="Generate review panels for highband (30-450 Hz) clusters")
    parser.add_argument("--cluster", type=str, default=None,
                        help="Single cluster ID (e.g., highband_1). "
                             "Default: all clusters")
    parser.add_argument("--strategy", type=str, default="stratified",
                        choices=["stratified", "random"])
    parser.add_argument("--n", type=int, default=15,
                        help="Panels per cluster (default: 15)")
    parser.add_argument("--all", action="store_true",
                        help="Include mega-cluster")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading highband data...")
    df = load_data()

    if args.cluster:
        clusters = [args.cluster]
    else:
        cluster_sizes = df["cluster_id"].value_counts()
        clusters = [c for c in cluster_sizes.index
                    if c != "highband_noise"]
        if not args.all:
            biggest = cluster_sizes.idxmax()
            clusters = [c for c in clusters if c != biggest]
        clusters = sorted(clusters, key=lambda c: cluster_sizes[c])

    print(f"Generating {args.strategy} panels for {len(clusters)} clusters, "
          f"{args.n} panels each")

    for cluster_id in clusters:
        cluster_df = df[df["cluster_id"] == cluster_id]
        cluster_size = len(cluster_df)
        print(f"\n--- {cluster_id} ({cluster_size:,} events) ---")

        if args.strategy == "stratified":
            selected = select_stratified(cluster_df, n=args.n)
        else:
            selected = select_random(cluster_df, n=args.n)

        panel_num = 0
        for _, ev in selected.iterrows():
            panel_num += 1
            snip = extract_snippet(ev)
            if snip is None:
                print(f"  Panel {panel_num}: extraction failed — skipped")
                continue
            outpath = plot_single_event(
                cluster_id, panel_num, ev, snip, cluster_size, args.strategy
            )
            print(f"  Panel {panel_num}: {outpath.name}")

    print(f"\nDone. {len(clusters)} clusters x {args.n} panels = "
          f"{len(clusters) * args.n} images generated.")


if __name__ == "__main__":
    main()
