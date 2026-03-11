#!/usr/bin/env python3
"""
make_gold_single_panels.py — Generate one large waveform+spectrogram image
per event for detailed visual review.

Two sampling strategies per cluster:
  - stratified: events sampled evenly across distance-from-centroid quantiles
    (core → edge), giving representation of cluster quality gradient
  - random: pure random sample for unbiased purity estimate

Usage:
    uv run python make_gold_single_panels.py --cluster low_0 --strategy stratified
    uv run python make_gold_single_panels.py --cluster low_0 --strategy random
    uv run python make_gold_single_panels.py --cluster low_0 --strategy stratified --n 30
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

# === Spectrogram parameters ===
NPERSEG = 256
NOVERLAP = 192
FREQ_MAX = 250
WINDOW_SEC = 15      # Fixed window: 5s before pick + 10s after pick
PAD_SEC = 5.0        # Pre-pick padding

# === Bandpass filter ranges per detection band ===
BAND_FILTERS = {
    "low":  (1.0, 15.0),
    "mid":  (15.0, 30.0),
    "high": (30.0, 250.0),
}

# === Spectrogram display range per band (show ~2× above bandpass) ===
BAND_FREQ_MAX = {
    "low":  50,
    "mid":  60,
    "high": 250,
}

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
    nyq = fs / 2
    high = min(high, nyq * 0.99)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data)


def load_data():
    umap_df = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    features = pd.read_parquet(DATA_DIR / "event_features.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])

    merged = umap_df.merge(cat, on="event_id", suffixes=("", "_cat"))
    merged = merged.merge(
        features[["event_id", "peak_freq_hz", "peak_power_db",
                   "spectral_slope", "bandwidth_hz", "spectral_centroid_hz"]],
        on="event_id", suffixes=("", "_feat"), how="left"
    )
    return merged


def compute_centroid_distances(cluster_df):
    """Add _centroid_dist column to cluster_df."""
    cx = cluster_df["umap_1"].mean()
    cy = cluster_df["umap_2"].mean()
    dist = np.sqrt((cluster_df["umap_1"] - cx)**2 +
                   (cluster_df["umap_2"] - cy)**2)
    cluster_df = cluster_df.copy()
    cluster_df["_centroid_dist"] = dist.values
    return cluster_df


def select_stratified(cluster_df, n=30, seed=42):
    """Sample events evenly across distance-from-centroid quantiles.

    Divides the cluster into quantile bins by centroid distance, then
    samples equally from each bin. This gives representation from
    the cluster core (best members) through the periphery (edge cases).
    """
    cluster_df = compute_centroid_distances(cluster_df)
    n = min(n, len(cluster_df))

    # Use 5 quantile bins (or fewer if cluster is small)
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
    """Pure random sample for unbiased purity estimate."""
    cluster_df = compute_centroid_distances(cluster_df)
    n = min(n, len(cluster_df))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cluster_df), size=n, replace=False)
    return cluster_df.iloc[idx].copy()


def select_nearest_centroid(cluster_df, n=10):
    """Legacy: select n events nearest to cluster centroid."""
    cluster_df = compute_centroid_distances(cluster_df)
    return cluster_df.nsmallest(min(n, len(cluster_df)), "_centroid_dist")


def extract_snippet(event_row, band):
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

    total_window = WINDOW_SEC  # Fixed 15s: 5s before pick + 10s after
    t_start = onset - timedelta(seconds=PAD_SEC)

    offset_s = (t_start - file_ts).total_seconds()
    start_samp = int(offset_s * SAMPLE_RATE)
    end_samp = start_samp + int(total_window * SAMPLE_RATE)

    if start_samp < 0 or end_samp > file_nsamples:
        return None

    segment = data[start_samp:end_samp].astype(np.float64)

    filt_low, filt_high = BAND_FILTERS.get(band, (1.0, 250.0))
    waveform_filt = bandpass(segment, filt_low, filt_high, SAMPLE_RATE)
    t_wave = np.arange(len(segment)) / SAMPLE_RATE

    freqs, times, Sxx = spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    freq_mask = freqs <= FREQ_MAX
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
        "total_window": total_window,
        "filt_low": filt_low,
        "filt_high": filt_high,
    }


def plot_single_event(cluster_id, panel_num, ev, snip, cluster_size, band,
                      strategy="stratified"):
    """One large figure per event: waveform on top, spectrogram on bottom."""
    fig, (ax_wave, ax_spec) = plt.subplots(
        2, 1, figsize=(14, 7), height_ratios=[1, 1.2],
        gridspec_kw={"hspace": 0.25}
    )

    filt_low, filt_high = BAND_FILTERS.get(band, (1.0, 250.0))

    # --- Metadata ---
    mooring = ev["mooring"].upper()
    time_str = ev["onset_utc"].strftime("%Y-%m-%d %H:%M:%S")
    eid = ev["event_id"]
    pf = ev.get("peak_freq_hz", ev.get("peak_freq_hz_feat", 0))
    if pd.isna(pf):
        pf = 0
    dur = ev["duration_s"]
    snr = ev.get("snr", ev.get("snr_cat", 0))
    if pd.isna(snr):
        snr = 0
    pp = ev.get("peak_power_db", 0)
    if pd.isna(pp):
        pp = 0
    slope = ev.get("spectral_slope", ev.get("spectral_slope_feat", 0))
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
        f"Waveform — Bandpass {filt_low:.0f}–{filt_high:.0f} Hz",
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
    display_fmax = BAND_FREQ_MAX.get(band, FREQ_MAX)
    ax_spec.set_ylim(0, display_fmax)
    ax_spec.axhline(filt_low, color="cyan", linewidth=0.7,
                    linestyle=":", alpha=0.6)
    ax_spec.axhline(filt_high, color="cyan", linewidth=0.7,
                    linestyle=":", alpha=0.6)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=10)
    ax_spec.set_xlabel("Time (s)", fontsize=10)
    ax_spec.set_title(f"Spectrogram — 0–{display_fmax} Hz", fontsize=10, loc="left")
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
        description="Generate individual event panels for gold-standard review")
    parser.add_argument("--cluster", type=str, required=True,
                        help="Cluster ID (e.g., low_0)")
    parser.add_argument("--strategy", type=str, default="stratified",
                        choices=["stratified", "random"],
                        help="Sampling strategy (default: stratified)")
    parser.add_argument("--n", type=int, default=30,
                        help="Number of events (default: 30)")
    args = parser.parse_args()

    cluster_id = args.cluster
    strategy = args.strategy
    cid_str = str(cluster_id)
    if cid_str.startswith("low"):
        band = "low"
    elif cid_str.startswith("mid"):
        band = "mid"
    elif cid_str.startswith("high"):
        band = "high"
    else:
        band = "low"

    print(f"Generating {args.n} {strategy} panels for cluster {cluster_id}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    cluster_df = df[df["cluster_id"] == cluster_id]
    cluster_size = len(cluster_df)
    print(f"  Cluster size: {cluster_size:,}")

    is_noise = cid_str.endswith("_noise")
    if is_noise:
        selected = select_random(cluster_df, n=args.n, seed=42)
    elif strategy == "stratified":
        selected = select_stratified(cluster_df, n=args.n)
    elif strategy == "random":
        selected = select_random(cluster_df, n=args.n)
    else:
        selected = select_stratified(cluster_df, n=args.n)

    panel_num = 0
    for _, ev in selected.iterrows():
        panel_num += 1
        snip = extract_snippet(ev, band)
        if snip is None:
            print(f"  Panel {panel_num}: extraction failed — skipped")
            continue
        outpath = plot_single_event(
            cluster_id, panel_num, ev, snip, cluster_size, band,
            strategy=strategy
        )
        print(f"  Panel {panel_num}: {outpath.name}")

    print(f"\nDone. Review panels in {FIG_DIR}")


if __name__ == "__main__":
    main()
