#!/usr/bin/env python3
"""
make_gold_standard_montages.py — Find high-confidence events per cluster
and generate waveform + spectrogram montages for visual review.

For each HDBSCAN cluster in each frequency band, selects the 10 events
nearest the UMAP centroid and plots a side-by-side montage:
  - Left column: bandpass-filtered waveform (timeseries)
  - Right column: spectrogram

These montages support hand-picking "gold standard" template events for
synthetic training data generation (Phase 3 classification pipeline).

Usage:
    uv run python make_gold_standard_montages.py
    uv run python make_gold_standard_montages.py --n-per-cluster 20
    uv run python make_gold_standard_montages.py --band low
    uv run python make_gold_standard_montages.py --cluster low_0

Output:
    outputs/figures/exploratory/gold_standard/gold_montage_{cluster_id}.png
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
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "gold_standard"

# === Spectrogram parameters ===
NPERSEG = 256
NOVERLAP = 192
FREQ_MAX = 250       # Full range for spectrogram display
WINDOW_SEC = 20      # Context window per event (seconds)
PAD_SEC = 5.0        # Pre-event padding

# === Bandpass filter ranges per detection band ===
BAND_FILTERS = {
    "low":  (1.0, 15.0),
    "mid":  (15.0, 30.0),
    "high": (30.0, 250.0),
}

# === Display ===
N_PER_CLUSTER = 10

# === Data cache ===
_data_cache = {}
MAX_CACHE = 5


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


def bandpass(data, low, high, fs, order=4):
    """Apply Butterworth bandpass filter."""
    nyq = fs / 2
    # Clamp high to just below Nyquist
    high = min(high, nyq * 0.99)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data)


def load_data():
    """Load UMAP coordinates, event catalogue, and features."""
    umap_df = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    features = pd.read_parquet(DATA_DIR / "event_features.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])

    # Merge all three
    merged = umap_df.merge(cat, on="event_id", suffixes=("", "_cat"))
    merged = merged.merge(
        features[["event_id", "peak_freq_hz", "peak_power_db",
                   "spectral_slope", "bandwidth_hz", "spectral_centroid_hz"]],
        on="event_id", suffixes=("", "_feat"), how="left"
    )
    return merged


def select_nearest_centroid(cluster_df, n=N_PER_CLUSTER):
    """Select n events nearest to the UMAP centroid of the cluster."""
    cx = cluster_df["umap_1"].mean()
    cy = cluster_df["umap_2"].mean()
    dist = np.sqrt((cluster_df["umap_1"] - cx)**2 +
                   (cluster_df["umap_2"] - cy)**2)
    cluster_df = cluster_df.copy()
    cluster_df["_centroid_dist"] = dist.values
    return cluster_df.nsmallest(min(n, len(cluster_df)), "_centroid_dist")


def extract_snippet(event_row):
    """Extract waveform + spectrogram for one event."""
    mooring = event_row["mooring"]
    file_num = event_row["file_number"]
    onset = event_row["onset_utc"]
    duration = event_row["duration_s"]
    band = event_row.get("detection_band",
                         event_row.get("detection_band_cat", "low"))

    info = MOORINGS[mooring]
    mooring_dir = DATA_ROOT / info["data_dir"]
    dat_path = mooring_dir / f"{file_num:08d}.DAT"

    if not dat_path.exists():
        return None

    file_ts, data = get_data(dat_path)
    file_nsamples = len(data)

    # Window centered on event with pre-context
    total_window = max(WINDOW_SEC, duration + 2 * PAD_SEC)
    t_start = onset - timedelta(seconds=PAD_SEC)
    t_end = t_start + timedelta(seconds=total_window)

    offset_s = (t_start - file_ts).total_seconds()
    start_samp = int(offset_s * SAMPLE_RATE)
    end_samp = start_samp + int(total_window * SAMPLE_RATE)

    if start_samp < 0 or end_samp > file_nsamples:
        return None

    segment = data[start_samp:end_samp].astype(np.float64)

    # Bandpass filter for waveform display
    filt_low, filt_high = BAND_FILTERS.get(band, (1.0, 250.0))
    waveform_filt = bandpass(segment, filt_low, filt_high, SAMPLE_RATE)
    t_wave = np.arange(len(segment)) / SAMPLE_RATE

    # Spectrogram (full bandwidth for visual context)
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


def plot_gold_montage(cluster_id, events, snippets, cluster_size, band):
    """Plot a 2-column montage: waveform (left) + spectrogram (right).

    Each row is one event. 10 rows per cluster.
    """
    n = sum(1 for s in snippets if s is not None)
    if n == 0:
        return None

    fig = plt.figure(figsize=(16, 2.4 * n + 1.5))
    gs = GridSpec(n, 2, figure=fig,
                  hspace=0.45, wspace=0.15,
                  top=0.94, bottom=0.03, left=0.07, right=0.97,
                  width_ratios=[1, 1])

    filt_low, filt_high = BAND_FILTERS.get(band, (1.0, 250.0))
    fig.suptitle(
        f"Gold Standard Candidates — Cluster {cluster_id} "
        f"({cluster_size:,} events)\n"
        f"Bandpass: {filt_low:.0f}–{filt_high:.0f} Hz | "
        f"Showing {n} events nearest UMAP centroid",
        fontsize=13, fontweight="bold"
    )

    plot_idx = 0
    for idx in range(len(events)):
        if snippets[idx] is None:
            continue

        ev = events.iloc[idx]
        snip = snippets[idx]

        # --- Left: Waveform ---
        ax_wave = fig.add_subplot(gs[plot_idx, 0])
        ax_wave.plot(snip["t_wave"], snip["waveform"],
                     color="black", linewidth=0.4)

        # Event window shading
        ax_wave.axvspan(snip["ev_start"], snip["ev_end"],
                        color="red", alpha=0.1)
        ax_wave.axvline(snip["ev_start"], color="red", linewidth=1.2,
                        linestyle="--", alpha=0.8)
        ax_wave.axvline(snip["ev_end"], color="red", linewidth=0.8,
                        linestyle=":", alpha=0.6)
        ax_wave.set_xlim(0, snip["total_window"])
        ax_wave.set_ylabel("Amplitude", fontsize=7)
        ax_wave.tick_params(labelsize=6)

        # Event metadata title
        mooring = ev["mooring"].upper()
        time_str = ev["onset_utc"].strftime("%Y-%m-%d %H:%M:%S")
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

        ax_wave.set_title(
            f"#{plot_idx+1} {mooring} | {time_str} | "
            f"pf={pf:.1f}Hz  dur={dur:.1f}s  SNR={snr:.1f}  "
            f"pwr={pp:.1f}dB  slope={slope:.2f}  "
            f"d_centroid={cdist:.3f}",
            fontsize=7, fontweight="bold", pad=3
        )

        if plot_idx == n - 1:
            ax_wave.set_xlabel("Time (s)", fontsize=7)

        # --- Right: Spectrogram ---
        ax_spec = fig.add_subplot(gs[plot_idx, 1])
        vmin = np.percentile(snip["Sxx_dB"], 5)
        vmax = np.percentile(snip["Sxx_dB"], 95)
        ax_spec.pcolormesh(snip["times"], snip["freqs"], snip["Sxx_dB"],
                           vmin=vmin, vmax=vmax, cmap="viridis",
                           shading="auto", rasterized=True)
        ax_spec.axvline(snip["ev_start"], color="white", linewidth=1.2,
                        linestyle="--", alpha=0.8)
        ax_spec.axvline(snip["ev_end"], color="white", linewidth=0.8,
                        linestyle=":", alpha=0.6)

        # Show full spectrogram range but highlight the band
        ax_spec.set_ylim(0, FREQ_MAX)
        ax_spec.axhline(filt_low, color="cyan", linewidth=0.5,
                        linestyle=":", alpha=0.5)
        ax_spec.axhline(filt_high, color="cyan", linewidth=0.5,
                        linestyle=":", alpha=0.5)

        ax_spec.set_ylabel("Freq (Hz)", fontsize=7)
        ax_spec.tick_params(labelsize=6)

        if plot_idx == n - 1:
            ax_spec.set_xlabel("Time (s)", fontsize=7)

        plot_idx += 1

    outpath = FIG_DIR / f"gold_montage_{cluster_id}.png"
    fig.savefig(outpath, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return outpath


def main():
    parser = argparse.ArgumentParser(
        description="Generate gold-standard candidate montages per cluster")
    parser.add_argument("--n-per-cluster", type=int, default=N_PER_CLUSTER,
                        help=f"Events per montage (default: {N_PER_CLUSTER})")
    parser.add_argument("--band", type=str, default=None,
                        choices=["low", "mid", "high"],
                        help="Process only one band")
    parser.add_argument("--cluster", type=str, default=None,
                        help="Process only one cluster ID (e.g., low_0)")
    args = parser.parse_args()

    print("=" * 60)
    print("BRAVOSEIS Gold Standard Candidate Montages")
    print(f"  Events per cluster: {args.n_per_cluster}")
    if args.band:
        print(f"  Band filter: {args.band}")
    if args.cluster:
        print(f"  Cluster filter: {args.cluster}")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df):,} events")

    # Get all cluster IDs
    all_cluster_ids = sorted(df["cluster_id"].unique(), key=str)

    # Filter by band if requested
    if args.band:
        all_cluster_ids = [cid for cid in all_cluster_ids
                           if str(cid).startswith(args.band)]

    # Filter to specific cluster if requested
    if args.cluster:
        all_cluster_ids = [cid for cid in all_cluster_ids
                           if str(cid) == args.cluster]

    # Separate noise from real clusters
    noise_ids = [cid for cid in all_cluster_ids
                 if str(cid).endswith("_noise") or cid == -1]
    real_ids = [cid for cid in all_cluster_ids if cid not in noise_ids]

    # Process real clusters first, then noise
    cluster_ids = real_ids + noise_ids
    print(f"Clusters to process: {len(cluster_ids)}")
    for cid in cluster_ids:
        print(f"  {cid}")

    results = []
    for cid in cluster_ids:
        cluster_df = df[df["cluster_id"] == cid]
        cluster_size = len(cluster_df)
        is_noise = str(cid).endswith("_noise") or cid == -1

        # Determine band from cluster ID
        cid_str = str(cid)
        if cid_str.startswith("low"):
            band = "low"
        elif cid_str.startswith("mid"):
            band = "mid"
        elif cid_str.startswith("high"):
            band = "high"
        else:
            band = "low"

        # Select events
        if is_noise:
            rng = np.random.default_rng(42)
            n = min(args.n_per_cluster, len(cluster_df))
            idx = rng.choice(len(cluster_df), size=n, replace=False)
            selected = cluster_df.iloc[idx].copy()
            selected["_centroid_dist"] = 0.0
        else:
            selected = select_nearest_centroid(cluster_df,
                                               n=args.n_per_cluster)

        # Extract snippets
        print(f"\n  Cluster {cid} ({cluster_size:,} events, band={band}):")
        snippets = []
        for i, (_, ev) in enumerate(selected.iterrows()):
            snip = extract_snippet(ev)
            snippets.append(snip)

        n_ok = sum(1 for s in snippets if s is not None)
        print(f"    Extracted {n_ok}/{len(selected)} snippets")

        # Plot
        outpath = plot_gold_montage(
            cid, selected.reset_index(drop=True), snippets,
            cluster_size, band
        )
        if outpath:
            print(f"    Saved: {outpath}")
            results.append((cid, cluster_size, band, n_ok))
        else:
            print(f"    No valid snippets — skipped")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'Cluster':>16s}  {'Size':>8s}  {'Band':>5s}  {'Plotted':>7s}")
    print(f"{'-'*16}  {'-'*8}  {'-'*5}  {'-'*7}")
    for cid, size, band, n_ok in results:
        print(f"{str(cid):>16s}  {size:>8,}  {band:>5s}  {n_ok:>7}")
    print(f"\nMontages saved to: {FIG_DIR}")
    print("Review these montages to select gold-standard templates "
          "for synthetic training data generation.")


if __name__ == "__main__":
    main()
