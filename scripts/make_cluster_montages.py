#!/usr/bin/env python3
"""
make_cluster_montages.py — Generate spectrogram montages for each cluster.

For each HDBSCAN cluster, selects 20 events nearest to the cluster centroid
and plots their spectrogram patches in a 4×5 grid. Used for visual inspection
and labeling during Phase 1c.

Usage:
    uv run python make_cluster_montages.py
    uv run python make_cluster_montages.py --n-per-cluster 30

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
from scipy.signal import spectrogram

from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "clustering"
PATCH_DIR = DATA_DIR / "event_patches"

# === Parameters ===
N_PER_CLUSTER = 20
NCOLS = 5
NROWS = 4
NPERSEG = 256
NOVERLAP = 192
FREQ_MAX = 250
WINDOW_SEC = 10  # context window per event
PAD_SEC = 2.0

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

    # Merge to get full event info with cluster assignments
    merged = umap_df.merge(cat, on="event_id", suffixes=("", "_cat"))
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
    """Extract spectrogram snippet for one event."""
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

    # Center window on event onset with pre-context
    pre_context = WINDOW_SEC * 0.3
    t_start = onset - timedelta(seconds=pre_context)
    t_end = t_start + timedelta(seconds=WINDOW_SEC)

    offset_s = (t_start - file_ts).total_seconds()
    start_samp = int(offset_s * SAMPLE_RATE)
    end_samp = start_samp + int(WINDOW_SEC * SAMPLE_RATE)

    if start_samp < 0 or end_samp > file_nsamples:
        return None

    segment = data[start_samp:end_samp]

    freqs, times, Sxx = spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    freq_mask = freqs <= FREQ_MAX
    freqs = freqs[freq_mask]
    Sxx_dB = 10 * np.log10(Sxx[freq_mask, :] + 1e-20)

    ev_start = pre_context
    ev_end = ev_start + duration

    return {
        "times": times,
        "freqs": freqs,
        "Sxx_dB": Sxx_dB,
        "ev_start": ev_start,
        "ev_end": ev_end,
    }


def plot_cluster_montage(cluster_id, events, snippets, cluster_size):
    """Plot a montage for one cluster."""
    n = len(events)
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(NROWS, NCOLS, figure=fig,
                  hspace=0.4, wspace=0.2,
                  top=0.93, bottom=0.03, left=0.04, right=0.98)

    fig.suptitle(
        f"Cluster {cluster_id} — {cluster_size:,} events "
        f"(showing {n} nearest centroid)",
        fontsize=14, fontweight="bold"
    )

    for idx in range(NROWS * NCOLS):
        row, col = idx // NCOLS, idx % NCOLS
        ax = fig.add_subplot(gs[row, col])

        if idx >= n or snippets[idx] is None:
            ax.set_visible(False)
            continue

        ev = events.iloc[idx]
        snip = snippets[idx]

        vmin = np.percentile(snip["Sxx_dB"], 5)
        vmax = np.percentile(snip["Sxx_dB"], 95)

        ax.pcolormesh(snip["times"], snip["freqs"], snip["Sxx_dB"],
                      vmin=vmin, vmax=vmax, cmap="viridis",
                      shading="auto", rasterized=True)
        ax.axvline(snip["ev_start"], color="white", linewidth=1.0,
                   linestyle="--", alpha=0.7)
        ax.axvline(snip["ev_end"], color="white", linewidth=0.8,
                   linestyle=":", alpha=0.5)
        ax.set_ylim(0, FREQ_MAX)
        ax.tick_params(labelsize=6)

        # Title with key info
        mooring = ev["mooring"].upper()
        band = ev.get("detection_band", ev.get("detection_band_cat", "?"))
        pf = ev.get("peak_freq_hz", ev.get("peak_freq_hz_cat", 0))
        dur = ev["duration_s"]
        snr = ev.get("snr", ev.get("snr_cat", 0))
        time_str = ev["onset_utc"].strftime("%m-%d %H:%M")
        ax.set_title(
            f"{mooring} {band} {pf:.0f}Hz {dur:.1f}s SNR={snr:.1f}\n"
            f"{time_str}",
            fontsize=6, fontweight="bold", pad=2
        )

        if col == 0:
            ax.set_ylabel("Hz", fontsize=7)
        else:
            ax.set_yticklabels([])
        if row == NROWS - 1:
            ax.set_xlabel("s", fontsize=7)
        else:
            ax.set_xticklabels([])

    outpath = FIG_DIR / f"cluster_montage_{cluster_id}.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    return outpath


def main():
    parser = argparse.ArgumentParser(
        description="Generate spectrogram montages for each cluster")
    parser.add_argument("--n-per-cluster", type=int, default=N_PER_CLUSTER,
                        help=f"Events per montage (default: {N_PER_CLUSTER})")
    args = parser.parse_args()

    print("=" * 60)
    print("BRAVOSEIS Cluster Montages — Phase 1c")
    print(f"  Events per cluster: {args.n_per_cluster}")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df):,} clustered events")

    # Get unique clusters — handle both string (per-band) and numeric IDs
    all_cluster_ids = sorted(df["cluster_id"].unique(), key=str)

    # Separate noise clusters from real clusters
    noise_ids = [cid for cid in all_cluster_ids
                 if str(cid).endswith("_noise") or cid == -1]
    real_ids = [cid for cid in all_cluster_ids if cid not in noise_ids]

    cluster_ids = noise_ids + real_ids
    print(f"Clusters to visualize: {len(cluster_ids)}")

    for cid in cluster_ids:
        cluster_df = df[df["cluster_id"] == cid]
        cluster_size = len(cluster_df)
        is_noise = str(cid).endswith("_noise") or cid == -1
        label = f"noise ({cid})" if is_noise else f"cluster {cid}"

        # Select representative events
        if is_noise:
            # Random sample for noise
            rng = np.random.default_rng(42)
            n = min(args.n_per_cluster, len(cluster_df))
            idx = rng.choice(len(cluster_df), size=n, replace=False)
            selected = cluster_df.iloc[idx]
        else:
            selected = select_nearest_centroid(cluster_df,
                                               n=args.n_per_cluster)

        # Extract snippets
        snippets = []
        for _, ev in selected.iterrows():
            snip = extract_snippet(ev)
            snippets.append(snip)

        n_ok = sum(1 for s in snippets if s is not None)
        print(f"  {label}: {cluster_size:,} events, "
              f"{n_ok}/{len(selected)} snippets extracted")

        # Plot
        outpath = plot_cluster_montage(
            cid, selected.reset_index(drop=True), snippets, cluster_size
        )
        print(f"    Saved: {outpath}")

    # Summary table
    print(f"\n{'=' * 60}")
    print("Cluster summary:")
    for cid in cluster_ids:
        n = (df["cluster_id"] == cid).sum()
        print(f"  {str(cid):16s}: {n:6,}")

    print(f"\nDone. Review montages and create "
          f"outputs/data/cluster_labels.json")


if __name__ == "__main__":
    main()
