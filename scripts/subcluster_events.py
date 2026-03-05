#!/usr/bin/env python3
"""
subcluster_events.py — Re-cluster a mega-cluster to expose internal structure.

The initial per-band UMAP + HDBSCAN pass separates obvious signal types
(fin whale, noise, seismic_mixed) but leaves a large "catch-all" mega-cluster
(e.g., mid_3 with 124K events). This script re-runs UMAP + HDBSCAN on just
that subset with tighter parameters to tease apart sub-populations.

Usage:
    uv run python subcluster_events.py mid_3
    uv run python subcluster_events.py mid_3 --min-cluster-size 50
    uv run python subcluster_events.py mid_3 --min-dist 0.005
    uv run python subcluster_events.py low_2 --n-seeds 5

Outputs:
    - outputs/data/subclusters_{cluster_id}.parquet  (UMAP coords + sub-cluster labels)
    - outputs/figures/exploratory/clustering/subcluster_{cluster_id}_map.png
    - outputs/figures/exploratory/clustering/subcluster_{cluster_id}_features.png
    - outputs/figures/exploratory/clustering/subcluster_montage_{cluster_id}_{sub_id}.png

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import hdbscan

from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "clustering"
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")

# === Feature columns (must match extract_features.py / cluster_events.py) ===
BAND_POWER_COLS = [f"band_power_{i}" for i in range(10)]
FEATURE_COLS = (
    BAND_POWER_COLS +
    ["peak_freq_hz", "peak_power_db", "bandwidth_hz",
     "duration_s", "rise_time_s", "decay_time_s",
     "spectral_slope", "freq_modulation", "spectral_centroid_hz"]
)

# === UMAP defaults ===
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 2
UMAP_METRIC = "euclidean"

# === Montage parameters ===
N_PER_CLUSTER = 20
NCOLS = 5
NROWS = 4
NPERSEG = 256
NOVERLAP = 192
FREQ_MAX = 250
WINDOW_SEC = 10

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


def load_cluster_events(cluster_id):
    """Load events belonging to a specific cluster.

    Returns
    -------
    umap_df : pd.DataFrame
        UMAP coordinates for the target cluster.
    features_df : pd.DataFrame
        Full feature table for those events.
    cat_df : pd.DataFrame
        Event catalogue entries for those events.
    """
    umap_all = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")
    features_all = pd.read_parquet(DATA_DIR / "event_features.parquet")
    cat_all = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat_all["onset_utc"] = pd.to_datetime(cat_all["onset_utc"])
    cat_all["end_utc"] = pd.to_datetime(cat_all["end_utc"])

    # Filter to target cluster
    mask = umap_all["cluster_id"] == cluster_id
    umap_df = umap_all[mask].copy()
    event_ids = set(umap_df["event_id"])

    features_df = features_all[features_all["event_id"].isin(event_ids)].copy()
    cat_df = cat_all[cat_all["event_id"].isin(event_ids)].copy()

    print(f"Cluster {cluster_id}: {len(umap_df):,} events")
    return umap_df, features_df, cat_df


def prepare_features(df):
    """Standardize feature matrix for sub-clustering."""
    X_raw = df[FEATURE_COLS].values
    valid_mask = ~np.any(np.isnan(X_raw), axis=1)
    print(f"  Valid features: {valid_mask.sum():,} / {len(df):,} "
          f"({100*valid_mask.mean():.1f}%)")

    X_valid = X_raw[valid_mask]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    return X_scaled, valid_mask, scaler


def run_umap(X, min_dist=0.005, seed=42):
    """Run UMAP dimensionality reduction."""
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=min_dist,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=seed,
    )
    embedding = reducer.fit_transform(X)
    return embedding


def run_hdbscan(embedding, min_cluster_size, cluster_selection_method="eom"):
    """Run HDBSCAN clustering on UMAP embedding."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(embedding)
    return labels, clusterer


def stability_analysis(X, n_seeds, min_dist, min_cluster_size,
                       cluster_selection_method):
    """Run multiple seeds to check clustering stability."""
    results = []
    for seed in range(n_seeds):
        print(f"    Seed {seed+1}/{n_seeds}...")
        emb = run_umap(X, min_dist=min_dist, seed=seed)
        labels, _ = run_hdbscan(emb, min_cluster_size, cluster_selection_method)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = (labels == -1).mean()

        non_noise = labels >= 0
        if non_noise.sum() > 1 and n_clusters > 1:
            sil = silhouette_score(emb[non_noise], labels[non_noise])
        else:
            sil = 0.0

        results.append({
            "seed": seed,
            "n_clusters": n_clusters,
            "noise_frac": round(noise_frac, 4),
            "silhouette": round(sil, 4),
        })
    return results


def plot_umap_subclusters(embedding, labels, parent_id, outpath):
    """UMAP scatter plot colored by sub-cluster."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_mask = labels == -1

    fig, ax = plt.subplots(figsize=(12, 10))

    if noise_mask.any():
        ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                   c="lightgray", s=1, alpha=0.3, label="noise", rasterized=True)

    cluster_ids = sorted(set(labels) - {-1})
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))

    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        n_in = mask.sum()
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(i % 20)], s=2, alpha=0.5,
                   label=f"{parent_id}_{cid} (n={n_in:,})", rasterized=True)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"Sub-clustering {parent_id} — {n_clusters} sub-clusters, "
                 f"{noise_mask.sum():,} noise points",
                 fontsize=14, fontweight="bold")

    if n_clusters <= 25:
        ax.legend(fontsize=7, markerscale=4, loc="best",
                  ncol=2 if n_clusters > 12 else 1)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_umap_features(embedding, df_valid, parent_id, outpath):
    """UMAP colored by individual features for interpretation."""
    features_to_plot = [
        ("peak_freq_hz", "Peak Frequency (Hz)"),
        ("duration_s", "Duration (s)"),
        ("spectral_slope", "Spectral Slope"),
        ("bandwidth_hz", "Bandwidth (Hz)"),
        ("spectral_centroid_hz", "Spectral Centroid (Hz)"),
        ("freq_modulation", "Freq Modulation (std)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, features_to_plot):
        vals = df_valid[col].values
        vmin, vmax = np.percentile(vals[np.isfinite(vals)], [2, 98])
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=vals, s=1, alpha=0.3, cmap="viridis",
                        vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        plt.colorbar(sc, ax=ax, shrink=0.8)

    fig.suptitle(f"Sub-clustering {parent_id} — UMAP Colored by Spectral Features",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


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

    pre_context = WINDOW_SEC * 0.3
    t_start = onset - timedelta(seconds=pre_context)
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
        "times": times, "freqs": freqs, "Sxx_dB": Sxx_dB,
        "ev_start": ev_start, "ev_end": ev_end,
    }


def select_nearest_centroid(cluster_df, n=N_PER_CLUSTER):
    """Select n events nearest to the UMAP centroid."""
    cx = cluster_df["umap_1_sub"].mean()
    cy = cluster_df["umap_2_sub"].mean()
    dist = np.sqrt((cluster_df["umap_1_sub"] - cx)**2 +
                   (cluster_df["umap_2_sub"] - cy)**2)
    cluster_df = cluster_df.copy()
    cluster_df["_centroid_dist"] = dist.values
    return cluster_df.nsmallest(min(n, len(cluster_df)), "_centroid_dist")


def plot_subcluster_montage(sub_id, events, snippets, cluster_size):
    """Plot a montage for one sub-cluster."""
    n = len(events)
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(NROWS, NCOLS, figure=fig,
                  hspace=0.4, wspace=0.2,
                  top=0.93, bottom=0.03, left=0.04, right=0.98)

    fig.suptitle(
        f"Sub-cluster {sub_id} — {cluster_size:,} events "
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

        mooring = ev["mooring"].upper()
        pf = ev.get("peak_freq_hz", 0)
        dur = ev["duration_s"]
        snr = ev.get("snr", 0)
        time_str = ev["onset_utc"].strftime("%m-%d %H:%M")
        ax.set_title(
            f"{mooring} {pf:.0f}Hz {dur:.1f}s SNR={snr:.1f}\n{time_str}",
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

    outpath = FIG_DIR / f"subcluster_montage_{sub_id}.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    return outpath


def main():
    parser = argparse.ArgumentParser(
        description="Sub-cluster a mega-cluster to expose internal structure")
    parser.add_argument("cluster_id", type=str,
                        help="Parent cluster to sub-cluster (e.g., mid_3, low_2)")
    parser.add_argument("--min-dist", type=float, default=0.005,
                        help="UMAP min_dist (tighter than parent; default: 0.005)")
    parser.add_argument("--min-cluster-size", type=int, default=200,
                        help="HDBSCAN min_cluster_size (default: 200)")
    parser.add_argument("--selection", type=str, default="eom",
                        choices=["eom", "leaf"],
                        help="HDBSCAN cluster selection method (default: eom)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds for stability analysis")
    parser.add_argument("--skip-stability", action="store_true",
                        help="Skip stability analysis")
    parser.add_argument("--skip-montages", action="store_true",
                        help="Skip montage generation (faster)")
    args = parser.parse_args()

    parent_id = args.cluster_id

    print("=" * 60)
    print(f"BRAVOSEIS Sub-Clustering — {parent_id}")
    print(f"  UMAP min_dist={args.min_dist}")
    print(f"  HDBSCAN min_cluster_size={args.min_cluster_size}, "
          f"selection={args.selection}")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load events for this cluster
    umap_df, features_df, cat_df = load_cluster_events(parent_id)

    # Prepare features (re-standardize for just this subset)
    X, valid_mask, scaler = prepare_features(features_df)
    features_valid = features_df[valid_mask].reset_index(drop=True)

    # Stability analysis
    if not args.skip_stability:
        print(f"\n  Stability analysis ({args.n_seeds} seeds)...")
        stability = stability_analysis(
            X, args.n_seeds, args.min_dist,
            args.min_cluster_size, args.selection
        )
        for r in stability:
            print(f"    Seed {r['seed']}: {r['n_clusters']} clusters, "
                  f"{r['noise_frac']*100:.1f}% noise, "
                  f"sil={r['silhouette']:.3f}")

        stab_df = pd.DataFrame(stability)
        stab_df["parent_cluster"] = parent_id
        stab_path = DATA_DIR / f"subcluster_stability_{parent_id}.csv"
        stab_df.to_csv(stab_path, index=False)
        print(f"  Saved stability: {stab_path}")

    # Primary run (seed=42)
    print(f"\n  Primary UMAP + HDBSCAN (seed=42)...")
    embedding = run_umap(X, min_dist=args.min_dist, seed=42)
    labels, clusterer = run_hdbscan(embedding, args.min_cluster_size,
                                     args.selection)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()
    non_noise = labels >= 0

    print(f"  Results: {n_clusters} sub-clusters, "
          f"{(labels == -1).sum():,} noise ({noise_frac*100:.1f}%)")

    if non_noise.sum() > 1 and n_clusters > 1:
        sil = silhouette_score(embedding[non_noise], labels[non_noise])
        print(f"  Silhouette: {sil:.3f}")

    # Build sub-cluster IDs
    sub_cluster_ids = [
        f"{parent_id}_{cid}" if cid >= 0 else f"{parent_id}_noise"
        for cid in labels
    ]

    # Print cluster sizes
    print(f"\n  Sub-cluster sizes:")
    for cid in sorted(set(labels)):
        n = (labels == cid).sum()
        label = f"{parent_id}_{cid}" if cid >= 0 else f"{parent_id}_noise"
        print(f"    {label}: {n:,}")

    # Build result dataframe
    result_df = features_valid[["event_id"]].copy()
    result_df["parent_cluster"] = parent_id
    result_df["umap_1_sub"] = embedding[:, 0]
    result_df["umap_2_sub"] = embedding[:, 1]
    result_df["subcluster_numeric"] = labels
    result_df["subcluster_id"] = sub_cluster_ids

    outpath = DATA_DIR / f"subclusters_{parent_id}.parquet"
    result_df.to_parquet(outpath, index=False)
    print(f"\n  Saved: {outpath}")

    # UMAP scatter plot
    plot_umap_subclusters(
        embedding, labels, parent_id,
        FIG_DIR / f"subcluster_{parent_id}_map.png"
    )

    # UMAP feature maps
    plot_umap_features(
        embedding, features_valid, parent_id,
        FIG_DIR / f"subcluster_{parent_id}_features.png"
    )

    # Generate montages for each sub-cluster
    if not args.skip_montages:
        print(f"\n  Generating montages...")

        # Merge with catalogue for snippet extraction
        merged = result_df.merge(cat_df, on="event_id")

        all_sub_ids = sorted(set(sub_cluster_ids), key=str)
        for sub_id in all_sub_ids:
            sub_df = merged[merged["subcluster_id"] == sub_id]
            sub_size = len(sub_df)

            is_noise = sub_id.endswith("_noise")
            if is_noise:
                rng = np.random.default_rng(42)
                n = min(N_PER_CLUSTER, len(sub_df))
                idx = rng.choice(len(sub_df), size=n, replace=False)
                selected = sub_df.iloc[idx]
            else:
                selected = select_nearest_centroid(sub_df, n=N_PER_CLUSTER)

            snippets = []
            for _, ev in selected.iterrows():
                snip = extract_snippet(ev)
                snippets.append(snip)

            n_ok = sum(1 for s in snippets if s is not None)
            print(f"    {sub_id}: {sub_size:,} events, "
                  f"{n_ok}/{len(selected)} snippets")

            montage_path = plot_subcluster_montage(
                sub_id, selected.reset_index(drop=True), snippets, sub_size
            )
            print(f"      Saved: {montage_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Sub-clustering of {parent_id} complete.")
    print(f"  {n_clusters} sub-clusters + "
          f"{(labels == -1).sum():,} noise points")
    print(f"  Review montages in {FIG_DIR}/")
    print(f"  Sub-cluster data: {outpath}")


if __name__ == "__main__":
    main()
