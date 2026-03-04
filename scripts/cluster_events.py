#!/usr/bin/env python3
"""
cluster_events.py — Per-band UMAP + HDBSCAN clustering of acoustic events.

Clusters each detection band (low, mid, high) independently, removing the
dominant frequency-regime axis that caused the initial all-band pass to
collapse into a single mega-cluster (99% of events). Per-band clustering
exposes within-band structure: T-phases vs. local earthquakes in the low
band, fin whale calls vs. background in the mid band, tonal whale calls
vs. ice cracking in the high band.

Usage:
    uv run python cluster_events.py
    uv run python cluster_events.py --min-cluster-size 100
    uv run python cluster_events.py --n-seeds 5
    uv run python cluster_events.py --band low   # single band only

Spec: specs/002-event-discrimination/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import hdbscan

# === Paths ===
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "clustering"

# === Feature columns (must match extract_features.py) ===
BAND_POWER_COLS = [f"band_power_{i}" for i in range(10)]
FEATURE_COLS = (
    BAND_POWER_COLS +
    ["peak_freq_hz", "peak_power_db", "bandwidth_hz",
     "duration_s", "rise_time_s", "decay_time_s",
     "spectral_slope", "freq_modulation", "spectral_centroid_hz"]
)

# === Detection bands ===
BANDS = ["low", "mid", "high"]

# === Per-band tuning ===
# Low: "leaf" selection to force finer splitting of the stubborn mega-cluster
# Mid: min_dist=0.01 worked well — increase min_cluster_size to absorb tiny splinters
# High: min_dist=0.01 was too aggressive (90 clusters, 30% noise) — back off
BAND_PARAMS = {
    "low": {
        "umap_min_dist": 0.01,
        "min_cluster_size": 500,
        "cluster_selection_method": "eom",
    },
    "mid": {
        "umap_min_dist": 0.01,
        "min_cluster_size": 100,
        "cluster_selection_method": "eom",
    },
    "high": {
        "umap_min_dist": 0.03,
        "min_cluster_size": 100,
        "cluster_selection_method": "eom",
    },
}

# === UMAP defaults ===
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 2
UMAP_METRIC = "euclidean"

# === HDBSCAN defaults ===
HDBSCAN_MIN_SAMPLES = None  # let HDBSCAN choose

# === Colors ===
BAND_COLORS = {"low": "#E69F00", "mid": "#56B4E9", "high": "#009E73"}


def load_features():
    """Load feature table."""
    path = DATA_DIR / "event_features.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} events from {path}")
    return df


def prepare_features(df):
    """Extract and standardize feature matrix.

    Returns
    -------
    X : np.ndarray (n_events, n_features)
        Standardized feature matrix.
    valid_mask : np.ndarray (n_events,) bool
        Which rows have complete features.
    scaler : StandardScaler
        Fitted scaler for inverse transforms.
    """
    X_raw = df[FEATURE_COLS].values
    valid_mask = ~np.any(np.isnan(X_raw), axis=1)

    print(f"  Valid features: {valid_mask.sum():,} / {len(df):,} "
          f"({100*valid_mask.mean():.1f}%)")

    X_valid = X_raw[valid_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    return X_scaled, valid_mask, scaler


def run_umap(X, min_dist=0.01, seed=42):
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
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(embedding)
    return labels, clusterer


def stability_analysis(X, n_seeds=5, min_cluster_size=50,
                       min_dist=0.01, cluster_selection_method="eom"):
    """Run UMAP + HDBSCAN with different seeds to check stability.

    Returns
    -------
    results : list of dict
        Each with 'seed', 'n_clusters', 'noise_frac', 'silhouette'.
    """
    results = []
    for seed in range(n_seeds):
        print(f"    Seed {seed+1}/{n_seeds}...")
        emb = run_umap(X, min_dist=min_dist, seed=seed)
        labels, _ = run_hdbscan(emb, min_cluster_size, cluster_selection_method)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = (labels == -1).mean()

        # Silhouette on non-noise points
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


def plot_umap_clusters(embedding, labels, band, outpath):
    """UMAP scatter plot colored by cluster for one band."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_mask = labels == -1

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot noise points first (gray, small)
    if noise_mask.any():
        ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                   c="lightgray", s=1, alpha=0.3, label="noise", rasterized=True)

    # Plot clustered points
    cluster_ids = sorted(set(labels) - {-1})
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))

    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        n_in = mask.sum()
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(i % 20)], s=2, alpha=0.5,
                   label=f"{band}_{cid} (n={n_in:,})", rasterized=True)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"{band.upper()} band — {n_clusters} clusters, "
                 f"{noise_mask.sum():,} noise points",
                 fontsize=14, fontweight="bold")

    # Legend (only if manageable number of clusters)
    if n_clusters <= 25:
        ax.legend(fontsize=7, markerscale=4, loc="best",
                  ncol=2 if n_clusters > 12 else 1)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_umap_features(embedding, df_valid, band, outpath):
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
        # Clip outliers for better coloring
        vmin, vmax = np.percentile(vals[np.isfinite(vals)], [2, 98])
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=vals, s=1, alpha=0.3, cmap="viridis",
                        vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        plt.colorbar(sc, ax=ax, shrink=0.8)

    fig.suptitle(f"{band.upper()} Band — UMAP Colored by Spectral Features",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def cluster_one_band(df_band, band, args):
    """Run the full clustering pipeline for one detection band.

    Returns
    -------
    result_df : pd.DataFrame
        UMAP coordinates and cluster labels for this band.
    """
    # Get per-band parameters
    bp = BAND_PARAMS[band]
    min_dist = bp["umap_min_dist"]
    min_cluster_size = bp["min_cluster_size"]
    selection_method = bp["cluster_selection_method"]

    print(f"\n{'—' * 50}")
    print(f"  Band: {band.upper()} ({len(df_band):,} events)")
    print(f"  UMAP min_dist={min_dist}, HDBSCAN min_cluster_size="
          f"{min_cluster_size}, selection={selection_method}")
    print(f"{'—' * 50}")

    # Prepare features (standardize per band)
    X, valid_mask, scaler = prepare_features(df_band)
    df_valid = df_band[valid_mask].reset_index(drop=True)

    # Stability analysis
    if not args.skip_stability:
        print(f"  Stability analysis ({args.n_seeds} seeds)...")
        stability = stability_analysis(
            X, n_seeds=args.n_seeds,
            min_cluster_size=min_cluster_size,
            min_dist=min_dist,
            cluster_selection_method=selection_method,
        )
        for r in stability:
            print(f"    Seed {r['seed']}: {r['n_clusters']} clusters, "
                  f"{r['noise_frac']*100:.1f}% noise, "
                  f"sil={r['silhouette']:.3f}")

        stab_df = pd.DataFrame(stability)
        stab_df["band"] = band
        stab_path = DATA_DIR / f"cluster_stability_{band}.csv"
        stab_df.to_csv(stab_path, index=False)

    # Primary run (seed=42)
    print(f"  Primary UMAP + HDBSCAN (seed=42)...")
    embedding = run_umap(X, min_dist=min_dist, seed=42)
    labels, clusterer = run_hdbscan(embedding, min_cluster_size,
                                     selection_method)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()
    non_noise = labels >= 0

    print(f"  Results: {n_clusters} clusters, "
          f"{(labels == -1).sum():,} noise ({noise_frac*100:.1f}%)")

    if non_noise.sum() > 1 and n_clusters > 1:
        sil = silhouette_score(embedding[non_noise], labels[non_noise])
        print(f"  Silhouette: {sil:.3f}")

    # Cluster sizes
    for cid in sorted(set(labels) - {-1}):
        n = (labels == cid).sum()
        print(f"    {band}_{cid}: {n:,}")

    # Band-prefix cluster IDs (noise stays -1)
    cluster_ids_str = [
        f"{band}_{cid}" if cid >= 0 else f"{band}_noise"
        for cid in labels
    ]

    # Build result dataframe
    result_df = df_valid[["event_id", "mooring", "detection_band"]].copy()
    result_df["umap_1"] = embedding[:, 0]
    result_df["umap_2"] = embedding[:, 1]
    result_df["cluster_id_numeric"] = labels
    result_df["cluster_id"] = cluster_ids_str

    # Generate figures
    plot_umap_clusters(
        embedding, labels, band,
        FIG_DIR / f"umap_{band}_cluster_map.png"
    )
    plot_umap_features(
        embedding, df_valid, band,
        FIG_DIR / f"umap_{band}_feature_maps.png"
    )

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Per-band UMAP + HDBSCAN clustering of acoustic events")
    parser.add_argument("--min-cluster-size", type=int, default=None,
                        help="Override HDBSCAN min cluster size for all bands "
                             "(default: use per-band settings)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds for stability analysis")
    parser.add_argument("--skip-stability", action="store_true",
                        help="Skip stability analysis (faster)")
    parser.add_argument("--band", type=str, default=None,
                        choices=BANDS,
                        help="Cluster a single band (default: all)")
    args = parser.parse_args()

    bands_to_run = [args.band] if args.band else BANDS

    # Apply CLI override if given
    if args.min_cluster_size is not None:
        for b in BAND_PARAMS:
            BAND_PARAMS[b]["min_cluster_size"] = args.min_cluster_size

    print("=" * 60)
    print("BRAVOSEIS Per-Band Event Clustering — Phase 1b")
    print(f"  Bands: {', '.join(bands_to_run)}")
    for b in bands_to_run:
        bp = BAND_PARAMS[b]
        print(f"    {b}: min_dist={bp['umap_min_dist']}, "
              f"min_cluster_size={bp['min_cluster_size']}, "
              f"selection={bp['cluster_selection_method']}")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load features
    df = load_features()

    # Cluster each band independently
    all_results = []
    for band in bands_to_run:
        df_band = df[df["detection_band"] == band].copy()
        if len(df_band) == 0:
            print(f"\nSkipping {band} — no events")
            continue
        result = cluster_one_band(df_band, band, args)
        all_results.append(result)

    # Combine results across bands
    combined = pd.concat(all_results, ignore_index=True)

    outpath = DATA_DIR / "umap_coordinates.parquet"
    combined.to_parquet(outpath, index=False)
    print(f"\nSaved combined results: {outpath}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Per-band clustering summary:")
    for band in bands_to_run:
        band_df = combined[combined["detection_band"] == band]
        n_clusters = band_df["cluster_id"].nunique()
        noise = (band_df["cluster_id_numeric"] == -1).sum()
        print(f"  {band.upper():5s}: {len(band_df):,} events, "
              f"{n_clusters} clusters (incl. noise), "
              f"{noise:,} noise points")

    print(f"\nDone. Review UMAP figures and proceed to montage generation.")


if __name__ == "__main__":
    main()
