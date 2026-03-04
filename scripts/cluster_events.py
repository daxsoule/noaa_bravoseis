#!/usr/bin/env python3
"""
cluster_events.py — UMAP + HDBSCAN clustering of acoustic events.

Loads extracted spectral features, standardizes them, projects to 2D with
UMAP, clusters with HDBSCAN, and generates visualization figures.

Usage:
    uv run python cluster_events.py
    uv run python cluster_events.py --min-cluster-size 100
    uv run python cluster_events.py --n-seeds 5

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
FIG_DIR = OUTPUT_DIR / "figures" / "journal"

# === Feature columns (must match extract_features.py) ===
BAND_POWER_COLS = [f"band_power_{i}" for i in range(10)]
FEATURE_COLS = (
    BAND_POWER_COLS +
    ["peak_freq_hz", "peak_power_db", "bandwidth_hz",
     "duration_s", "rise_time_s", "decay_time_s",
     "spectral_slope", "freq_modulation", "spectral_centroid_hz"]
)

# === UMAP defaults ===
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2
UMAP_METRIC = "euclidean"

# === HDBSCAN defaults ===
HDBSCAN_MIN_CLUSTER_SIZE = 50
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

    print(f"Valid features: {valid_mask.sum():,} / {len(df):,} "
          f"({100*valid_mask.mean():.1f}%)")

    X_valid = X_raw[valid_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    return X_scaled, valid_mask, scaler


def run_umap(X, seed=42):
    """Run UMAP dimensionality reduction."""
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=seed,
    )
    embedding = reducer.fit_transform(X)
    return embedding


def run_hdbscan(embedding, min_cluster_size):
    """Run HDBSCAN clustering on UMAP embedding."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embedding)
    return labels, clusterer


def stability_analysis(X, n_seeds=5, min_cluster_size=50):
    """Run UMAP + HDBSCAN with different seeds to check stability.

    Returns
    -------
    results : list of dict
        Each with 'seed', 'n_clusters', 'noise_frac', 'silhouette'.
    """
    results = []
    for seed in range(n_seeds):
        print(f"  Stability run {seed+1}/{n_seeds} (seed={seed})...")
        emb = run_umap(X, seed=seed)
        labels, _ = run_hdbscan(emb, min_cluster_size)

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


def plot_umap_clusters(embedding, labels, df_valid, outpath):
    """UMAP scatter plot colored by cluster."""
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
                   label=f"C{cid} (n={n_in:,})", rasterized=True)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"UMAP + HDBSCAN Clustering — {n_clusters} clusters, "
                 f"{noise_mask.sum():,} noise points",
                 fontsize=14, fontweight="bold")

    # Legend (only if manageable number of clusters)
    if n_clusters <= 25:
        ax.legend(fontsize=7, markerscale=4, loc="best",
                  ncol=2 if n_clusters > 12 else 1)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_umap_features(embedding, df_valid, outpath):
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

    fig.suptitle("UMAP Colored by Spectral Features",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_umap_bands(embedding, df_valid, outpath):
    """UMAP colored by detection band."""
    fig, ax = plt.subplots(figsize=(12, 10))

    for band, color in BAND_COLORS.items():
        mask = df_valid["detection_band"].values == band
        if mask.any():
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, s=1, alpha=0.3,
                       label=f"{band} (n={mask.sum():,})", rasterized=True)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("UMAP Colored by Detection Band",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, markerscale=6)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="UMAP + HDBSCAN clustering of acoustic events")
    parser.add_argument("--min-cluster-size", type=int,
                        default=HDBSCAN_MIN_CLUSTER_SIZE,
                        help=f"HDBSCAN min cluster size (default: "
                             f"{HDBSCAN_MIN_CLUSTER_SIZE})")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds for stability analysis")
    parser.add_argument("--skip-stability", action="store_true",
                        help="Skip stability analysis (faster)")
    args = parser.parse_args()

    print("=" * 60)
    print("BRAVOSEIS Event Clustering — Phase 1b")
    print(f"  UMAP: n_neighbors={UMAP_N_NEIGHBORS}, "
          f"min_dist={UMAP_MIN_DIST}")
    print(f"  HDBSCAN: min_cluster_size={args.min_cluster_size}")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load features
    df = load_features()

    # Prepare feature matrix
    X, valid_mask, scaler = prepare_features(df)
    df_valid = df[valid_mask].reset_index(drop=True)

    # Stability analysis
    if not args.skip_stability:
        print(f"\nRunning stability analysis ({args.n_seeds} seeds)...")
        stability = stability_analysis(
            X, n_seeds=args.n_seeds,
            min_cluster_size=args.min_cluster_size
        )
        print("\nStability results:")
        for r in stability:
            print(f"  Seed {r['seed']}: {r['n_clusters']} clusters, "
                  f"{r['noise_frac']*100:.1f}% noise, "
                  f"silhouette={r['silhouette']:.3f}")

        # Save stability results
        stab_df = pd.DataFrame(stability)
        stab_path = DATA_DIR / "cluster_stability.csv"
        stab_df.to_csv(stab_path, index=False)
        print(f"Saved: {stab_path}")

    # Primary run (seed=42)
    print("\nRunning primary UMAP + HDBSCAN (seed=42)...")
    embedding = run_umap(X, seed=42)
    labels, clusterer = run_hdbscan(embedding, args.min_cluster_size)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()
    non_noise = labels >= 0

    print(f"\nClustering results:")
    print(f"  Clusters:    {n_clusters}")
    print(f"  Noise:       {(labels == -1).sum():,} ({noise_frac*100:.1f}%)")
    print(f"  Clustered:   {non_noise.sum():,} ({100*(1-noise_frac):.1f}%)")

    if non_noise.sum() > 1 and n_clusters > 1:
        sil = silhouette_score(embedding[non_noise], labels[non_noise])
        print(f"  Silhouette:  {sil:.3f}")

    # Cluster size distribution
    print(f"\nCluster sizes:")
    for cid in sorted(set(labels) - {-1}):
        n = (labels == cid).sum()
        print(f"  Cluster {cid:3d}: {n:6,}")

    # Save UMAP coordinates + cluster labels
    umap_df = df_valid[["event_id", "mooring", "detection_band"]].copy()
    umap_df["umap_1"] = embedding[:, 0]
    umap_df["umap_2"] = embedding[:, 1]
    umap_df["cluster_id"] = labels

    outpath = DATA_DIR / "umap_coordinates.parquet"
    umap_df.to_parquet(outpath, index=False)
    print(f"\nSaved: {outpath}")

    # Generate figures
    print("\nGenerating figures...")
    plot_umap_clusters(
        embedding, labels, df_valid,
        FIG_DIR / "umap_cluster_map.png"
    )
    plot_umap_features(
        embedding, df_valid,
        FIG_DIR / "umap_feature_maps.png"
    )
    plot_umap_bands(
        embedding, df_valid,
        FIG_DIR / "umap_band_map.png"
    )

    print(f"\n{'=' * 60}")
    print("Done. Review the UMAP figures and proceed to montage generation.")


if __name__ == "__main__":
    main()
