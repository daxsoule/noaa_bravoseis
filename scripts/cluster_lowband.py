#!/usr/bin/env python3
"""
cluster_lowband.py — UMAP + HDBSCAN clustering of lowband (1–14 Hz) events.

Filters out whale-contaminated events (catalogue peak_freq > 14 Hz) before
clustering. Uses the 6-band lowband features from extract_features_lowband.py.

Usage:
    uv run python cluster_lowband.py
    uv run python cluster_lowband.py --min-cluster-size 300
    uv run python cluster_lowband.py --skip-stability
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
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "clustering" / "phase3_lowband"

# === Feature columns (from extract_features_lowband.py — 6 bands) ===
BAND_POWER_COLS = [f"band_power_{i}" for i in range(6)]
FEATURE_COLS = (
    BAND_POWER_COLS +
    ["peak_freq_hz", "peak_power_db", "bandwidth_hz",
     "duration_s", "rise_time_s", "decay_time_s",
     "spectral_slope", "freq_modulation", "spectral_centroid_hz"]
)

# === Whale filter ===
PEAK_FREQ_MAX = 17.0  # Hz — remove events with catalogue peak_freq above this
# Raised from 14.0 to 17.0: cross-validation showed 14-17 Hz events are
# genuine T-phases (matching accepted population in duration, slope, SNR),
# not whale contamination. Hard 14 Hz cutoff was losing ~4K real seismic events.

# === UMAP parameters ===
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 2
UMAP_METRIC = "euclidean"
UMAP_MIN_DIST = 0.01

# === HDBSCAN parameters ===
MIN_CLUSTER_SIZE = 500
CLUSTER_SELECTION_METHOD = "eom"
HDBSCAN_MIN_SAMPLES = None


def load_and_filter():
    """Load lowband features and filter out whale-contaminated events."""
    features = pd.read_parquet(DATA_DIR / "event_features_lowband.parquet")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")

    n_before = len(features)

    # Get catalogue peak_freq for filtering
    cat_freq = cat[["event_id", "peak_freq_hz"]].rename(
        columns={"peak_freq_hz": "cat_peak_freq_hz"}
    )
    features = features.merge(cat_freq, on="event_id", how="left")

    # Filter: keep only events with catalogue peak_freq <= 14 Hz
    features = features[features["cat_peak_freq_hz"] <= PEAK_FREQ_MAX].copy()
    features = features.drop(columns=["cat_peak_freq_hz"])

    n_after = len(features)
    print(f"Loaded {n_before:,} lowband events")
    print(f"Filtered to {n_after:,} events (removed {n_before - n_after:,} "
          f"with catalogue peak_freq > {PEAK_FREQ_MAX} Hz)")

    return features


def prepare_features(df):
    """Standardize feature matrix."""
    X_raw = df[FEATURE_COLS].values
    valid_mask = ~np.any(np.isnan(X_raw), axis=1)

    print(f"  Valid features: {valid_mask.sum():,} / {len(df):,} "
          f"({100 * valid_mask.mean():.1f}%)")

    X_valid = X_raw[valid_mask]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    return X_scaled, valid_mask, scaler


def run_umap(X, min_dist=UMAP_MIN_DIST, seed=42):
    """Run UMAP dimensionality reduction."""
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=min_dist,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=seed,
    )
    return reducer.fit_transform(X)


def run_hdbscan(embedding, min_cluster_size=MIN_CLUSTER_SIZE):
    """Run HDBSCAN clustering."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
    )
    labels = clusterer.fit_predict(embedding)
    return labels, clusterer


def stability_analysis(X, n_seeds=5, min_cluster_size=MIN_CLUSTER_SIZE):
    """Run UMAP+HDBSCAN with different seeds to check stability."""
    results = []
    for seed in range(n_seeds):
        print(f"    Seed {seed + 1}/{n_seeds}...")
        emb = run_umap(X, seed=seed)
        labels, _ = run_hdbscan(emb, min_cluster_size)

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


def plot_umap_clusters(embedding, labels, outpath):
    """UMAP scatter colored by cluster."""
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
                   label=f"lowband_{cid} (n={n_in:,})", rasterized=True)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(
        f"Lowband (1–14 Hz, whale-filtered) — {n_clusters} clusters, "
        f"{noise_mask.sum():,} noise",
        fontsize=14, fontweight="bold"
    )

    if n_clusters <= 25:
        ax.legend(fontsize=7, markerscale=4, loc="best",
                  ncol=2 if n_clusters > 12 else 1)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_umap_features(embedding, df_valid, outpath):
    """UMAP colored by features."""
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

    fig.suptitle("Lowband (1–14 Hz, whale-filtered) — UMAP by Feature",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="UMAP+HDBSCAN clustering of lowband events (whale-filtered)")
    parser.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--skip-stability", action="store_true")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load and filter
    df = load_and_filter()

    # Prepare features
    print("\nPreparing features...")
    X, valid_mask, scaler = prepare_features(df)
    df_valid = df[valid_mask].reset_index(drop=True)

    # Stability analysis
    if not args.skip_stability:
        print(f"\nStability analysis ({args.n_seeds} seeds)...")
        stability = stability_analysis(X, n_seeds=args.n_seeds,
                                       min_cluster_size=args.min_cluster_size)
        for r in stability:
            print(f"  Seed {r['seed']}: {r['n_clusters']} clusters, "
                  f"{r['noise_frac'] * 100:.1f}% noise, sil={r['silhouette']:.3f}")

        stab_df = pd.DataFrame(stability)
        stab_path = DATA_DIR / "cluster_stability_lowband.csv"
        stab_df.to_csv(stab_path, index=False)
        print(f"  Saved: {stab_path}")

    # Primary run
    print(f"\nPrimary UMAP + HDBSCAN (seed=42, min_cluster_size={args.min_cluster_size})...")
    embedding = run_umap(X, seed=42)
    labels, clusterer = run_hdbscan(embedding, args.min_cluster_size)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = (labels == -1).mean()
    non_noise = labels >= 0

    print(f"  Results: {n_clusters} clusters, "
          f"{(labels == -1).sum():,} noise ({noise_frac * 100:.1f}%)")

    if non_noise.sum() > 1 and n_clusters > 1:
        sil = silhouette_score(embedding[non_noise], labels[non_noise])
        print(f"  Silhouette: {sil:.3f}")

    # Cluster sizes
    for cid in sorted(set(labels) - {-1}):
        n = (labels == cid).sum()
        print(f"    lowband_{cid}: {n:,}")

    # Build cluster IDs
    cluster_ids_str = [
        f"lowband_{cid}" if cid >= 0 else "lowband_noise"
        for cid in labels
    ]

    # Result dataframe
    result_df = df_valid[["event_id", "mooring", "detection_band"]].copy()
    result_df["umap_1"] = embedding[:, 0]
    result_df["umap_2"] = embedding[:, 1]
    result_df["cluster_id_numeric"] = labels
    result_df["cluster_id"] = cluster_ids_str

    # Save
    outpath = DATA_DIR / "umap_coordinates_lowband.parquet"
    result_df.to_parquet(outpath, index=False)
    print(f"\nSaved: {outpath} ({len(result_df):,} events)")

    # Figures
    plot_umap_clusters(embedding, labels,
                       FIG_DIR / "umap_lowband_cluster_map.png")
    plot_umap_features(embedding, df_valid,
                       FIG_DIR / "umap_lowband_feature_maps.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
