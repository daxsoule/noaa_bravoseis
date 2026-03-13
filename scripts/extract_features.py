#!/usr/bin/env python3
"""
extract_features.py — Extract spectral features from detected events.

For each event in the catalogue, loads the corresponding DAT file segment,
computes a spectrogram patch, and extracts ~20 handcrafted features for
unsupervised clustering (UMAP + HDBSCAN).

Features extracted:
  - 10 band powers (25 Hz bands: 0–25, 25–50, ..., 225–250 Hz)
  - Duration, rise time, decay time
  - Peak frequency, peak power (dB), bandwidth (90% energy)
  - Spectral slope (log-power vs log-frequency)
  - Frequency modulation metric

Usage:
    uv run python extract_features.py                  # full catalogue
    uv run python extract_features.py --mooring m1     # single mooring
    uv run python extract_features.py --file 00001282  # single file

Spec: specs/002-event-discrimination/
"""

import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from scipy.signal import spectrogram

from read_dat import read_dat, MOORINGS, SAMPLE_RATE, get_data_dir

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
PATCH_DIR = DATA_DIR / "event_patches"
CHECKPOINT_DIR = DATA_DIR / "feature_checkpoints"

# === Spectrogram parameters (matching spec 001) ===
NPERSEG = 1024
NOVERLAP = 512
FREQ_MAX = 250

# === Feature extraction parameters ===
PRE_PICK_SEC = 5.0   # seconds before onset pick
POST_PICK_SEC = 10.0  # seconds after onset pick
WINDOW_SEC = PRE_PICK_SEC + POST_PICK_SEC  # fixed 15s window
N_FREQ_BANDS = 10  # 25 Hz bands from 0–250 Hz
BAND_WIDTH_HZ = FREQ_MAX / N_FREQ_BANDS  # 25 Hz each


def compute_spectrogram_patch(data, onset_samp, end_samp, file_nsamples):
    """Extract spectrogram patch using fixed 15s window around onset pick.

    Window: onset - 5s to onset + 10s (fixed, regardless of event duration).

    Returns
    -------
    freqs : np.ndarray
    times : np.ndarray
    Sxx_dB : np.ndarray
        Spectrogram in dB (clipped to 0–250 Hz).
    onset_rel : float
        Event onset time relative to patch start (seconds).
    end_rel : float
        Event end time relative to patch start (seconds).
    """
    pre_samp = int(PRE_PICK_SEC * SAMPLE_RATE)
    post_samp = int(POST_PICK_SEC * SAMPLE_RATE)

    seg_start = onset_samp - pre_samp
    seg_end = onset_samp + post_samp

    if seg_start < 0 or seg_end > file_nsamples:
        return None

    segment = data[seg_start:seg_end]

    if len(segment) < NPERSEG:
        return None

    freqs, times, Sxx = spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
    )

    # Clip to frequency range
    freq_mask = freqs <= FREQ_MAX
    freqs = freqs[freq_mask]
    Sxx = Sxx[freq_mask, :]

    if Sxx.size == 0:
        return None

    Sxx_dB = 10 * np.log10(Sxx + 1e-20)

    onset_rel = PRE_PICK_SEC
    end_rel = PRE_PICK_SEC + (end_samp - onset_samp) / SAMPLE_RATE

    return freqs, times, Sxx_dB, Sxx, onset_rel, end_rel


def extract_features_from_patch(freqs, times, Sxx_dB, Sxx_linear,
                                onset_rel, end_rel):
    """Extract ~20 handcrafted features from a spectrogram patch.

    Returns dict of feature name → value.
    """
    features = {}

    # Event time mask
    t_mask = (times >= onset_rel) & (times <= end_rel)
    if not np.any(t_mask):
        t_mask = np.ones(len(times), dtype=bool)

    event_Sxx = Sxx_linear[:, t_mask]
    event_dB = Sxx_dB[:, t_mask]

    # --- Band powers (10 bands, 25 Hz each) ---
    for i in range(N_FREQ_BANDS):
        f_lo = i * BAND_WIDTH_HZ
        f_hi = (i + 1) * BAND_WIDTH_HZ
        band_mask = (freqs >= f_lo) & (freqs < f_hi)
        if np.any(band_mask):
            features[f"band_power_{i}"] = float(
                np.mean(event_dB[band_mask, :])
            )
        else:
            features[f"band_power_{i}"] = -200.0

    # --- Mean power spectrum across event ---
    mean_power = event_Sxx.mean(axis=1)  # linear, averaged over time
    mean_dB = event_dB.mean(axis=1)

    # --- Peak frequency ---
    peak_idx = np.argmax(mean_power)
    features["peak_freq_hz"] = float(freqs[peak_idx])

    # --- Peak power (dB) ---
    features["peak_power_db"] = float(event_dB.max())

    # --- Bandwidth (90% energy) ---
    total_power = mean_power.sum()
    if total_power > 0:
        cumsum = np.cumsum(mean_power)
        frac = cumsum / total_power
        low_idx = np.searchsorted(frac, 0.05)
        high_idx = np.searchsorted(frac, 0.95)
        features["bandwidth_hz"] = float(
            freqs[min(high_idx, len(freqs) - 1)] - freqs[max(low_idx, 0)]
        )
    else:
        features["bandwidth_hz"] = 0.0

    # --- Duration ---
    features["duration_s"] = end_rel - onset_rel

    # --- Rise time and decay time ---
    # Power envelope over time (summed across frequencies)
    time_envelope = event_Sxx.sum(axis=0)
    if len(time_envelope) > 1:
        event_times = times[t_mask]
        peak_t_idx = np.argmax(time_envelope)
        peak_t = event_times[peak_t_idx] if len(event_times) > 0 else onset_rel
        features["rise_time_s"] = float(peak_t - onset_rel)
        features["decay_time_s"] = float(end_rel - peak_t)
    else:
        features["rise_time_s"] = features["duration_s"] / 2
        features["decay_time_s"] = features["duration_s"] / 2

    # --- Spectral slope (log-power vs log-frequency) ---
    # Fit in log-log space, excluding DC (freq=0)
    valid_f = freqs > 0
    if np.sum(valid_f) > 2 and np.any(mean_power[valid_f] > 0):
        log_f = np.log10(freqs[valid_f])
        log_p = np.log10(mean_power[valid_f] + 1e-20)
        # Linear fit
        coeffs = np.polyfit(log_f, log_p, 1)
        features["spectral_slope"] = float(coeffs[0])
    else:
        features["spectral_slope"] = 0.0

    # --- Frequency modulation metric ---
    # Variance of instantaneous peak frequency over time
    if event_Sxx.shape[1] > 2:
        inst_peak_freq = freqs[np.argmax(event_Sxx, axis=0)]
        features["freq_modulation"] = float(np.std(inst_peak_freq))
    else:
        features["freq_modulation"] = 0.0

    # --- Spectral centroid ---
    if total_power > 0:
        features["spectral_centroid_hz"] = float(
            np.sum(freqs * mean_power) / total_power
        )
    else:
        features["spectral_centroid_hz"] = 0.0

    return features


def load_catalogue():
    """Load event catalogue."""
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])
    return cat


def _process_file_group(args):
    """Worker function: extract features for all events in one file.

    Returns list of dicts (one per event, with _event_id key).
    """
    mooring, file_num, events_data, data_root = args

    info = MOORINGS[mooring]
    mooring_dir = data_root / get_data_dir(info, data_root)
    dat_path = mooring_dir / f"{file_num:08d}.DAT"

    results = []
    if not dat_path.exists():
        for ev in events_data:
            results.append({"_event_id": ev["event_id"]})
        return results

    file_ts, data, _ = read_dat(dat_path)
    file_nsamples = len(data)

    for ev in events_data:
        onset_offset_s = (ev["onset_utc"] - file_ts).total_seconds()
        end_offset_s = (ev["end_utc"] - file_ts).total_seconds()
        onset_samp = int(onset_offset_s * SAMPLE_RATE)
        end_samp = int(end_offset_s * SAMPLE_RATE)

        if onset_samp < 0 or onset_samp >= file_nsamples:
            results.append({"_event_id": ev["event_id"]})
            continue

        end_samp = min(end_samp, file_nsamples)

        result = compute_spectrogram_patch(
            data, onset_samp, end_samp, file_nsamples
        )

        if result is None:
            results.append({"_event_id": ev["event_id"]})
            continue

        freqs, times, Sxx_dB, Sxx_linear, onset_rel, end_rel = result

        feats = extract_features_from_patch(
            freqs, times, Sxx_dB, Sxx_linear, onset_rel, end_rel
        )
        feats["_event_id"] = ev["event_id"]
        results.append(feats)

    return results


def process_mooring_features(cat, mooring, data_root, n_workers=1):
    """Extract features for all events in one mooring.

    Returns DataFrame with feature columns + event_id.
    """
    mcat = cat[cat["mooring"] == mooring].copy()
    if len(mcat) == 0:
        return pd.DataFrame()

    # Build work items grouped by file
    groups = mcat.groupby("file_number")
    work_items = []
    for file_num, group in groups:
        events_data = [
            {"event_id": row["event_id"],
             "onset_utc": row["onset_utc"],
             "end_utc": row["end_utc"]}
            for _, row in group.iterrows()
        ]
        work_items.append((mooring, file_num, events_data, data_root))

    n_groups = len(work_items)
    all_features = []

    if n_workers > 1 and n_groups > 1:
        completed = 0
        with mp.Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_process_file_group, work_items,
                                              chunksize=4):
                all_features.extend(result)
                completed += 1
                if completed % 100 == 0 or completed == n_groups:
                    print(f"    {mooring}: {completed}/{n_groups} files, "
                          f"{len(all_features)} events")
    else:
        for i, item in enumerate(work_items):
            result = _process_file_group(item)
            all_features.extend(result)
            if (i + 1) % 50 == 0 or (i + 1) == n_groups:
                print(f"    {mooring}: {i+1}/{n_groups} files, "
                      f"{len(all_features)} events")

    feat_df = pd.DataFrame(all_features)
    return feat_df


def process_catalogue(cat, mooring_filter=None, file_filter=None,
                      save_patches=True, data_root=None):
    """Extract features for all events, grouped by (mooring, file_number).

    Returns DataFrame with feature columns appended.
    Legacy interface for backward compatibility (single-threaded).
    """
    if data_root is None:
        data_root = DATA_ROOT
    if mooring_filter:
        cat = cat[cat["mooring"] == mooring_filter].copy()
    if file_filter:
        cat = cat[cat["file_number"] == int(file_filter)].copy()

    if len(cat) == 0:
        print("No events to process after filtering.")
        return cat, []

    # Group by (mooring, file_number)
    groups = cat.groupby(["mooring", "file_number"])
    n_groups = len(groups)
    n_events_done = 0
    n_processed = 0

    all_features = []
    all_patches = {}  # mooring -> list of (event_id, patch_data)

    for (mooring, file_num), group_idx in groups.groups.items():
        group = cat.loc[group_idx].sort_values("onset_utc")

        # Resolve file path
        info = MOORINGS[mooring]
        mooring_dir = data_root / get_data_dir(info, data_root)
        dat_path = mooring_dir / f"{file_num:08d}.DAT"

        if not dat_path.exists():
            # Fill with NaN features
            for idx in group.index:
                all_features.append({"_idx": idx})
            n_processed += 1
            continue

        # Read file once
        file_ts, data, _ = read_dat(dat_path)
        file_nsamples = len(data)

        for idx, ev in group.iterrows():
            # Compute sample positions
            onset_offset_s = (ev["onset_utc"] - file_ts).total_seconds()
            end_offset_s = (ev["end_utc"] - file_ts).total_seconds()
            onset_samp = int(onset_offset_s * SAMPLE_RATE)
            end_samp = int(end_offset_s * SAMPLE_RATE)

            if onset_samp < 0 or onset_samp >= file_nsamples:
                all_features.append({"_idx": idx})
                continue

            end_samp = min(end_samp, file_nsamples)

            result = compute_spectrogram_patch(
                data, onset_samp, end_samp, file_nsamples
            )

            if result is None:
                all_features.append({"_idx": idx})
                continue

            freqs, times, Sxx_dB, Sxx_linear, onset_rel, end_rel = result

            # Extract features
            feats = extract_features_from_patch(
                freqs, times, Sxx_dB, Sxx_linear, onset_rel, end_rel
            )
            feats["_idx"] = idx
            all_features.append(feats)

            # Save patch for CNN later
            if save_patches:
                if mooring not in all_patches:
                    all_patches[mooring] = []
                all_patches[mooring].append({
                    "event_id": ev["event_id"],
                    "Sxx_dB": Sxx_dB.astype(np.float32),
                    "freqs": freqs.astype(np.float32),
                    "times": times.astype(np.float32),
                    "onset_rel": onset_rel,
                    "end_rel": end_rel,
                })

        n_processed += 1
        n_events_done += len(group)
        if n_processed % 50 == 0 or n_processed == n_groups:
            print(f"  {n_processed}/{n_groups} file groups, "
                  f"{n_events_done}/{len(cat)} events")

        # Save patches periodically per mooring to avoid memory buildup
        if save_patches:
            for mk in list(all_patches.keys()):
                if len(all_patches[mk]) >= 5000:
                    _save_patches(mk, all_patches[mk])
                    all_patches[mk] = []

    # Save remaining patches
    if save_patches:
        for mk, patches in all_patches.items():
            if patches:
                _save_patches(mk, patches)

    # Build feature DataFrame
    feat_df = pd.DataFrame(all_features)
    if "_idx" in feat_df.columns:
        feat_df = feat_df.set_index("_idx")

    return cat, feat_df


def _save_patches(mooring, patches):
    """Save accumulated patches for a mooring as NPZ."""
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing file to append
    outpath = PATCH_DIR / f"patches_{mooring}.npz"
    existing = {}
    if outpath.exists():
        with np.load(outpath, allow_pickle=True) as f:
            existing = dict(f)

    for p in patches:
        key = p["event_id"]
        existing[key] = {
            "Sxx_dB": p["Sxx_dB"],
            "freqs": p["freqs"],
            "times": p["times"],
            "onset_rel": p["onset_rel"],
            "end_rel": p["end_rel"],
        }

    # Save as dict of arrays
    np.savez_compressed(
        outpath,
        **{k: v for k, v in existing.items()}
    )


def print_summary(feat_df):
    """Print feature extraction summary."""
    # Count non-NaN features
    feature_cols = [c for c in feat_df.columns if not c.startswith("_")]
    n_complete = feat_df[feature_cols].dropna(how="any").shape[0]
    n_total = len(feat_df)

    print(f"\n{'=' * 60}")
    print("Feature Extraction Summary")
    print(f"{'=' * 60}")
    print(f"Total events:        {n_total:,}")
    print(f"Features extracted:  {n_complete:,} ({100*n_complete/n_total:.1f}%)")
    print(f"Missing features:    {n_total - n_complete:,}")
    print(f"Feature dimensions:  {len(feature_cols)}")

    if n_complete > 0:
        print(f"\nFeature ranges (non-NaN events):")
        for col in feature_cols:
            vals = feat_df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:25s}: median={vals.median():8.2f}, "
                      f"IQR=[{vals.quantile(0.25):8.2f}, "
                      f"{vals.quantile(0.75):8.2f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Extract spectral features from detected events")
    parser.add_argument("--mooring", type=str, default=None,
                        help="Process single mooring (e.g., m1)")
    parser.add_argument("--file", type=str, default=None,
                        help="Process single file number (e.g., 00001282)")
    parser.add_argument("--no-patches", action="store_true",
                        help="Skip saving spectrogram patches (faster)")
    parser.add_argument("--catalogue", type=str, default=None,
                        help="Path to event catalogue parquet "
                             "(default: outputs/data/event_catalogue.parquet)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory containing mooring data folders")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers per mooring (default: 1)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from per-mooring checkpoints")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DATA_ROOT

    print("=" * 60)
    print("BRAVOSEIS Feature Extraction")
    print(f"  Data root: {data_root}")
    print(f"  Workers: {args.workers}")
    print(f"  Spectrogram: nperseg={NPERSEG}, noverlap={NOVERLAP}")
    print(f"  Frequency bands: {N_FREQ_BANDS} × {BAND_WIDTH_HZ:.0f} Hz")
    print(f"  Fixed window: {WINDOW_SEC:.0f}s "
          f"(pick − {PRE_PICK_SEC:.0f}s, pick + {POST_PICK_SEC:.0f}s)")
    print("=" * 60)

    # Load catalogue
    if args.catalogue:
        cat_path = Path(args.catalogue)
    else:
        cat_path = DATA_DIR / "event_catalogue.parquet"
    cat = pd.read_parquet(cat_path)
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])
    print(f"Loaded {len(cat):,} events from {cat_path.name}")

    # Determine suffix for output files
    suffix = ""
    if args.catalogue and "_full" in str(args.catalogue):
        suffix += "_full"
    if args.mooring:
        suffix += f"_{args.mooring}"
    if args.file:
        suffix += f"_{args.file}"

    # Use legacy single-threaded path for small runs
    if args.file or (args.mooring and args.workers <= 1):
        cat, feat_df = process_catalogue(
            cat,
            mooring_filter=args.mooring,
            file_filter=args.file,
            save_patches=not args.no_patches,
            data_root=data_root,
        )
        if len(feat_df) == 0:
            print("No features extracted!")
            return
        print_summary(feat_df)
        feature_cols = [c for c in feat_df.columns if not c.startswith("_")]
        output_df = cat[["event_id", "mooring", "file_number", "detection_band",
                          "onset_utc", "duration_s", "snr"]].copy()
        for col in feature_cols:
            output_df[col] = feat_df[col].values
        outpath = DATA_DIR / f"event_features{suffix}.parquet"
        output_df.to_parquet(outpath, index=False)
        print(f"\nSaved: {outpath} ({len(output_df):,} events)")
        return

    # Parallel per-mooring processing with checkpoints
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mooring:
        moorings = [args.mooring]
    else:
        moorings = sorted(cat["mooring"].unique())

    # Load existing checkpoints if resuming
    completed_moorings = {}
    if args.resume:
        for mkey in moorings:
            ckpt = CHECKPOINT_DIR / f"features{suffix}_{mkey}.parquet"
            if ckpt.exists():
                ckpt_df = pd.read_parquet(ckpt)
                completed_moorings[mkey] = ckpt_df
                print(f"  RESUME: loaded {len(ckpt_df):,} features for {mkey}")

    all_dfs = list(completed_moorings.values())

    for mkey in moorings:
        if mkey in completed_moorings:
            print(f"\n--- {mkey}: skipped (checkpoint exists) ---")
            continue

        n_events = len(cat[cat["mooring"] == mkey])
        print(f"\n--- {mkey} ({n_events:,} events) ---")

        feat_df = process_mooring_features(
            cat, mkey, data_root, n_workers=args.workers
        )

        if len(feat_df) == 0:
            print(f"  {mkey}: no features extracted")
            continue

        # Merge with catalogue metadata
        mcat = cat[cat["mooring"] == mkey].copy()
        feature_cols = [c for c in feat_df.columns
                        if not c.startswith("_")]
        # Avoid column collisions: drop feature cols that also exist in catalogue
        cat_keep_cols = ["event_id", "mooring", "file_number", "detection_band",
                         "onset_utc", "snr"]
        merged = mcat[cat_keep_cols].merge(
            feat_df.rename(columns={"_event_id": "event_id"}),
            on="event_id", how="left"
        )
        output_df = merged[cat_keep_cols + feature_cols].copy()

        # Save checkpoint
        ckpt_path = CHECKPOINT_DIR / f"features{suffix}_{mkey}.parquet"
        output_df.to_parquet(ckpt_path, index=False)
        print(f"  Checkpoint: {ckpt_path} ({len(output_df):,} events)")
        all_dfs.append(output_df)

    # Merge all moorings
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df.sort_values("onset_utc").reset_index(drop=True)
    else:
        print("No features extracted!")
        return

    print_summary(final_df)

    outpath = DATA_DIR / f"event_features{suffix}.parquet"
    final_df.to_parquet(outpath, index=False)
    print(f"\nSaved: {outpath} ({len(final_df):,} events)")
    print("=" * 60)


if __name__ == "__main__":
    main()
