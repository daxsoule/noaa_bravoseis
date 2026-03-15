#!/usr/bin/env python3
"""
extract_features_highband.py — Extract features for high-band events
with a 30–450 Hz bandpass applied BEFORE computing spectrogram features.

Targets icequake and vessel noise classification (Phase 3 frequency-band
approach). Isolates high-frequency energy above the whale-dominated 14–30 Hz
band.

Usage:
    uv run python scripts/extract_features_highband.py
    uv run python scripts/extract_features_highband.py --workers 4
    uv run python scripts/extract_features_highband.py --resume

Output:
    outputs/data/event_features_highband.parquet
"""

import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import spectrogram, butter, sosfilt

from read_dat import read_dat, MOORINGS, SAMPLE_RATE, get_data_dir

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
CHECKPOINT_DIR = DATA_DIR / "feature_checkpoints_highband"

# === Bandpass filter ===
FMIN = 30.0    # Hz
FMAX = 450.0   # Hz (below Nyquist at 500 Hz)
FILTER_ORDER = 4

# === Spectrogram parameters ===
# nperseg=1024 gives ~1 Hz bins at 1 kHz — good resolution for >30 Hz
NPERSEG = 1024
NOVERLAP = 512

# === Feature extraction parameters ===
PRE_PICK_SEC = 2.0    # shorter window — icequakes are brief
POST_PICK_SEC = 8.0
WINDOW_SEC = PRE_PICK_SEC + POST_PICK_SEC  # 10s

# Frequency bands for feature extraction (30–450 Hz, 6 bands)
FREQ_BANDS = [
    (30, 60),     # low-icequake / T-phase tail
    (60, 100),    # mid-icequake
    (100, 160),   # upper-icequake / vessel
    (160, 250),   # vessel harmonics
    (250, 350),   # high vessel / cavitation
    (350, 450),   # near-Nyquist
]
N_FREQ_BANDS = len(FREQ_BANDS)


def make_bandpass_filter():
    """Create Butterworth bandpass filter coefficients."""
    sos = butter(FILTER_ORDER, [FMIN, FMAX], btype='band',
                 fs=SAMPLE_RATE, output='sos')
    return sos


def compute_spectrogram_patch(data, onset_samp, end_samp, file_nsamples, sos):
    """Extract spectrogram patch with bandpass filter applied first."""
    pre_samp = int(PRE_PICK_SEC * SAMPLE_RATE)
    post_samp = int(POST_PICK_SEC * SAMPLE_RATE)

    seg_start = onset_samp - pre_samp
    seg_end = onset_samp + post_samp

    if seg_start < 0 or seg_end > file_nsamples:
        return None

    segment = data[seg_start:seg_end].astype(np.float64)

    # Apply bandpass filter
    segment = sosfilt(sos, segment)

    if len(segment) < NPERSEG:
        return None

    freqs, times, Sxx = spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
    )

    # Clip to our frequency range
    freq_mask = (freqs >= FMIN) & (freqs <= FMAX)
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
    """Extract features from a bandpass-filtered spectrogram patch."""
    features = {}

    # Event time mask
    t_mask = (times >= onset_rel) & (times <= end_rel)
    if not np.any(t_mask):
        t_mask = np.ones(len(times), dtype=bool)

    event_Sxx = Sxx_linear[:, t_mask]
    event_dB = Sxx_dB[:, t_mask]

    # --- Band powers (6 bands within 30–450 Hz) ---
    for i, (f_lo, f_hi) in enumerate(FREQ_BANDS):
        band_mask = (freqs >= f_lo) & (freqs < f_hi)
        if np.any(band_mask):
            features[f"band_power_{i}"] = float(np.mean(event_dB[band_mask, :]))
        else:
            features[f"band_power_{i}"] = -200.0

    # --- Mean power spectrum across event ---
    mean_power = event_Sxx.mean(axis=1)
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

    # --- Spectral slope ---
    valid_f = freqs > 0
    if np.sum(valid_f) > 2 and np.any(mean_power[valid_f] > 0):
        log_f = np.log10(freqs[valid_f])
        log_p = np.log10(mean_power[valid_f] + 1e-20)
        coeffs = np.polyfit(log_f, log_p, 1)
        features["spectral_slope"] = float(coeffs[0])
    else:
        features["spectral_slope"] = 0.0

    # --- Frequency modulation ---
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


def _process_file_group(args):
    """Worker: extract features for all events in one file."""
    mooring, file_num, events_data, data_root = args

    sos = make_bandpass_filter()

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
            data, onset_samp, end_samp, file_nsamples, sos
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract features for high-band events with 30-450 Hz bandpass")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers per mooring (default: 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print("BRAVOSEIS High-Band Feature Extraction (30–450 Hz)")
    print(f"  Bandpass: {FMIN}–{FMAX} Hz (Butterworth order {FILTER_ORDER})")
    print(f"  Spectrogram: nperseg={NPERSEG}, noverlap={NOVERLAP}")
    print(f"  Freq resolution: {SAMPLE_RATE/NPERSEG:.2f} Hz/bin")
    print(f"  Feature bands: {N_FREQ_BANDS} ({FREQ_BANDS})")
    print(f"  Workers: {args.workers}")
    print("=" * 60)

    # Load catalogue — only high-band events
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])
    cat = cat[cat["detection_band"] == "high"].copy()
    print(f"High-band events: {len(cat):,}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    moorings = sorted(cat["mooring"].unique())

    # Load checkpoints if resuming
    completed = {}
    if args.resume:
        for m in moorings:
            ckpt = CHECKPOINT_DIR / f"features_{m}.parquet"
            if ckpt.exists():
                completed[m] = pd.read_parquet(ckpt)
                print(f"  RESUME: {m} — {len(completed[m]):,} events")

    all_dfs = list(completed.values())

    cat_keep_cols = ["event_id", "mooring", "file_number", "detection_band",
                     "onset_utc", "snr"]

    for mkey in moorings:
        if mkey in completed:
            print(f"\n--- {mkey}: skipped (checkpoint) ---")
            continue

        mcat = cat[cat["mooring"] == mkey]
        n_events = len(mcat)
        print(f"\n--- {mkey} ({n_events:,} events) ---")

        # Build work items
        groups = mcat.groupby("file_number")
        work_items = []
        for file_num, group in groups:
            events_data = [
                {"event_id": row["event_id"],
                 "onset_utc": row["onset_utc"],
                 "end_utc": row["end_utc"]}
                for _, row in group.iterrows()
            ]
            work_items.append((mkey, file_num, events_data, DATA_ROOT))

        n_groups = len(work_items)
        all_features = []

        if args.workers > 1 and n_groups > 1:
            done = 0
            with mp.Pool(processes=args.workers) as pool:
                for result in pool.imap_unordered(_process_file_group,
                                                   work_items, chunksize=4):
                    all_features.extend(result)
                    done += 1
                    if done % 100 == 0 or done == n_groups:
                        print(f"    {mkey}: {done}/{n_groups} files, "
                              f"{len(all_features)} events")
        else:
            for i, item in enumerate(work_items):
                result = _process_file_group(item)
                all_features.extend(result)
                if (i + 1) % 50 == 0 or (i + 1) == n_groups:
                    print(f"    {mkey}: {i+1}/{n_groups} files, "
                          f"{len(all_features)} events")

        feat_df = pd.DataFrame(all_features)
        if len(feat_df) == 0:
            print(f"  {mkey}: no features extracted")
            continue

        feature_cols = [c for c in feat_df.columns if not c.startswith("_")]
        merged = mcat[cat_keep_cols].merge(
            feat_df.rename(columns={"_event_id": "event_id"}),
            on="event_id", how="left"
        )
        output_df = merged[cat_keep_cols + feature_cols].copy()

        ckpt_path = CHECKPOINT_DIR / f"features_{mkey}.parquet"
        output_df.to_parquet(ckpt_path, index=False)
        print(f"  Checkpoint: {ckpt_path} ({len(output_df):,} events)")
        all_dfs.append(output_df)

    # Merge all
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df.sort_values("onset_utc").reset_index(drop=True)
    else:
        print("No features extracted!")
        return

    # Summary
    feature_cols = [c for c in final_df.columns
                    if c not in cat_keep_cols and not c.startswith("_")]
    n_complete = final_df[feature_cols].dropna(how="any").shape[0]
    print(f"\n{'=' * 60}")
    print(f"High-band feature extraction complete")
    print(f"  Total events:     {len(final_df):,}")
    print(f"  Complete features: {n_complete:,} ({100*n_complete/len(final_df):.1f}%)")
    print(f"  Peak freq stats:")
    pf = final_df["peak_freq_hz"].dropna()
    print(f"    median: {pf.median():.1f} Hz")
    print(f"    IQR: [{pf.quantile(0.25):.1f}, {pf.quantile(0.75):.1f}] Hz")

    outpath = DATA_DIR / "event_features_highband.parquet"
    final_df.to_parquet(outpath, index=False)
    print(f"\nSaved: {outpath} ({len(final_df):,} events)")
    print("=" * 60)


if __name__ == "__main__":
    main()
