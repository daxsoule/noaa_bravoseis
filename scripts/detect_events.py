#!/usr/bin/env python3
"""
detect_events.py — STA/LTA event detection for BRAVOSEIS hydrophone data.

Detects discrete acoustic events across all 6 moorings using a Short-Term
Average / Long-Term Average energy detector in 4 frequency bands. Each
detection is characterized by onset time, duration, peak frequency,
bandwidth, and SNR.

Usage:
    uv run python detect_events.py                  # all 717 files
    uv run python detect_events.py --tune           # tuning subset only
    uv run python detect_events.py --mooring m1     # single mooring
    uv run python detect_events.py --file 00001282  # single file number

Spec: specs/001-event-detection/
"""

import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from scipy.signal import butter, sosfilt, spectrogram

from read_dat import read_dat, read_header, list_mooring_files, MOORINGS, SAMPLE_RATE, get_data_dir


def classic_sta_lta(data, nsta, nlta):
    """Classic STA/LTA using cumulative sum (fully vectorized).

    Computes ratio of short-term to long-term average energy.
    Output is valid from index nlta onward.
    """
    ndat = len(data)

    # Cumulative sum with a leading zero for easy windowing
    cs = np.zeros(ndat + 1, dtype=np.float64)
    np.cumsum(data, out=cs[1:])

    # STA and LTA as vectorized differences
    sta = np.zeros(ndat, dtype=np.float64)
    lta = np.zeros(ndat, dtype=np.float64)

    # cs has length ndat+1; window sums use cs[i+1] - cs[i+1-n]
    sta[nsta:] = (cs[nsta + 1:] - cs[1:-nsta]) / nsta
    lta[nlta:] = (cs[nlta + 1:] - cs[1:-nlta]) / nlta

    # Ratio (avoid div by zero)
    cft = np.zeros(ndat, dtype=np.float64)
    valid = lta > 0
    cft[valid] = sta[valid] / lta[valid]

    return cft


def trigger_onset(cft, threshold_on, threshold_off):
    """Extract trigger on/off pairs from STA/LTA function."""
    on = False
    triggers = []
    on_idx = 0
    for i in range(len(cft)):
        if not on and cft[i] >= threshold_on:
            on = True
            on_idx = i
        elif on and cft[i] <= threshold_off:
            on = False
            triggers.append([on_idx, i])
    # If still triggered at end of data, close it
    if on:
        triggers.append([on_idx, len(cft) - 1])
    return np.array(triggers, dtype=np.int64) if triggers else np.empty((0, 2), dtype=np.int64)

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "data"
CHECKPOINT_DIR = OUTPUT_DIR / "detection_checkpoints"

# === Detection parameters (from plan R2) ===
STA_SEC = 2.0       # Short-term average window (seconds)
LTA_SEC = 60.0      # Long-term average window (seconds)
TRIGGER = 3.0        # STA/LTA trigger threshold
DETRIGGER = 1.5      # STA/LTA detrigger threshold
MIN_DURATION = 0.5   # Minimum event duration (seconds)
MIN_GAP = 2.0        # Minimum inter-event gap (seconds)

# Convert to samples
NSTA = int(STA_SEC * SAMPLE_RATE)
NLTA = int(LTA_SEC * SAMPLE_RATE)
MIN_DUR_SAMP = int(MIN_DURATION * SAMPLE_RATE)
MIN_GAP_SAMP = int(MIN_GAP * SAMPLE_RATE)

# === Three-pass detection strategy ===
# Non-overlapping frequency passes so each STA/LTA sees only its own
# energy regime. Breakpoints at 15 Hz and 30 Hz chosen from spectral
# analysis: dominant earthquake/T-phase energy < 15 Hz; fin whale calls
# ~20 Hz in the mid band; ice quakes span all three passes.
PASSES = {
    1: {
        "label": "low",
        "filter": "lowpass",
        "cutoff": 15,
        "band": (1, 15),
        "targets": "Earthquakes, T-phases, ice quakes (low-freq component)",
    },
    2: {
        "label": "mid",
        "filter": "bandpass",
        "cutoff": (15, 30),
        "band": (15, 30),
        "targets": "Fin whale calls (~20 Hz), ice quakes, mixed seismicity",
    },
    3: {
        "label": "high",
        "filter": "highpass",
        "cutoff": 30,
        "band": (30, 250),
        "targets": "Ice quakes (high-freq), other whale calls, biological",
    },
}

# Legacy band definitions (kept for backward compatibility with figures)
BANDS = {
    "low":  (1, 15),
    "mid":  (15, 30),
    "high": (30, 250),
}

# === Spectrogram parameters (matching make_spectrogram.py) ===
NPERSEG = 1024
NOVERLAP = 512
FREQ_MAX = 250

# === Tuning file numbers ===
TUNE_FILES = {"00001282", "00001283", "00002166"}

# === Mooring order ===
MOORING_KEYS = sorted(MOORINGS.keys())


def lowpass_filter(data, cutoff_hz, fs=SAMPLE_RATE, order=4):
    """Apply a Butterworth lowpass filter."""
    nyq = fs / 2
    wn = min(cutoff_hz / nyq, 0.999)
    sos = butter(order, wn, btype='lowpass', output='sos')
    return sosfilt(sos, data)


def bandpass_filter(data, low_hz, high_hz, fs=SAMPLE_RATE, order=4):
    """Apply a Butterworth bandpass filter."""
    nyq = fs / 2
    low = max(low_hz / nyq, 0.001)
    high = min(high_hz / nyq, 0.999)
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfilt(sos, data)


def highpass_filter(data, cutoff_hz, fs=SAMPLE_RATE, order=4):
    """Apply a Butterworth highpass filter."""
    nyq = fs / 2
    wn = max(cutoff_hz / nyq, 0.001)
    sos = butter(order, wn, btype='highpass', output='sos')
    return sosfilt(sos, data)


def apply_pass_filter(data, pass_cfg):
    """Apply the pre-filter for a detection pass."""
    ftype = pass_cfg["filter"]
    cutoff = pass_cfg["cutoff"]
    if ftype == "lowpass":
        return lowpass_filter(data, cutoff)
    elif ftype == "bandpass":
        return bandpass_filter(data, cutoff[0], cutoff[1])
    elif ftype == "highpass":
        return highpass_filter(data, cutoff)
    else:
        raise ValueError(f"Unknown filter type: {ftype}")


def detect_in_band(data, band_name, band_range, file_ts):
    """Run STA/LTA detection on one frequency band.

    Returns list of dicts, one per detection.
    """
    low_hz, high_hz = band_range

    # Bandpass filter
    filtered = bandpass_filter(data, low_hz, high_hz)

    # Energy envelope (squared amplitude)
    envelope = filtered ** 2

    # STA/LTA
    cft = classic_sta_lta(envelope, NSTA, NLTA)

    # Trigger/detrigger
    try:
        on_off = trigger_onset(cft, TRIGGER, DETRIGGER)
    except Exception:
        return []

    if len(on_off) == 0:
        return []

    # Filter by min duration and min gap
    events = []
    prev_off = -MIN_GAP_SAMP - 1

    for on_samp, off_samp in on_off:
        duration_samp = off_samp - on_samp
        if duration_samp < MIN_DUR_SAMP:
            continue
        if on_samp - prev_off < MIN_GAP_SAMP:
            # Merge with previous event
            if events:
                events[-1]["off_samp"] = off_samp
                prev_off = off_samp
                continue
        events.append({
            "on_samp": int(on_samp),
            "off_samp": int(off_samp),
        })
        prev_off = off_samp

    # Characterize each event from the raw spectrogram
    detections = []
    for ev in events:
        on_s = ev["on_samp"]
        off_s = ev["off_samp"]

        # Pad ±2s for spectrogram context
        pad = 2 * SAMPLE_RATE
        seg_start = max(0, on_s - pad)
        seg_end = min(len(data), off_s + pad)
        segment = data[seg_start:seg_end]

        # Compute spectrogram of segment
        if len(segment) < NPERSEG:
            continue

        freqs, times, Sxx = spectrogram(
            segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
        )

        # Clip to frequency range
        freq_mask = freqs <= FREQ_MAX
        freqs = freqs[freq_mask]
        Sxx = Sxx[freq_mask, :]

        if Sxx.size == 0:
            continue

        # Power in dB
        Sxx_dB = 10 * np.log10(Sxx + 1e-20)

        # Event region within the padded spectrogram
        event_t_start = (on_s - seg_start) / SAMPLE_RATE
        event_t_end = (off_s - seg_start) / SAMPLE_RATE
        t_mask = (times >= event_t_start) & (times <= event_t_end)

        if not np.any(t_mask):
            # Fallback: use all times
            t_mask = np.ones(len(times), dtype=bool)

        event_Sxx = Sxx[:, t_mask]
        event_dB = Sxx_dB[:, t_mask]

        # Peak frequency (frequency with max power)
        mean_power = event_Sxx.mean(axis=1)
        peak_freq_idx = np.argmax(mean_power)
        peak_freq_hz = freqs[peak_freq_idx]

        # Peak power (dB)
        peak_db = float(event_dB.max())

        # Bandwidth (frequency range containing 90% of energy)
        total_power = mean_power.sum()
        if total_power > 0:
            cumsum = np.cumsum(mean_power)
            frac = cumsum / total_power
            low_idx = np.searchsorted(frac, 0.05)
            high_idx = np.searchsorted(frac, 0.95)
            bandwidth_hz = float(freqs[min(high_idx, len(freqs)-1)]
                                 - freqs[max(low_idx, 0)])
        else:
            bandwidth_hz = 0.0

        # SNR (peak STA/LTA ratio in the event window)
        cft_event = cft[on_s:off_s]
        snr = float(cft_event.max()) if len(cft_event) > 0 else 0.0

        # Timing
        onset_sec = on_s / SAMPLE_RATE
        duration_sec = (off_s - on_s) / SAMPLE_RATE
        onset_utc = file_ts + timedelta(seconds=onset_sec)
        end_utc = onset_utc + timedelta(seconds=duration_sec)

        detections.append({
            "onset_utc": onset_utc,
            "duration_s": round(duration_sec, 3),
            "end_utc": end_utc,
            "peak_freq_hz": round(float(peak_freq_hz), 1),
            "bandwidth_hz": round(bandwidth_hz, 1),
            "peak_db": round(peak_db, 1),
            "snr": round(snr, 2),
            "detection_band": band_name,
            "on_samp": on_s,
            "off_samp": off_s,
        })

    return detections


def deduplicate_bands(detections):
    """Remove duplicate detections of the same event across bands.

    If two detections on the same mooring/file overlap in time, keep the
    one with highest SNR and record all bands.
    """
    if not detections:
        return detections

    # Sort by onset
    detections.sort(key=lambda d: d["onset_utc"])

    merged = []
    current = detections[0].copy()
    current["all_bands"] = {current["detection_band"]}

    for det in detections[1:]:
        # Check overlap: does this detection start before the current one ends?
        if det["onset_utc"] < current["end_utc"] + timedelta(seconds=MIN_GAP):
            # Overlap — merge
            current["all_bands"].add(det["detection_band"])
            if det["snr"] > current["snr"]:
                # Keep the higher-SNR detection's characterization
                best_band = det["detection_band"]
                for key in det:
                    if key != "all_bands":
                        current[key] = det[key]
            # Extend end time if needed
            if det["end_utc"] > current["end_utc"]:
                current["end_utc"] = det["end_utc"]
                current["duration_s"] = (
                    current["end_utc"] - current["onset_utc"]
                ).total_seconds()
        else:
            # No overlap — finalize current, start new
            current["all_bands"] = ",".join(sorted(current["all_bands"]))
            merged.append(current)
            current = det.copy()
            current["all_bands"] = {current["detection_band"]}

    current["all_bands"] = ",".join(sorted(current["all_bands"]))
    merged.append(current)

    return merged


def process_file(filepath, mooring_key, pass_nums=None):
    """Process a single DAT file: detect events in specified passes.

    Each pass pre-filters into a non-overlapping frequency band, then
    runs STA/LTA on the filtered signal. No cross-band deduplication
    is needed because bands don't overlap.

    Parameters
    ----------
    pass_nums : list of int, optional
        Which passes to run (1, 2, 3). Default: all three.
    """
    if pass_nums is None:
        pass_nums = list(PASSES.keys())

    file_ts, data, meta = read_dat(filepath)

    all_detections = []
    for pnum in pass_nums:
        pcfg = PASSES[pnum]
        # Pre-filter into this pass's frequency band
        filtered_data = apply_pass_filter(data, pcfg)
        # Run STA/LTA on the pre-filtered signal (single band)
        # Note: detect_in_band() applies bandpass_filter() again with the same
        # passband. This is intentional — the double 4th-order Butterworth
        # effectively gives an 8th-order rolloff, providing steeper rejection
        # of out-of-band energy for spectral characterization.
        band_name = pcfg["label"]
        band_range = pcfg["band"]
        dets = detect_in_band(filtered_data, band_name, band_range, file_ts)
        for det in dets:
            det["detection_pass"] = pnum
            det["mooring"] = mooring_key
            det["file_number"] = meta["file_number"]
            det["instrument_id"] = meta["instrument_id"]
        all_detections.extend(dets)

    return all_detections


def _process_file_worker(args):
    """Top-level wrapper for multiprocessing Pool.map."""
    filepath, mooring_key, pass_nums = args
    return process_file(filepath, mooring_key, pass_nums=pass_nums)


def process_mooring(mooring_key, file_filter=None, pass_nums=None,
                    data_root=None, n_workers=1):
    """Process all files for one mooring.

    Parameters
    ----------
    n_workers : int
        Number of parallel workers. 1 = sequential (original behavior).
    """
    if data_root is None:
        data_root = DATA_ROOT
    info = MOORINGS[mooring_key]
    mooring_dir = data_root / get_data_dir(info, data_root)

    if not mooring_dir.exists():
        print(f"  WARNING: {mooring_dir} not found, skipping")
        return []

    catalog = list_mooring_files(mooring_dir, sort_by="filename")

    if file_filter is not None:
        catalog = [f for f in catalog
                   if f"{f['file_number']:08d}" in file_filter]

    if n_workers > 1 and len(catalog) > 1:
        # Parallel processing
        work_items = [(entry["path"], mooring_key, pass_nums)
                      for entry in catalog]
        all_events = []
        completed = 0
        with mp.Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_process_file_worker, work_items,
                                              chunksize=4):
                all_events.extend(result)
                completed += 1
                if completed % 50 == 0 or completed == len(catalog):
                    print(f"    {mooring_key}: {completed}/{len(catalog)} files, "
                          f"{len(all_events)} events so far")
    else:
        # Sequential processing
        all_events = []
        for i, entry in enumerate(catalog):
            filepath = entry["path"]
            detections = process_file(filepath, mooring_key, pass_nums=pass_nums)
            all_events.extend(detections)

            if (i + 1) % 10 == 0 or (i + 1) == len(catalog):
                print(f"    {mooring_key}: {i+1}/{len(catalog)} files, "
                      f"{len(all_events)} events so far")

    return all_events


def build_catalogue(events):
    """Convert event list to a DataFrame with event IDs."""
    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)

    # Drop sample-level columns (internal use only)
    df = df.drop(columns=["on_samp", "off_samp"], errors="ignore")

    # Sort by onset time
    df = df.sort_values("onset_utc").reset_index(drop=True)

    # Assign event IDs
    df.insert(0, "event_id",
              [f"E{i:06d}" for i in range(len(df))])

    return df


def main():
    parser = argparse.ArgumentParser(
        description="STA/LTA event detection for BRAVOSEIS hydrophones")
    parser.add_argument("--tune", action="store_true",
                        help="Run on tuning subset only (3 files per mooring)")
    parser.add_argument("--mooring", type=str, default=None,
                        help="Process single mooring (e.g., m1)")
    parser.add_argument("--file", type=str, default=None,
                        help="Process single file number (e.g., 00001282)")
    parser.add_argument("--pass", type=str, default="all", dest="det_pass",
                        help="Which pass(es) to run: 1, 2, 3, or 'all' (default: all)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory containing mooring data folders "
                             "(default: subset at /home/jovyan/my_data/bravoseis/NOAA)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers per mooring (default: 1)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from per-mooring checkpoints (skip completed moorings)")
    args = parser.parse_args()

    # Resolve data root
    data_root = Path(args.data_root) if args.data_root else DATA_ROOT

    # Parse pass selection
    if args.det_pass == "all":
        pass_nums = list(PASSES.keys())
    else:
        pass_nums = [int(p.strip()) for p in args.det_pass.split(",")]
        for p in pass_nums:
            if p not in PASSES:
                parser.error(f"Invalid pass number: {p}. Must be 1, 2, or 3.")

    pass_descs = [f"Pass {p}: {PASSES[p]['label']} ({PASSES[p]['filter']} "
                  f"{PASSES[p]['cutoff']} Hz)" for p in pass_nums]

    n_workers = args.workers

    print("=" * 60)
    print("BRAVOSEIS Event Detection — Three-Pass Strategy")
    print(f"  Data root: {data_root}")
    print(f"  Workers: {n_workers}")
    print(f"  STA={STA_SEC}s, LTA={LTA_SEC}s, "
          f"trigger={TRIGGER}, detrigger={DETRIGGER}")
    for desc in pass_descs:
        print(f"  {desc}")
    print("=" * 60)

    # Determine file filter
    file_filter = None
    if args.tune:
        file_filter = TUNE_FILES
        print(f"  TUNING MODE: processing files {sorted(file_filter)}")
    elif args.file:
        file_filter = {args.file}
        print(f"  SINGLE FILE: {args.file}")

    # Determine moorings
    if args.mooring:
        moorings = [args.mooring]
    else:
        moorings = MOORING_KEYS

    # Checkpoint setup
    suffix = ""
    if args.data_root:
        suffix += "_full"
    if args.det_pass != "all":
        suffix += f"_pass{args.det_pass}"
    if args.tune:
        suffix += "_tune"
    elif args.file:
        suffix += f"_{args.file}"
    elif args.mooring:
        suffix += f"_{args.mooring}"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoints if resuming
    completed_moorings = {}
    if args.resume:
        for mkey in moorings:
            ckpt = CHECKPOINT_DIR / f"checkpoint{suffix}_{mkey}.parquet"
            if ckpt.exists():
                ckpt_df = pd.read_parquet(ckpt)
                completed_moorings[mkey] = ckpt_df
                print(f"  RESUME: loaded {len(ckpt_df)} events for {mkey} from checkpoint")

    # Process
    all_events = []
    for mkey in moorings:
        if mkey in completed_moorings:
            print(f"\n--- {mkey}: skipped (checkpoint exists) ---")
            continue
        info = MOORINGS[mkey]
        print(f"\n--- {mkey} / {info['name']} ({info['hydrophone']}) ---")
        events = process_mooring(mkey, file_filter=file_filter,
                                pass_nums=pass_nums, data_root=data_root,
                                n_workers=n_workers)
        all_events.extend(events)
        print(f"  {mkey}: {len(events)} events detected")

        # Save per-mooring checkpoint
        if events:
            ckpt_df = build_catalogue(events)
            ckpt_path = CHECKPOINT_DIR / f"checkpoint{suffix}_{mkey}.parquet"
            ckpt_df.to_parquet(ckpt_path, index=False)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Merge checkpointed moorings with newly processed ones
    checkpoint_dfs = list(completed_moorings.values())
    if all_events:
        checkpoint_dfs.append(pd.DataFrame(all_events))
    if checkpoint_dfs:
        merged = pd.concat(checkpoint_dfs, ignore_index=True)
        # Drop event_id from checkpoints (will re-assign)
        if "event_id" in merged.columns:
            merged = merged.drop(columns=["event_id"])
    else:
        merged = pd.DataFrame()

    # Build final catalogue
    df = build_catalogue(merged.to_dict("records") if len(merged) > 0 else [])

    if len(df) == 0:
        print("\nNo events detected!")
        return

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Total events: {len(df)}")
    print(f"\nPer mooring:")
    print(df.groupby("mooring").size().to_string())
    print(f"\nPer band:")
    print(df.groupby("detection_band").size().to_string())
    print(f"\nDuration: median={df['duration_s'].median():.1f}s, "
          f"IQR=[{df['duration_s'].quantile(0.25):.1f}, "
          f"{df['duration_s'].quantile(0.75):.1f}]s")
    print(f"Peak freq: median={df['peak_freq_hz'].median():.0f} Hz, "
          f"IQR=[{df['peak_freq_hz'].quantile(0.25):.0f}, "
          f"{df['peak_freq_hz'].quantile(0.75):.0f}] Hz")
    print(f"SNR: median={df['snr'].median():.1f}, "
          f"IQR=[{df['snr'].quantile(0.25):.1f}, "
          f"{df['snr'].quantile(0.75):.1f}]")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / f"event_catalogue{suffix}.parquet"
    df.to_parquet(outpath, index=False)
    print(f"\nSaved: {outpath} ({len(df)} events)")

    # Also save CSV for quick inspection
    csv_path = outpath.with_suffix(".csv")
    df.head(200).to_csv(csv_path, index=False)
    print(f"Saved preview: {csv_path} (first 200 rows)")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
