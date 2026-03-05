#!/usr/bin/env python3
"""
make_tphase_montage.py — Curated T-phase montage for expert review.

Extracts the best T-phase candidates from labeled seismic clusters and
generates a multi-page PDF with waveform + spectrogram pairs. Designed
for sending to Bob Dziak (NOAA/PMEL) for confirmation.

Selection criteria for "best" T-phase candidates:
    - Duration 3-30 s (emergent, sustained signals)
    - Peak frequency 2-15 Hz (classic T-phase band)
    - Higher SNR preferred
    - Sorted by SNR descending, top N selected

Usage:
    uv run python make_tphase_montage.py
    uv run python make_tphase_montage.py --n-events 60
    uv run python make_tphase_montage.py --events-per-page 6

Output:
    outputs/figures/tphase_montage_for_review.pdf
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import timedelta
from scipy.signal import spectrogram, butter, sosfilt

from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures"

# === T-phase source clusters ===
TPHASE_CLUSTERS = ["low_0", "low_1", "mid_0", "mid_3_1"]

# === T-phase selection criteria ===
MIN_DURATION_S = 3.0
MAX_DURATION_S = 30.0
MIN_PEAK_FREQ_HZ = 2.0
MAX_PEAK_FREQ_HZ = 15.0

# === Spectrogram parameters ===
NPERSEG = 256
NOVERLAP = 192
FREQ_MAX_SPEC = 50  # Show up to 50 Hz for T-phase detail
WINDOW_SEC = 20  # Wider window for T-phases (longer signals)
PAD_SEC = 5.0  # Pre-event context

# === Bandpass filter for waveform display ===
FILT_LOW = 1.0
FILT_HIGH = 30.0

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
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data)


def load_tphase_candidates():
    """Load and filter T-phase candidates from labeled clusters."""
    # Load data
    umap_df = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    features = pd.read_parquet(DATA_DIR / "event_features.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])

    # Also load sub-cluster assignments
    subcluster_files = list(DATA_DIR.glob("subclusters_*.parquet"))
    sub_map = {}
    for f in subcluster_files:
        sub_df = pd.read_parquet(f)
        for _, row in sub_df.iterrows():
            sub_map[row["event_id"]] = row["subcluster_id"]

    # Get events from T-phase clusters
    # For top-level clusters (low_0, low_1, mid_0): use cluster_id from umap_df
    # For sub-clusters (mid_3_1): use subcluster_id from sub_map
    top_level = [c for c in TPHASE_CLUSTERS if "_" not in c[c.index("_")+1:]]
    sub_level = [c for c in TPHASE_CLUSTERS if c not in top_level]

    # Collect event IDs
    event_ids = set()

    # Top-level clusters
    for cid in ["low_0", "low_1", "mid_0"]:
        mask = umap_df["cluster_id"] == cid
        event_ids.update(umap_df.loc[mask, "event_id"])

    # Sub-clusters
    for scid in ["mid_3_1"]:
        for eid, sid in sub_map.items():
            if sid == scid:
                event_ids.add(eid)

    print(f"  Total events from T-phase clusters: {len(event_ids):,}")

    # Merge catalogue + features
    cat_filt = cat[cat["event_id"].isin(event_ids)].copy()
    merged = cat_filt.merge(features, on="event_id", suffixes=("", "_feat"))

    # Add cluster source
    cluster_source = {}
    for eid in event_ids:
        if eid in sub_map and sub_map[eid] in TPHASE_CLUSTERS:
            cluster_source[eid] = sub_map[eid]
        else:
            row = umap_df[umap_df["event_id"] == eid]
            if len(row) > 0:
                cluster_source[eid] = row.iloc[0]["cluster_id"]
    merged["source_cluster"] = merged["event_id"].map(cluster_source)

    print(f"  Merged with features: {len(merged):,} events")

    # Apply T-phase selection criteria
    mask = (
        (merged["duration_s"] >= MIN_DURATION_S) &
        (merged["duration_s"] <= MAX_DURATION_S) &
        (merged["peak_freq_hz"] >= MIN_PEAK_FREQ_HZ) &
        (merged["peak_freq_hz"] <= MAX_PEAK_FREQ_HZ)
    )
    filtered = merged[mask].copy()
    print(f"  After T-phase criteria (dur {MIN_DURATION_S}-{MAX_DURATION_S}s, "
          f"freq {MIN_PEAK_FREQ_HZ}-{MAX_PEAK_FREQ_HZ} Hz): {len(filtered):,}")

    # Sort by SNR descending
    if "snr" in filtered.columns:
        filtered = filtered.sort_values("snr", ascending=False)

    return filtered


def extract_tphase_snippet(event_row):
    """Extract waveform + spectrogram for one T-phase candidate."""
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

    # Window centered on event with generous context
    t_start = onset - timedelta(seconds=PAD_SEC)
    total_window = max(WINDOW_SEC, duration + 2 * PAD_SEC)
    t_end = t_start + timedelta(seconds=total_window)

    offset_s = (t_start - file_ts).total_seconds()
    start_samp = int(offset_s * SAMPLE_RATE)
    end_samp = start_samp + int(total_window * SAMPLE_RATE)

    if start_samp < 0 or end_samp > file_nsamples:
        return None

    segment = data[start_samp:end_samp].astype(np.float64)

    # Bandpass filter for waveform
    waveform_filt = bandpass(segment, FILT_LOW, FILT_HIGH, SAMPLE_RATE)
    t_wave = np.arange(len(segment)) / SAMPLE_RATE

    # Spectrogram
    freqs, times, Sxx = spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    freq_mask = freqs <= FREQ_MAX_SPEC
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
    }


def plot_tphase_page(pdf, events, snippets, page_num, total_pages):
    """Plot one page of the T-phase montage PDF."""
    n = len(events)
    fig = plt.figure(figsize=(11, 17))  # Letter portrait
    gs = GridSpec(n, 2, figure=fig,
                  hspace=0.35, wspace=0.15,
                  top=0.94, bottom=0.03, left=0.08, right=0.95)

    fig.suptitle(
        f"BRAVOSEIS T-Phase Candidates — Page {page_num}/{total_pages}\n"
        f"Bandpass {FILT_LOW}-{FILT_HIGH} Hz | Spectrogram 0-{FREQ_MAX_SPEC} Hz",
        fontsize=12, fontweight="bold"
    )

    for idx in range(n):
        ev = events.iloc[idx]
        snip = snippets[idx]

        if snip is None:
            continue

        # Left panel: waveform
        ax_wave = fig.add_subplot(gs[idx, 0])
        ax_wave.plot(snip["t_wave"], snip["waveform"],
                     color="black", linewidth=0.3)
        ax_wave.axvline(snip["ev_start"], color="red", linewidth=1.0,
                        linestyle="--", alpha=0.8, label="onset")
        ax_wave.axvline(snip["ev_end"], color="red", linewidth=0.8,
                        linestyle=":", alpha=0.6)
        ax_wave.set_xlim(0, snip["total_window"])
        ax_wave.set_ylabel("Amplitude", fontsize=7)
        ax_wave.tick_params(labelsize=6)

        # Event info title
        mooring = ev["mooring"].upper()
        time_str = ev["onset_utc"].strftime("%Y-%m-%d %H:%M:%S")
        pf = ev.get("peak_freq_hz", 0)
        dur = ev["duration_s"]
        snr = ev.get("snr", 0)
        cluster = ev.get("source_cluster", "?")
        ax_wave.set_title(
            f"{mooring} | {time_str} | {pf:.1f} Hz | {dur:.1f}s | "
            f"SNR={snr:.1f} | {cluster}",
            fontsize=7, fontweight="bold", pad=3
        )

        if idx == n - 1:
            ax_wave.set_xlabel("Time (s)", fontsize=7)

        # Right panel: spectrogram
        ax_spec = fig.add_subplot(gs[idx, 1])
        vmin = np.percentile(snip["Sxx_dB"], 5)
        vmax = np.percentile(snip["Sxx_dB"], 95)
        ax_spec.pcolormesh(snip["times"], snip["freqs"], snip["Sxx_dB"],
                           vmin=vmin, vmax=vmax, cmap="viridis",
                           shading="auto", rasterized=True)
        ax_spec.axvline(snip["ev_start"], color="white", linewidth=1.0,
                        linestyle="--", alpha=0.8)
        ax_spec.axvline(snip["ev_end"], color="white", linewidth=0.8,
                        linestyle=":", alpha=0.6)
        ax_spec.set_ylim(0, FREQ_MAX_SPEC)
        ax_spec.set_ylabel("Frequency (Hz)", fontsize=7)
        ax_spec.tick_params(labelsize=6)

        if idx == n - 1:
            ax_spec.set_xlabel("Time (s)", fontsize=7)

    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate T-phase montage PDF for expert review")
    parser.add_argument("--n-events", type=int, default=60,
                        help="Total number of T-phase candidates to show "
                             "(default: 60)")
    parser.add_argument("--events-per-page", type=int, default=6,
                        help="Events per PDF page (default: 6)")
    args = parser.parse_args()

    print("=" * 60)
    print("BRAVOSEIS T-Phase Montage for Expert Review")
    print(f"  Source clusters: {', '.join(TPHASE_CLUSTERS)}")
    print(f"  Selection: dur {MIN_DURATION_S}-{MAX_DURATION_S}s, "
          f"freq {MIN_PEAK_FREQ_HZ}-{MAX_PEAK_FREQ_HZ} Hz")
    print(f"  Top {args.n_events} by SNR, {args.events_per_page} per page")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load and filter candidates
    candidates = load_tphase_candidates()

    if len(candidates) == 0:
        print("No T-phase candidates found. Check cluster labels and criteria.")
        return

    # Take top N by SNR
    selected = candidates.head(args.n_events)
    print(f"\n  Selected {len(selected)} events for montage")

    # Cluster breakdown
    print(f"  Source breakdown:")
    for cluster, count in selected["source_cluster"].value_counts().items():
        print(f"    {cluster}: {count}")

    # Extract snippets
    print(f"\n  Extracting waveforms and spectrograms...")
    snippets = []
    for i, (_, ev) in enumerate(selected.iterrows()):
        snip = extract_tphase_snippet(ev)
        snippets.append(snip)
        if (i + 1) % 10 == 0:
            n_ok = sum(1 for s in snippets if s is not None)
            print(f"    {i+1}/{len(selected)} extracted ({n_ok} ok)")

    n_ok = sum(1 for s in snippets if s is not None)
    print(f"  Total: {n_ok}/{len(selected)} snippets extracted")

    # Filter out failed extractions
    valid = [(ev, snip) for (_, ev), snip
             in zip(selected.iterrows(), snippets) if snip is not None]
    if not valid:
        print("No valid snippets extracted.")
        return

    valid_events = pd.DataFrame([ev for ev, _ in valid])
    valid_snippets = [snip for _, snip in valid]

    # Generate PDF
    outpath = FIG_DIR / "tphase_montage_for_review.pdf"
    n_pages = (len(valid_events) + args.events_per_page - 1) // args.events_per_page

    print(f"\n  Generating {n_pages}-page PDF...")
    with PdfPages(outpath) as pdf:
        for page in range(n_pages):
            start = page * args.events_per_page
            end = min(start + args.events_per_page, len(valid_events))
            page_events = valid_events.iloc[start:end]
            page_snippets = valid_snippets[start:end]

            plot_tphase_page(pdf, page_events, page_snippets,
                             page + 1, n_pages)

            print(f"    Page {page+1}/{n_pages}")

    print(f"\n  Saved: {outpath}")
    print(f"  {len(valid_events)} T-phase candidates across {n_pages} pages")
    print(f"\nDone. Send to Bob Dziak for review.")


if __name__ == "__main__":
    main()
