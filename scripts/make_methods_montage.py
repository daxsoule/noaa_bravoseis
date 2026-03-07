#!/usr/bin/env python3
"""
make_methods_montage.py — Publication-quality event type showcase montage.

Generates a 3×4 panel figure showing curated examples of each event class
(T-phase, Icequake, Vessel) with waveform + spectrogram + AIC onset pick.
Events are selected for high SNR and onset quality (grade A) to demonstrate
the best-case detection and picking performance.

Usage:
    uv run python make_methods_montage.py

Output:
    outputs/figures/paper/event_type_montage.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.signal import butter, sosfilt
from pathlib import Path
from datetime import timedelta

from read_dat import read_dat, list_mooring_files, MOORINGS, SAMPLE_RATE
from make_bathy_map import add_caption_justified

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "paper"

# === Spectrogram parameters ===
NPERSEG = 256
NOVERLAP = int(NPERSEG * 0.875)  # 87.5% overlap (matches CNN)
FREQ_MAX_DEFAULT = 100  # Hz for T-phase and Icequake
FREQ_MAX_VESSEL = 250   # Hz for Vessel (broadband, peak > 100 Hz)

# === Paper font sizes (14pt title, 10pt axis per project standard) ===
FS_TITLE = 16
FS_ROW_LABEL = 14
FS_PANEL_LABEL = 11
FS_AXIS = 10
FS_TICK = 9
FS_CAPTION = 10

# === Event selection parameters ===
N_EXAMPLES = 4  # columns per class

# === Band-specific filter for waveform display ===
BAND_FILTERS = {
    "low": (1, 15),
    "mid": (5, 30),   # Extended to 5 Hz for T-phases in mid band
    "high": (30, 250),
}


def get_filter(band, fs=SAMPLE_RATE):
    """Return a bandpass SOS filter for the given detection band."""
    lo, hi = BAND_FILTERS[band]
    return butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")


def select_events():
    """Select high-quality, high-SNR events for each class.

    Uses Phase 1 cluster labels for T-phases and icequakes (most reliable),
    and the feature-based vessel identification for vessel noise.
    """
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["onset_utc_refined"] = pd.to_datetime(cat["onset_utc_refined"])

    umap = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")
    feat = pd.read_parquet(DATA_DIR / "event_features.parquet")

    # Phase 1 T-phase clusters (expert-confirmed)
    tphase_clusters = {"low_0", "low_1", "mid_0"}
    tphase_ids = set(umap[umap["cluster_id"].isin(tphase_clusters)]["event_id"])

    # Phase 1 icequake: feature-based (duration > 3s, power > 48 dB)
    ice_candidates = feat[
        (feat["duration_s"] > 3.0)
        & (feat["peak_power_db"] > 48)
        & (feat["peak_freq_hz"] < 30)
        & (feat["spectral_slope"] < -0.2)
        & (feat["spectral_slope"] > -0.5)
    ]["event_id"]
    icequake_ids = set(ice_candidates) - tphase_ids

    # Vessel noise: Type A broadband transients
    vessel_candidates = feat[
        (feat["spectral_slope"] > 0)
        & (feat["peak_freq_hz"] > 100)
    ]["event_id"]
    vessel_ids = set(vessel_candidates) - tphase_ids - icequake_ids

    selections = {}
    for label, event_ids in [("T-phase", tphase_ids),
                              ("Icequake", icequake_ids),
                              ("Vessel", vessel_ids)]:
        # Filter catalogue to these events, grade A, high SNR
        subset = cat[
            (cat["event_id"].isin(event_ids))
            & (cat["onset_grade"] == "A")
            & (cat["snr"] > 5.0)
        ].copy()

        if len(subset) < N_EXAMPLES:
            # Relax to grade A/B
            subset = cat[
                (cat["event_id"].isin(event_ids))
                & (cat["onset_grade"].isin(["A", "B"]))
            ].copy()

        # Pick events spread across SNR range and moorings for variety
        subset = subset.sort_values("snr", ascending=False)

        # Sample from different SNR quartiles for variety
        picked = []
        moorings_used = set()
        if len(subset) >= N_EXAMPLES * 4:
            # Pick one from each SNR quartile, preferring different moorings
            quartile_size = len(subset) // N_EXAMPLES
            for q in range(N_EXAMPLES):
                q_start = q * quartile_size
                q_end = min((q + 1) * quartile_size, len(subset))
                q_subset = subset.iloc[q_start:q_end]
                # Prefer unseen mooring
                for _, row in q_subset.iterrows():
                    if row["mooring"] not in moorings_used:
                        picked.append(row)
                        moorings_used.add(row["mooring"])
                        break
                else:
                    picked.append(q_subset.iloc[0])
        else:
            for _, row in subset.iterrows():
                picked.append(row)
                if len(picked) >= N_EXAMPLES:
                    break

        picked_df = pd.DataFrame(picked).head(N_EXAMPLES)
        selections[label] = picked_df
        print(f"  {label}: {len(picked_df)} events selected "
              f"(SNR range: {picked_df['snr'].min():.1f}–{picked_df['snr'].max():.1f})")

    return selections


def load_event_waveform(event_row, pre_pad_s=2.0, post_pad_s=6.0):
    """Load the waveform segment for a single event.

    Returns (time_array, waveform, onset_sample_in_segment) or None.
    """
    mooring = event_row["mooring"]
    info = MOORINGS[mooring]
    mooring_dir = DATA_ROOT / info["data_dir"]

    if not mooring_dir.exists():
        return None

    # Find the DAT file by file_number
    file_number = int(event_row["file_number"])
    catalog = list_mooring_files(mooring_dir, sort_by="filename")
    dat_path = None
    for entry in catalog:
        if entry["file_number"] == file_number:
            dat_path = entry["path"]
            break

    if dat_path is None:
        return None

    file_ts, data, meta = read_dat(dat_path)

    # Use refined onset
    onset = event_row["onset_utc_refined"]
    if pd.isna(onset):
        onset = event_row["onset_utc"]

    onset_offset_s = (onset - file_ts).total_seconds()
    start_s = onset_offset_s - pre_pad_s
    end_s = onset_offset_s + post_pad_s

    start_samp = max(0, int(start_s * SAMPLE_RATE))
    end_samp = min(len(data), int(end_s * SAMPLE_RATE))

    if end_samp - start_samp < SAMPLE_RATE:  # Need at least 1 s
        return None

    segment = data[start_samp:end_samp].astype(np.float64)
    segment -= segment.mean()

    # Apply band-specific filter
    band = event_row["detection_band"]
    sos = get_filter(band)
    segment_filt = sosfilt(sos, segment)

    # Time array relative to onset
    t = np.arange(len(segment)) / SAMPLE_RATE - pre_pad_s
    onset_samp = int(pre_pad_s * SAMPLE_RATE)

    return t, segment, segment_filt, onset_samp, band


def _make_one_row(cls, events, color, freq_max, caption_text):
    """Create a single-row figure (1 class, 4 examples) with its own caption.

    Returns the saved path.
    """
    n_cols = len(events)
    fig = plt.figure(figsize=(14, 4.8))

    # Panels fill the top; caption goes in a reserved bottom strip
    panel_gs = GridSpec(2, n_cols, figure=fig,
                        hspace=0.08, wspace=0.25,
                        height_ratios=[1, 1.5],
                        top=0.88, bottom=0.32,
                        left=0.08, right=0.95)

    for col_idx in range(n_cols):
        event_row = events.iloc[col_idx]
        result = load_event_waveform(event_row)
        if result is None:
            continue

        t, segment_raw, segment_filt, onset_samp, band = result

        # --- Waveform panel ---
        ax_wave = fig.add_subplot(panel_gs[0, col_idx])
        ax_wave.plot(t, segment_filt, color="black", linewidth=0.3)
        ax_wave.axvline(x=0, color="red", linewidth=1.2, linestyle="-",
                        zorder=5)

        dur = event_row["duration_s"]
        ax_wave.axvspan(0, dur, color=color, alpha=0.15, zorder=1)

        ax_wave.set_xlim(t[0], t[-1])
        yl = np.percentile(np.abs(segment_filt), 99.5) * 1.3
        if yl > 0:
            ax_wave.set_ylim(-yl, yl)
        ax_wave.tick_params(labelsize=FS_TICK, length=2)
        ax_wave.set_xticklabels([])

        mooring_name = MOORINGS[event_row["mooring"]]["name"]
        ax_wave.set_title(
            f"{mooring_name}  SNR={event_row['snr']:.1f}  "
            f"Grade {event_row['onset_grade']}",
            fontsize=FS_PANEL_LABEL, pad=3)

        if col_idx == 0:
            ax_wave.set_ylabel("Amplitude", fontsize=FS_AXIS)

        # --- Spectrogram panel ---
        ax_spec = fig.add_subplot(panel_gs[1, col_idx])

        freqs, times_spec, Sxx = scipy_spectrogram(
            segment_raw, fs=SAMPLE_RATE,
            nperseg=NPERSEG, noverlap=NOVERLAP)
        freq_mask = freqs <= freq_max
        freqs = freqs[freq_mask]
        Sxx = Sxx[freq_mask, :]

        dB = 10 * np.log10(Sxx + 1e-20)
        vmin, vmax = np.percentile(dB, [2, 98])

        times_plot = np.linspace(t[0], t[-1], Sxx.shape[1])

        ax_spec.pcolormesh(times_plot, freqs, dB,
                           cmap="viridis", vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        ax_spec.axvline(x=0, color="red", linewidth=1.2, linestyle="-",
                        zorder=5)

        ax_spec.set_xlim(t[0], t[-1])
        ax_spec.set_ylim(0, freq_max)
        ax_spec.tick_params(labelsize=FS_TICK, length=2)

        if col_idx == 0:
            ax_spec.set_ylabel("Freq (Hz)", fontsize=FS_AXIS)
        ax_spec.set_xlabel("Time (s)", fontsize=FS_AXIS)

    # Row label on the left
    fig.text(0.01, 0.60, cls, fontsize=FS_ROW_LABEL, fontweight="bold",
             color=color, rotation=90, va="center", ha="center")

    # Title
    fig.suptitle(f"{cls} — Waveform and Spectrogram with AIC Onset Pick",
                 fontsize=FS_TITLE, fontweight="bold")

    # Per-row caption
    add_caption_justified(fig, caption_text, caption_width=0.85,
                          fontsize=FS_CAPTION, caption_left=0.08,
                          bold_prefix="Temporary Caption:")

    slug = cls.lower().replace("-", "").replace(" ", "_")
    outpath = FIG_DIR / f"event_montage_{slug}.png"
    fig.savefig(outpath, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


def make_montage(selections):
    """Create three separate event-type montage figures, one per class."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    class_configs = {
        "T-phase": {
            "color": "#d62728",
            "freq_max": FREQ_MAX_DEFAULT,
            "caption": (
                "Temporary Caption: T-phase (earthquake) examples from the "
                "BRAVOSEIS hydrophone array. Each column shows one event with "
                "bandpass-filtered waveform (1–15 Hz, top) and spectrogram "
                "(0–100 Hz, bottom). Red vertical lines mark the AIC-refined "
                "onset time (Maeda, 1985); red shading indicates the STA/LTA-"
                "detected event duration. Events are selected across a range "
                "of signal-to-noise ratios (SNR) and from different moorings. "
                "T-phases are the hydroacoustic signature of regional "
                "earthquakes: seismic energy converts to acoustic energy at "
                "the seafloor-water interface and propagates efficiently "
                "through the SOFAR channel. They are characterized by "
                "impulsive onsets, dominant energy below 15 Hz, steep negative "
                "spectral slopes (< −0.5), and short durations (typically "
                "≤3 s). The AIC picker identifies the noise-to-signal "
                "transition by minimizing the Akaike Information Criterion on "
                "the squared envelope within a 7 s window (5 s pre-trigger + "
                "2 s post-trigger). Grade A picks (quality ≥ 0.7) indicate "
                "high-confidence onsets suitable for TDOA source location. "
                "Spectrogram: nperseg=256, 87.5% overlap, Hann window."
            ),
        },
        "Icequake": {
            "color": "#1f77b4",
            "freq_max": FREQ_MAX_DEFAULT,
            "caption": (
                "Temporary Caption: Icequake (cryogenic) examples from the "
                "BRAVOSEIS hydrophone array. Each column shows one event with "
                "bandpass-filtered waveform (5–30 Hz, top) and spectrogram "
                "(0–100 Hz, bottom). Red vertical lines mark the AIC-refined "
                "onset time; blue shading indicates the detected event "
                "duration. Icequakes are distinguished from T-phases by their "
                "longer duration (>3 s), moderate spectral slope (−0.2 to "
                "−0.5), and sustained energy envelopes rather than impulsive "
                "onsets. They are attributed to glacial calving, ice shelf "
                "fracture, and sea ice cracking — sources concentrated near "
                "the Antarctic Peninsula coast, South Shetland Islands, and "
                "nearby ice shelves. The display filter extends to 5 Hz "
                "(below the 15 Hz detection passband) to capture leading-edge "
                "energy that may arrive before the mid-band trigger. Onset "
                "picking follows the same AIC procedure as T-phases, though "
                "the emergent character of many icequakes makes precise onset "
                "identification more challenging — reflected in a higher "
                "fraction of grade B picks for this class. "
                "Spectrogram: nperseg=256, 87.5% overlap, Hann window."
            ),
        },
        "Vessel": {
            "color": "#ff7f0e",
            "freq_max": FREQ_MAX_VESSEL,
            "caption": (
                "Temporary Caption: Vessel noise examples from the BRAVOSEIS "
                "hydrophone array. Each column shows one event with bandpass-"
                "filtered waveform (30–250 Hz, top) and spectrogram (0–250 Hz, "
                "bottom). Red vertical lines mark the AIC-refined onset time; "
                "orange shading indicates the detected event duration. Vessel "
                "noise is identified by its distinctive spectral character: "
                "positive spectral slope, peak energy above 100 Hz, broad "
                "bandwidth (~211 Hz), and high frequency modulation — the "
                "acoustic signature of propeller cavitation and ship machinery. "
                "These events were initially labeled 'Type A broadband "
                "transients' during unsupervised clustering and subsequently "
                "identified as vessel traffic based on temporal burst pattern "
                "(~24 passages over 13 months), multi-mooring simultaneity "
                "(47% of 200 s time bins show 2+ moorings detecting), and "
                "seasonal correlation with krill fishing fleet activity in "
                "CCAMLR Subarea 48.1 (peak May–Sep). The spectrogram is "
                "displayed to 250 Hz to show the full broadband character. "
                "Spectrogram: nperseg=256, 87.5% overlap, Hann window."
            ),
        },
    }

    paths = []
    for cls, cfg in class_configs.items():
        if cls not in selections or len(selections[cls]) == 0:
            continue
        p = _make_one_row(cls, selections[cls], cfg["color"],
                          cfg["freq_max"], cfg["caption"])
        paths.append(p)

    return paths


def main():
    print("=" * 60)
    print("Event Type Montage — Methods Figure")
    print("=" * 60)

    print("\nSelecting high-quality examples...")
    selections = select_events()

    print("\nGenerating montage...")
    make_montage(selections)

    print("\nDone.")


if __name__ == "__main__":
    main()
