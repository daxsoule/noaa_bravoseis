#!/usr/bin/env python3
"""
make_lowband2_pdf.py — PDF montage of lowband cluster 2 events (tonal 2–5 Hz
signals) at multiple time scales for review by Bob Dziak.

Shows representative events at 15 s, 60 s, and 5 min windows to characterize
these quasi-monochromatic signals.

Usage:
    uv run python make_lowband2_pdf.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import timedelta
from scipy.signal import spectrogram, butter, sosfilt

from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
PDF_PATH = OUTPUT_DIR / "docs" / "lowband2_tonal_signals_for_dziak.pdf"

# === Bandpass ===
BP_LOW = 1.0
BP_HIGH = 14.0

# === Time scales ===
SCALES = [
    {"label": "15 s (event detail)", "window_sec": 15, "pad_sec": 5,
     "nperseg": 2048, "noverlap": 1536, "fmax": 20},
    {"label": "60 s (near context)", "window_sec": 60, "pad_sec": 15,
     "nperseg": 2048, "noverlap": 1536, "fmax": 20},
    {"label": "5 min (broad context)", "window_sec": 300, "pad_sec": 30,
     "nperseg": 4096, "noverlap": 3584, "fmax": 20},
]

# Number of representative events to show
N_EVENTS = 8

_data_cache = {}
MAX_CACHE = 20


def get_data(filepath):
    key = str(filepath)
    if key not in _data_cache:
        if len(_data_cache) >= MAX_CACHE:
            oldest = next(iter(_data_cache))
            del _data_cache[oldest]
        ts, data, _ = read_dat(filepath)
        _data_cache[key] = (ts, data)
    return _data_cache[key]


def bandpass(data, low, high, fs, order=4):
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data)


def extract_multiscale(event_row, scale):
    """Extract waveform and spectrogram at a given time scale."""
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

    pad = scale["pad_sec"]
    window = scale["window_sec"]

    # Center the event in the window
    event_center = onset + timedelta(seconds=duration / 2)
    t_start = event_center - timedelta(seconds=window / 2)
    offset_s = (t_start - file_ts).total_seconds()
    start_samp = int(offset_s * SAMPLE_RATE)
    end_samp = start_samp + int(window * SAMPLE_RATE)

    # Clamp to file bounds
    if start_samp < 0:
        start_samp = 0
    if end_samp > file_nsamples:
        end_samp = file_nsamples
    if end_samp - start_samp < scale["nperseg"]:
        return None

    segment = data[start_samp:end_samp].astype(np.float64)
    waveform_filt = bandpass(segment, BP_LOW, BP_HIGH, SAMPLE_RATE)
    t_wave = np.arange(len(segment)) / SAMPLE_RATE

    # Event position relative to segment start
    ev_onset_rel = (onset - file_ts).total_seconds() - start_samp / SAMPLE_RATE
    ev_end_rel = ev_onset_rel + duration

    # Spectrogram on filtered signal
    freqs, times, Sxx = spectrogram(
        waveform_filt, fs=SAMPLE_RATE,
        nperseg=scale["nperseg"], noverlap=scale["noverlap"]
    )
    freq_mask = freqs <= scale["fmax"]
    freqs = freqs[freq_mask]
    Sxx_dB = 10 * np.log10(Sxx[freq_mask, :] + 1e-20)

    return {
        "t_wave": t_wave,
        "waveform": waveform_filt,
        "times": times,
        "freqs": freqs,
        "Sxx_dB": Sxx_dB,
        "ev_start": ev_onset_rel,
        "ev_end": ev_end_rel,
        "window_sec": window,
    }


def make_context_page(pdf, cluster_df, cat_stats):
    """First page: explanation and summary stats."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")

    text = """$\\bf{Lowband\\ Cluster\\ 2\\ —\\ Tonal\\ 2{-}5\\ Hz\\ Signals}$

BRAVOSEIS Lowband (1–14 Hz) Gold Standard Review
Phase 3 Frequency-Band Reclassification

OBSERVATION
During review of lowband (1–14 Hz) clusters, cluster lowband_2 (312 events)
shows highly distinctive quasi-monochromatic waveforms at 2–5 Hz that do not
resemble impulsive seismic events or T-phases.

CHARACTERISTICS
  - Peak frequency: 2.4–5.4 Hz (median 3.4 Hz)
  - Duration: 5–11 s (much longer than typical T-phases at this frequency)
  - Spectral slope: -1.7 to -4.1 (steep negative — energy concentrated)
  - Waveform morphology: very regular sinusoidal oscillations, not impulsive
  - Spectrograms show persistent horizontal energy bands at 2–4 Hz
    with harmonics at ~5 Hz and ~7 Hz
  - SNR: 3.0–12.7
  - Dates: mostly Oct 2019 – Feb 2020 (austral spring/summer)
  - Moorings: M1, M2, M3, M5 (distributed across array)

ASSESSMENT
These do NOT appear to be seismic events. The highly tonal, long-duration,
regular oscillations are more consistent with:
  1. Ocean microseism / infragravity wave signals
  2. Ice-related tremor (ice shelf or sea ice processes)
  3. Volcanic harmonic tremor
  4. Biological source (whale call not filtered by 14 Hz cutoff)

QUESTION FOR BOB
Can you identify these signals? Are they consistent with any known source
in the Bransfield Strait region?

SUMMARY STATISTICS"""

    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top", horizontalalignment="left")

    # Stats table
    table_data = [
        ["Events", f"{len(cluster_df):,}"],
        ["Peak freq (median)", f"{cat_stats['peak_freq_median']:.1f} Hz"],
        ["Peak freq (range)", f"{cat_stats['peak_freq_min']:.1f}–{cat_stats['peak_freq_max']:.1f} Hz"],
        ["Duration (median)", f"{cat_stats['duration_median']:.1f} s"],
        ["Duration (range)", f"{cat_stats['duration_min']:.1f}–{cat_stats['duration_max']:.1f} s"],
        ["SNR (median)", f"{cat_stats['snr_median']:.1f}"],
        ["Date range", f"{cat_stats['date_min']} to {cat_stats['date_max']}"],
        ["Moorings", cat_stats["moorings"]],
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="bottom",
        bbox=[0.1, 0.02, 0.8, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#3C3C50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F0F0F8" if key[0] % 2 == 0 else "white")

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_multiscale_page(pdf, ev, snippets, page_num, total):
    """One page per event showing all time scales."""
    n_scales = len(snippets)
    fig, axes = plt.subplots(
        n_scales * 2, 1, figsize=(11, 4 * n_scales),
        gridspec_kw={"hspace": 0.35}
    )

    mooring = ev["mooring"].upper()
    time_str = ev["onset_utc"].strftime("%Y-%m-%d %H:%M:%S")
    eid = ev["event_id"]
    pf = ev.get("peak_freq_hz", 0)
    dur = ev["duration_s"]
    snr = ev.get("snr", 0)

    fig.suptitle(
        f"Event {eid} — {mooring} — {time_str}\n"
        f"peak_freq={pf:.1f} Hz   dur={dur:.1f} s   SNR={snr:.1f}   "
        f"[Page {page_num}/{total}]",
        fontsize=12, fontweight="bold", y=0.99
    )

    for i, (scale, snip) in enumerate(zip(SCALES, snippets)):
        if snip is None:
            continue

        ax_wave = axes[i * 2]
        ax_spec = axes[i * 2 + 1]

        # Waveform
        ax_wave.plot(snip["t_wave"], snip["waveform"],
                     color="black", linewidth=0.4)
        ax_wave.axvspan(snip["ev_start"], snip["ev_end"],
                        color="red", alpha=0.12)
        ax_wave.axvline(snip["ev_start"], color="red", linewidth=1.2,
                        linestyle="--", alpha=0.7)
        ax_wave.axvline(snip["ev_end"], color="red", linewidth=0.8,
                        linestyle=":", alpha=0.5)
        ax_wave.set_xlim(0, len(snip["waveform"]) / SAMPLE_RATE)
        ax_wave.set_ylabel("Amplitude", fontsize=9)
        ax_wave.set_title(
            f"Waveform — {scale['label']} — Bandpass {BP_LOW:.0f}–{BP_HIGH:.0f} Hz",
            fontsize=10, loc="left", fontweight="bold"
        )
        ax_wave.tick_params(labelsize=8)

        # Spectrogram
        vmin = np.percentile(snip["Sxx_dB"], 5)
        vmax = np.percentile(snip["Sxx_dB"], 95)
        im = ax_spec.pcolormesh(
            snip["times"], snip["freqs"], snip["Sxx_dB"],
            vmin=vmin, vmax=vmax, cmap="viridis",
            shading="auto", rasterized=True
        )
        ax_spec.axvline(snip["ev_start"], color="white", linewidth=1.2,
                        linestyle="--", alpha=0.7)
        ax_spec.axvline(snip["ev_end"], color="white", linewidth=0.8,
                        linestyle=":", alpha=0.5)
        ax_spec.set_ylim(0, scale["fmax"])
        ax_spec.set_ylabel("Freq (Hz)", fontsize=9)
        ax_spec.set_xlabel("Time (s)", fontsize=9)
        ax_spec.set_title(
            f"Spectrogram — {scale['label']}",
            fontsize=10, loc="left"
        )
        ax_spec.tick_params(labelsize=8)

        cbar = fig.colorbar(im, ax=ax_spec, pad=0.02, aspect=20)
        cbar.set_label("dB", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading data...")
    umap_df = pd.read_parquet(DATA_DIR / "umap_coordinates_lowband.parquet")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    feat = pd.read_parquet(DATA_DIR / "event_features_lowband.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat["end_utc"] = pd.to_datetime(cat["end_utc"])

    # Get lowband_2 events
    lb2_ids = umap_df[umap_df["cluster_id"] == "lowband_2"]["event_id"]
    cluster_df = cat[cat["event_id"].isin(lb2_ids)].copy()
    cluster_df = cluster_df.merge(
        feat[["event_id", "peak_freq_hz"]].rename(
            columns={"peak_freq_hz": "peak_freq_hz_lb"}),
        on="event_id", how="left"
    )
    print(f"Cluster lowband_2: {len(cluster_df)} events")

    # Summary stats for context page
    cat_stats = {
        "peak_freq_median": cluster_df["peak_freq_hz_lb"].median(),
        "peak_freq_min": cluster_df["peak_freq_hz_lb"].min(),
        "peak_freq_max": cluster_df["peak_freq_hz_lb"].max(),
        "duration_median": cluster_df["duration_s"].median(),
        "duration_min": cluster_df["duration_s"].min(),
        "duration_max": cluster_df["duration_s"].max(),
        "snr_median": cluster_df["snr"].median(),
        "date_min": cluster_df["onset_utc"].min().strftime("%Y-%m-%d"),
        "date_max": cluster_df["onset_utc"].max().strftime("%Y-%m-%d"),
        "moorings": ", ".join(sorted(
            cluster_df["mooring"].str.upper().unique())),
    }

    # Select representative events: stratified by centroid distance
    # Use the UMAP coordinates to compute centroid distance
    lb2_umap = umap_df[umap_df["cluster_id"] == "lowband_2"].copy()
    cx = lb2_umap["umap_1"].mean()
    cy = lb2_umap["umap_2"].mean()
    lb2_umap["cdist"] = np.sqrt(
        (lb2_umap["umap_1"] - cx)**2 + (lb2_umap["umap_2"] - cy)**2
    )

    # Pick N_EVENTS spread across centroid distance
    lb2_umap = lb2_umap.sort_values("cdist")
    indices = np.linspace(0, len(lb2_umap) - 1, N_EVENTS, dtype=int)
    selected_ids = lb2_umap.iloc[indices]["event_id"].values

    selected = cluster_df[cluster_df["event_id"].isin(selected_ids)]
    # Preserve centroid-distance order
    id_order = {eid: i for i, eid in enumerate(selected_ids)}
    selected = selected.copy()
    selected["_order"] = selected["event_id"].map(id_order)
    selected = selected.sort_values("_order")

    print(f"Selected {len(selected)} representative events")

    # Generate PDF
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_pages = len(selected) + 1

    with PdfPages(str(PDF_PATH)) as pdf:
        # Context page
        print("Generating context page...")
        make_context_page(pdf, cluster_df, cat_stats)

        # Event pages
        for idx, (_, ev) in enumerate(selected.iterrows()):
            page_num = idx + 2
            eid = ev["event_id"]
            print(f"  Page {page_num}/{total_pages}: {eid} "
                  f"({ev['mooring'].upper()}, "
                  f"peak={ev['peak_freq_hz_lb']:.1f} Hz)")

            snippets = []
            for scale in SCALES:
                snip = extract_multiscale(ev, scale)
                snippets.append(snip)

            if all(s is None for s in snippets):
                print(f"    All scales failed — skipped")
                continue

            make_multiscale_page(pdf, ev, snippets, page_num, total_pages)

    print(f"\nPDF saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
