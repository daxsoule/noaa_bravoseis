#!/usr/bin/env python3
"""
make_high3_whale_figures.py — Generate multi-scale figures of suspected whale
calls from high_3 cluster for species identification.

142/143 events from file 2013 (2019-12-14), all 6 moorings.

Usage:
    uv run python make_high3_whale_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import timedelta
from scipy.signal import spectrogram, butter, sosfilt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "high3_whale"
GOLD_DIR = OUTPUT_DIR / "figures" / "exploratory" / "gold_standard"

# === Signal source ===
FILE_NUM = 2013
MOORINGS_TO_PLOT = ["m3", "m5", "m1"]

# === Display params ===
BANDPASS = (5, 250)


def load_file(mooring, file_num):
    info = MOORINGS[mooring]
    dat_path = DATA_ROOT / info["data_dir"] / f"{file_num:08d}.DAT"
    ts, data, _ = read_dat(dat_path)
    return ts, data


def bandpass_filter(data, low, high, fs, order=4):
    nyq = fs / 2
    high = min(high, nyq * 0.99)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data)


def make_figure(mooring, file_num, file_ts, data, start_min, duration_min,
                freq_max, nperseg, noverlap, tag):
    """Generate a waveform + spectrogram figure."""
    start_samp = int(start_min * 60 * SAMPLE_RATE)
    end_samp = start_samp + int(duration_min * 60 * SAMPLE_RATE)
    end_samp = min(end_samp, len(data))
    segment = data[start_samp:end_samp].astype(np.float64)

    t_seconds = np.arange(len(segment)) / SAMPLE_RATE

    if duration_min >= 5:
        t_plot = t_seconds / 60
        xlabel = "Time (minutes from start)"
    else:
        t_plot = t_seconds
        xlabel = "Time (seconds from start)"

    filt = bandpass_filter(segment, BANDPASS[0], BANDPASS[1], SAMPLE_RATE)

    freqs, times, Sxx = spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap
    )
    freq_mask = freqs <= freq_max
    freqs = freqs[freq_mask]
    Sxx_dB = 10 * np.log10(Sxx[freq_mask, :] + 1e-20)

    if duration_min >= 5:
        times_plot = times / 60
    else:
        times_plot = times

    window_start = file_ts + timedelta(minutes=start_min)
    window_end = file_ts + timedelta(minutes=start_min + duration_min)

    fig, (ax_wave, ax_spec) = plt.subplots(
        2, 1, figsize=(16, 8), height_ratios=[1, 1.3],
        gridspec_kw={"hspace": 0.3}
    )

    ax_wave.plot(t_plot, filt, color="black", linewidth=0.4)
    ax_wave.set_xlim(t_plot[0], t_plot[-1])
    ax_wave.set_ylabel("Amplitude", fontsize=12)
    dur_label = f"{duration_min:.0f} min" if duration_min >= 1 else f"{duration_min*60:.0f} s"
    ax_wave.set_title(
        f"{mooring.upper()} — File {file_num} — "
        f"Bandpass {BANDPASS[0]}–{BANDPASS[1]} Hz\n"
        f"{window_start.strftime('%Y-%m-%d %H:%M:%S')} to "
        f"{window_end.strftime('%H:%M:%S')} UTC "
        f"({dur_label})",
        fontsize=13, fontweight="bold"
    )
    ax_wave.tick_params(labelsize=10)

    vmin = np.percentile(Sxx_dB, 5)
    vmax = np.percentile(Sxx_dB, 95)
    im = ax_spec.pcolormesh(times_plot, freqs, Sxx_dB,
                             vmin=vmin, vmax=vmax, cmap="viridis",
                             shading="auto", rasterized=True)
    ax_spec.set_ylim(0, freq_max)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=12)
    ax_spec.set_xlabel(xlabel, fontsize=12)
    ax_spec.set_title(f"Spectrogram — 0–{freq_max} Hz",
                       fontsize=12, fontweight="bold", loc="left")
    ax_spec.tick_params(labelsize=10)

    cbar = fig.colorbar(im, ax=ax_spec, pad=0.02, aspect=30)
    cbar.set_label("Power (dB)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    outpath = FIG_DIR / f"whale_{mooring}_{tag}.png"
    fig.savefig(outpath, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {outpath.name}")
    return outpath


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Suspected Whale Calls (high_3) — Multi-scale Figures")
    print(f"  Source: file {FILE_NUM} (2019-12-14)")
    print(f"  Moorings: {', '.join(m.upper() for m in MOORINGS_TO_PLOT)}")
    print(f"  143 events, all 6 moorings, austral summer")
    print("=" * 60)

    all_figures = []

    for mooring in MOORINGS_TO_PLOT:
        print(f"\n--- {mooring.upper()} ---")
        file_ts, data = load_file(mooring, FILE_NUM)
        file_duration_min = len(data) / SAMPLE_RATE / 60
        print(f"  File start: {file_ts}, duration: {file_duration_min:.1f} min")

        # 60-min overview — first hour
        all_figures.append(make_figure(
            mooring, FILE_NUM, file_ts, data,
            start_min=0, duration_min=60,
            freq_max=250, nperseg=512, noverlap=448,
            tag="60min_hour1"))

        # 60-min overview — second hour
        if file_duration_min > 60:
            all_figures.append(make_figure(
                mooring, FILE_NUM, file_ts, data,
                start_min=60, duration_min=min(60, file_duration_min - 60),
                freq_max=250, nperseg=512, noverlap=448,
                tag="60min_hour2"))

        # 5-min zoom windows
        for i, start in enumerate([5, 30, 55]):
            if start + 5 <= file_duration_min:
                all_figures.append(make_figure(
                    mooring, FILE_NUM, file_ts, data,
                    start_min=start, duration_min=5,
                    freq_max=250, nperseg=256, noverlap=224,
                    tag=f"5min_window{i+1}"))

        # 1-min detail windows
        for i, start in enumerate([6, 31, 56]):
            if start + 1 <= file_duration_min:
                all_figures.append(make_figure(
                    mooring, FILE_NUM, file_ts, data,
                    start_min=start, duration_min=1,
                    freq_max=250, nperseg=256, noverlap=224,
                    tag=f"1min_detail{i+1}"))

    # === Build PDF ===
    print("\nBuilding PDF...")
    from PIL import Image as PILImage

    outpath = OUTPUT_DIR / "figures" / "exploratory" / "high3_suspected_whale_calls.pdf"

    # Collect gold standard panels too
    gold_panels = sorted(GOLD_DIR.glob("gold_high_3_stratified_panel*.png"))[:10]

    FIGURE_SPECS = [
        ("60min_hour1", "{mooring} — 60-minute overview (hour 1). "
         "Suspected whale calls visible as repeated broadband pulses. "
         "File 2013, 2019-12-14 (austral summer)."),
        ("60min_hour2", "{mooring} — 60-minute overview (hour 2). "
         "Continuation of the same recording. "
         "File 2013, 2019-12-14."),
        ("5min_window1", "{mooring} — 5-minute zoom (starting ~5 min). "
         "Individual calls become resolvable. Note frequency content and "
         "call structure. File 2013, 2019-12-14."),
        ("5min_window2", "{mooring} — 5-minute zoom (starting ~30 min). "
         "Second window for comparison. File 2013, 2019-12-14."),
        ("5min_window3", "{mooring} — 5-minute zoom (starting ~55 min). "
         "Third window. File 2013, 2019-12-14."),
        ("1min_detail1", "{mooring} — 1-minute detail (starting ~6 min). "
         "Close-up of individual calls showing spectral structure. "
         "File 2013, 2019-12-14."),
        ("1min_detail2", "{mooring} — 1-minute detail (starting ~31 min). "
         "Second detail window. File 2013, 2019-12-14."),
        ("1min_detail3", "{mooring} — 1-minute detail (starting ~56 min). "
         "Third detail window. File 2013, 2019-12-14."),
    ]

    with PdfPages(str(outpath)) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.70,
                 "Suspected Whale Calls — BRAVOSEIS high_3 Cluster",
                 ha="center", va="center", fontsize=22, fontweight="bold")
        fig.text(0.5, 0.58,
                 "File 2013 — 2019-12-14 — Bransfield Strait",
                 ha="center", va="center", fontsize=16)
        fig.text(0.5, 0.42,
                 "Signal characteristics:\n"
                 "  - 143 events across all 6 moorings\n"
                 "  - Broadband: energy from <10 Hz to >200 Hz\n"
                 "  - Individual call duration: ~2–3 s\n"
                 "  - Peak frequency: median 16.6 Hz (range 3–66 Hz)\n"
                 "  - Spectral centroid: ~35 Hz\n"
                 "  - Date: 2019-12-14 (austral summer)\n"
                 "  - 142/143 events from single 4-hour file\n"
                 "  - KEY FEATURE: significant energy above 200 Hz\n"
                 "  - Preliminary ID: blue whale?",
                 ha="center", va="center", fontsize=13,
                 family="monospace", linespacing=1.6)
        fig.text(0.5, 0.18,
                 "Figures organized: gold standard panels → 60-min → 5-min → 1-min\n"
                 "3 moorings (M3, M5, M1) shown for multi-scale views\n\n"
                 "The key diagnostic feature is the energy concentrated above 200 Hz.\n"
                 "Question: Can you confirm species identification?",
                 ha="center", va="center", fontsize=12, style="italic")
        fig.text(0.5, 0.06,
                 "Draft for colleague review — generated 2026-03-11",
                 ha="center", va="center", fontsize=10, color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        page_num = 1

        # Gold standard panels first (best individual examples)
        for gp in gold_panels:
            page_num += 1
            img = PILImage.open(gp)
            fig = plt.figure(figsize=(11, 8.5))
            ax_img = fig.add_axes([0.02, 0.12, 0.96, 0.86])
            ax_img.imshow(img)
            ax_img.axis("off")
            panel_name = gp.stem.replace("gold_high_3_stratified_", "Panel ")
            fig.text(0.05, 0.06,
                     f"Figure {page_num}. Gold standard review — {panel_name}. "
                     f"15s window (pick − 5s, pick + 10s), unfiltered spectrogram 0–250 Hz, "
                     f"bandpass waveform 30–250 Hz.",
                     ha="left", va="top", fontsize=10, wrap=True,
                     transform=fig.transFigure)
            fig.text(0.95, 0.02, f"Page {page_num}",
                     ha="right", va="bottom", fontsize=8, color="gray")
            pdf.savefig(fig)
            plt.close(fig)

        # Multi-scale figures
        for tag, caption_template in FIGURE_SPECS:
            for mooring in MOORINGS_TO_PLOT:
                fpath = FIG_DIR / f"whale_{mooring}_{tag}.png"
                if not fpath.exists():
                    continue
                page_num += 1
                caption = caption_template.format(mooring=mooring.upper())
                img = PILImage.open(fpath)
                fig = plt.figure(figsize=(11, 8.5))
                ax_img = fig.add_axes([0.02, 0.15, 0.96, 0.83])
                ax_img.imshow(img)
                ax_img.axis("off")
                fig.text(0.05, 0.08, f"Figure {page_num}. {caption}",
                         ha="left", va="top", fontsize=10, wrap=True,
                         transform=fig.transFigure)
                fig.text(0.95, 0.02, f"Page {page_num}",
                         ha="right", va="bottom", fontsize=8, color="gray")
                pdf.savefig(fig)
                plt.close(fig)

    print(f"\nSaved: {outpath}")
    print(f"  {page_num} pages (1 title + {len(gold_panels)} gold panels + "
          f"{page_num - 1 - len(gold_panels)} multi-scale figures)")


if __name__ == "__main__":
    main()
