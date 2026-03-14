#!/usr/bin/env python3
"""
make_5hz_montage_pdf.py — Generate a PDF montage of the ~4.9 Hz repeating
signal observed on M6 on 2019-10-26 for review by Bob Dziak.

Includes:
- Context page with explanation and event table
- Individual waveform+spectrogram panels for representative events
- Multi-mooring comparison showing the signal across the array
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
PDF_PATH = OUTPUT_DIR / "docs" / "5hz_signal_montage_for_dziak.pdf"

# === Parameters ===
NPERSEG = 512
NOVERLAP = 448
FREQ_MAX = 50
WINDOW_SEC = 15
PAD_SEC = 5.0

_data_cache = {}
MAX_CACHE = 10


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
    nyq = fs / 2
    high = min(high, nyq * 0.99)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data)


def extract_snippet(event_row):
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
    t_start = onset - timedelta(seconds=PAD_SEC)
    offset_s = (t_start - file_ts).total_seconds()
    start_samp = int(offset_s * SAMPLE_RATE)
    end_samp = start_samp + int(WINDOW_SEC * SAMPLE_RATE)

    if start_samp < 0 or end_samp > len(data):
        return None

    segment = data[start_samp:end_samp].astype(np.float64)
    waveform_filt = bandpass(segment, 1.0, 15.0, SAMPLE_RATE)
    t_wave = np.arange(len(segment)) / SAMPLE_RATE

    freqs, times, Sxx = spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    freq_mask = freqs <= FREQ_MAX
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
    }


def make_context_page(pdf, events_df, all_lf):
    """First page: explanation and event table."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")

    title = "Unknown ~4.9 Hz Repeating Signal — Bransfield Strait"
    subtitle = "3-Day Episode: 2019-10-26 to 2019-10-28"

    text = f"""$\\bf{{{title}}}$

{subtitle}

OBSERVATION
During gold standard review of BRAVOSEIS low-band detections, a repeating
signal at ~4.9 Hz (and harmonics at ~6.8 Hz) was identified on mooring M6
on 2019-10-26 around 14:00 UTC. Further investigation reveals this is part
of a 3-day episode (Oct 26-28) with shifting mooring patterns.

CHARACTERISTICS
  - Peak frequency: 4.9 Hz (with related detections at 6.8 Hz)
  - Duration per pulse: ~1.6-2.3 s
  - Inter-pulse spacing: Irregular (not whale-like IPI)
  - SNR range: 3.1-22.7
  - Singer daily notes: No events logged for 2019-10-26

3-DAY EVOLUTION (3-10 Hz band, all moorings)
  - Oct 26: 841 events. M6 dominates (64%). Strongest 13:30-14:30 UTC.
  - Oct 27: 223 events (relative lull). Activity spreads across moorings.
            M1 shows earliest 4.9 Hz cluster at 15:38-15:40, then
            propagates to M3, M2, M4. Different pattern from Oct 26.
  - Oct 28: 1,284 events (highest day in October). M6 leads but M4 surges.
            Peak activity 15:00-18:00 UTC.

MOORING PATTERN SHIFT
  - Oct 26: M6=64%, M5=24%, M4=5%, M1=2%  (M6-dominated)
  - Oct 27: M6=32%, M5=27%, M4=16%, M1=9% (distributed)
  - Oct 28: M6=42%, M4=27%, M5=23%         (M4 emergence)
  The migration away from M6 argues against a fixed source.

CONTEXT
  - R/V Sarmiento de Gamboa airgun survey was Jan-Feb 2019 (not active)
  - No other known seismic survey in Bransfield Strait in Oct 2019
  - October is early austral spring (ice breakup beginning)
  - M6 is the southernmost mooring, closest to Deception Island

HYPOTHESES
  1. Volcanic harmonic tremor (Deception Island / Orca seamount)
     - Oct 26 M6-dominance fits, but migration to M1/M4 complicates this
  2. Ice-related source (migrating breakup front?)
  3. Distant airgun survey (Drake Passage / Scotia Sea, SOFAR channel)
  4. Unknown anthropogenic source

EVENT TABLE (~4.9 Hz detections, Oct 26, 13:30-14:30)
"""

    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top", horizontalalignment="left")

    # Add event table
    table_data = []
    for _, ev in events_df.iterrows():
        table_data.append([
            ev["event_id"],
            ev["mooring"].upper(),
            ev["onset_utc"].strftime("%H:%M:%S"),
            f"{ev['duration_s']:.1f}",
            f"{ev['snr']:.1f}",
            f"{ev['peak_freq_hz']:.1f}",
        ])

    if table_data:
        col_labels = ["Event ID", "Mooring", "Time (UTC)", "Dur (s)", "SNR", "Peak Hz"]
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="bottom",
            bbox=[0.02, 0.02, 0.96, 0.22],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_facecolor("#3C3C50")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#F0F0F8" if key[0] % 2 == 0 else "white")

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_event_page(pdf, ev, snip, page_num, total):
    """One waveform+spectrogram per event."""
    fig, (ax_wave, ax_spec) = plt.subplots(
        2, 1, figsize=(11, 6), height_ratios=[1, 1.2],
        gridspec_kw={"hspace": 0.3}
    )

    mooring = ev["mooring"].upper()
    time_str = ev["onset_utc"].strftime("%Y-%m-%d %H:%M:%S")
    eid = ev["event_id"]

    fig.suptitle(
        f"Event {eid} — {mooring} — {time_str}\n"
        f"peak_freq={ev['peak_freq_hz']:.1f} Hz   "
        f"dur={ev['duration_s']:.1f} s   "
        f"SNR={ev['snr']:.1f}   "
        f"[Page {page_num}/{total}]",
        fontsize=11, fontweight="bold"
    )

    # Waveform
    ax_wave.plot(snip["t_wave"], snip["waveform"], color="black", linewidth=0.5)
    ax_wave.axvspan(snip["ev_start"], snip["ev_end"], color="red", alpha=0.12)
    ax_wave.axvline(snip["ev_start"], color="red", linewidth=1.5,
                    linestyle="--", alpha=0.8, label="onset")
    ax_wave.axvline(snip["ev_end"], color="red", linewidth=1.0,
                    linestyle=":", alpha=0.6, label="end")
    ax_wave.set_xlim(0, WINDOW_SEC)
    ax_wave.set_ylabel("Amplitude", fontsize=10)
    ax_wave.set_title("Waveform — Bandpass 1–15 Hz", fontsize=10, loc="left")
    ax_wave.legend(fontsize=8, loc="upper right")

    # Spectrogram
    vmin = np.percentile(snip["Sxx_dB"], 5)
    vmax = np.percentile(snip["Sxx_dB"], 95)
    im = ax_spec.pcolormesh(snip["times"], snip["freqs"], snip["Sxx_dB"],
                            vmin=vmin, vmax=vmax, cmap="viridis",
                            shading="auto", rasterized=True)
    ax_spec.axvline(snip["ev_start"], color="white", linewidth=1.5,
                    linestyle="--", alpha=0.8)
    ax_spec.axvline(snip["ev_end"], color="white", linewidth=1.0,
                    linestyle=":", alpha=0.6)
    ax_spec.axhline(4.9, color="cyan", linewidth=1.0, linestyle="--",
                    alpha=0.8, label="4.9 Hz")
    ax_spec.axhline(6.8, color="orange", linewidth=1.0, linestyle="--",
                    alpha=0.8, label="6.8 Hz")
    ax_spec.set_ylim(0, FREQ_MAX)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=10)
    ax_spec.set_xlabel("Time (s)", fontsize=10)
    ax_spec.set_title("Spectrogram — 0–50 Hz", fontsize=10, loc="left")
    ax_spec.legend(fontsize=8, loc="upper right")

    cbar = fig.colorbar(im, ax=ax_spec, pad=0.02, aspect=30)
    cbar.set_label("Power (dB)", fontsize=9)

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading event catalogue...")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])

    # === Oct 26 events ===
    # Select ~4.9 Hz events across all moorings in the time window
    mask = (
        (cat["onset_utc"] >= "2019-10-26 13:30:00")
        & (cat["onset_utc"] <= "2019-10-26 14:30:00")
        & (cat["peak_freq_hz"] >= 4.5)
        & (cat["peak_freq_hz"] <= 5.5)
    )
    events_49_oct26 = cat[mask].sort_values("onset_utc").copy()
    print(f"Oct 26: Found {len(events_49_oct26)} events at ~4.9 Hz")

    # Also grab some 6.8 Hz events from M6 for comparison
    mask_68 = (
        (cat["onset_utc"] >= "2019-10-26 13:30:00")
        & (cat["onset_utc"] <= "2019-10-26 14:30:00")
        & (cat["mooring"] == "m6")
        & (cat["peak_freq_hz"] >= 6.3)
        & (cat["peak_freq_hz"] <= 7.3)
    )
    events_68_oct26 = cat[mask_68].sort_values("snr", ascending=False).head(3)
    print(f"Oct 26: Found {len(events_68_oct26)} high-SNR events at ~6.8 Hz on M6")

    # === Oct 27 events ===
    mask_49_oct27 = (
        (cat["onset_utc"] >= "2019-10-27 15:20:00")
        & (cat["onset_utc"] <= "2019-10-27 16:10:00")
        & (cat["peak_freq_hz"] >= 4.5)
        & (cat["peak_freq_hz"] <= 5.5)
    )
    events_49_oct27 = cat[mask_49_oct27].sort_values("onset_utc").copy()
    print(f"Oct 27: Found {len(events_49_oct27)} events at ~4.9 Hz")

    mask_68_oct27 = (
        (cat["onset_utc"] >= "2019-10-27 15:20:00")
        & (cat["onset_utc"] <= "2019-10-27 16:10:00")
        & (cat["peak_freq_hz"] >= 6.3)
        & (cat["peak_freq_hz"] <= 7.3)
    )
    events_68_oct27 = cat[mask_68_oct27].sort_values("snr", ascending=False).head(3)
    print(f"Oct 27: Found {len(events_68_oct27)} high-SNR events at ~6.8 Hz")

    # Combine all events
    all_events = pd.concat([
        events_49_oct26, events_68_oct26,
        events_49_oct27, events_68_oct27,
    ]).drop_duplicates("event_id").sort_values("onset_utc")

    # Events for context page table (Oct 26 only)
    events_49 = events_49_oct26

    # Also get all low-freq for context page table
    mask_all_lf = (
        (cat["onset_utc"] >= "2019-10-26 13:30:00")
        & (cat["onset_utc"] <= "2019-10-26 14:30:00")
        & (cat["peak_freq_hz"] >= 3)
        & (cat["peak_freq_hz"] <= 7)
    )
    all_lf = cat[mask_all_lf]

    # Generate PDF
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_pages = len(all_events) + 1

    with PdfPages(str(PDF_PATH)) as pdf:
        # Context page
        print("Generating context page...")
        make_context_page(pdf, events_49, all_lf)

        # Event pages
        for idx, (_, ev) in enumerate(all_events.iterrows()):
            page_num = idx + 2
            print(f"  Page {page_num}/{total_pages}: {ev['event_id']} "
                  f"({ev['mooring'].upper()}, {ev['peak_freq_hz']:.1f} Hz)")
            snip = extract_snippet(ev)
            if snip is None:
                print(f"    Extraction failed — skipped")
                continue
            make_event_page(pdf, ev, snip, page_num, total_pages)

    print(f"\nPDF saved to {PDF_PATH}")


if __name__ == "__main__":
    main()
