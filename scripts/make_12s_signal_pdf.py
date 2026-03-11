#!/usr/bin/env python3
"""
make_12s_signal_pdf.py — Compile all ~12s signal figures into a single PDF
with temporary captions, organized from longest to shortest time scale.

Usage:
    uv run python make_12s_signal_pdf.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from PIL import Image

FIG_DIR = Path(__file__).parent.parent / "outputs" / "figures" / "exploratory" / "12s_signal"
OUT_DIR = Path(__file__).parent.parent / "outputs" / "figures" / "exploratory"

MOORINGS = ["m3", "m5", "m1"]

# Ordered longest → shortest, with caption templates
FIGURE_SPECS = [
    # 60-min overviews
    {
        "tag": "60min_hour1",
        "caption": "{mooring} — 60-minute overview (hour 1). "
                   "Recurring broadband pulses with ~12 s periodicity visible in "
                   "both the bandpass-filtered waveform (10–40 Hz) and the spectrogram. "
                   "Signal energy is concentrated between 15–35 Hz. "
                   "File 943, 2019-06-19.",
    },
    {
        "tag": "60min_hour2",
        "caption": "{mooring} — 60-minute overview (hour 2). "
                   "Continuation of the same recording. Note whether the signal "
                   "persists, intensifies, or fades during the second hour. "
                   "File 943, 2019-06-19.",
    },
    # 5-min zooms
    {
        "tag": "5min_window1",
        "caption": "{mooring} — 5-minute zoom (starting ~10 min into file). "
                   "Individual pulses become resolvable. Note the regular spacing "
                   "(~12 s between pulse onsets) and the broadband spectral signature "
                   "of each pulse. File 943, 2019-06-19.",
    },
    {
        "tag": "5min_window2",
        "caption": "{mooring} — 5-minute zoom (starting ~40 min into file). "
                   "Second 5-min window for comparison. Is the pulse rate consistent? "
                   "Does the amplitude change? File 943, 2019-06-19.",
    },
    {
        "tag": "5min_window3",
        "caption": "{mooring} — 5-minute zoom (starting ~70 min into file). "
                   "Third 5-min window. By this point we are past the first hour — "
                   "does the signal character change? File 943, 2019-06-19.",
    },
    # 1-min details
    {
        "tag": "1min_detail1",
        "caption": "{mooring} — 1-minute detail (starting ~12 min into file). "
                   "Close-up of individual pulses. Each pulse appears as a "
                   "broadband burst spanning ~15–35 Hz with duration ~2–3 s "
                   "and inter-pulse interval ~12 s. File 943, 2019-06-19.",
    },
    {
        "tag": "1min_detail2",
        "caption": "{mooring} — 1-minute detail (starting ~42 min into file). "
                   "Second 1-min detail window. File 943, 2019-06-19.",
    },
    {
        "tag": "1min_detail3",
        "caption": "{mooring} — 1-minute detail (starting ~72 min into file). "
                   "Third 1-min detail window. File 943, 2019-06-19.",
    },
]


def main():
    outpath = OUT_DIR / "12s_recurring_signal_investigation.pdf"

    with PdfPages(str(outpath)) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.65,
                 "Recurring ~12 s Signal — BRAVOSEIS",
                 ha="center", va="center", fontsize=24, fontweight="bold")
        fig.text(0.5, 0.52,
                 "File 943 — 2019-06-19 — All moorings affected",
                 ha="center", va="center", fontsize=16)
        fig.text(0.5, 0.38,
                 "Signal characteristics:\n"
                 "  - Broadband pulses, energy concentrated 15–35 Hz\n"
                 "  - Inter-pulse interval ~12 seconds\n"
                 "  - Individual pulse duration ~2–3 s\n"
                 "  - Present on at least 3 moorings (M1, M3, M5)\n"
                 "  - Persists for at least 2 hours\n"
                 "  - 812 of 980 mid-band cluster 0 events from this file",
                 ha="center", va="center", fontsize=13,
                 family="monospace", linespacing=1.6)
        fig.text(0.5, 0.18,
                 "Figures organized: 60-min overview → 5-min zoom → 1-min detail\n"
                 "3 moorings shown for each time scale",
                 ha="center", va="center", fontsize=12, style="italic")
        fig.text(0.5, 0.06,
                 "Draft for colleague review — generated 2026-03-11",
                 ha="center", va="center", fontsize=10, color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        page_num = 1
        for spec in FIGURE_SPECS:
            for mooring in MOORINGS:
                fname = f"12s_{mooring}_{spec['tag']}.png"
                fpath = FIG_DIR / fname
                if not fpath.exists():
                    continue

                page_num += 1
                caption = spec["caption"].format(mooring=mooring.upper())

                # Load image
                img = Image.open(fpath)
                w, h = img.size
                aspect = h / w

                fig = plt.figure(figsize=(11, 8.5))

                # Image axes — leave room for caption at bottom
                ax_img = fig.add_axes([0.02, 0.15, 0.96, 0.83])
                ax_img.imshow(img)
                ax_img.axis("off")

                # Caption
                fig.text(0.05, 0.08, f"Figure {page_num}. {caption}",
                         ha="left", va="top", fontsize=10,
                         wrap=True,
                         transform=fig.transFigure)
                fig.text(0.95, 0.02, f"Page {page_num}",
                         ha="right", va="bottom", fontsize=8, color="gray")

                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved: {outpath}")
    print(f"  {page_num} pages (1 title + {page_num - 1} figures)")


if __name__ == "__main__":
    main()
