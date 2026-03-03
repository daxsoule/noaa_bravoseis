#!/usr/bin/env python3
"""
make_detection_figures.py — Detection catalogue figures for BRAVOSEIS.

Generates 4 journal-quality figures from the event catalogue and
cross-mooring associations:

  1. Detection rate timeline (events per recording window, stacked by band)
  2. Duration vs. peak frequency scatter (colored by detection band)
  3. Example detections (array spectrograms with trigger overlays)
  4. Cross-mooring statistics (bar chart by number of moorings)

Usage:
    uv run python make_detection_figures.py

Spec: specs/001-event-detection/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.dates import DateFormatter, MonthLocator
from pathlib import Path
from datetime import timedelta
from scipy.signal import spectrogram

from read_dat import read_dat, list_mooring_files, MOORINGS, SAMPLE_RATE
from make_bathy_map import add_caption_justified

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory"

# === Paper-tier font sizes ===
FS_TITLE = 14
FS_AXIS_LABEL = 10
FS_TICK = 9
FS_LEGEND = 9
FS_CAPTION = 10
FS_PANEL_LABEL = 10

# === Spectrogram parameters (matching detect_events.py) ===
NPERSEG = 1024
NOVERLAP = 512
FREQ_MAX = 250

# === Colorblind-safe band colors (Okabe-Ito subset) ===
BAND_COLORS = {
    "low":       "#E69F00",   # orange
    "mid":       "#56B4E9",   # sky blue
    "high":      "#009E73",   # bluish green
    "broadband": "#CC79A7",   # reddish purple
}

BAND_LABELS = {
    "low":       "Low (1–50 Hz)",
    "mid":       "Mid (10–200 Hz)",
    "high":      "High (50–250 Hz)",
    "broadband": "Broadband (1–250 Hz)",
}

MOORING_KEYS = sorted(MOORINGS.keys())


def load_data():
    """Load event catalogue and associations."""
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    assoc = pd.read_parquet(DATA_DIR / "cross_mooring_associations.parquet")
    assoc["earliest_utc"] = pd.to_datetime(assoc["earliest_utc"])
    return cat, assoc


# ======================================================================
# Figure 1: Detection Rate Timeline
# ======================================================================

def fig_detection_rate_timeline(cat):
    """Events per day, stacked by detection band."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(bottom=0.35)

    # Bin by day
    cat = cat.copy()
    cat["date"] = cat["onset_utc"].dt.date

    bands = ["low", "mid", "high", "broadband"]
    # Pivot: rows=date, columns=band, values=count
    daily = cat.groupby(["date", "detection_band"]).size().unstack(fill_value=0)
    # Ensure all bands present
    for b in bands:
        if b not in daily.columns:
            daily[b] = 0
    daily = daily[bands]
    daily.index = pd.to_datetime(daily.index)

    # Stacked bar
    bottom = np.zeros(len(daily))
    bar_width = 1.5  # days
    for band in bands:
        ax.bar(daily.index, daily[band], bottom=bottom, width=bar_width,
               color=BAND_COLORS[band], label=BAND_LABELS[band],
               edgecolor="none")
        bottom += daily[band].values

    ax.set_ylabel("Events per day", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_xlabel("Date (UTC)", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_major_locator(MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.legend(fontsize=FS_LEGEND, loc="upper right", framealpha=0.9)
    ax.set_title("Event Detection Rate", fontsize=FS_TITLE, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Caption
    caption = (
        "Daily event counts across all 6 BRAVOSEIS hydrophone moorings, "
        "stacked by detection frequency band. STA/LTA detector parameters: "
        "STA=2 s, LTA=60 s, trigger=3.0, detrigger=1.5. Gaps correspond to "
        "the ~40-hour off-duty periods in the recording duty cycle (~5% "
        "temporal coverage). Events detected in the broadband band (1–250 Hz) "
        "are those not attributable to a narrower band."
    )
    add_caption_justified(fig, caption, caption_width=0.85, fontsize=FS_CAPTION,
                          caption_left=0.08, bold_prefix="Temporary Caption:")

    outpath = FIG_DIR / "detection_rate_timeline.png"
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ======================================================================
# Figure 2: Duration vs. Peak Frequency
# ======================================================================

def fig_duration_vs_peak_freq(cat):
    """Scatter plot of event duration vs. peak frequency."""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.subplots_adjust(bottom=0.30)

    bands = ["low", "mid", "high", "broadband"]
    for band in bands:
        subset = cat[cat["detection_band"] == band]
        ax.scatter(subset["peak_freq_hz"], subset["duration_s"],
                   c=BAND_COLORS[band], label=BAND_LABELS[band],
                   s=3, alpha=0.3, edgecolors="none", rasterized=True)

    ax.set_xlabel("Peak Frequency (Hz)", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_ylabel("Duration (s)", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(0, FREQ_MAX)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc="upper right", framealpha=0.9,
              markerscale=3)
    ax.set_title("Event Duration vs. Peak Frequency", fontsize=FS_TITLE,
                 fontweight="bold")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    caption = (
        "Scatter plot of event duration versus peak frequency for all "
        f"{len(cat):,} detected events across 6 moorings. Each point is "
        "colored by the frequency band in which it was detected. Duration "
        "is plotted on a logarithmic scale. Peak frequency is the frequency "
        "with maximum mean power in the event spectrogram. Events cluster "
        "by band as expected: low-band events peak below 50 Hz, high-band "
        "events above 50 Hz."
    )
    add_caption_justified(fig, caption, caption_width=0.85, fontsize=FS_CAPTION,
                          caption_left=0.08, bold_prefix="Temporary Caption:")

    outpath = FIG_DIR / "duration_vs_peak_freq.png"
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ======================================================================
# Figure 3: Example Detections (Array Spectrograms with Overlays)
# ======================================================================

def fig_example_detections(cat, assoc):
    """Array spectrograms for selected multi-mooring events with detection
    markers overlaid."""

    # Pick 3 associations with the most moorings, spread across time
    assoc_sorted = assoc.sort_values("n_moorings", ascending=False)
    # Get top associations with >= 4 moorings, pick 3 spread in time
    top = assoc_sorted[assoc_sorted["n_moorings"] >= 4].head(50)
    if len(top) < 3:
        top = assoc_sorted.head(50)

    # Spread across time: pick early, middle, late
    top = top.sort_values("earliest_utc")
    if len(top) >= 3:
        indices = [0, len(top) // 2, len(top) - 1]
        selected = top.iloc[indices]
    else:
        selected = top.head(min(3, len(top)))

    if len(selected) == 0:
        print("No multi-mooring associations to plot!")
        return

    for idx, (_, row) in enumerate(selected.iterrows()):
        _plot_one_example(cat, row, idx)


def _plot_one_example(cat, assoc_row, fig_idx):
    """Plot one array spectrogram centered on an association."""
    event_ids = assoc_row["event_ids"].split(",")
    events = cat[cat["event_id"].isin(event_ids)]
    center_time = assoc_row["earliest_utc"]

    # Window: 2 minutes centered on the association
    window_sec = 120
    half = timedelta(seconds=window_sec / 2)
    t_start = center_time - half
    t_end = center_time + half

    fig = plt.figure(figsize=(7, 9))

    # Layout: 6 rows stacked
    plot_left = 0.14
    plot_right = 0.88
    plot_top = 0.92
    plot_bottom = 0.30
    total_height = plot_top - plot_bottom
    n_rows = 6
    gap_frac = 0.08
    row_height = total_height / (n_rows + (n_rows - 1) * gap_frac)
    row_gap = row_height * gap_frac

    all_dB = []
    axes = []
    panel_data = []

    for r, mkey in enumerate(MOORING_KEYS):
        y_bottom = plot_top - (r + 1) * row_height - r * row_gap
        ax = fig.add_axes([plot_left, y_bottom, plot_right - plot_left, row_height])
        axes.append(ax)

        # Find the DAT file that contains this time window
        info = MOORINGS[mkey]
        mooring_dir = DATA_ROOT / info["data_dir"]
        if not mooring_dir.exists():
            panel_data.append(None)
            continue

        catalog = list_mooring_files(mooring_dir, sort_by="filename")
        found = False
        for entry in catalog:
            file_ts, data, meta = read_dat(entry["path"])
            file_end = file_ts + timedelta(seconds=len(data) / SAMPLE_RATE)
            if file_ts <= t_start and file_end >= t_end:
                # Extract segment
                offset_s = (t_start - file_ts).total_seconds()
                start_samp = int(offset_s * SAMPLE_RATE)
                end_samp = start_samp + int(window_sec * SAMPLE_RATE)
                if end_samp > len(data):
                    end_samp = len(data)
                segment = data[start_samp:end_samp]

                freqs, times, Sxx = spectrogram(
                    segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
                )
                freq_mask = freqs <= FREQ_MAX
                freqs = freqs[freq_mask]
                Sxx = Sxx[freq_mask, :]
                Sxx_dB = 10 * np.log10(Sxx + 1e-20)
                all_dB.append(Sxx_dB)
                panel_data.append((freqs, times, Sxx_dB))
                found = True
                break

        if not found:
            panel_data.append(None)

    # Shared color limits
    if all_dB:
        combined = np.concatenate([d.ravel() for d in all_dB])
        vmin = np.percentile(combined, 2)
        vmax = np.percentile(combined, 98)
    else:
        vmin, vmax = -80, -20

    # Plot each panel
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for r, (ax, mkey) in enumerate(zip(axes, MOORING_KEYS)):
        info = MOORINGS[mkey]
        label = f"{info['name']} / {mkey.upper()}"

        if panel_data[r] is None:
            ax.set_facecolor("#e0e0e0")
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    fontsize=12, color="#888888", transform=ax.transAxes)
            ax.set_yticks([0, 50, 100, 150, 200, 250])
            ax.set_ylim(0, FREQ_MAX)
        else:
            freqs, times, Sxx_dB = panel_data[r]
            ax.pcolormesh(times / 60, freqs, Sxx_dB, vmin=vmin, vmax=vmax,
                          cmap="viridis", shading="auto", rasterized=True)
            ax.set_ylim(0, FREQ_MAX)
            ax.set_yticks([0, 50, 100, 150, 200, 250])

            # Overlay detection markers
            m_events = events[events["mooring"] == mkey]
            for _, ev in m_events.iterrows():
                ev_onset = ev["onset_utc"]
                ev_end = ev["end_utc"]
                # Convert to minutes relative to window start
                t0_min = (ev_onset - t_start).total_seconds() / 60
                t1_min = (ev_end - t_start).total_seconds() / 60
                ax.axvspan(t0_min, t1_min, color="red", alpha=0.3, linewidth=0)
                ax.axvline(t0_min, color="red", linewidth=1.0, alpha=0.8)

        ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center",
                      labelpad=5)
        ax.tick_params(labelsize=FS_TICK)
        ax.text(0.01, 0.92, panel_labels[r], transform=ax.transAxes,
                fontsize=FS_PANEL_LABEL, fontweight="bold", va="top",
                color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5))

        if r < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (minutes)", fontsize=FS_AXIS_LABEL,
                          fontweight="bold")

    # Shared y-axis label
    fig.text(0.02, (plot_top + plot_bottom) / 2, "Frequency (Hz)",
             fontsize=FS_AXIS_LABEL, fontweight="bold",
             rotation=90, ha="center", va="center")

    # Shared colorbar
    cbar_ax = fig.add_axes([plot_right + 0.02, plot_bottom, 0.02,
                            plot_top - plot_bottom])
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Power (dB)", fontsize=FS_AXIS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)

    # Title
    n_m = assoc_row["n_moorings"]
    moorings_str = assoc_row["moorings"]
    fig.suptitle(
        f"{center_time.strftime('%Y-%m-%d %H:%M UTC')} — "
        f"{n_m} moorings ({moorings_str})",
        fontsize=FS_TITLE, fontweight="bold", y=0.96)

    caption = (
        f"Two-minute spectrogram array centered on association "
        f"{assoc_row['assoc_id']} detected on {n_m} moorings "
        f"({moorings_str}). Red shading marks detected event windows; "
        f"red vertical lines mark onset times. Spectrogram parameters: "
        f"nperseg=1024, 50% overlap, fs=1000 Hz, 0–250 Hz. "
        f"Shared colorscale (2nd–98th percentile)."
    )
    add_caption_justified(fig, caption, caption_width=0.82, fontsize=FS_CAPTION,
                          caption_left=0.08, bold_prefix="Temporary Caption:")

    ts_str = center_time.strftime("%Y%m%d_%H%M")
    outpath = FIG_DIR / f"example_detection_{ts_str}.png"
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ======================================================================
# Figure 4: Cross-Mooring Statistics
# ======================================================================

def fig_cross_mooring_stats(cat, assoc):
    """Bar chart of events by number of moorings detected on."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.subplots_adjust(bottom=0.35)

    # Count events in associations by n_moorings
    # Also count isolated (single-mooring) events
    all_assoc_events = set()
    for eids in assoc["event_ids"]:
        all_assoc_events.update(eids.split(","))

    n_isolated = len(cat) - len(all_assoc_events)

    # Build counts: 1 mooring (isolated), 2, 3, 4, 5, 6
    counts_by_n = {1: n_isolated}
    for n_m in sorted(assoc["n_moorings"].unique()):
        # Count events in associations with this many moorings
        subset = assoc[assoc["n_moorings"] == n_m]
        n_events = sum(len(eids.split(",")) for eids in subset["event_ids"])
        counts_by_n[n_m] = n_events

    x_vals = sorted(counts_by_n.keys())
    y_vals = [counts_by_n[x] for x in x_vals]
    colors = ["#999999" if x == 1 else "#0072B2" for x in x_vals]

    bars = ax.bar(x_vals, y_vals, color=colors, edgecolor="black", linewidth=0.5)

    # Add count labels on bars
    for bar, y in zip(bars, y_vals):
        if y > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{y:,}", ha="center", va="bottom", fontsize=FS_TICK)

    ax.set_xlabel("Number of Moorings", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_ylabel("Number of Events", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_title("Event Detection Across Moorings", fontsize=FS_TITLE,
                 fontweight="bold")
    ax.set_xticks(x_vals)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_yscale("log")

    legend_elements = [
        Patch(facecolor="#999999", edgecolor="black", label="Isolated (1 mooring)"),
        Patch(facecolor="#0072B2", edgecolor="black", label="Multi-mooring association"),
    ]
    ax.legend(handles=legend_elements, fontsize=FS_LEGEND, loc="upper right")

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    caption = (
        f"Distribution of events by the number of moorings on which they "
        f"were detected. Gray: isolated events detected on a single mooring "
        f"only. Blue: events associated across multiple moorings using "
        f"pair-specific travel-time windows derived from in-situ XBT sound "
        f"speed profiles (BRAVOSEIS cruise, Jan 2019; effective speed ~1456 "
        f"m/s, 15% safety factor). Windows range from 21 s (M4–M5, 27 km) "
        f"to 139 s (M1–M6, 176 km). Total events: {len(cat):,}; "
        f"associations: {len(assoc):,}. Note: high association rates may "
        f"include coincidental matches due to event density; source location "
        f"via ray tracing is needed to confirm true multi-mooring detections."
    )
    add_caption_justified(fig, caption, caption_width=0.85, fontsize=FS_CAPTION,
                          caption_left=0.08, bold_prefix="Temporary Caption:")

    outpath = FIG_DIR / "cross_mooring_statistics.png"
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("Detection Catalogue Figures")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cat, assoc = load_data()
    print(f"Loaded {len(cat):,} events, {len(assoc):,} associations")

    print("\n--- Figure 1: Detection Rate Timeline ---")
    fig_detection_rate_timeline(cat)

    print("\n--- Figure 2: Duration vs. Peak Frequency ---")
    fig_duration_vs_peak_freq(cat)

    print("\n--- Figure 3: Example Detections ---")
    fig_example_detections(cat, assoc)

    print("\n--- Figure 4: Cross-Mooring Statistics ---")
    fig_cross_mooring_stats(cat, assoc)

    print(f"\n{'=' * 60}")
    print("All figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
