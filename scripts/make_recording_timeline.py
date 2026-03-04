#!/usr/bin/env python3
"""
make_recording_timeline.py — BRAVOSEIS recording timeline

Gantt-style figure showing when each of the 6 hydrophone moorings was
recording across the full 13-month deployment.  Each 4-hour DAT file
becomes one horizontal bar.  Reveals the duty cycle pattern and data gaps.

Usage:
    uv run python make_recording_timeline.py
"""

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from read_dat import MOORINGS, list_mooring_files
from make_bathy_map import add_caption_justified

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "figures" / "exploratory" / "detection"

# === Font sizes (paper tier) ===
FS_TITLE = 14
FS_AXIS_LABEL = 10
FS_TICK = 9
FS_LEGEND = 9
FS_CAPTION = 10
FS_STATS = 8

# === Okabe-Ito palette (6 colors) ===
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
]

# Mooring order (top to bottom on y-axis)
MOORING_KEYS = ["m1", "m2", "m3", "m4", "m5", "m6"]

FILE_DURATION = timedelta(hours=4)


def scan_all_moorings():
    """Scan DAT headers for all 6 moorings. Returns dict of catalogs."""
    catalogs = {}
    for key in MOORING_KEYS:
        info = MOORINGS[key]
        mdir = DATA_ROOT / info["data_dir"]
        if not mdir.exists():
            print(f"WARNING: {key} directory not found: {mdir}")
            catalogs[key] = []
            continue
        catalog = list_mooring_files(mdir)
        catalogs[key] = catalog
        print(f"  {key} / {info['name']} ({info['hydrophone']}): "
              f"{len(catalog)} files, "
              f"{catalog[0]['timestamp']} → {catalog[-1]['timestamp']}"
              if catalog else f"  {key}: no files")
    total = sum(len(c) for c in catalogs.values())
    print(f"  Total: {total} files")
    return catalogs


def make_recording_timeline():
    """Build and save the recording timeline figure."""
    print("Scanning DAT file headers...")
    catalogs = scan_all_moorings()

    total_files = sum(len(c) for c in catalogs.values())

    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_axes([0.12, 0.35, 0.82, 0.55])

    bar_height = 0.6

    for i, key in enumerate(MOORING_KEYS):
        info = MOORINGS[key]
        catalog = catalogs[key]
        color = OKABE_ITO[i]
        y = len(MOORING_KEYS) - 1 - i  # M1 at top

        # Draw each 4-hour recording as a horizontal bar
        for entry in catalog:
            t0 = entry["timestamp"]
            ax.barh(y, FILE_DURATION, left=t0, height=bar_height,
                    color=color, edgecolor='none', alpha=0.85)

        # Deploy / recover markers
        ax.plot(info["deployed"], y, marker='|', color='black',
                markersize=12, markeredgewidth=2.0, zorder=5)
        ax.plot(info["recovered"], y, marker='|', color='black',
                markersize=12, markeredgewidth=2.0, zorder=5)

    # --- Y-axis ---
    y_labels = []
    for key in MOORING_KEYS:
        info = MOORINGS[key]
        y_labels.append(f"{info['name']} / {key.upper()}\n({info['hydrophone']})")
    ax.set_yticks(range(len(MOORING_KEYS) - 1, -1, -1))
    ax.set_yticklabels(y_labels, fontsize=FS_TICK, family='sans-serif')
    ax.set_ylim(-0.5, len(MOORING_KEYS) - 0.5)

    # --- X-axis ---
    # Pad 5 days on each side
    all_times = []
    for cat in catalogs.values():
        for entry in cat:
            all_times.append(entry["timestamp"])
    if all_times:
        x_min = min(all_times) - timedelta(days=5)
        x_max = max(all_times) + FILE_DURATION + timedelta(days=5)
        ax.set_xlim(x_min, x_max)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.tick_params(axis='x', labelsize=FS_TICK)
    ax.set_xlabel("Date (UTC)", fontsize=FS_AXIS_LABEL, fontweight='bold',
                  family='sans-serif')

    # --- Grid ---
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # --- Spines ---
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # --- Title ---
    ax.set_title("BRAVOSEIS Hydrophone Recording Timeline",
                 fontsize=FS_TITLE, fontweight='bold', family='sans-serif',
                 pad=10)

    # --- Legend ---
    legend_handles = []
    for i, key in enumerate(MOORING_KEYS):
        info = MOORINGS[key]
        patch = mpatches.Patch(color=OKABE_ITO[i], alpha=0.85,
                               label=f"{info['name']}")
        legend_handles.append(patch)
    # Deploy/recover marker
    marker_handle = mlines.Line2D([], [], color='black', marker='|',
                                  linestyle='None', markersize=8,
                                  markeredgewidth=2.0,
                                  label='Deploy / Recover')
    legend_handles.append(marker_handle)
    ax.legend(handles=legend_handles, loc='upper right', fontsize=FS_LEGEND,
              frameon=True, fancybox=False, edgecolor='black',
              framealpha=0.9)

    # --- Stats box ---
    stats_lines = []
    for key in MOORING_KEYS:
        info = MOORINGS[key]
        n = len(catalogs[key])
        stats_lines.append(f"{info['name']:>5s}: {n:3d} files")
    stats_lines.append(f"{'Total':>5s}: {total_files:3d} files")
    stats_text = "\n".join(stats_lines)
    ax.text(0.01, 0.98, stats_text, transform=ax.transAxes,
            fontsize=FS_STATS, family='monospace', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='black', alpha=0.9))

    # --- Caption ---
    caption = (
        "Temporary Caption: "
        "Recording timeline for six BRAVOSEIS autonomous hydrophone moorings "
        "(BRA28\u2013BRA33) deployed in Bransfield Strait from January 2019 to "
        "February 2020. Each horizontal bar represents one 4-hour DAT file "
        "(1000 Hz, 24-bit). The duty cycle was approximately 8 hours of "
        "recording followed by 40 hours off (~5% temporal coverage). Black pipe "
        f"markers indicate deployment and recovery dates ({total_files} files total)."
    )
    add_caption_justified(fig, caption, caption_width=0.82, fontsize=FS_CAPTION,
                          caption_left=0.12, bold_prefix="Temporary Caption:")

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / "recording_timeline.png"
    fig.savefig(outpath, dpi=300, facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    make_recording_timeline()
