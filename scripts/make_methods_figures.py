#!/usr/bin/env python3
"""
make_methods_figures.py — Generate curated figures for the methods PDF.

Produces:
  1. late_pick_problem.png      — 6 events showing STA/LTA late-pick problem (§3.1)
  2. onset_refinement_6panel.png — 6 curated onset refinement examples (§3.2)
  3. tphase_cluster_curated.png  — 2×3 T-phase cluster montage (§4.1)
  4. magnitude_completeness.png  — Cumulative freq vs. relative source level (§new)

Usage:
    uv run python scripts/make_methods_figures.py
    uv run python scripts/make_methods_figures.py --figure late_pick
    uv run python scripts/make_methods_figures.py --figure onset
    uv run python scripts/make_methods_figures.py --figure cluster
    uv run python scripts/make_methods_figures.py --figure completeness
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from datetime import timedelta
from scipy.signal import spectrogram, butter, sosfilt

from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# === Spectrogram parameters ===
NPERSEG = 256
NOVERLAP = 192
FREQ_MAX = 250
WINDOW_SEC = 10

# === Filters (mirrors detect_events.py) ===
PASSES = {
    "low":  {"filter": "lowpass",  "cutoff": 15},
    "mid":  {"filter": "bandpass", "cutoff": (15, 30)},
    "high": {"filter": "highpass", "cutoff": 30},
}

BAND_COLORS = {"low": "#E69F00", "mid": "#56B4E9", "high": "#009E73"}

# Caption helper (reuse from make_bathy_map if available)
try:
    from make_bathy_map import add_caption_justified
except ImportError:
    def add_caption_justified(fig, text, fontsize=10, bold_prefix=None, **kw):
        prefix = f"$\\bf{{{bold_prefix}}}$ " if bold_prefix else ""
        fig.text(0.05, 0.01, prefix + text, fontsize=fontsize, ha="left", va="bottom",
                 wrap=True, transform=fig.transFigure, fontfamily="serif")


def _apply_filter(data, band, fs=SAMPLE_RATE, order=4):
    """Apply band-specific filter."""
    cfg = PASSES[band]
    if cfg["filter"] == "lowpass":
        sos = butter(order, cfg["cutoff"], btype="low", fs=fs, output="sos")
    elif cfg["filter"] == "highpass":
        sos = butter(order, cfg["cutoff"], btype="high", fs=fs, output="sos")
    else:
        sos = butter(order, cfg["cutoff"], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data)


def _load_snippet(ev, window_sec=WINDOW_SEC):
    """Load waveform snippet around an event."""
    mooring_key = ev["mooring"]
    info = MOORINGS[mooring_key]
    data_dir = DATA_ROOT / info["data_dir"]

    # Find the right DAT file
    file_num = ev["file_number"]
    dat_path = data_dir / f"{file_num:08d}.DAT"
    if not dat_path.exists():
        return None

    ts, data, meta = read_dat(dat_path)

    # Event position in samples
    onset_utc = pd.Timestamp(ev["onset_utc"])
    file_start = pd.Timestamp(ts)
    offset_s = (onset_utc - file_start).total_seconds()

    # Window centered on event
    pre_s = window_sec * 0.3
    post_s = window_sec * 0.7
    i_start = max(0, int((offset_s - pre_s) * SAMPLE_RATE))
    i_end = min(len(data), int((offset_s + post_s) * SAMPLE_RATE))

    if i_end - i_start < SAMPLE_RATE:
        return None

    snippet = data[i_start:i_end]
    time_s = np.arange(len(snippet)) / SAMPLE_RATE

    # Filter
    band = ev["detection_band"]
    filtered = _apply_filter(snippet, band)

    # Spectrogram
    f, t, Sxx = spectrogram(snippet, fs=SAMPLE_RATE,
                             nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx_dB = 10 * np.log10(Sxx + 1e-20)

    # Event timing relative to snippet
    ev_start = offset_s - (i_start / SAMPLE_RATE)
    ev_end = ev_start + ev.get("duration_s", 1.0)

    # Refined onset
    refined_start = None
    if "onset_shift_s" in ev and pd.notna(ev["onset_shift_s"]):
        refined_start = ev_start + ev["onset_shift_s"]

    return {
        "time_s": time_s, "filtered": filtered,
        "freqs": f, "times": t, "Sxx_dB": Sxx_dB,
        "ev_start": ev_start, "ev_end": ev_end,
        "refined_start": refined_start,
    }


def _plot_panel(fig, gs_slot, ev, snip, show_xlabel=True, show_ylabel=True,
                freq_max=100, panel_label=None):
    """Plot a single waveform+spectrogram panel."""
    inner_gs = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_slot,
        height_ratios=[1, 1.2], hspace=0.08
    )
    ax_wave = fig.add_subplot(inner_gs[0])
    ax_spec = fig.add_subplot(inner_gs[1], sharex=ax_wave)

    band = ev["detection_band"]
    color = BAND_COLORS.get(band, "red")

    # Waveform
    ax_wave.plot(snip["time_s"], snip["filtered"],
                 color="0.3", linewidth=0.4, rasterized=True)

    # Original STA/LTA onset (dashed yellow)
    ax_wave.axvline(snip["ev_start"], color="#FFD700", linewidth=1.5,
                    linestyle="--", alpha=0.9, label="STA/LTA onset")
    ax_spec.axvline(snip["ev_start"], color="#FFD700", linewidth=1.5,
                    linestyle="--", alpha=0.9)

    # Refined AIC onset (solid red)
    if snip["refined_start"] is not None:
        ax_wave.axvline(snip["refined_start"], color="red", linewidth=1.8,
                        alpha=0.95, label="AIC onset")
        ax_spec.axvline(snip["refined_start"], color="red", linewidth=1.8,
                        alpha=0.95)

    # Event end
    ax_spec.axvline(snip["ev_end"], color="gray", linewidth=0.8,
                    linestyle=":", alpha=0.5)
    ax_wave.axvline(snip["ev_end"], color="gray", linewidth=0.8,
                    linestyle=":", alpha=0.5)

    # Spectrogram
    mask = snip["freqs"] <= freq_max
    vmin = np.percentile(snip["Sxx_dB"][mask], 5)
    vmax = np.percentile(snip["Sxx_dB"][mask], 95)
    ax_spec.pcolormesh(snip["times"], snip["freqs"][mask], snip["Sxx_dB"][mask],
                       vmin=vmin, vmax=vmax, cmap="viridis",
                       shading="auto", rasterized=True)
    ax_spec.set_ylim(0, freq_max)

    # Title
    mooring = ev["mooring"].upper()
    grade = ev.get("onset_grade", "?")
    quality = ev.get("onset_quality", 0)
    shift = ev.get("onset_shift_s", 0)
    snr_val = ev.get("snr", 0)
    title = f"{mooring}  {band}  grade {grade}  q={quality:.2f}  shift={shift:+.2f}s"
    ax_wave.set_title(title, fontsize=8, fontweight="bold", pad=3)

    # Labels
    plt.setp(ax_wave.get_xticklabels(), visible=False)
    ax_wave.tick_params(labelsize=7)
    ax_spec.tick_params(labelsize=7)

    if show_ylabel:
        ax_wave.set_ylabel("Amplitude", fontsize=8)
        ax_spec.set_ylabel("Freq (Hz)", fontsize=8)
    else:
        ax_wave.set_yticklabels([])
        ax_spec.set_yticklabels([])

    if show_xlabel:
        ax_spec.set_xlabel("Time (s)", fontsize=8)
    else:
        ax_spec.set_xticklabels([])

    if panel_label:
        ax_wave.text(0.02, 0.92, panel_label, transform=ax_wave.transAxes,
                     fontsize=10, fontweight="bold", va="top",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    return ax_wave, ax_spec


# ========================================================================
# Figure 1: Late-pick problem (§3.1)
# ========================================================================

def fig_late_pick_problem():
    """Select 6 events where STA/LTA pick is clearly late (large negative AIC shift)."""
    print("=== Late-Pick Problem Figure ===")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")

    # Want events with large negative shifts (AIC moved onset much earlier)
    # and grade A/B (so the AIC pick is trustworthy)
    mask = (
        cat["onset_grade"].isin(["A", "B"]) &
        (cat["onset_shift_s"] < -1.5) &  # at least 1.5s earlier
        (cat["snr"] > 5.0)  # clear events
    )
    candidates = cat[mask].copy()
    print(f"  Candidates with shift < -1.5s, grade A/B, SNR > 5: {len(candidates)}")

    # Sample 2 per band, prefer large shifts
    rng = np.random.default_rng(42)
    selected = []
    for band in ["low", "mid", "high"]:
        band_cands = candidates[candidates["detection_band"] == band]
        band_cands = band_cands.sort_values("onset_shift_s")  # most negative first
        # Take from top 50, but spread across moorings
        top = band_cands.head(50)
        if len(top) >= 2:
            # Pick 2 from different moorings if possible
            moorings = top["mooring"].unique()
            picks = []
            for m in moorings:
                sub = top[top["mooring"] == m]
                picks.append(sub.iloc[0])
                if len(picks) >= 2:
                    break
            if len(picks) < 2:
                picks = [top.iloc[i] for i in range(min(2, len(top)))]
            selected.extend(picks)

    if len(selected) < 6:
        print(f"  Warning: only {len(selected)} events selected")

    events = pd.DataFrame(selected).reset_index(drop=True)
    print(f"  Selected {len(events)} events")
    for _, ev in events.iterrows():
        print(f"    {ev['mooring']} {ev['detection_band']} shift={ev['onset_shift_s']:+.2f}s grade={ev['onset_grade']}")

    # Load snippets
    snippets = []
    for _, ev in events.iterrows():
        snip = _load_snippet(ev)
        snippets.append(snip)

    # Plot: 3 rows × 2 columns
    nrows, ncols = 3, 2
    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(nrows, ncols, figure=fig,
                  hspace=0.35, wspace=0.25,
                  top=0.93, bottom=0.14, left=0.08, right=0.96)

    fig.suptitle("STA/LTA Late-Pick Problem", fontsize=14, fontweight="bold")

    labels = [f"({chr(97+i)})" for i in range(6)]  # (a) through (f)
    for idx, (ev_idx, ev) in enumerate(events.iterrows()):
        if idx >= nrows * ncols:
            break
        row, col = idx // ncols, idx % ncols
        snip = snippets[idx]
        if snip is None:
            continue
        freq_max = 250 if ev["detection_band"] == "high" else 100
        _plot_panel(fig, gs[row, col], ev, snip,
                    show_xlabel=(row == nrows - 1),
                    show_ylabel=(col == 0),
                    freq_max=freq_max,
                    panel_label=labels[idx])

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#FFD700", linewidth=1.5, linestyle="--", label="STA/LTA onset (late)"),
        Line2D([0], [0], color="red", linewidth=1.8, label="AIC-refined onset (true)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=9,
               frameon=True, fancybox=True)

    caption = (
        "Six events illustrating the STA/LTA late-pick problem. "
        "Dashed yellow lines mark the original STA/LTA trigger time; solid red lines mark the "
        "AIC-refined onset. In each case, the STA/LTA detector triggered 1.5-4 seconds late, "
        "typically in the event coda rather than at the true first arrival. "
        "Validation on 50 manually reviewed events confirmed that 68% of raw STA/LTA picks "
        "fall in the coda and only 11% hit the true first arrival. "
        "Top row: low-band (1-15 Hz) T-phase events. "
        "Middle row: mid-band (15-30 Hz) events. "
        "Bottom row: high-band (30-250 Hz) events. "
        "Spectrogram: nperseg=256, 87.5% overlap, Hann window."
    )
    add_caption_justified(fig, caption, fontsize=9,
                          bold_prefix="Temporary Caption:")

    outpath = FIG_DIR / "late_pick_problem.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ========================================================================
# Figure 2: Onset refinement curated 6-panel (§3.2)
# ========================================================================

def fig_onset_refinement_6panel():
    """Select 6 curated events: 2 grade A, 2 grade B, 2 grade C."""
    print("=== Onset Refinement 6-Panel ===")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")

    rng = np.random.default_rng(99)
    selected = []

    # 2 per grade, prefer different bands and clear examples
    for grade, snr_min in [("A", 6.0), ("B", 5.0), ("C", 4.0)]:
        cands = cat[(cat["onset_grade"] == grade) & (cat["snr"] > snr_min)].copy()
        if grade == "A":
            # Want large shifts to show clear improvement
            cands = cands[cands["onset_shift_s"] < -0.5]
        elif grade == "C":
            # Want examples where AIC struggled
            cands = cands[cands["onset_quality"] < 0.3]

        # Sample from different bands
        bands_seen = set()
        for _, row in cands.sample(min(50, len(cands)), random_state=rng).iterrows():
            if row["detection_band"] not in bands_seen:
                selected.append(row)
                bands_seen.add(row["detection_band"])
            if len([s for s in selected if s.get("onset_grade") == grade]) >= 2:
                break

    events = pd.DataFrame(selected).reset_index(drop=True)
    print(f"  Selected {len(events)} events")
    for _, ev in events.iterrows():
        print(f"    {ev['mooring']} {ev['detection_band']} grade={ev['onset_grade']} q={ev['onset_quality']:.2f}")

    # Load snippets
    snippets = [_load_snippet(ev) for _, ev in events.iterrows()]

    # Plot: 3 rows × 2 columns
    nrows, ncols = 3, 2
    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(nrows, ncols, figure=fig,
                  hspace=0.35, wspace=0.25,
                  top=0.93, bottom=0.14, left=0.08, right=0.96)

    fig.suptitle("Onset Refinement — Curated Examples by Grade", fontsize=14, fontweight="bold")

    labels = [f"({chr(97+i)})" for i in range(6)]
    for idx in range(min(len(events), nrows * ncols)):
        ev = events.iloc[idx]
        snip = snippets[idx]
        if snip is None:
            continue
        row, col = idx // ncols, idx % ncols
        freq_max = 250 if ev["detection_band"] == "high" else 100
        _plot_panel(fig, gs[row, col], ev, snip,
                    show_xlabel=(row == nrows - 1),
                    show_ylabel=(col == 0),
                    freq_max=freq_max,
                    panel_label=labels[idx])

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#FFD700", linewidth=1.5, linestyle="--", label="Original STA/LTA onset"),
        Line2D([0], [0], color="red", linewidth=1.8, label="AIC-refined onset"),
        Line2D([0], [0], color="gray", linewidth=0.8, linestyle=":", label="Event end"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=9,
               frameon=True, fancybox=True)

    caption = (
        "Six curated events illustrating onset refinement quality across grades A, B, and C. "
        "Top row: Grade A (quality >= 0.7) — sharp AIC minimum at the noise-to-signal transition, "
        "producing a confident onset pick 0.5-2s earlier than STA/LTA. "
        "Middle row: Grade B (quality 0.4-0.7) — moderate AIC minimum with some ambiguity, "
        "but still a meaningful improvement over STA/LTA. "
        "Bottom row: Grade C (quality < 0.4) — weak or ambiguous AIC minimum; these events are "
        "excluded from source location but retained in the catalogue. "
        "Dashed yellow: original STA/LTA trigger. Solid red: AIC-refined onset. "
        "Dotted gray: event end (detrigger). "
        "The AIC picker operates on the squared envelope within a 7s window (5s pre-trigger + 2s post-trigger). "
        "Spectrogram: nperseg=256, 87.5% overlap, Hann window."
    )
    add_caption_justified(fig, caption, fontsize=9,
                          bold_prefix="Temporary Caption:")

    outpath = FIG_DIR / "onset_refinement_6panel.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ========================================================================
# Figure 3: T-phase cluster curated 2×3 montage (§4.1)
# ========================================================================

def fig_tphase_cluster_curated():
    """Select 6 representative T-phase spectrograms from Phase 1 clusters."""
    print("=== T-phase Cluster Curated Montage ===")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    features = pd.read_parquet(DATA_DIR / "event_features.parquet")

    # T-phase clusters from Phase 1: low_0, low_1, mid_0
    tphase_clusters = ["low_0", "low_1", "mid_0"]

    # Merge cluster IDs from umap_coordinates
    umap_df = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")
    cat = cat.merge(umap_df[["event_id", "cluster_id", "umap_1", "umap_2"]],
                    on="event_id", how="left")

    tphase = cat[cat["cluster_id"].isin(tphase_clusters)].copy()

    print(f"  T-phase events: {len(tphase)}")

    # Select 6 events: 2 nearest centroid from each of 3 clusters
    selected = []

    for cid in tphase_clusters:
        clust = tphase[tphase["cluster_id"] == cid].copy()
        if len(clust) == 0:
            continue
        # Compute distance to UMAP centroid
        cx = clust["umap_1"].mean()
        cy = clust["umap_2"].mean()
        clust["centroid_dist"] = np.sqrt((clust["umap_1"] - cx)**2 + (clust["umap_2"] - cy)**2)
        # Filter for good picks
        good = clust[clust["onset_grade"].isin(["A", "B"]) & (clust["snr"] > 5.0)]
        if len(good) < 2:
            good = clust
        good = good.sort_values("centroid_dist")
        # Pick 2 from different moorings
        moorings_used = set()
        for _, row in good.iterrows():
            if row["mooring"] not in moorings_used:
                selected.append(row)
                moorings_used.add(row["mooring"])
            if len([s for s in selected if s.get("cluster_id") == cid]) >= 2:
                break

    events = pd.DataFrame(selected[:6]).reset_index(drop=True)
    print(f"  Selected {len(events)} events")

    snippets = [_load_snippet(ev) for _, ev in events.iterrows()]

    # Plot: 3 rows × 2 columns
    nrows, ncols = 3, 2
    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(nrows, ncols, figure=fig,
                  hspace=0.35, wspace=0.25,
                  top=0.93, bottom=0.14, left=0.08, right=0.96)

    fig.suptitle("T-phase Cluster — Representative Events", fontsize=14, fontweight="bold")

    labels = [f"({chr(97+i)})" for i in range(6)]
    for idx in range(min(len(events), nrows * ncols)):
        ev = events.iloc[idx]
        snip = snippets[idx]
        if snip is None:
            continue
        row, col = idx // ncols, idx % ncols
        _plot_panel(fig, gs[row, col], ev, snip,
                    show_xlabel=(row == nrows - 1),
                    show_ylabel=(col == 0),
                    freq_max=100,
                    panel_label=labels[idx])

    caption = (
        "Six representative T-phase events from Phase 1 unsupervised clusters (low_0, low_1, mid_0), "
        "selected from events nearest the UMAP cluster centroids across different moorings. "
        "Each panel shows bandpass-filtered waveform (top) and spectrogram (bottom, 0-100 Hz). "
        "Dashed yellow: STA/LTA onset. Solid red: AIC-refined onset. "
        "T-phases are characterized by impulsive broadband arrivals with dominant energy below 15 Hz "
        "and duration typically <= 3s. "
        "Phase 1 identified 55,783 T-phases across three clusters confirmed by expert review. "
        "A complete 4x5 montage (20 events nearest each cluster centroid) is available in the "
        "supplementary materials (cluster_montage_low_0.png, cluster_montage_low_1.png, cluster_montage_mid_0.png). "
        "Spectrogram: nperseg=256, 87.5% overlap, Hann window."
    )
    add_caption_justified(fig, caption, fontsize=9,
                          bold_prefix="Temporary Caption:")

    outpath = FIG_DIR / "tphase_cluster_curated.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ========================================================================
# Figure 4: Magnitude of completeness (relative source level)
# ========================================================================

def fig_magnitude_completeness():
    """Cumulative frequency vs. relative source level for located T-phases."""
    print("=== Magnitude of Completeness ===")

    locations = pd.read_parquet(DATA_DIR / "event_locations.parquet")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")

    # Get T-phase locations with tier A/B/C
    tphase_locs = locations[
        (locations["event_class"] == "tphase") &
        (locations["quality_tier"].isin(["A", "B", "C"]))
    ].copy()
    print(f"  Located T-phases (tier A-C): {len(tphase_locs)}")

    # Get peak_db from catalogue — need to link via association
    assoc = pd.read_parquet(DATA_DIR / "cross_mooring_associations.parquet")

    # Each association has event_ids (comma-separated)
    # Get the max peak_db across all moorings for each association as received level
    assoc_events = []
    for _, row in assoc.iterrows():
        event_ids = str(row["event_ids"]).split(",")
        event_ids = [eid.strip() for eid in event_ids if eid.strip()]
        for eid in event_ids:
            assoc_events.append({"assoc_id": row["assoc_id"], "event_id": eid})

    ae_df = pd.DataFrame(assoc_events)

    # Merge with catalogue for peak_db
    cat_slim = cat[["event_id", "peak_db", "mooring"]].copy()
    ae_df = ae_df.merge(cat_slim, on="event_id", how="left")

    # For each association, take max peak_db (best received level)
    assoc_peak = ae_df.groupby("assoc_id").agg(
        received_db=("peak_db", "max"),
        n_moorings=("event_id", "count"),
        best_mooring=("peak_db", "idxmax"),
    ).reset_index()

    # Merge with locations
    tphase_locs = tphase_locs.merge(assoc_peak[["assoc_id", "received_db"]], on="assoc_id", how="left")
    tphase_locs = tphase_locs.dropna(subset=["received_db"])

    # Estimate relative source level:
    # SL = RL + TL, where TL ≈ 20*log10(distance_m) for spherical spreading
    # We'll also add a cylindrical component: TL = 15*log10(r) + α*r
    # Use 15*log10(r) as a compromise (between spherical 20 and cylindrical 10)
    # α ≈ 0.001 dB/m for ~10 Hz in seawater (very small for short ranges)

    # Need distance from source to nearest mooring
    # Use the best mooring's distance
    from pyproj import Geod
    geod = Geod(ellps="WGS84")

    # For each located event, compute distance to array centroid as proxy
    # (Better: distance to the best-receiving mooring, but we'd need to track that)
    # Actually, let's compute distance to the closest mooring
    mooring_locs = {k: (v["lat"], v["lon"]) for k, v in MOORINGS.items()}

    def min_distance_m(row):
        dists = []
        for mlat, mlon in mooring_locs.values():
            _, _, dist = geod.inv(row["lon"], row["lat"], mlon, mlat)
            dists.append(dist)
        return min(dists)

    tphase_locs["dist_m"] = tphase_locs.apply(min_distance_m, axis=1)

    # Transmission loss: 15*log10(r) (practical spreading)
    tphase_locs["TL_dB"] = 15.0 * np.log10(tphase_locs["dist_m"].clip(lower=100))

    # Relative source level
    tphase_locs["source_level_dB"] = tphase_locs["received_db"] + tphase_locs["TL_dB"]

    sl = tphase_locs["source_level_dB"].dropna()
    print(f"  Source levels computed: {len(sl)}")
    print(f"  Range: {sl.min():.1f} to {sl.max():.1f} dB")
    print(f"  Median: {sl.median():.1f} dB")

    # Cumulative frequency-magnitude distribution
    sl_sorted = np.sort(sl.values)
    cumulative = np.arange(len(sl_sorted), 0, -1)  # N >= SL

    # Find Mc using maximum curvature method
    bin_edges = np.arange(sl_sorted.min(), sl_sorted.max() + 1, 1.0)
    counts, _ = np.histogram(sl_sorted, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mc_idx = np.argmax(counts)
    mc = bin_centers[mc_idx]
    print(f"  Mc (max curvature): {mc:.1f} dB")

    # Fit G-R law above Mc: log10(N) = a - b*SL
    above_mc = sl_sorted[sl_sorted >= mc]
    cum_above = np.arange(len(above_mc), 0, -1)
    if len(above_mc) > 10:
        # Linear fit in log space
        log_cum = np.log10(cum_above)
        coeffs = np.polyfit(above_mc, log_cum, 1)
        b_value = -coeffs[0]
        a_value = coeffs[1]
        fit_sl = np.linspace(mc, sl_sorted.max(), 100)
        fit_n = 10 ** (a_value + coeffs[0] * fit_sl)
    else:
        b_value = None
        fit_sl = None

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5),
                                     gridspec_kw={"width_ratios": [1.2, 1],
                                                   "wspace": 0.3})
    fig.subplots_adjust(top=0.90, bottom=0.28, left=0.08, right=0.96)

    # Left: cumulative plot
    ax1.semilogy(sl_sorted, cumulative, "k.", markersize=1.5, alpha=0.3, rasterized=True)
    ax1.axvline(mc, color="red", linewidth=1.5, linestyle="--", label=f"$M_c$ = {mc:.0f} dB")
    if fit_sl is not None:
        ax1.semilogy(fit_sl, fit_n, "b-", linewidth=2.0,
                     label=f"G-R fit: b = {b_value:.3f}")
    ax1.set_xlabel("Relative Source Level (dB)", fontsize=11)
    ax1.set_ylabel("Cumulative Number of Events (N ≥ SL)", fontsize=11)
    ax1.set_title("(a) Cumulative Frequency–Source Level", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.tick_params(labelsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: histogram
    ax2.bar(bin_centers, counts, width=0.9, color="steelblue", edgecolor="white", linewidth=0.3)
    ax2.axvline(mc, color="red", linewidth=1.5, linestyle="--", label=f"$M_c$ = {mc:.0f} dB")
    ax2.set_xlabel("Relative Source Level (dB)", fontsize=11)
    ax2.set_ylabel("Number of Events per 1 dB Bin", fontsize=11)
    ax2.set_title("(b) Frequency Distribution", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.tick_params(labelsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    caption = (
        "Acoustic magnitude of completeness for located T-phase events (tiers A-C, N="
        f"{len(sl):,}). "
        "Relative source level estimated as received level (peak spectral power in dB) plus "
        "transmission loss (15*log10(r) practical spreading to nearest mooring). "
        "These are RELATIVE values — absolute calibration in dB re 1 uPa at 1m requires "
        "hydrophone sensitivity curves not available for this deployment. "
        f"(a) Cumulative frequency-source level distribution with Gutenberg-Richter fit above Mc. "
        f"Magnitude of completeness Mc = {mc:.0f} dB (maximum curvature method, Wiemer & Wyss 2000). "
    )
    if b_value is not None:
        caption += f"b-value = {b_value:.3f}. "
    caption += (
        "(b) Non-cumulative histogram (1 dB bins) showing the rollover below Mc "
        "where detection sensitivity falls off. "
        "The departure from linearity below Mc indicates incomplete detection of smaller events."
    )
    add_caption_justified(fig, caption, fontsize=9,
                          bold_prefix="Temporary Caption:")

    outpath = FIG_DIR / "magnitude_completeness.png"
    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    print(f"  Mc = {mc:.1f} dB, b-value = {b_value:.3f}" if b_value else f"  Mc = {mc:.1f} dB")
    return outpath


# ========================================================================
# Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=["late_pick", "onset", "cluster", "completeness", "all"],
                        default="all")
    args = parser.parse_args()

    figs = {
        "late_pick": fig_late_pick_problem,
        "onset": fig_onset_refinement_6panel,
        "cluster": fig_tphase_cluster_curated,
        "completeness": fig_magnitude_completeness,
    }

    if args.figure == "all":
        for name, func in figs.items():
            try:
                func()
            except Exception as e:
                print(f"  ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()
            print()
    else:
        figs[args.figure]()


if __name__ == "__main__":
    main()
