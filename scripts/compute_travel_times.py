#!/usr/bin/env python3
"""
compute_travel_times.py — Pair-specific acoustic travel times from XBT profiles.

Parses in-situ XBT sound speed profiles from the BRAVOSEIS deployment cruise
(January 2019) and computes effective horizontal sound speed at hydrophone
depth. Derives pair-specific maximum travel time windows for all 15 mooring
pairs, replacing the previous constant 120-s global window.

Usage:
    uv run python compute_travel_times.py

Outputs:
    outputs/data/travel_times.json
    outputs/figures/exploratory/association/sound_speed_profile.png

Spec: specs/001-event-detection/
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from pyproj import Geod

from read_dat import MOORINGS

# === Paths ===
XBT_DIR = Path("/home/jovyan/my_data/bravoseis/XBT/XBT")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "association"

# === Parameters ===
SAFETY_FACTOR = 1.15  # 15% safety margin on travel times
MOORING_KEYS = sorted(MOORINGS.keys())


def parse_asvp(filepath):
    """Parse an .asvp sound speed profile.

    Handles three formats found in the BRAVOSEIS XBT data:
      1. Simple CSV: header lines then depth,speed rows
      2. SVP_VERSION_2: bracket header then depth speed rows (space-separated)
      3. SoundVelocity: parenthesized header then depth speed rows

    Returns
    -------
    depth : np.ndarray
        Depth in meters.
    speed : np.ndarray
        Sound speed in m/s.
    metadata : dict
        Parsed header info (lat, lon, date, etc.).
    """
    filepath = Path(filepath)
    text = filepath.read_text()
    lines = text.strip().splitlines()

    if not lines:
        raise ValueError(f"Empty file: {filepath}")

    first_line = lines[0].strip()

    if first_line.startswith("[SVP_VERSION_2]"):
        return _parse_svp_v2(lines, filepath)
    elif first_line.startswith("("):
        return _parse_soundvelocity(lines, filepath)
    else:
        return _parse_csv(lines, filepath)


def _parse_csv(lines, filepath):
    """Parse simple CSV format: Date, Time, Lat, Lon header then depth,speed."""
    metadata = {}
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Date"):
            metadata["date"] = stripped.split(",")[1].strip()
        elif stripped.startswith("Time"):
            metadata["time"] = stripped.split(",")[1].strip()
        elif stripped.startswith("Latitude"):
            metadata["lat"] = float(stripped.split(",")[1].strip())
        elif stripped.startswith("Longitude"):
            metadata["lon"] = float(stripped.split(",")[1].strip())
        elif stripped.startswith("depth"):
            data_start = i + 1
            break

    depths, speeds = [], []
    for line in lines[data_start:]:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                depths.append(float(parts[0]))
                speeds.append(float(parts[1]))
            except ValueError:
                continue

    metadata["source"] = filepath.name
    return np.array(depths), np.array(speeds), metadata


def _parse_svp_v2(lines, filepath):
    """Parse SVP_VERSION_2 format: bracket header, then depth speed rows."""
    metadata = {"source": filepath.name}
    # Line 2: filename, Line 3: "Section YYYY-DDD HH:MM:SS lat lon"
    if len(lines) >= 3:
        section = lines[2].strip()
        parts = section.split()
        if len(parts) >= 5:
            metadata["date"] = parts[1]
            metadata["time"] = parts[2]
            # lat/lon in DMS format: -DD:MM:SS.ss
            metadata["lat_dms"] = parts[3]
            metadata["lon_dms"] = parts[4]

    depths, speeds = [], []
    for line in lines[3:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                depths.append(float(parts[0]))
                speeds.append(float(parts[1]))
            except ValueError:
                continue

    return np.array(depths), np.array(speeds), metadata


def _parse_soundvelocity(lines, filepath):
    """Parse SoundVelocity format: parenthesized header then depth speed rows."""
    metadata = {"source": filepath.name}
    header = lines[0].strip()
    # Extract lat/lon from header: ( SoundVelocity 1.0 0 YYMMDDHHMMSS lat lon ... )
    parts = header.strip("() ").split()
    if len(parts) >= 6:
        metadata["lat"] = float(parts[4])
        metadata["lon"] = float(parts[5])

    depths, speeds = [], []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                depths.append(float(parts[0]))
                speeds.append(float(parts[1]))
            except ValueError:
                continue

    return np.array(depths), np.array(speeds), metadata


def compute_effective_speed(depth, speed, z_max):
    """Compute effective horizontal sound speed via harmonic mean.

    c_eff = z_max / integral(1/c(z) dz, 0, z_max)

    This is the correct average for horizontal propagation through
    a vertically stratified medium.

    Parameters
    ----------
    depth : np.ndarray
        Depth values (m), must be sorted ascending.
    speed : np.ndarray
        Sound speed values (m/s).
    z_max : float
        Maximum depth for integration (hydrophone depth).

    Returns
    -------
    float
        Effective horizontal sound speed (m/s).
    """
    # Clip to z_max
    mask = depth <= z_max
    if mask.sum() < 2:
        return np.mean(speed[:2])

    z = depth[mask]
    c = speed[mask]

    # Trapezoidal integration of 1/c(z)
    integral = np.trapezoid(1.0 / c, z)

    if integral <= 0:
        return np.mean(c)

    return z[-1] / integral


def compute_mooring_distances():
    """Compute pairwise geodesic distances between all mooring pairs."""
    geod = Geod(ellps="WGS84")
    distances = {}
    for i, k1 in enumerate(MOORING_KEYS):
        for k2 in MOORING_KEYS[i + 1 :]:
            _, _, dist_m = geod.inv(
                MOORINGS[k1]["lon"],
                MOORINGS[k1]["lat"],
                MOORINGS[k2]["lon"],
                MOORINGS[k2]["lat"],
            )
            distances[(k1, k2)] = dist_m / 1000  # km
    return distances


def plot_sound_speed_profile(profiles, primary_name, hydrophone_depths):
    """Plot sound speed profiles with hydrophone depth range marked."""
    fig, ax = plt.subplots(figsize=(6, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))
    for (name, depth, speed), color in zip(profiles, colors):
        lw = 2.0 if name == primary_name else 0.8
        alpha = 1.0 if name == primary_name else 0.5
        ax.plot(speed, depth, label=name, color=color, linewidth=lw, alpha=alpha)

    # Mark hydrophone depth range
    z_min = min(hydrophone_depths.values())
    z_max = max(hydrophone_depths.values())
    ax.axhspan(z_min, z_max, color="red", alpha=0.15, label="Hydrophone depths")
    ax.axhline(np.mean(list(hydrophone_depths.values())), color="red",
               linestyle="--", linewidth=1, alpha=0.6, label="Mean hydro. depth")

    ax.set_xlabel("Sound Speed (m/s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=12, fontweight="bold")
    ax.set_title("XBT Sound Speed Profiles — BRAVOSEIS Jan 2019",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(1450, 1485)
    ax.set_ylim(bottom=1100, top=0)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()
    outpath = FIG_DIR / "sound_speed_profile.png"
    fig.savefig(outpath, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    print("=" * 60)
    print("Pair-Specific Acoustic Travel Times from XBT Profiles")
    print("=" * 60)

    # --- Parse all XBT profiles ---
    asvp_files = sorted(XBT_DIR.glob("*.asvp"))
    print(f"\nFound {len(asvp_files)} .asvp files")

    profiles = []
    for f in asvp_files:
        try:
            depth, speed, meta = parse_asvp(f)
            # Skip profiles with non-monotonic depth (messy data)
            if len(depth) > 10 and depth[-1] > 100:
                profiles.append((f.stem, depth, speed, meta))
                print(f"  {f.name}: {len(depth)} pts, "
                      f"{depth[0]:.0f}–{depth[-1]:.0f} m, "
                      f"{speed[0]:.1f}–{speed[-1]:.1f} m/s")
        except Exception as e:
            print(f"  {f.name}: SKIP ({e})")

    if not profiles:
        print("ERROR: No valid profiles found!")
        return

    # --- Select primary profile ---
    # T5_18_01_19: deepest clean profile (1022 m), mid-array location
    primary_name = "T5_18_01_19"
    primary = None
    for name, depth, speed, meta in profiles:
        if name == primary_name:
            primary = (depth, speed, meta)
            break

    if primary is None:
        print(f"WARNING: Primary profile {primary_name} not found, using deepest")
        profiles.sort(key=lambda x: x[1][-1], reverse=True)
        primary_name = profiles[0][0]
        primary = (profiles[0][1], profiles[0][2], profiles[0][3])

    depth_p, speed_p, meta_p = primary
    print(f"\nPrimary profile: {primary_name}")
    print(f"  Depth range: {depth_p[0]:.1f}–{depth_p[-1]:.1f} m")
    print(f"  Speed range: {speed_p[0]:.1f}–{speed_p[-1]:.1f} m/s")

    # --- Hydrophone depths ---
    hydrophone_depths = {k: v["hydrophone_depth_m"] for k, v in MOORINGS.items()}
    mean_hydro_depth = np.mean(list(hydrophone_depths.values()))
    print(f"\nHydrophone depths: {hydrophone_depths}")
    print(f"Mean hydrophone depth: {mean_hydro_depth:.1f} m")

    # --- Compute effective speed at mean hydrophone depth ---
    c_eff = compute_effective_speed(depth_p, speed_p, mean_hydro_depth)
    print(f"\nEffective horizontal speed (0–{mean_hydro_depth:.0f} m): {c_eff:.1f} m/s")
    print(f"  (vs. previous assumption of 1480 m/s — "
          f"difference: {1480 - c_eff:.1f} m/s)")

    # --- Per-pair effective speed using pair-specific hydrophone depths ---
    # For each pair, use the deeper hydrophone depth as integration limit
    print(f"\nPer-pair effective speeds:")
    pair_c_eff = {}
    for i, k1 in enumerate(MOORING_KEYS):
        for k2 in MOORING_KEYS[i + 1 :]:
            z_max = max(hydrophone_depths[k1], hydrophone_depths[k2])
            c = compute_effective_speed(depth_p, speed_p, z_max)
            pair_c_eff[(k1, k2)] = c
            print(f"  {k1}-{k2}: z_max={z_max:.0f} m → c_eff={c:.1f} m/s")

    # --- Compute pairwise distances ---
    distances = compute_mooring_distances()

    # --- Compute travel times ---
    print(f"\nPair-specific travel times (with {SAFETY_FACTOR:.0%} safety factor):")
    print(f"{'Pair':<10} {'Dist (km)':>10} {'c_eff':>8} {'t_travel':>10} "
          f"{'t_max':>8} {'Old':>6}")

    travel_times = {}
    for (k1, k2), dist_km in sorted(distances.items()):
        c = pair_c_eff[(k1, k2)]
        t_travel = dist_km * 1000 / c
        t_max = t_travel * SAFETY_FACTOR
        t_max_rounded = round(t_max, 1)
        travel_times[f"{k1}-{k2}"] = {
            "distance_km": round(dist_km, 1),
            "c_eff_ms": round(c, 1),
            "travel_time_s": round(t_travel, 1),
            "max_travel_time_s": t_max_rounded,
        }
        print(f"  {k1}-{k2}:  {dist_km:8.1f} km  {c:7.1f}  {t_travel:8.1f} s  "
              f"{t_max_rounded:6.1f}  {120:4d}")

    global_max = max(v["max_travel_time_s"] for v in travel_times.values())
    global_min = min(v["max_travel_time_s"] for v in travel_times.values())
    print(f"\nGlobal max travel time: {global_max:.1f} s")
    print(f"Global min travel time: {global_min:.1f} s")

    # --- Save JSON ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "Pair-specific maximum acoustic travel times derived from "
                       "in-situ XBT sound speed profiles (BRAVOSEIS cruise, Jan 2019)",
        "primary_profile": primary_name,
        "safety_factor": SAFETY_FACTOR,
        "effective_speed_mean_ms": round(c_eff, 1),
        "global_max_travel_time_s": global_max,
        "pairs": travel_times,
    }
    json_path = DATA_DIR / "travel_times.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # --- Plot ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_profiles = [(name, d, s) for name, d, s, _ in profiles]
    plot_sound_speed_profile(plot_profiles, primary_name, hydrophone_depths)

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
