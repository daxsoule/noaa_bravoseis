#!/usr/bin/env python3
"""
locate_land_events.py — Locate well-constrained events using P/S picks from
BRAVOSEIS 5M land stations, then calibrate the hydroacoustic catalogue.

Steps:
  1. Locate 12 GT events using iterative Geiger's method (P + S arrivals)
  2. Compute residual vectors vs hydroacoustic TDOA locations
  3. Fit a spatial correction model (mooring-dependent bias)
  4. Apply correction to full hydroacoustic catalogue
  5. Validate against 153 S-P distance constraints

Usage:
    uv run python locate_land_events.py
    uv run python locate_land_events.py --vp 6.2 --vpvs 1.74

Output:
    outputs/data/seismic_locations.parquet       (12 GT event locations)
    outputs/data/location_residuals.parquet       (GT vs hydroacoustic)
    outputs/data/catalogue_corrected.parquet      (corrected full catalogue)
    outputs/data/sp_validation.parquet            (S-P validation results)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "outputs" / "data"
PICKS_PATH = DATA_DIR / "land_station_picks.parquet"
HYDRO_LOCS_PATH = DATA_DIR / "tapaas_locations.parquet"

GEOD = Geod(ellps="WGS84")

# === Station coordinates ===
STATIONS = {
    "AST": (-63.3273, -58.7027),
    "BYE": (-62.6665, -61.0992),
    "DCP": (-62.9775, -60.6782),
    "ERJ": (-62.02436, -57.64911),
    "FER": (-62.08976, -58.40655),
    "FRE": (-62.2068, -58.9607),
    "GUR": (-62.30753, -59.19597),
    "HMI": (-62.5958, -59.90387),
    "LVN": (-62.6627, -60.3875),
    "OHI": (-63.3221, -57.8973),
    "PEN": (-62.09932, -57.93673),
    "ROB": (-62.37935, -59.70353),
    "SNW": (-62.72787, -61.2003),
    "TOW": (-63.5921, -59.7828),
    "JUBA": (-62.237301, -58.662701),
    "ESPZ": (-63.398102, -56.996399),
}


# ============================================================
# Step 1: Grid-search event location using P and S arrivals
# ============================================================

def geodesic_dist_km(lat1, lon1, lat2, lon2):
    """Geodesic distance in km between two points."""
    _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0


def locate_gridsearch(picks_df, vp, vs, depth_km=10.0,
                      grid_center=None, grid_extent=2.0,
                      coarse_spacing=0.02, fine_spacing=0.002):
    """Locate one event using two-pass grid search over lat/lon.

    For each grid point, finds the optimal origin time analytically, then
    computes the weighted RMS of travel-time residuals.

    Parameters
    ----------
    picks_df : DataFrame
        Picks for one event with columns: station, phase, pick_time, probability
    vp, vs : float
        P and S wave velocities (km/s)
    depth_km : float
        Fixed source depth (km)
    grid_center : tuple (lat, lon) or None
        Center of search grid. If None, uses station centroid.
    grid_extent : float
        Half-width of search grid in degrees.
    coarse_spacing, fine_spacing : float
        Grid spacing for coarse and fine passes (degrees).

    Returns
    -------
    result : dict with lat, lon, depth_km, origin_time, rms_residual, etc.
    """
    # Keep only the highest-probability pick per station per phase
    picks_df = picks_df.copy()
    picks_df["pick_time"] = pd.to_datetime(picks_df["pick_time"])
    best = picks_df.sort_values("probability", ascending=False).drop_duplicates(
        ["station", "phase"])

    if len(best) < 3:
        return None

    stations = best["station"].values
    phases = best["phase"].values
    times = best["pick_time"].values
    probs = best["probability"].values

    ref_time = times.min()
    obs_times = (times - ref_time) / np.timedelta64(1, "s")

    station_lats = np.array([STATIONS[s][0] for s in stations])
    station_lons = np.array([STATIONS[s][1] for s in stations])
    velocities = np.array([vp if p == "P" else vs for p in phases])
    weights = probs / probs.sum()

    n_obs = len(best)

    if grid_center is not None:
        clat, clon = grid_center
    else:
        clat = np.mean(station_lats)
        clon = np.mean(station_lons)

    def _search(lat_min, lat_max, lon_min, lon_max, spacing):
        lats = np.arange(lat_min, lat_max + spacing, spacing)
        lons = np.arange(lon_min, lon_max + spacing, spacing)

        best_rms = np.inf
        best_lat, best_lon, best_t0 = clat, clon, 0.0

        for lat in lats:
            for lon in lons:
                # Compute travel times from this grid point to all stations
                pred_tt = np.zeros(n_obs)
                for k in range(n_obs):
                    dist = geodesic_dist_km(lat, lon,
                                            station_lats[k], station_lons[k])
                    hypo = np.sqrt(dist**2 + depth_km**2)
                    pred_tt[k] = hypo / velocities[k]

                # Optimal origin time: minimize weighted sum of
                # (obs_time - (t0 + pred_tt))^2
                # => t0 = weighted_mean(obs_times - pred_tt)
                residuals_no_t0 = obs_times - pred_tt
                t0 = np.average(residuals_no_t0, weights=weights)

                # Weighted RMS residual
                residuals = obs_times - (t0 + pred_tt)
                wrms = np.sqrt(np.average(residuals**2, weights=weights))

                if wrms < best_rms:
                    best_rms = wrms
                    best_lat = lat
                    best_lon = lon
                    best_t0 = t0

        return best_lat, best_lon, best_t0, best_rms

    # Pass 1: coarse grid
    lat1, lon1, t0_1, rms1 = _search(
        clat - grid_extent, clat + grid_extent,
        clon - grid_extent, clon + grid_extent,
        coarse_spacing)

    # Pass 2: fine grid around coarse minimum
    fine_extent = coarse_spacing * 5
    lat2, lon2, t0_2, rms2 = _search(
        lat1 - fine_extent, lat1 + fine_extent,
        lon1 - fine_extent, lon1 + fine_extent,
        fine_spacing)

    # Final residuals at best location
    final_residuals = np.zeros(n_obs)
    station_dists = np.zeros(n_obs)
    for k in range(n_obs):
        dist = geodesic_dist_km(lat2, lon2, station_lats[k], station_lons[k])
        station_dists[k] = dist
        hypo = np.sqrt(dist**2 + depth_km**2)
        pred_tt = hypo / velocities[k]
        final_residuals[k] = obs_times[k] - (t0_2 + pred_tt)

    rms = np.sqrt(np.mean(final_residuals**2))
    wrms = np.sqrt(np.average(final_residuals**2, weights=weights))

    # Origin time
    origin_time = ref_time + np.timedelta64(int(t0_2 * 1e9), "ns")

    # Approximate horizontal uncertainty from residual spread
    # Using the rule: err ~ (rms * v) / sqrt(N)
    avg_v = np.mean(velocities)
    horiz_err_km = rms * avg_v / np.sqrt(n_obs)

    return {
        "lat": lat2,
        "lon": lon2,
        "depth_km": depth_km,
        "origin_time": origin_time,
        "rms_residual_s": rms,
        "wrms_residual_s": wrms,
        "horiz_err_km": horiz_err_km,
        "n_picks": n_obs,
        "n_p": int((phases == "P").sum()),
        "n_s": int((phases == "S").sum()),
        "n_stations": len(set(stations)),
        "station_residuals": {
            f"{s}_{p}": f"{r:.2f}s"
            for s, p, r in zip(stations, phases, final_residuals)
        },
    }


# ============================================================
# Step 2-5: Calibration pipeline
# ============================================================

def compute_residuals(seismic_locs, hydro_locs):
    """Compute location residual vectors (hydroacoustic - seismic)."""
    merged = seismic_locs.merge(
        hydro_locs[["assoc_id", "lat", "lon", "n_moorings", "detection_band"]],
        on="assoc_id", suffixes=("_seis", "_hydro"))

    residuals = []
    for _, row in merged.iterrows():
        dlat = row["lat_hydro"] - row["lat_seis"]
        dlon = row["lon_hydro"] - row["lon_seis"]

        # Vector distance
        _, _, dist_m = GEOD.inv(
            row["lon_seis"], row["lat_seis"],
            row["lon_hydro"], row["lat_hydro"])
        dist_km = dist_m / 1000.0

        # Azimuth of offset
        az, _, _ = GEOD.inv(
            row["lon_seis"], row["lat_seis"],
            row["lon_hydro"], row["lat_hydro"])

        residuals.append({
            "assoc_id": row["assoc_id"],
            "lat_seis": row["lat_seis"],
            "lon_seis": row["lon_seis"],
            "lat_hydro": row["lat_hydro"],
            "lon_hydro": row["lon_hydro"],
            "dlat": dlat,
            "dlon": dlon,
            "offset_km": dist_km,
            "offset_azimuth": az,
            "n_moorings": row["n_moorings"],
        })

    return pd.DataFrame(residuals)


def fit_correction(residuals_df, hydro_locs):
    """Fit spatial bias correction from GT residuals and apply to catalogue.

    Uses a simple approach: mean bias correction, with optional spatial
    weighting using inverse-distance to GT events.
    """
    # Global mean correction
    mean_dlat = residuals_df["dlat"].mean()
    mean_dlon = residuals_df["dlon"].mean()
    median_dlat = residuals_df["dlat"].median()
    median_dlon = residuals_df["dlon"].median()

    print(f"\n  Global bias (mean):   dlat={mean_dlat:.4f}°  dlon={mean_dlon:.4f}°")
    print(f"  Global bias (median): dlat={median_dlat:.4f}°  dlon={median_dlon:.4f}°")
    print(f"  Mean offset: {residuals_df['offset_km'].mean():.1f} km")
    print(f"  Median offset: {residuals_df['offset_km'].median():.1f} km")

    # Apply inverse-distance-weighted correction
    gt_lats = residuals_df["lat_seis"].values
    gt_lons = residuals_df["lon_seis"].values
    gt_dlat = residuals_df["dlat"].values
    gt_dlon = residuals_df["dlon"].values

    corrected = hydro_locs.copy()
    corr_dlat = np.zeros(len(corrected))
    corr_dlon = np.zeros(len(corrected))

    for i, (_, row) in enumerate(corrected.iterrows()):
        # Distances from this event to all GT events
        dists = np.array([
            geodesic_dist_km(row["lat"], row["lon"], gt_lat, gt_lon)
            for gt_lat, gt_lon in zip(gt_lats, gt_lons)
        ])

        # Inverse-distance weights (with smoothing to avoid division by zero)
        smoothing_km = 20.0  # characteristic scale
        w = 1.0 / (dists + smoothing_km)**2
        w /= w.sum()

        corr_dlat[i] = np.dot(w, gt_dlat)
        corr_dlon[i] = np.dot(w, gt_dlon)

    corrected["lat_corrected"] = corrected["lat"] - corr_dlat
    corrected["lon_corrected"] = corrected["lon"] - corr_dlon
    corrected["correction_dlat"] = -corr_dlat
    corrected["correction_dlon"] = -corr_dlon
    corrected["correction_km"] = np.sqrt(
        (corr_dlat * 111.0)**2 +
        (corr_dlon * 111.0 * np.cos(np.radians(corrected["lat"])))**2
    )

    return corrected


def validate_sp(corrected_locs, picks_df, vp, vs):
    """Validate corrected locations against S-P distance constraints."""
    # Find events with S-P pairs at the same station
    sp_pairs = picks_df.groupby(["assoc_id", "station"])["phase"].apply(set)
    sp_pairs = sp_pairs[sp_pairs.apply(lambda x: "P" in x and "S" in x)]
    sp_events = sp_pairs.reset_index()[["assoc_id", "station"]].drop_duplicates()

    results = []
    for _, row in sp_events.iterrows():
        assoc_id = row["assoc_id"]
        station = row["station"]

        # Get P and S times (best pick per phase)
        ev_picks = picks_df[
            (picks_df.assoc_id == assoc_id) &
            (picks_df.station == station)
        ]
        p_pick = ev_picks[ev_picks.phase == "P"].sort_values(
            "probability", ascending=False).iloc[0]
        s_pick = ev_picks[ev_picks.phase == "S"].sort_values(
            "probability", ascending=False).iloc[0]

        sp_time = (pd.to_datetime(s_pick.pick_time) -
                   pd.to_datetime(p_pick.pick_time)).total_seconds()

        if sp_time <= 0:
            continue

        # S-P distance
        sp_dist_km = sp_time * vp * vs / (vp - vs)

        # Get corrected location
        ev_loc = corrected_locs[corrected_locs.assoc_id == assoc_id]
        if len(ev_loc) == 0:
            continue
        ev_loc = ev_loc.iloc[0]

        sta_lat, sta_lon = STATIONS[station]

        # Distance from corrected location to station
        loc_dist_km = geodesic_dist_km(
            ev_loc["lat_corrected"], ev_loc["lon_corrected"],
            sta_lat, sta_lon)

        # Distance from original location to station
        orig_dist_km = geodesic_dist_km(
            ev_loc["lat"], ev_loc["lon"],
            sta_lat, sta_lon)

        results.append({
            "assoc_id": assoc_id,
            "station": station,
            "sp_time_s": sp_time,
            "sp_dist_km": sp_dist_km,
            "corrected_dist_km": loc_dist_km,
            "original_dist_km": orig_dist_km,
            "corrected_error_km": abs(loc_dist_km - sp_dist_km),
            "original_error_km": abs(orig_dist_km - sp_dist_km),
            "p_prob": p_pick.probability,
            "s_prob": s_pick.probability,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Locate GT events and calibrate hydroacoustic catalogue")
    parser.add_argument("--vp", type=float, default=6.0,
                        help="P-wave velocity km/s (default: 6.0)")
    parser.add_argument("--vpvs", type=float, default=1.73,
                        help="Vp/Vs ratio (default: 1.73)")
    parser.add_argument("--fix-depth", type=float, default=None,
                        help="Fix depth in km (default: solve for depth)")
    args = parser.parse_args()

    vp = args.vp
    vs = vp / args.vpvs

    print("=" * 60)
    print("Land Station Event Location & Catalogue Calibration")
    print("=" * 60)
    print(f"Vp = {vp:.2f} km/s, Vs = {vs:.2f} km/s (Vp/Vs = {args.vpvs:.2f})")
    if args.fix_depth is not None:
        print(f"Fixed depth: {args.fix_depth:.1f} km")

    # Load data
    picks = pd.read_parquet(PICKS_PATH)
    hydro_locs = pd.read_parquet(HYDRO_LOCS_PATH)

    # --------------------------------------------------------
    # Step 1: Identify and locate well-constrained events
    # --------------------------------------------------------
    print("\n--- Step 1: Locate GT events ---")

    # Find events with >=3 unique stations and >=4 total phases
    sta_per_ev = picks.groupby("assoc_id")["station"].nunique()
    p_sta = picks[picks.phase == "P"].groupby("assoc_id")["station"].nunique()
    s_sta = picks[picks.phase == "S"].groupby("assoc_id")["station"].nunique()
    total_sta = p_sta.add(s_sta, fill_value=0)
    gt_ids = sta_per_ev[(sta_per_ev >= 3) & (total_sta >= 4)].index.tolist()

    print(f"GT events to locate: {len(gt_ids)}")

    depth_km = args.fix_depth if args.fix_depth is not None else 10.0

    results = []
    for assoc_id in gt_ids:
        ev_picks = picks[picks.assoc_id == assoc_id]

        # Use hydroacoustic location as grid center
        hloc = hydro_locs[hydro_locs.assoc_id == assoc_id]
        center = (hloc.iloc[0].lat, hloc.iloc[0].lon) if len(hloc) else None

        loc = locate_gridsearch(ev_picks, vp, vs, depth_km=depth_km,
                                grid_center=center)
        if loc is None:
            print(f"  {assoc_id}: FAILED (insufficient data)")
            continue

        loc["assoc_id"] = assoc_id
        results.append(loc)

        # Get hydroacoustic location for comparison
        hloc = hydro_locs[hydro_locs.assoc_id == assoc_id]
        if len(hloc):
            h = hloc.iloc[0]
            offset_km = geodesic_dist_km(loc["lat"], loc["lon"], h.lat, h.lon)
            print(f"  {assoc_id}: ({loc['lat']:.3f}, {loc['lon']:.3f}) "
                  f"z={loc['depth_km']:.1f}km  "
                  f"RMS={loc['rms_residual_s']:.2f}s  "
                  f"err={loc['horiz_err_km']:.1f}km  "
                  f"{loc['n_p']}P+{loc['n_s']}S  "
                  f"offset={offset_km:.1f}km from hydro")
        else:
            print(f"  {assoc_id}: ({loc['lat']:.3f}, {loc['lon']:.3f}) "
                  f"z={loc['depth_km']:.1f}km  "
                  f"RMS={loc['rms_residual_s']:.2f}s  "
                  f"{loc['n_p']}P+{loc['n_s']}S")

    if not results:
        print("No events located. Exiting.")
        return

    # Save seismic locations
    seis_locs = pd.DataFrame([{
        k: v for k, v in r.items() if k != "station_residuals"
    } for r in results])
    seis_locs.to_parquet(DATA_DIR / "seismic_locations.parquet", index=False)
    print(f"\nSaved {len(seis_locs)} seismic locations")

    # --------------------------------------------------------
    # Step 2: Compute residuals
    # --------------------------------------------------------
    print("\n--- Step 2: Compute residuals ---")
    residuals = compute_residuals(seis_locs, hydro_locs)
    residuals.to_parquet(DATA_DIR / "location_residuals.parquet", index=False)

    print(f"  Residuals computed for {len(residuals)} events:")
    for _, row in residuals.iterrows():
        print(f"    {row.assoc_id}: offset={row.offset_km:.1f}km "
              f"az={row.offset_azimuth:.0f}° "
              f"dlat={row.dlat:.4f}° dlon={row.dlon:.4f}°")

    # --------------------------------------------------------
    # Step 3-4: Fit and apply correction
    # --------------------------------------------------------
    print("\n--- Step 3-4: Fit correction and apply to catalogue ---")
    corrected = fit_correction(residuals, hydro_locs)
    corrected.to_parquet(DATA_DIR / "catalogue_corrected.parquet", index=False)
    print(f"\n  Corrected {len(corrected):,} events")
    print(f"  Mean correction magnitude: {corrected['correction_km'].mean():.1f} km")
    print(f"  Max correction magnitude: {corrected['correction_km'].max():.1f} km")

    # --------------------------------------------------------
    # Step 5: Validate against S-P constraints
    # --------------------------------------------------------
    print("\n--- Step 5: Validate with S-P distances ---")
    validation = validate_sp(corrected, picks, vp, vs)
    if len(validation) > 0:
        validation.to_parquet(DATA_DIR / "sp_validation.parquet", index=False)

        print(f"  S-P validation pairs: {len(validation)}")
        print(f"  Original location error vs S-P distance:")
        print(f"    Mean: {validation['original_error_km'].mean():.1f} km")
        print(f"    Median: {validation['original_error_km'].median():.1f} km")
        print(f"  Corrected location error vs S-P distance:")
        print(f"    Mean: {validation['corrected_error_km'].mean():.1f} km")
        print(f"    Median: {validation['corrected_error_km'].median():.1f} km")

        # Improvement
        improved = (validation["corrected_error_km"] <
                    validation["original_error_km"]).sum()
        print(f"  Improved: {improved}/{len(validation)} "
              f"({100*improved/len(validation):.0f}%)")
    else:
        print("  No S-P validation pairs found.")

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Seismic locations:   {DATA_DIR / 'seismic_locations.parquet'}")
    print(f"  Residuals:           {DATA_DIR / 'location_residuals.parquet'}")
    print(f"  Corrected catalogue: {DATA_DIR / 'catalogue_corrected.parquet'}")
    print(f"  S-P validation:      {DATA_DIR / 'sp_validation.parquet'}")


if __name__ == "__main__":
    main()
