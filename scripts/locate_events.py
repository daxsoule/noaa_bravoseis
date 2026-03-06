#!/usr/bin/env python3
"""
locate_events.py — Grid-search TDOA event location for BRAVOSEIS.

Locates cross-mooring associated events using time-difference-of-arrival
(TDOA) grid search. For each association with >=3 moorings, finds the
geographic location that minimizes the RMS residual between observed and
predicted inter-station travel time differences.

Includes:
  - Jackknife (leave-one-out) validation for multipath detection
  - Quality tiers (A/B/C/D) based on residual and stability
  - Map output plotted on Bransfield Strait bathymetry

Usage:
    uv run python locate_events.py
    uv run python locate_events.py --skip-map
    uv run python locate_events.py --grid-spacing 0.005

Output:
    outputs/data/event_locations.parquet
    outputs/figures/exploratory/location/event_location_map.png
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from pathlib import Path
from pyproj import Geod
from scipy.interpolate import griddata

from read_dat import MOORINGS

# === Paths ===
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory" / "location"
BATHY_REGIONAL_PATH = Path(
    "/home/jovyan/my_data/bravoseis/bathymetry/bransfield.xyz"
)
BATHY_ORCA_PATH = Path(
    "/home/jovyan/my_data/bravoseis/bathymetry/MGDS_Download/BRAVOSEIS/"
    "Orca_bathymetry.nc"
)

# === Map extent (matches make_bathy_map.py) ===
MAP_LON_MIN, MAP_LON_MAX = -61.5, -55.5
MAP_LAT_MIN, MAP_LAT_MAX = -63.1, -61.9

# === Location parameters ===
MOORING_KEYS = sorted(MOORINGS.keys())
GEOD = Geod(ellps="WGS84")

# Quality tier thresholds (seconds)
TIER_A_MAX_RESIDUAL = 1.0   # Excellent: tight fit, jackknife-stable
TIER_B_MAX_RESIDUAL = 2.0   # Good: usable with caveat
TIER_C_MAX_RESIDUAL = 5.0   # Suspect: flagged
# Above C → tier D (bearing only or failed)

# Jackknife instability threshold: if dropping one mooring moves location
# by more than this (km), the location is downgraded
JACKKNIFE_SHIFT_THRESHOLD_KM = 15.0

# Maximum distance from array centroid for a valid location (km)
# The array spans ~175 km end-to-end; allow 50% beyond that
MAX_DIST_FROM_CENTROID_KM = 150.0

# Per-mooring outlier threshold: if one mooring's individual residual
# exceeds this multiple of the median residual, try dropping it
OUTLIER_RESIDUAL_FACTOR = 3.0


def load_travel_times():
    """Load effective sound speed from travel_times.json."""
    json_path = DATA_DIR / "travel_times.json"
    with open(json_path) as f:
        data = json.load(f)
    return data["effective_speed_mean_ms"]


def build_grid(spacing_deg):
    """Build a lat/lon grid over the study area.

    Extends slightly beyond the map extent to allow locations near edges.
    """
    pad = 0.5  # degrees beyond map extent
    lons = np.arange(MAP_LON_MIN - pad, MAP_LON_MAX + pad, spacing_deg)
    lats = np.arange(MAP_LAT_MIN - pad, MAP_LAT_MAX + pad, spacing_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lon_grid, lat_grid


def precompute_distances(lon_grid, lat_grid):
    """Compute geodesic distance from every grid point to every mooring.

    Returns
    -------
    dist_km : dict
        {mooring_key: 2D array of distances in km, same shape as lon_grid}
    """
    dist_km = {}
    for mkey in MOORING_KEYS:
        m_lon = MOORINGS[mkey]["lon"]
        m_lat = MOORINGS[mkey]["lat"]
        # Broadcast mooring position to grid shape
        m_lon_arr = np.full_like(lon_grid, m_lon)
        m_lat_arr = np.full_like(lat_grid, m_lat)
        _, _, dist_m = GEOD.inv(
            lon_grid.ravel(), lat_grid.ravel(),
            m_lon_arr.ravel(), m_lat_arr.ravel(),
        )
        dist_km[mkey] = (np.array(dist_m) / 1000.0).reshape(lon_grid.shape)
    return dist_km


def locate_one(mooring_onsets, dist_km_grids, c_eff):
    """Grid-search TDOA location for one association.

    Parameters
    ----------
    mooring_onsets : dict
        {mooring_key: onset_time_seconds} for each mooring in this association.
    dist_km_grids : dict
        Precomputed distance grids from precompute_distances().
    c_eff : float
        Effective sound speed (m/s).

    Returns
    -------
    result : dict or None
        Location result with lat, lon, residual_s, etc.
    """
    moorings = sorted(mooring_onsets.keys())
    n_moorings = len(moorings)

    if n_moorings < 2:
        return None

    # Compute observed TDOAs (all pairs)
    obs_tdoa = []
    pair_labels = []
    for i in range(n_moorings):
        for j in range(i + 1, n_moorings):
            dt = mooring_onsets[moorings[j]] - mooring_onsets[moorings[i]]
            obs_tdoa.append(dt)
            pair_labels.append((moorings[i], moorings[j]))
    obs_tdoa = np.array(obs_tdoa)

    # Predicted TDOAs at each grid point
    # t_pred(i→j) = (dist_j - dist_i) / c_eff * 1000 (km→m conversion)
    pred_tdoa_stack = []
    for mi, mj in pair_labels:
        pred_dt = (dist_km_grids[mj] - dist_km_grids[mi]) * 1000.0 / c_eff
        pred_tdoa_stack.append(pred_dt)

    # RMS residual at each grid point
    n_pairs = len(obs_tdoa)
    rms_grid = np.zeros_like(pred_tdoa_stack[0])
    for k in range(n_pairs):
        rms_grid += (pred_tdoa_stack[k] - obs_tdoa[k]) ** 2
    rms_grid = np.sqrt(rms_grid / n_pairs)

    # Find minimum
    min_idx = np.unravel_index(np.argmin(rms_grid), rms_grid.shape)
    best_rms = rms_grid[min_idx]

    # Get lat/lon from grid indices
    # We need the actual grid arrays — they're embedded in dist_km_grids shape
    # but we need them passed separately. We'll return indices and resolve outside.
    return {
        "grid_idx": min_idx,
        "residual_s": float(best_rms),
        "n_pairs": n_pairs,
        "obs_tdoa": obs_tdoa,
        "pair_labels": pair_labels,
        "pred_tdoa_stack": pred_tdoa_stack,
        "rms_grid": rms_grid,
    }


def locate_association(assoc_row, cat_df, dist_km_grids, lon_grid, lat_grid,
                       c_eff, do_jackknife=True):
    """Locate one association with optional jackknife validation.

    Returns a dict with location, quality tier, and diagnostics.
    """
    event_ids = assoc_row["event_ids"].split(",")
    mooring_list = assoc_row["moorings"].split(",")
    n_moorings = assoc_row["n_moorings"]

    # Get onset times for each mooring
    events = cat_df[cat_df["event_id"].isin(event_ids)].copy()
    if len(events) == 0:
        return None

    mooring_onsets = {}
    mooring_event_ids = {}
    for _, ev in events.iterrows():
        mk = ev["mooring"]
        t = ev["onset_utc"].timestamp()
        # Keep earliest onset per mooring (first arrival)
        if mk not in mooring_onsets or t < mooring_onsets[mk]:
            mooring_onsets[mk] = t
            mooring_event_ids[mk] = ev["event_id"]

    if len(mooring_onsets) < 2:
        return None

    # --- Main location ---
    result = locate_one(mooring_onsets, dist_km_grids, c_eff)
    if result is None:
        return None

    lat_loc = float(lat_grid[result["grid_idx"]])
    lon_loc = float(lon_grid[result["grid_idx"]])
    residual = result["residual_s"]

    # --- Per-mooring residuals ---
    per_mooring_resid = {}
    moorings_used = sorted(mooring_onsets.keys())
    for mk in moorings_used:
        # Compute predicted arrival time at this mooring from the located source
        i_row, i_col = result["grid_idx"]
        dist_m = dist_km_grids[mk][i_row, i_col] * 1000.0
        t_pred = dist_m / c_eff
        # Relative to earliest mooring
        t_earliest = min(mooring_onsets.values())
        t_obs_rel = mooring_onsets[mk] - t_earliest
        # The predicted relative time needs a reference too
        # Use the mooring with earliest predicted arrival
        pred_arrivals = {m: dist_km_grids[m][i_row, i_col] * 1000.0 / c_eff
                         for m in moorings_used}
        t_pred_earliest = min(pred_arrivals.values())
        t_pred_rel = pred_arrivals[mk] - t_pred_earliest
        per_mooring_resid[mk] = t_obs_rel - t_pred_rel

    # --- Check for outlier mooring ---
    dropped_mooring = None
    if len(moorings_used) >= 4:
        abs_resids = {mk: abs(r) for mk, r in per_mooring_resid.items()}
        median_resid = np.median(list(abs_resids.values()))
        if median_resid > 0:
            for mk, ar in abs_resids.items():
                if ar > OUTLIER_RESIDUAL_FACTOR * median_resid and ar > 1.0:
                    # Try without this mooring
                    reduced_onsets = {m: t for m, t in mooring_onsets.items()
                                      if m != mk}
                    if len(reduced_onsets) >= 3:
                        r2 = locate_one(reduced_onsets, dist_km_grids, c_eff)
                        if r2 and r2["residual_s"] < residual * 0.7:
                            # Significant improvement — use reduced solution
                            lat_loc = float(lat_grid[r2["grid_idx"]])
                            lon_loc = float(lon_grid[r2["grid_idx"]])
                            residual = r2["residual_s"]
                            dropped_mooring = mk
                            result = r2
                            moorings_used = sorted(reduced_onsets.keys())
                            n_moorings = len(moorings_used)
                    break  # Only try dropping the worst one

    # --- Jackknife (leave-one-out) ---
    jackknife_shift_km = 0.0
    jackknife_stable = True
    if do_jackknife and len(moorings_used) >= 4:
        jk_locs = []
        for mk_drop in moorings_used:
            reduced = {m: t for m, t in mooring_onsets.items()
                       if m != mk_drop and m in moorings_used}
            if len(reduced) < 3:
                continue
            r_jk = locate_one(reduced, dist_km_grids, c_eff)
            if r_jk:
                jk_lat = float(lat_grid[r_jk["grid_idx"]])
                jk_lon = float(lon_grid[r_jk["grid_idx"]])
                _, _, shift_m = GEOD.inv(lon_loc, lat_loc, jk_lon, jk_lat)
                jk_locs.append(shift_m / 1000.0)

        if jk_locs:
            jackknife_shift_km = max(jk_locs)
            if jackknife_shift_km > JACKKNIFE_SHIFT_THRESHOLD_KM:
                jackknife_stable = False

    # --- Distance from array centroid check ---
    centroid_lat = np.mean([MOORINGS[m]["lat"] for m in MOORING_KEYS])
    centroid_lon = np.mean([MOORINGS[m]["lon"] for m in MOORING_KEYS])
    _, _, dist_from_centroid_m = GEOD.inv(
        lon_loc, lat_loc, centroid_lon, centroid_lat)
    dist_from_centroid_km = dist_from_centroid_m / 1000.0

    # --- Assign quality tier ---
    if n_moorings < 3:
        tier = "D"
    elif dist_from_centroid_km > MAX_DIST_FROM_CENTROID_KM:
        tier = "D"  # Too far from array — unreliable
    elif residual <= TIER_A_MAX_RESIDUAL and jackknife_stable and n_moorings >= 4:
        tier = "A"
    elif residual <= TIER_B_MAX_RESIDUAL:
        tier = "B"
    elif residual <= TIER_C_MAX_RESIDUAL:
        tier = "C"
    else:
        tier = "D"

    # Downgrade if jackknife unstable (only for tier A/B)
    if not jackknife_stable and tier in ("A", "B"):
        tier = "C"

    # --- Location uncertainty from residual surface curvature ---
    # Estimate semi-axes of the uncertainty ellipse by fitting a 2D
    # parabola to the RMS surface near the minimum.  The Hessian
    # eigenvalues give curvature in the principal directions; the
    # uncertainty scale is  sigma ~ sqrt(rms_min / eigenvalue) in grid
    # units, converted to km.
    uncertainty_km = np.nan
    rms_grid = result.get("rms_grid")
    if rms_grid is not None:
        ir, ic = result["grid_idx"]
        ny, nx = rms_grid.shape
        # Need at least 1 cell of padding for finite-difference Hessian
        if 1 <= ir < ny - 1 and 1 <= ic < nx - 1:
            # Second derivatives via central differences
            d2r_dy2 = (rms_grid[ir + 1, ic] - 2 * rms_grid[ir, ic]
                       + rms_grid[ir - 1, ic])
            d2r_dx2 = (rms_grid[ir, ic + 1] - 2 * rms_grid[ir, ic]
                       + rms_grid[ir, ic - 1])
            d2r_dxdy = (rms_grid[ir + 1, ic + 1] - rms_grid[ir + 1, ic - 1]
                        - rms_grid[ir - 1, ic + 1]
                        + rms_grid[ir - 1, ic - 1]) / 4.0
            H = np.array([[d2r_dy2, d2r_dxdy],
                          [d2r_dxdy, d2r_dx2]])
            eigvals = np.linalg.eigvalsh(H)
            # Both eigenvalues should be positive at a minimum
            if eigvals[0] > 0 and eigvals[1] > 0:
                # Grid spacing in km (approximate at this latitude)
                deg2km_lat = 111.0  # ~constant
                deg2km_lon = 111.0 * np.cos(np.radians(lat_loc))
                grid_spacing_deg = abs(
                    float(lat_grid[1, 0]) - float(lat_grid[0, 0]))
                dy_km = grid_spacing_deg * deg2km_lat
                dx_km = grid_spacing_deg * deg2km_lon
                # Semi-axes in km (1-sigma level: rms increases by rms_min)
                rms_min = max(residual, 1e-6)
                semi_y = np.sqrt(rms_min / eigvals[0]) * dy_km
                semi_x = np.sqrt(rms_min / eigvals[1]) * dx_km
                # Report geometric mean as scalar uncertainty
                uncertainty_km = round(float(np.sqrt(semi_x * semi_y)), 2)

    return {
        "assoc_id": assoc_row["assoc_id"],
        "lat": lat_loc,
        "lon": lon_loc,
        "residual_s": round(residual, 3),
        "n_moorings": n_moorings,
        "moorings": ",".join(moorings_used),
        "quality_tier": tier,
        "jackknife_shift_km": round(jackknife_shift_km, 2),
        "jackknife_stable": jackknife_stable,
        "dropped_mooring": dropped_mooring,
        "earliest_utc": assoc_row["earliest_utc"],
        "detection_band": assoc_row["detection_band"],
        "uncertainty_km": uncertainty_km,
    }


def classify_located_events(loc_df, cat_df, umap_df, cnn_df):
    """Add event classification labels to located events.

    Uses Phase 1 cluster labels (from train_cnn.py label assignment logic)
    and Phase 2 CNN predictions (cnn_label column) to classify each
    association by majority vote of its constituent events.
    """
    # Phase 1: cluster-based labels (same logic as train_cnn.py)
    tphase_clusters = {"low_0", "low_1", "mid_0"}
    phase1_map = {}
    for _, row in umap_df.iterrows():
        cid = row["cluster_id"]
        if cid in tphase_clusters:
            phase1_map[row["event_id"]] = "tphase"

    # Phase 2: CNN predictions for bulk events
    cnn_map = {}
    if cnn_df is not None:
        for _, row in cnn_df.iterrows():
            cnn_map[row["event_id"]] = row["cnn_label"]

    # Map associations to classes via their constituent events
    assoc_df = pd.read_parquet(DATA_DIR / "cross_mooring_associations.parquet")

    assoc_classes = {}
    for _, assoc_row in assoc_df.iterrows():
        eids = assoc_row["event_ids"].split(",")
        classes = []
        for eid in eids:
            if eid in phase1_map:
                classes.append(phase1_map[eid])
            elif eid in cnn_map:
                classes.append(cnn_map[eid])
        if not classes:
            assoc_classes[assoc_row["assoc_id"]] = "unclassified"
        elif "tphase" in classes:
            assoc_classes[assoc_row["assoc_id"]] = "tphase"
        elif "icequake" in classes:
            assoc_classes[assoc_row["assoc_id"]] = "icequake"
        elif "vessel" in classes:
            assoc_classes[assoc_row["assoc_id"]] = "vessel"
        else:
            assoc_classes[assoc_row["assoc_id"]] = "unclassified"

    loc_df["event_class"] = loc_df["assoc_id"].map(assoc_classes)
    loc_df["event_class"] = loc_df["event_class"].fillna("unclassified")
    return loc_df


# Maximum distance from coast/ice for icequake classification (km).
# In summer (ice-free), icequakes must be near glaciers/ice shelves.
# In winter, sea ice extends across the strait and mid-strait events
# can be legitimate (sea-ice cracking, pressure ridging).
ICEQUAKE_MAX_COAST_DIST_KM = 30.0

# Sea ice concentration threshold — if a location falls in an area
# with monthly SIC >= this value, it is considered ice-covered and
# the coast-distance filter is relaxed.
SEA_ICE_CONC_THRESHOLD = 0.15

# Path to cached sea ice data (from PolarWatch NSIDC CDR)
SEA_ICE_PATH = DATA_DIR / "sea_ice_monthly.npz"


def _load_sea_ice():
    """Load monthly sea ice concentration grid, or None if unavailable."""
    if not SEA_ICE_PATH.exists():
        return None
    sic = np.load(SEA_ICE_PATH)
    ice_conc = sic["ice_conc"]
    # Mask flag values (>1.0 = pole hole, land, etc.)
    ice_conc = np.where((ice_conc >= 0) & (ice_conc <= 1.0), ice_conc, np.nan)
    times = list(sic["times"])
    lon_grid = sic["lon_grid"]
    lat_grid = sic["lat_grid"]
    return {"conc": ice_conc, "times": times,
            "lon": lon_grid, "lat": lat_grid}


def _get_ice_conc_at(sic_data, lon, lat, month_str):
    """Look up sea ice concentration at a point for a given month.

    Uses nearest-neighbour lookup on the 25 km polar stereographic grid.
    Returns NaN if the month is not in the data or the point is outside
    the grid.
    """
    if sic_data is None or month_str not in sic_data["times"]:
        return np.nan
    tidx = sic_data["times"].index(month_str)
    ice_slice = sic_data["conc"][tidx]  # (ygrid, xgrid)
    # Nearest-neighbour: find grid cell closest to (lon, lat)
    dist2 = (sic_data["lon"] - lon) ** 2 + (sic_data["lat"] - lat) ** 2
    idx = np.unravel_index(np.argmin(dist2), dist2.shape)
    return float(ice_slice[idx])


def filter_icequake_locations(loc_df):
    """Reclassify icequakes located far from coastlines/ice shelves.

    Uses Natural Earth 10 m coastlines + Antarctic ice shelf polygons to
    compute geodesic distance from each located icequake to the nearest
    land/ice feature.

    Applies a **seasonally varying threshold**:
    - If the event's location has sea ice concentration >= 15% for that
      month (NSIDC CDR), the event is retained regardless of coast distance
      (sea-ice cracking is physically plausible in ice-covered water).
    - Otherwise, events beyond ICEQUAKE_MAX_COAST_DIST_KM (30 km) are
      reclassified as 'unclassified'.

    Adds columns:
        dist_to_coast_km  — geodesic distance to nearest land/ice (all events)
        sea_ice_conc      — monthly SIC at the event location (0–1)
        icequake_filtered — True if reclassified away from icequake
    """
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    from shapely.ops import nearest_points, unary_union

    print("  Loading coastline + ice shelf geometries...")
    coast_shp = shpreader.natural_earth(
        resolution="10m", category="physical", name="land")
    ice_shp = shpreader.natural_earth(
        resolution="10m", category="physical", name="antarctic_ice_shelves_polys")

    relevant = []
    for g in (list(shpreader.Reader(coast_shp).geometries())
              + list(shpreader.Reader(ice_shp).geometries())):
        minx, miny, maxx, maxy = g.bounds
        if maxy < -66 or miny > -58 or maxx < -70 or minx > -48:
            continue
        relevant.append(g)
    land_union = unary_union(relevant)
    print(f"  {len(relevant)} land/ice polygons in study region")

    # Load sea ice data
    sic_data = _load_sea_ice()
    if sic_data is not None:
        print(f"  Sea ice data loaded: {len(sic_data['times'])} months")
    else:
        print("  WARNING: No sea ice data — using fixed coast-distance filter")

    # Compute distance and SIC for located events (A/B/C with valid coords)
    located_mask = (
        loc_df["quality_tier"].isin(["A", "B", "C"])
        & loc_df["lat"].notna()
    )
    geod = GEOD

    distances = np.full(len(loc_df), np.nan)
    sic_vals = np.full(len(loc_df), np.nan)
    loc_df_times = pd.to_datetime(loc_df["earliest_utc"])

    for idx in loc_df.index[located_mask]:
        pos = loc_df.index.get_loc(idx)
        lon_val = loc_df.at[idx, "lon"]
        lat_val = loc_df.at[idx, "lat"]

        # Distance to coast
        pt = Point(lon_val, lat_val)
        near_pt = nearest_points(pt, land_union)[1]
        _, _, dist_m = geod.inv(pt.x, pt.y, near_pt.x, near_pt.y)
        distances[pos] = abs(dist_m) / 1000.0

        # Sea ice concentration for this month
        month_str = loc_df_times.iloc[pos].strftime("%Y-%m")
        sic_vals[pos] = _get_ice_conc_at(sic_data, lon_val, lat_val, month_str)

    loc_df["dist_to_coast_km"] = distances
    loc_df["sea_ice_conc"] = sic_vals

    # Apply seasonally varying filter to icequakes
    is_icequake = loc_df["event_class"] == "icequake"
    is_far = loc_df["dist_to_coast_km"] > ICEQUAKE_MAX_COAST_DIST_KM
    is_ice_covered = loc_df["sea_ice_conc"] >= SEA_ICE_CONC_THRESHOLD

    # Reclassify: far from coast AND NOT in ice-covered water
    ice_filter = is_icequake & located_mask & is_far & ~is_ice_covered

    n_ice_total = (is_icequake & located_mask).sum()
    n_near_coast = (is_icequake & located_mask & ~is_far).sum()
    n_ice_covered = (is_icequake & located_mask & is_far & is_ice_covered).sum()
    n_reclassified = ice_filter.sum()

    loc_df["icequake_filtered"] = False
    loc_df.loc[ice_filter, "icequake_filtered"] = True
    loc_df.loc[ice_filter, "event_class"] = "unclassified"

    print(f"  Icequake coast-distance + sea-ice filter:")
    print(f"    {n_ice_total:,} located icequakes before filter")
    print(f"    {n_near_coast:,} retained (within {ICEQUAKE_MAX_COAST_DIST_KM} km"
          f" of coast)")
    print(f"    {n_ice_covered:,} retained (>{ICEQUAKE_MAX_COAST_DIST_KM} km but"
          f" in ice-covered water, SIC>={SEA_ICE_CONC_THRESHOLD})")
    print(f"    {n_reclassified:,} reclassified to 'unclassified' (far from"
          f" coast, no ice cover)")
    print(f"    {n_ice_total - n_reclassified:,} retained total")

    return loc_df


# --- Swarm coherence QC ---
# T-phase swarms (temporally clustered events from the same seismic
# source) should be spatially coherent.  Events within a temporal swarm
# that locate far from the swarm's spatial centroid are likely mislocation
# artifacts.
SWARM_TIME_GAP_S = 3600.0      # Max gap between adjacent events in a swarm (1 hr)
SWARM_MIN_EVENTS = 10           # Min events to define a swarm
SWARM_OUTLIER_SIGMA = 3.0       # Flag events > 3 MAD from swarm centroid


def compute_swarm_coherence(loc_df):
    """Flag T-phase locations that are spatial outliers within temporal swarms.

    Groups temporally adjacent T-phase events into swarms, computes each
    swarm's spatial centroid, and flags events whose distance from the
    centroid exceeds SWARM_OUTLIER_SIGMA × MAD of the swarm's distances.

    Adds columns:
        swarm_id     — integer swarm label (NaN for non-swarm / non-T-phase)
        swarm_dist_km — distance from swarm centroid
        swarm_outlier — True if spatial outlier within its swarm
    """
    located_tph = loc_df[
        (loc_df["event_class"] == "tphase")
        & (loc_df["quality_tier"].isin(["A", "B", "C"]))
        & loc_df["lat"].notna()
    ].copy()

    if len(located_tph) == 0:
        loc_df["swarm_id"] = np.nan
        loc_df["swarm_dist_km"] = np.nan
        loc_df["swarm_outlier"] = False
        return loc_df

    # Sort by time
    located_tph["t"] = pd.to_datetime(located_tph["earliest_utc"])
    located_tph = located_tph.sort_values("t")

    # Assign swarm IDs by detecting gaps > SWARM_TIME_GAP_S
    dt = located_tph["t"].diff().dt.total_seconds().fillna(0)
    swarm_labels = (dt > SWARM_TIME_GAP_S).cumsum()
    located_tph["_swarm"] = swarm_labels.values

    # Only keep swarms with enough events
    swarm_counts = located_tph["_swarm"].value_counts()
    valid_swarms = swarm_counts[swarm_counts >= SWARM_MIN_EVENTS].index

    # Compute centroid and flag outliers
    swarm_ids = np.full(len(loc_df), np.nan)
    swarm_dists = np.full(len(loc_df), np.nan)
    swarm_outliers = np.zeros(len(loc_df), dtype=bool)

    geod = GEOD
    n_outliers = 0
    n_swarms = 0

    for sw_id in valid_swarms:
        sw = located_tph[located_tph["_swarm"] == sw_id]
        c_lat = sw["lat"].median()
        c_lon = sw["lon"].median()

        # Distance of each event from centroid
        dists_km = []
        for _, row in sw.iterrows():
            _, _, d_m = geod.inv(row["lon"], row["lat"], c_lon, c_lat)
            dists_km.append(abs(d_m) / 1000.0)
        dists_km = np.array(dists_km)

        mad = np.median(np.abs(dists_km - np.median(dists_km)))
        if mad < 1.0:
            mad = 1.0  # Floor to avoid flagging tight swarms

        threshold = np.median(dists_km) + SWARM_OUTLIER_SIGMA * mad

        for i, (orig_idx, _) in enumerate(sw.iterrows()):
            pos = loc_df.index.get_loc(orig_idx)
            swarm_ids[pos] = int(sw_id)
            swarm_dists[pos] = round(dists_km[i], 2)
            if dists_km[i] > threshold:
                swarm_outliers[pos] = True
                n_outliers += 1
        n_swarms += 1

    loc_df["swarm_id"] = swarm_ids
    loc_df["swarm_dist_km"] = swarm_dists
    loc_df["swarm_outlier"] = swarm_outliers

    n_in_swarms = np.isfinite(swarm_ids).sum()
    print(f"  Swarm coherence QC:")
    print(f"    {n_swarms} swarms identified (>={SWARM_MIN_EVENTS} events,"
          f" <{SWARM_TIME_GAP_S:.0f} s gap)")
    print(f"    {n_in_swarms:,} T-phase events in swarms")
    print(f"    {n_outliers:,} spatial outliers flagged"
          f" (>{SWARM_OUTLIER_SIGMA}×MAD from centroid)")

    return loc_df


def load_bathy_grid():
    """Load and merge regional + Orca bathymetry (reused from make_bathy_map.py)."""
    print("  Loading regional bathymetry...")
    data = np.loadtxt(BATHY_REGIONAL_PATH)
    lon_raw, lat_raw, z_raw = data[:, 0], data[:, 1], data[:, 2]

    grid_spacing = 0.004
    lon_grid_1d = np.arange(lon_raw.min(), lon_raw.max(), grid_spacing)
    lat_grid_1d = np.arange(lat_raw.min(), lat_raw.max(), grid_spacing)
    lon_grid, lat_grid = np.meshgrid(lon_grid_1d, lat_grid_1d)

    z_grid = griddata(
        (lon_raw, lat_raw), z_raw,
        (lon_grid, lat_grid), method="linear"
    )
    del data, lon_raw, lat_raw, z_raw

    if BATHY_ORCA_PATH.exists():
        import xarray as xr
        print("  Overlaying MGDS Orca bathymetry...")
        ds = xr.open_dataset(BATHY_ORCA_PATH)
        orca_lat = ds["latitude"].values
        orca_lon = ds["longitude"].values
        orca_z = ds["data"].values

        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (orca_lat, orca_lon), orca_z,
            method="linear", bounds_error=False, fill_value=np.nan
        )
        lat_mask = (lat_grid_1d >= orca_lat.min()) & (lat_grid_1d <= orca_lat.max())
        lon_mask = (lon_grid_1d >= orca_lon.min()) & (lon_grid_1d <= orca_lon.max())
        lon_mg, lat_mg = np.meshgrid(lon_grid_1d[lon_mask], lat_grid_1d[lat_mask])
        pts = np.column_stack([lat_mg.ravel(), lon_mg.ravel()])
        z_interp = interp(pts).reshape(lat_mg.shape)

        valid = ~np.isnan(z_interp)
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        for jj, j in enumerate(lat_idx):
            for ii, i in enumerate(lon_idx):
                if valid[jj, ii]:
                    z_grid[j, i] = z_interp[jj, ii]
        ds.close()

    return lon_grid_1d, lat_grid_1d, z_grid


def make_location_map(loc_df, skip_bathy=False):
    """Plot event locations on Bransfield Strait bathymetry map."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cmocean

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    map_proj = ccrs.Mercator(central_longitude=-58.5)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.08, 0.12, 0.72, 0.78], projection=map_proj)
    ax.set_extent([MAP_LON_MIN, MAP_LON_MAX, MAP_LAT_MIN, MAP_LAT_MAX],
                  crs=data_crs)

    # --- Bathymetry background ---
    if not skip_bathy:
        try:
            lon_grid_1d, lat_grid_1d, z_grid = load_bathy_grid()
            z_ocean = np.where(z_grid <= 0, z_grid, np.nan)
            z_min, z_max = np.nanpercentile(z_ocean, [2, 98])
            ls = LightSource(azdeg=315, altdeg=45)
            bathy_cmap = cmocean.cm.ice.copy()
            bathy_cmap.set_bad(color="#d9d9d9")
            rgb = ls.shade(z_ocean, cmap=bathy_cmap, blend_mode="soft",
                           vmin=z_min, vmax=z_max)
            ax.imshow(rgb,
                      extent=[lon_grid_1d.min(), lon_grid_1d.max(),
                              lat_grid_1d.min(), lat_grid_1d.max()],
                      origin="lower", transform=data_crs)
        except Exception as e:
            print(f"  WARNING: Could not load bathymetry: {e}")
            ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")

    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="black",
                   linewidth=0.4, zorder=10)
    ax.coastlines(resolution="10m", linewidth=0.4, color="black", zorder=10)

    # --- Gridlines ---
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=0.5, color="white", alpha=0.3, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    # --- Plot locations by class and quality ---
    # Filter to locatable events (tier A, B, C)
    locatable = loc_df[loc_df["quality_tier"].isin(["A", "B", "C"])].copy()

    class_styles = {
        "tphase": {"color": "red", "marker": "o", "label": "T-phase"},
        "icequake": {"color": "dodgerblue", "marker": "s", "label": "Icequake"},
        "vessel": {"color": "orange", "marker": "D", "label": "Vessel"},
        "unclassified": {"color": "gray", "marker": ".", "label": "Unclassified"},
    }

    tier_sizes = {"A": 18, "B": 8, "C": 4}
    tier_alphas = {"A": 0.8, "B": 0.4, "C": 0.2}

    # Plot lower tiers first, higher on top
    for cls, style in class_styles.items():
        cls_df = locatable[locatable["event_class"] == cls]
        if len(cls_df) == 0:
            continue
        for tier in ["C", "B", "A"]:
            tier_df = cls_df[cls_df["quality_tier"] == tier]
            if len(tier_df) == 0:
                continue
            ax.scatter(
                tier_df["lon"].values, tier_df["lat"].values,
                c=style["color"], marker=style["marker"],
                s=tier_sizes[tier], alpha=tier_alphas[tier],
                edgecolors="none", transform=data_crs, zorder=12,
            )

    # Legend with class counts only
    for cls, style in class_styles.items():
        n = (locatable["event_class"] == cls).sum()
        if n > 0:
            ax.scatter([], [], c=style["color"], marker=style["marker"],
                       s=30, label=f"{style['label']} ({n:,})")

    # --- Mooring locations ---
    for key, info in sorted(MOORINGS.items()):
        ax.plot(info["lon"], info["lat"], marker="^", color="white",
                markersize=8, markeredgecolor="black", markeredgewidth=1.0,
                transform=data_crs, zorder=16)
        ax.text(info["lon"], info["lat"] + 0.04, info["name"],
                fontsize=7, fontweight="bold", ha="center",
                va="bottom", transform=data_crs, zorder=16,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.8, edgecolor="none"))

    ax.set_title("BRAVOSEIS Event Locations — Grid-Search TDOA",
                 fontsize=13, fontweight="bold", pad=10)

    # --- Legend ---
    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.9,
                    markerscale=1.2)
    leg.set_zorder(20)

    # --- Quality summary text ---
    tier_counts = loc_df["quality_tier"].value_counts()
    summary = "Quality: " + " | ".join(
        f"{t}: {tier_counts.get(t, 0):,}"
        for t in ["A", "B", "C", "D"]
    )
    ax.text(0.02, 0.02, summary, fontsize=8,
            transform=ax.transAxes, zorder=20,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    outpath = FIG_DIR / "event_location_map.png"
    fig.savefig(outpath, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


def _add_bathy_and_moorings(ax, data_crs, skip_bathy=False):
    """Add bathymetry background, coastlines, gridlines, and moorings to ax."""
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cmocean

    if not skip_bathy:
        try:
            lon_grid_1d, lat_grid_1d, z_grid = load_bathy_grid()
            z_ocean = np.where(z_grid <= 0, z_grid, np.nan)
            z_min, z_max = np.nanpercentile(z_ocean, [2, 98])
            ls = LightSource(azdeg=315, altdeg=45)
            bathy_cmap = cmocean.cm.ice.copy()
            bathy_cmap.set_bad(color="#d9d9d9")
            rgb = ls.shade(z_ocean, cmap=bathy_cmap, blend_mode="soft",
                           vmin=z_min, vmax=z_max)
            ax.imshow(rgb,
                      extent=[lon_grid_1d.min(), lon_grid_1d.max(),
                              lat_grid_1d.min(), lat_grid_1d.max()],
                      origin="lower", transform=data_crs)
        except Exception as e:
            print(f"  WARNING: Could not load bathymetry: {e}")
            ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")

    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="black",
                   linewidth=0.4, zorder=10)
    ax.coastlines(resolution="10m", linewidth=0.4, color="black", zorder=10)

    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=0.5, color="white", alpha=0.3, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    for key, info in sorted(MOORINGS.items()):
        ax.plot(info["lon"], info["lat"], marker="^", color="white",
                markersize=8, markeredgecolor="black", markeredgewidth=1.0,
                transform=data_crs, zorder=16)
        ax.text(info["lon"], info["lat"] + 0.04, info["name"],
                fontsize=7, fontweight="bold", ha="center",
                va="bottom", transform=data_crs, zorder=16,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.8, edgecolor="none"))


def _get_event_features_for_locations(loc_df):
    """Link located associations back to event features (max SNR per assoc)."""
    assoc_df = pd.read_parquet(DATA_DIR / "cross_mooring_associations.parquet")
    feat_df = pd.read_parquet(DATA_DIR / "event_features.parquet")

    # For each association, get the max SNR and peak_power across its events
    rows = []
    for _, arow in assoc_df.iterrows():
        eids = arow["event_ids"].split(",")
        match = feat_df[feat_df["event_id"].isin(eids)]
        if len(match) > 0:
            rows.append({
                "assoc_id": arow["assoc_id"],
                "max_snr": match["snr"].max(),
                "max_peak_power_db": match["peak_power_db"].max(),
                "max_duration_s": match["duration_s"].max(),
            })
    feat_agg = pd.DataFrame(rows)
    return loc_df.merge(feat_agg, on="assoc_id", how="left")


def make_class_maps(loc_df):
    """Make 6-panel (2-month bins) maps for T-phases and icequakes.

    Each panel shows one 2-month window. Dot color = time within the window,
    dot size = SNR (T-phase) or peak power (icequake).
    """
    import cartopy.crs as ccrs
    from matplotlib.colors import Normalize

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Get event features for sizing
    print("  Linking event features for dot sizing...")
    loc_feat = _get_event_features_for_locations(loc_df)
    locatable = loc_feat[loc_feat["quality_tier"].isin(["A", "B", "C"])].copy()
    locatable["earliest_utc"] = pd.to_datetime(locatable["earliest_utc"])

    # Load bathy once, reuse across all panels
    print("  Loading bathymetry (once)...")
    try:
        lon_grid_1d, lat_grid_1d, z_grid = load_bathy_grid()
        z_ocean = np.where(z_grid <= 0, z_grid, np.nan)
        z_min, z_max = np.nanpercentile(z_ocean, [2, 98])
        bathy_data = (lon_grid_1d, lat_grid_1d, z_ocean, z_min, z_max)
    except Exception as e:
        print(f"  WARNING: Could not load bathymetry: {e}")
        bathy_data = None

    # 6 two-month bins covering the deployment
    time_bins = [
        (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-03-01"), "Jan–Feb 2019"),
        (pd.Timestamp("2019-03-01"), pd.Timestamp("2019-05-01"), "Mar–Apr 2019"),
        (pd.Timestamp("2019-05-01"), pd.Timestamp("2019-07-01"), "May–Jun 2019"),
        (pd.Timestamp("2019-07-01"), pd.Timestamp("2019-09-01"), "Jul–Aug 2019"),
        (pd.Timestamp("2019-09-01"), pd.Timestamp("2019-11-01"), "Sep–Oct 2019"),
        (pd.Timestamp("2019-11-01"), pd.Timestamp("2020-03-01"), "Nov 2019–Feb 2020"),
    ]

    map_proj = ccrs.Mercator(central_longitude=-58.5)
    data_crs = ccrs.PlateCarree()

    class_configs = {
        "tphase": {
            "suptitle": "BRAVOSEIS T-Phase (Earthquake) Locations",
            "cmap": "hot_r",
            "filename": "tphase_location_6panel.png",
            "size_col": "max_snr",
            "size_label": "SNR",
            "size_range": (3, 50),
        },
        "icequake": {
            "suptitle": "BRAVOSEIS Icequake Locations",
            "cmap": "hot_r",
            "filename": "icequake_location_6panel.png",
            "size_col": "max_peak_power_db",
            "size_label": "Peak Power (dB)",
            "size_range": (8, 80),
        },
    }

    for cls, cfg in class_configs.items():
        all_cls = locatable[locatable["event_class"] == cls].copy()
        if len(all_cls) == 0:
            print(f"  No {cls} events to plot")
            continue

        # Global size scaling (consistent across panels)
        size_vals_all = all_cls[cfg["size_col"]].fillna(
            all_cls[cfg["size_col"]].median())
        s_min = size_vals_all.quantile(0.05)
        s_max = size_vals_all.quantile(0.95)
        if s_max == s_min:
            s_max = s_min + 1

        print(f"  Plotting {cls} 6-panel ({len(all_cls):,} events total)...")

        fig, axes = plt.subplots(
            2, 3, figsize=(18, 11),
            subplot_kw={"projection": map_proj},
        )
        fig.suptitle(
            f"{cfg['suptitle']} (n={len(all_cls):,})",
            fontsize=16, fontweight="bold", y=0.98,
        )

        for idx, (t_start, t_end, label) in enumerate(time_bins):
            ax = axes.flat[idx]
            ax.set_extent(
                [MAP_LON_MIN, MAP_LON_MAX, MAP_LAT_MIN, MAP_LAT_MAX],
                crs=data_crs,
            )

            # Bathy background
            import cartopy.feature as cfeature
            import cmocean
            if bathy_data is not None:
                lg1d, la1d, z_oc, zn, zx = bathy_data
                ls = LightSource(azdeg=315, altdeg=45)
                bcmap = cmocean.cm.ice.copy()
                bcmap.set_bad(color="#d9d9d9")
                rgb = ls.shade(z_oc, cmap=bcmap, blend_mode="soft",
                               vmin=zn, vmax=zx)
                ax.imshow(rgb,
                          extent=[lg1d.min(), lg1d.max(),
                                  la1d.min(), la1d.max()],
                          origin="lower", transform=data_crs)

            ax.add_feature(cfeature.LAND, facecolor="#d9d9d9",
                           edgecolor="black", linewidth=0.3, zorder=10)
            ax.coastlines(resolution="10m", linewidth=0.3, color="black",
                          zorder=10)

            # Moorings (small, unobtrusive)
            for key, info in sorted(MOORINGS.items()):
                ax.plot(info["lon"], info["lat"], marker="^", color="white",
                        markersize=5, markeredgecolor="black",
                        markeredgewidth=0.6, transform=data_crs, zorder=16)

            # Subset for this time bin
            mask = ((all_cls["earliest_utc"] >= t_start)
                    & (all_cls["earliest_utc"] < t_end))
            subset = all_cls[mask].copy()
            n_events = len(subset)

            if n_events > 0:
                # Time within this bin as color
                bin_seconds = (t_end - t_start).total_seconds()
                t_numeric = (
                    subset["earliest_utc"] - t_start
                ).dt.total_seconds()
                t_norm = Normalize(vmin=0, vmax=bin_seconds)

                # Size
                sv = subset[cfg["size_col"]].fillna(size_vals_all.median())
                size_frac = ((sv - s_min) / (s_max - s_min)).clip(0, 1)
                dot_sizes = (cfg["size_range"][0]
                             + size_frac * (cfg["size_range"][1]
                                            - cfg["size_range"][0]))

                # Sort by time (latest on top)
                sort_idx = t_numeric.argsort()
                subset = subset.iloc[sort_idx.values]
                t_numeric = t_numeric.iloc[sort_idx.values]
                dot_sizes = dot_sizes.iloc[sort_idx.values]

                ax.scatter(
                    subset["lon"].values, subset["lat"].values,
                    c=t_numeric.values, cmap=cfg["cmap"], norm=t_norm,
                    s=dot_sizes.values, alpha=0.75,
                    edgecolors="black", linewidths=0.15,
                    transform=data_crs, zorder=12,
                )

            ax.set_title(f"{label}  (n={n_events:,})",
                         fontsize=11, fontweight="bold", pad=4)

            # Gridlines only on edge panels
            from cartopy.mpl.gridliner import (LONGITUDE_FORMATTER,
                                               LATITUDE_FORMATTER)
            gl = ax.gridlines(
                crs=data_crs, draw_labels=True,
                linewidth=0.3, color="white", alpha=0.2, linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.bottom_labels = (idx >= 3)  # bottom row only
            gl.left_labels = (idx % 3 == 0)  # left column only
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {"size": 7}
            gl.ylabel_style = {"size": 7}

        # Size legend in bottom-right panel area
        ax_last = axes.flat[-1]
        legend_quantiles = [0.25, 0.50, 0.75]
        for q in legend_quantiles:
            val = size_vals_all.quantile(q)
            frac = (val - s_min) / (s_max - s_min)
            frac = max(0, min(1, frac))
            sz = cfg["size_range"][0] + frac * (
                cfg["size_range"][1] - cfg["size_range"][0])
            ax_last.scatter([], [], s=sz, c="gray", edgecolors="black",
                            linewidths=0.3, alpha=0.7,
                            label=f"{cfg['size_label']}={val:.1f}")
        leg = ax_last.legend(loc="lower right", fontsize=7, framealpha=0.9,
                             title=cfg["size_label"], title_fontsize=7)
        leg.set_zorder(20)

        # Color legend annotation
        axes.flat[0].text(
            0.02, 0.05,
            "Color: early → late\nwithin each 2-month window",
            fontsize=6, transform=axes.flat[0].transAxes, zorder=20,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        fig.subplots_adjust(
            left=0.04, right=0.97, top=0.93, bottom=0.04,
            wspace=0.08, hspace=0.15,
        )

        outpath = FIG_DIR / cfg["filename"]
        fig.savefig(outpath, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid-search TDOA event location for BRAVOSEIS")
    parser.add_argument("--grid-spacing", type=float, default=0.01,
                        help="Grid spacing in degrees (default: 0.01, ~1 km)")
    parser.add_argument("--skip-map", action="store_true",
                        help="Skip generating the map figure")
    parser.add_argument("--no-jackknife", action="store_true",
                        help="Skip jackknife validation (faster)")
    args = parser.parse_args()

    print("=" * 60)
    print("BRAVOSEIS Event Location — Grid-Search TDOA")
    print("=" * 60)

    # --- Load data ---
    print("\nLoading data...")
    cat_df = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    cat_df["onset_utc"] = pd.to_datetime(cat_df["onset_utc"])
    assoc_df = pd.read_parquet(DATA_DIR / "cross_mooring_associations.parquet")
    assoc_df["earliest_utc"] = pd.to_datetime(assoc_df["earliest_utc"])
    c_eff = load_travel_times()
    print(f"  Catalogue: {len(cat_df):,} events")
    print(f"  Associations: {len(assoc_df):,}")
    print(f"  Effective sound speed: {c_eff:.1f} m/s")

    # Filter to >=3 moorings for 2D location
    locatable = assoc_df[assoc_df["n_moorings"] >= 3].copy()
    bearings_only = assoc_df[assoc_df["n_moorings"] == 2].copy()
    print(f"  Locatable (>=3 moorings): {len(locatable):,}")
    print(f"  Bearing only (2 moorings): {len(bearings_only):,}")

    # --- Build search grid ---
    print(f"\nBuilding search grid (spacing={args.grid_spacing}°)...")
    lon_grid, lat_grid = build_grid(args.grid_spacing)
    print(f"  Grid size: {lon_grid.shape[1]} x {lon_grid.shape[0]} "
          f"= {lon_grid.size:,} points")

    print("Precomputing mooring distances...")
    dist_km_grids = precompute_distances(lon_grid, lat_grid)

    # --- Locate events ---
    print(f"\nLocating {len(locatable):,} associations...")
    do_jk = not args.no_jackknife
    if not do_jk:
        print("  (Jackknife disabled)")

    results = []
    for i, (_, assoc_row) in enumerate(locatable.iterrows()):
        loc = locate_association(
            assoc_row, cat_df, dist_km_grids, lon_grid, lat_grid,
            c_eff, do_jackknife=do_jk
        )
        if loc is not None:
            results.append(loc)

        if (i + 1) % 2000 == 0:
            n_ok = len(results)
            print(f"  {i+1:,}/{len(locatable):,} processed ({n_ok:,} located)")

    # Also add 2-mooring associations as tier D (no location, just metadata)
    for _, assoc_row in bearings_only.iterrows():
        results.append({
            "assoc_id": assoc_row["assoc_id"],
            "lat": np.nan,
            "lon": np.nan,
            "residual_s": np.nan,
            "n_moorings": assoc_row["n_moorings"],
            "moorings": assoc_row["moorings"],
            "quality_tier": "D",
            "jackknife_shift_km": np.nan,
            "jackknife_stable": False,
            "dropped_mooring": None,
            "earliest_utc": assoc_row["earliest_utc"],
            "detection_band": assoc_row["detection_band"],
            "uncertainty_km": np.nan,
        })

    loc_df = pd.DataFrame(results)
    print(f"\nLocated {len(loc_df):,} associations total")

    # --- Quality summary ---
    print("\nQuality tier breakdown:")
    for tier in ["A", "B", "C", "D"]:
        n = (loc_df["quality_tier"] == tier).sum()
        if tier != "D":
            tier_data = loc_df[loc_df["quality_tier"] == tier]
            if len(tier_data) > 0:
                med_r = tier_data["residual_s"].median()
                print(f"  {tier}: {n:,} (median residual {med_r:.2f} s)")
            else:
                print(f"  {tier}: {n:,}")
        else:
            print(f"  {tier}: {n:,} (2-mooring or failed)")

    located = loc_df[loc_df["quality_tier"].isin(["A", "B", "C"])]
    print(f"\nTotal located (A+B+C): {len(located):,}")
    if len(located) > 0:
        print(f"  Median residual: {located['residual_s'].median():.2f} s")
        jk_data = located[located["jackknife_shift_km"].notna()]
        if len(jk_data) > 0:
            n_unstable = (~jk_data["jackknife_stable"]).sum()
            print(f"  Jackknife unstable: {n_unstable:,} "
                  f"({100*n_unstable/len(jk_data):.1f}%)")
        n_dropped = located["dropped_mooring"].notna().sum()
        if n_dropped > 0:
            print(f"  Outlier mooring dropped: {n_dropped:,}")

    # --- Add event classification ---
    print("\nAdding event classifications...")
    umap_df = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")
    cnn_path = DATA_DIR / "cnn_predictions.parquet"
    cnn_df = pd.read_parquet(cnn_path) if cnn_path.exists() else None
    loc_df = classify_located_events(loc_df, cat_df, umap_df, cnn_df)

    class_counts = loc_df[loc_df["quality_tier"].isin(["A", "B", "C"])][
        "event_class"].value_counts()
    print("  Located events by class (before icequake filter):")
    for cls, n in class_counts.items():
        print(f"    {cls}: {n:,}")

    # --- Filter implausible icequake locations ---
    print("\nFiltering icequake locations by distance to coast/ice...")
    loc_df = filter_icequake_locations(loc_df)

    class_counts_post = loc_df[loc_df["quality_tier"].isin(["A", "B", "C"])][
        "event_class"].value_counts()
    print("  Located events by class (after icequake filter):")
    for cls, n in class_counts_post.items():
        print(f"    {cls}: {n:,}")

    # --- Swarm coherence QC ---
    print("\nRunning swarm coherence QC on T-phases...")
    loc_df = compute_swarm_coherence(loc_df)

    # --- Location uncertainty summary ---
    unc = loc_df[loc_df["quality_tier"].isin(["A", "B", "C"])]["uncertainty_km"]
    valid_unc = unc.dropna()
    if len(valid_unc) > 0:
        print(f"\nLocation uncertainty (from residual curvature):")
        print(f"  Computed for {len(valid_unc):,}/{len(unc):,} located events")
        print(f"  Median: {valid_unc.median():.1f} km")
        print(f"  IQR: [{valid_unc.quantile(0.25):.1f},"
              f" {valid_unc.quantile(0.75):.1f}] km")
        for tier in ["A", "B", "C"]:
            tu = loc_df[loc_df["quality_tier"] == tier]["uncertainty_km"].dropna()
            if len(tu) > 0:
                print(f"  Tier {tier}: median {tu.median():.1f} km"
                      f" (n={len(tu):,})")

    # --- Save ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "event_locations.parquet"
    loc_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # --- Maps ---
    if not args.skip_map:
        print("\nGenerating location maps...")
        make_location_map(loc_df)
        make_class_maps(loc_df)

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
