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
    print("  Located events by class:")
    for cls, n in class_counts.items():
        print(f"    {cls}: {n:,}")

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
