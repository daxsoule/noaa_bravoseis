#!/usr/bin/env python3
"""
make_location_accuracy_figure.py — Location accuracy vs distance from array.

Two-panel paper-quality figure:
  (a) Scatter: location offset (km, ours vs Orca OBS) vs distance from array
      centroid, colored by n_moorings, with LOWESS trend and array-extent line.
  (b) Map: arrows from Orca to our location, colored by offset magnitude,
      on IBCSO bathymetry with mooring positions.

Usage:
    uv run python scripts/make_location_accuracy_figure.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource, Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from pathlib import Path
from datetime import datetime, timedelta
from pyproj import Geod
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_dat import MOORINGS

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PHASE3_LOCS = DATA_DIR / "event_locations_phase3.parquet"
ORCA_PATH = Path("/home/jovyan/my_data/bravoseis/earthquakes/Orca_EQ_data.csv")

BATHY_IBCSO = Path("/home/jovyan/my_data/bravoseis/bathymetry/IBCSO_v2_bed_WGS84.nc")
BATHY_REGIONAL = Path("/home/jovyan/my_data/bravoseis/bathymetry/bransfield.xyz")
BATHY_ORCA = Path("/home/jovyan/my_data/bravoseis/bathymetry/MGDS_Download/BRAVOSEIS/Orca_bathymetry.nc")

# Array centroid
CENTROID_LAT = -62.560
CENTROID_LON = -58.591
ARRAY_HALF_APERTURE_KM = 80.0  # approximate

# Map extent
MAP_LON_MIN, MAP_LON_MAX = -62.5, -54.5
MAP_LAT_MIN, MAP_LAT_MAX = -64.5, -60.7

# Temporal matching tolerance (seconds)
MATCH_TOL_S = 30.0

# Geodetic calculator
geod = Geod(ellps="WGS84")


def matlab_datenum_to_datetime(d):
    """Convert MATLAB datenum to Python datetime."""
    return datetime(1, 1, 1) + timedelta(days=d - 367)


def haversine_km(lat1, lon1, lat2, lon2):
    """Geodesic distance in km between two points."""
    _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0


def load_orca():
    """Load Orca OBS catalogue with datetime conversion."""
    df = pd.read_csv(ORCA_PATH)
    # Strip trailing comma column if present
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["datetime"] = df["date"].apply(matlab_datenum_to_datetime)
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"Orca catalogue: {len(df):,} events")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def load_our_locations():
    """Load Phase 3 locations, filter to seismic/both with quality A/B/C."""
    df = pd.read_parquet(PHASE3_LOCS)
    mask = (
        df["phase3_class"].isin(["seismic", "both"])
        & df["quality_tier"].isin(["A", "B", "C"])
    )
    df = df[mask].copy()
    df["datetime"] = pd.to_datetime(df["earliest_utc"])
    print(f"Our seismic locations (A+B+C): {len(df):,}")
    return df


def match_events(ours, orca):
    """Match Orca events to our nearest event within MATCH_TOL_S."""
    # Convert to epoch seconds for fast matching
    our_epoch = ours["datetime"].values.astype("datetime64[s]").astype("int64")
    orca_epoch = orca["datetime"].values.astype("datetime64[s]").astype("int64")

    # Sort ours for binary search
    sort_idx = np.argsort(our_epoch)
    our_sorted = our_epoch[sort_idx]
    our_df_idx = ours.index.values[sort_idx]

    matches = []
    for i, t_orca in enumerate(orca_epoch):
        # Binary search for closest
        pos = np.searchsorted(our_sorted, t_orca)
        best_dt = np.inf
        best_j = -1
        for candidate in [pos - 1, pos]:
            if 0 <= candidate < len(our_sorted):
                dt = abs(our_sorted[candidate] - t_orca)
                if dt < best_dt:
                    best_dt = dt
                    best_j = candidate
        if best_dt <= MATCH_TOL_S:
            our_row = ours.loc[our_df_idx[best_j]]
            orca_row = orca.iloc[i]
            offset_km = haversine_km(
                our_row["lat"], our_row["lon"],
                orca_row["lat"], orca_row["lon"]
            )
            dist_from_centroid = haversine_km(
                our_row["lat"], our_row["lon"],
                CENTROID_LAT, CENTROID_LON
            )
            matches.append({
                "our_lat": our_row["lat"],
                "our_lon": our_row["lon"],
                "orca_lat": orca_row["lat"],
                "orca_lon": orca_row["lon"],
                "offset_km": offset_km,
                "dist_from_centroid_km": dist_from_centroid,
                "n_moorings": our_row["n_moorings"],
                "quality_tier": our_row["quality_tier"],
                "dt_s": best_dt,
            })

    matches_df = pd.DataFrame(matches)
    print(f"Matched events: {len(matches_df):,} (within {MATCH_TOL_S}s)")
    if len(matches_df) > 0:
        print(f"  Offset median: {matches_df['offset_km'].median():.1f} km")
        print(f"  Offset 90th pct: {matches_df['offset_km'].quantile(0.9):.1f} km")
    return matches_df


def load_bathy():
    """Load merged bathymetry: IBCSO v2 base + multibeam + Orca overlay."""
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    print("Loading IBCSO v2 base bathymetry...")
    ds_ibcso = xr.open_dataset(BATHY_IBCSO)
    sub = ds_ibcso.sel(lat=slice(MAP_LAT_MIN, MAP_LAT_MAX),
                       lon=slice(MAP_LON_MIN, MAP_LON_MAX))
    ibcso_lat = sub["lat"].values
    ibcso_lon = sub["lon"].values
    ibcso_z = sub["z"].values.astype(np.float64)
    ds_ibcso.close()

    grid_spacing = 0.004
    lon_1d = np.arange(MAP_LON_MIN, MAP_LON_MAX, grid_spacing)
    lat_1d = np.arange(MAP_LAT_MIN, MAP_LAT_MAX, grid_spacing)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    ibcso_interp = RegularGridInterpolator(
        (ibcso_lat, ibcso_lon), ibcso_z,
        method="linear", bounds_error=False, fill_value=np.nan
    )
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    z_grid = ibcso_interp(pts).reshape(lat_grid.shape)

    if BATHY_REGIONAL.exists():
        print("Overlaying BRAVOSEIS regional multibeam...")
        data = np.loadtxt(BATHY_REGIONAL)
        lon_raw, lat_raw, z_raw = data[:, 0], data[:, 1], data[:, 2]
        z_mb = griddata((lon_raw, lat_raw), z_raw, (lon_grid, lat_grid), method="linear")
        valid = ~np.isnan(z_mb)
        z_grid[valid] = z_mb[valid]
        del data, lon_raw, lat_raw, z_raw, z_mb

    if BATHY_ORCA.exists():
        print("Overlaying Orca high-res bathymetry...")
        ds = xr.open_dataset(BATHY_ORCA)
        orca_interp = RegularGridInterpolator(
            (ds["latitude"].values, ds["longitude"].values), ds["data"].values,
            method="linear", bounds_error=False, fill_value=np.nan
        )
        lat_mask = (lat_1d >= float(ds["latitude"].min())) & (lat_1d <= float(ds["latitude"].max()))
        lon_mask = (lon_1d >= float(ds["longitude"].min())) & (lon_1d <= float(ds["longitude"].max()))
        lon_mg, lat_mg = np.meshgrid(lon_1d[lon_mask], lat_1d[lat_mask])
        z_orca = orca_interp(np.column_stack([lat_mg.ravel(), lon_mg.ravel()])).reshape(lat_mg.shape)
        valid = ~np.isnan(z_orca)
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        for jj, j in enumerate(lat_idx):
            for ii, i in enumerate(lon_idx):
                if valid[jj, ii]:
                    z_grid[j, i] = z_orca[jj, ii]
        ds.close()

    return lon_grid, lat_grid, z_grid


def lowess_smooth(x, y, frac=0.3):
    """Simple LOWESS smoother using statsmodels if available, else moving average."""
    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess
        result = lowess(y, x, frac=frac, return_sorted=True)
        return result[:, 0], result[:, 1]
    except ImportError:
        # Fallback: running median in distance bins
        sort_idx = np.argsort(x)
        xs, ys = x[sort_idx], y[sort_idx]
        window = max(5, len(xs) // 20)
        smooth_y = pd.Series(ys).rolling(window, center=True, min_periods=3).median().values
        mask = ~np.isnan(smooth_y)
        return xs[mask], smooth_y[mask]


def main():
    print("=" * 60)
    print("Location Accuracy vs Distance from Array")
    print("=" * 60)

    # Load data
    ours = load_our_locations()
    orca = load_orca()
    matches = match_events(ours, orca)

    if len(matches) == 0:
        print("ERROR: No matched events found. Cannot produce figure.")
        return

    # Print statistics by n_moorings
    print("\n=== Offset statistics by n_moorings ===")
    for nm in sorted(matches["n_moorings"].unique()):
        sub = matches[matches["n_moorings"] == nm]
        print(f"  n_moorings={nm}: n={len(sub):,}, "
              f"median={sub['offset_km'].median():.1f} km, "
              f"90th={sub['offset_km'].quantile(0.9):.1f} km")

    # Load bathy for panel b
    lon_grid, lat_grid, z_grid = load_bathy()
    ls = LightSource(azdeg=315, altdeg=25)
    z_shade = z_grid.copy()
    z_shade[np.isnan(z_shade)] = 0
    hillshade = ls.hillshade(z_shade, vert_exag=0.005)

    # =====================================================================
    # Figure
    # =====================================================================
    map_proj = ccrs.Mercator(central_longitude=-58.5)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(7, 9))

    # --- Panel (a): Scatter ---
    ax_scatter = fig.add_axes([0.12, 0.56, 0.82, 0.36])

    # Color map for n_moorings
    nm_colors = {3: "#1f77b4", 4: "#ff7f0e", 5: "#2ca02c", 6: "#d62728"}
    nm_labels = {3: "3 moorings", 4: "4 moorings", 5: "5 moorings", 6: "6 moorings"}

    for nm in sorted(matches["n_moorings"].unique()):
        sub = matches[matches["n_moorings"] == nm]
        color = nm_colors.get(int(nm), "gray")
        ax_scatter.scatter(
            sub["dist_from_centroid_km"], sub["offset_km"],
            s=12, alpha=0.5, c=color, edgecolors="none",
            label=nm_labels.get(int(nm), f"{int(nm)} moorings"),
            rasterized=True, zorder=3,
        )

    # LOWESS trend
    x_all = matches["dist_from_centroid_km"].values
    y_all = matches["offset_km"].values
    lx, ly = lowess_smooth(x_all, y_all, frac=0.5)
    ax_scatter.plot(lx, ly, "k-", linewidth=2, label="LOWESS trend", zorder=5)

    # Array half-aperture line
    ax_scatter.axvline(
        ARRAY_HALF_APERTURE_KM, color="gray", linestyle="--", linewidth=1.2, zorder=4
    )

    # Region labels
    ylim_max = min(matches["offset_km"].quantile(0.99) * 1.3, matches["offset_km"].max())
    ax_scatter.set_ylim(0, ylim_max)
    xlim_max = matches["dist_from_centroid_km"].max() * 1.05
    ax_scatter.set_xlim(0, xlim_max)

    label_y = ylim_max * 0.92
    ax_scatter.text(
        ARRAY_HALF_APERTURE_KM * 0.45, label_y, "Inside array",
        ha="center", va="top", fontsize=9, fontstyle="italic", color="gray",
    )
    ax_scatter.text(
        ARRAY_HALF_APERTURE_KM + (xlim_max - ARRAY_HALF_APERTURE_KM) * 0.5,
        label_y, "Outside array",
        ha="center", va="top", fontsize=9, fontstyle="italic", color="gray",
    )

    ax_scatter.set_xlabel("Distance from array centroid (km)", fontsize=10)
    ax_scatter.set_ylabel("Location offset vs Orca OBS (km)", fontsize=10)
    ax_scatter.set_title("(a) Location accuracy degrades with distance", fontsize=14, fontweight="bold")
    ax_scatter.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=2)
    ax_scatter.tick_params(labelsize=9)
    ax_scatter.grid(True, alpha=0.3)

    # --- Panel (b): Map ---
    ax_map = fig.add_axes([0.08, 0.03, 0.84, 0.46], projection=map_proj)

    # Bathy
    ax_map.pcolormesh(
        lon_grid, lat_grid, z_grid,
        cmap=cmocean.cm.deep, vmin=-4000, vmax=0,
        transform=data_crs, shading="auto", rasterized=True, alpha=0.7, zorder=0,
    )
    ax_map.pcolormesh(
        lon_grid, lat_grid, hillshade,
        cmap="gray", vmin=0, vmax=1,
        transform=data_crs, shading="auto", rasterized=True, alpha=0.3, zorder=1,
    )
    ax_map.add_feature(cfeature.LAND, facecolor="#d4c5a9", zorder=2)
    ax_map.coastlines(resolution="10m", linewidth=0.5, zorder=3)

    # Gridlines
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5,
                          color="gray", linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    ax_map.set_extent([-62, -55, -64, -61.2], crs=data_crs)

    # Moorings
    for key, info in sorted(MOORINGS.items()):
        ax_map.plot(
            info["lon"], info["lat"], "w^", markersize=8,
            markeredgecolor="black", markeredgewidth=0.8,
            transform=data_crs, zorder=10,
        )
        ax_map.text(
            info["lon"] + 0.08, info["lat"] + 0.02, key.upper(),
            fontsize=7, fontweight="bold", transform=data_crs, zorder=10,
            color="white", path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        )

    # Array centroid
    ax_map.plot(
        CENTROID_LON, CENTROID_LAT, "k+", markersize=10, markeredgewidth=1.5,
        transform=data_crs, zorder=10,
    )

    # Arrows from Orca to our location, colored by offset
    offset_vals = matches["offset_km"].values
    norm = Normalize(vmin=0, vmax=min(np.percentile(offset_vals, 95), 50))
    cmap = plt.cm.plasma

    for _, row in matches.iterrows():
        color = cmap(norm(row["offset_km"]))
        ax_map.annotate(
            "",
            xy=(row["our_lon"], row["our_lat"]),
            xytext=(row["orca_lon"], row["orca_lat"]),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=0.6, mutation_scale=5,
            ),
            transform=data_crs, zorder=6,
        )

    # Colorbar for arrows
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_map, orientation="vertical",
                        fraction=0.03, pad=0.02, shrink=0.8)
    cbar.set_label("Location offset (km)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax_map.set_title(
        f"(b) Offset vectors: Orca OBS to this study (n = {len(matches):,})",
        fontsize=14, fontweight="bold",
    )

    # Save
    outpath = FIG_DIR / "location_accuracy_vs_distance.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
