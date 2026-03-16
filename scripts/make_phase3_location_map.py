#!/usr/bin/env python3
"""
make_phase3_location_map.py — Map of Phase 3 event locations by class.

Shows seismic (1–14 Hz) and cryogenic (>30 Hz) events on a bathymetric
map of the Bransfield Strait. Tier A+B+C locations only.

Usage:
    uv run python scripts/make_phase3_location_map.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.interpolate import griddata
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.patheffects as pe

from read_dat import MOORINGS

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BATHY_IBCSO = Path("/home/jovyan/my_data/bravoseis/bathymetry/IBCSO_v2_bed_WGS84.nc")
BATHY_REGIONAL = Path("/home/jovyan/my_data/bravoseis/bathymetry/bransfield.xyz")
BATHY_ORCA = Path("/home/jovyan/my_data/bravoseis/bathymetry/MGDS_Download/BRAVOSEIS/Orca_bathymetry.nc")

# Map extent (slightly wider than set_extent for data loading margin)
MAP_LON_MIN, MAP_LON_MAX = -62.5, -54.5
MAP_LAT_MIN, MAP_LAT_MAX = -64.5, -60.7


def load_bathy():
    """Load merged bathymetry: IBCSO v2 base + BRAVOSEIS multibeam + Orca high-res overlay."""
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    # --- Layer 1: IBCSO v2 (500 m, full coverage) ---
    print("Loading IBCSO v2 base bathymetry...")
    ds_ibcso = xr.open_dataset(BATHY_IBCSO)
    sub = ds_ibcso.sel(lat=slice(MAP_LAT_MIN, MAP_LAT_MAX),
                       lon=slice(MAP_LON_MIN, MAP_LON_MAX))
    ibcso_lat = sub["lat"].values
    ibcso_lon = sub["lon"].values
    ibcso_z = sub["z"].values.astype(np.float64)
    ds_ibcso.close()
    print(f"  IBCSO grid: {ibcso_z.shape[0]} x {ibcso_z.shape[1]}")

    # Build output grid at ~0.004° (~400 m) spacing to match multibeam resolution
    grid_spacing = 0.004
    lon_1d = np.arange(MAP_LON_MIN, MAP_LON_MAX, grid_spacing)
    lat_1d = np.arange(MAP_LAT_MIN, MAP_LAT_MAX, grid_spacing)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    # Interpolate IBCSO onto the output grid
    ibcso_interp = RegularGridInterpolator(
        (ibcso_lat, ibcso_lon), ibcso_z,
        method="linear", bounds_error=False, fill_value=np.nan
    )
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    z_grid = ibcso_interp(pts).reshape(lat_grid.shape)
    print(f"  Output grid: {z_grid.shape[0]} x {z_grid.shape[1]}")

    # --- Layer 2: BRAVOSEIS regional multibeam (higher res where available) ---
    if BATHY_REGIONAL.exists():
        print("Overlaying BRAVOSEIS regional multibeam...")
        data = np.loadtxt(BATHY_REGIONAL)
        lon_raw, lat_raw, z_raw = data[:, 0], data[:, 1], data[:, 2]
        z_mb = griddata((lon_raw, lat_raw), z_raw, (lon_grid, lat_grid), method="linear")
        valid = ~np.isnan(z_mb)
        z_grid[valid] = z_mb[valid]
        print(f"  Replaced {valid.sum():,} cells with multibeam data")
        del data, lon_raw, lat_raw, z_raw, z_mb

    # --- Layer 3: Orca high-res (highest res in central basin) ---
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
        n_replaced = valid.sum()
        ds.close()
        print(f"  Replaced {n_replaced:,} cells with Orca data")

    return lon_grid, lat_grid, z_grid


def main():
    print("=" * 60)
    print("Phase 3 Location Map")
    print("=" * 60)

    # Load locations
    locs = pd.read_parquet(DATA_DIR / "event_locations_phase3.parquet")

    # Filter to publishable tiers (A+B+C)
    good = locs[locs["quality_tier"].isin(["A", "B", "C"])].copy()
    print(f"Publishable locations (A+B+C): {len(good):,}")
    print(f"  seismic: {(good['phase3_class']=='seismic').sum():,}")
    print(f"  cryogenic: {(good['phase3_class']=='cryogenic').sum():,}")
    print(f"  both: {(good['phase3_class']=='both').sum():,}")
    print(f"  unclassified: {(good['phase3_class']=='unclassified').sum():,}")

    # Load bathy
    lon_grid, lat_grid, z_grid = load_bathy()

    # Shaded relief
    ls = LightSource(azdeg=315, altdeg=25)
    z_shade = z_grid.copy()
    z_shade[np.isnan(z_shade)] = 0
    hillshade = ls.hillshade(z_shade, vert_exag=0.005)

    # --- Figure: 2-panel map ---
    map_proj = ccrs.Mercator(central_longitude=-58.5)
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                              subplot_kw={"projection": map_proj})

    for ax in axes:
        # Bathy
        ax.pcolormesh(lon_grid, lat_grid, z_grid,
                      cmap=cmocean.cm.deep, vmin=-4000, vmax=0,
                      transform=data_crs, shading="auto", rasterized=True,
                      alpha=0.7, zorder=0)
        ax.pcolormesh(lon_grid, lat_grid, hillshade,
                      cmap="gray", vmin=0, vmax=1,
                      transform=data_crs, shading="auto", rasterized=True,
                      alpha=0.3, zorder=1)
        ax.add_feature(cfeature.LAND, facecolor="#d4c5a9", zorder=2)
        ax.coastlines(resolution="10m", linewidth=0.5, zorder=3)

        # Moorings
        for key, info in sorted(MOORINGS.items()):
            ax.plot(info["lon"], info["lat"], "w^", markersize=8,
                    markeredgecolor="black", markeredgewidth=0.8,
                    transform=data_crs, zorder=10)
            ax.text(info["lon"] + 0.08, info["lat"] + 0.02,
                    key.upper(), fontsize=7, fontweight="bold",
                    transform=data_crs, zorder=10,
                    color="white", path_effects=[
                        pe.withStroke(linewidth=2, foreground="black")
                    ])

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5,
                          color="gray", linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {"size": 9}
        gl.ylabel_style = {"size": 9}

        ax.set_extent([-62, -55, -64, -61.2], crs=data_crs)

    # --- Panel (a): Seismic events ---
    ax = axes[0]
    seis = good[good["phase3_class"].isin(["seismic", "both"])]
    ax.scatter(seis["lon"], seis["lat"], s=3, c="#1f77b4", alpha=0.3,
               transform=data_crs, zorder=5, rasterized=True, label="Seismic")
    ax.set_title(f"(a) Seismic events (1–14 Hz)\nn = {len(seis):,}",
                 fontsize=14, fontweight="bold")

    # --- Panel (b): Cryogenic events ---
    ax = axes[1]
    cryo = good[good["phase3_class"].isin(["cryogenic", "both"])]
    ax.scatter(cryo["lon"], cryo["lat"], s=3, c="#d62728", alpha=0.3,
               transform=data_crs, zorder=5, rasterized=True, label="Cryogenic")
    ax.set_title(f"(b) Cryogenic events (>30 Hz)\nn = {len(cryo):,}",
                 fontsize=14, fontweight="bold")

    fig.suptitle("Phase 3 Event Locations — Bransfield Strait (Tier A+B+C)",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = FIG_DIR / "phase3_location_map.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {outpath}")

    # --- Quick spatial stats ---
    print(f"\n=== Spatial statistics (A+B+C) ===")
    for cls in ["seismic", "cryogenic", "both", "unclassified"]:
        sub = good[good["phase3_class"] == cls]
        if len(sub) == 0:
            continue
        print(f"\n{cls} ({len(sub):,}):")
        print(f"  Lat: {sub['lat'].median():.3f} [{sub['lat'].quantile(.25):.3f}, {sub['lat'].quantile(.75):.3f}]")
        print(f"  Lon: {sub['lon'].median():.3f} [{sub['lon'].quantile(.25):.3f}, {sub['lon'].quantile(.75):.3f}]")
        if "dist_to_coast_km" in sub.columns:
            print(f"  Dist to coast: {sub['dist_to_coast_km'].median():.1f} km")


if __name__ == "__main__":
    main()
