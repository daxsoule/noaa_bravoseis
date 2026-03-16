#!/usr/bin/env python3
"""
make_bathy_map.py - Bransfield Strait bathymetric map

Single-panel shaded-relief map with contours, neatline, scale bar,
north arrow, gridlines, and colorbar.  Style matches the Axial Volcano
composite map in miso_my-analysis/make_composite_map.py.

Usage:
    uv run python make_bathy_map.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LightSource
from scipy.interpolate import griddata
from datetime import date
from pathlib import Path
import textwrap
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
from read_dat import MOORINGS


# === Paths ===
BATHY_IBCSO_PATH = Path(
    "/home/jovyan/my_data/bravoseis/bathymetry/IBCSO_v2_bed_WGS84.nc"
)
BATHY_REGIONAL_PATH = Path(
    "/home/jovyan/my_data/bravoseis/bathymetry/bransfield.xyz"
)
BATHY_ORCA_PATH = Path(
    "/home/jovyan/my_data/bravoseis/bathymetry/MGDS_Download/BRAVOSEIS/Orca_bathymetry.nc"
)
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "figures" / "exploratory" / "maps"

# === Font sizes (Paper tier: title >= 14pt, axis/caption >= 10pt,
#     feature labels >= 8pt, min DPI 300) ===
FS_TITLE = 14
FS_FEATURE_LABEL = 9
FS_SCALE_BAR = 9
FS_GRIDLINE = 9
FS_COLORBAR = 12
FS_COLORBAR_TICK = 9
FS_DATE_STAMP = 8
FS_CAPTION = 10


# ─── Shared helpers ──────────────────────────────────────────────────────────

def add_caption_justified(fig, caption_text, caption_width=0.85, fontsize=10,
                          caption_left=0.05, bold_prefix=None):
    """Add a renderer-based fully justified caption below the figure.

    Measures each word's pixel width via get_window_extent(renderer),
    distributes remaining horizontal space as equal gaps between words.
    Last line is left-aligned.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    caption_text : str
    caption_width : float
        Width of caption as fraction of figure width.
    fontsize : int
    caption_left : float
        Left edge of caption as fraction of figure width.
    bold_prefix : str or None
        If provided, the first occurrence of this string in caption_text
        will be rendered in bold.
    """
    caption_width_in = caption_width * fig.get_size_inches()[0]
    char_width_in = fontsize / 72 * 0.55
    wrap_chars = int(caption_width_in / char_width_in)
    lines = textwrap.wrap(caption_text, width=wrap_chars)

    caption_ax = fig.add_axes([caption_left, 0.02, caption_width, 0.15])
    caption_ax.axis('off')

    renderer = fig.canvas.get_renderer()
    ax_bbox = caption_ax.get_window_extent(renderer)

    # Measure line height
    sample = caption_ax.text(0, 0, "Tg", fontsize=fontsize, family='sans-serif',
                             transform=caption_ax.transAxes)
    sample_bbox = sample.get_window_extent(renderer)
    line_height = (sample_bbox.height * 1.35) / ax_bbox.height
    sample.remove()

    for i, line in enumerate(lines):
        y = 1.0 - i * line_height
        words = line.split()

        if i < len(lines) - 1 and len(words) > 1:
            # Measure each word's rendered width
            word_widths = []
            word_bolds = []
            for word in words:
                # Check if this word is part of the bold prefix
                is_bold = False
                if bold_prefix:
                    for bp_word in bold_prefix.split():
                        if word == bp_word or word.rstrip(':') == bp_word.rstrip(':'):
                            is_bold = True
                            break
                weight = 'bold' if is_bold else 'normal'
                t = caption_ax.text(0, 0, word, fontsize=fontsize, family='sans-serif',
                                    fontweight=weight, transform=caption_ax.transAxes)
                wb = t.get_window_extent(renderer)
                word_widths.append(wb.width / ax_bbox.width)
                word_bolds.append(is_bold)
                t.remove()

            total_word_width = sum(word_widths)
            remaining = 1.0 - total_word_width
            gap = remaining / (len(words) - 1)

            x = 0.0
            for j, word in enumerate(words):
                weight = 'bold' if word_bolds[j] else 'normal'
                caption_ax.text(x, y, word, fontsize=fontsize, family='sans-serif',
                                fontweight=weight,
                                transform=caption_ax.transAxes, va='top', ha='left')
                x += word_widths[j] + gap
        else:
            # Last line: left-aligned
            # Handle bold prefix on last line too
            if bold_prefix and i == 0:
                # Unlikely but handle it
                caption_ax.text(0.0, y, line, fontsize=fontsize, family='sans-serif',
                                transform=caption_ax.transAxes, va='top', ha='left')
            else:
                caption_ax.text(0.0, y, line, fontsize=fontsize, family='sans-serif',
                                transform=caption_ax.transAxes, va='top', ha='left')


def draw_neatline(ax, n_segments=12, linewidth=5):
    """Draw an alternating black/white ladder border (neatline) around the axes."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]

    seg_w = width / n_segments
    seg_h = height / n_segments

    for edge in ['bottom', 'top', 'left', 'right']:
        if edge == 'bottom':
            xs, ys = [xlim[0], xlim[1]], [ylim[0], ylim[0]]
        elif edge == 'top':
            xs, ys = [xlim[0], xlim[1]], [ylim[1], ylim[1]]
        elif edge == 'left':
            xs, ys = [xlim[0], xlim[0]], [ylim[0], ylim[1]]
        else:
            xs, ys = [xlim[1], xlim[1]], [ylim[0], ylim[1]]
        ax.plot(xs, ys, color='black', linewidth=linewidth + 2,
                transform=ax.transData, clip_on=False, zorder=19,
                solid_capstyle='butt')

    for i in range(n_segments):
        color = 'black' if i % 2 == 0 else 'white'
        ax.plot([xlim[0] + i * seg_w, xlim[0] + (i + 1) * seg_w],
                [ylim[0], ylim[0]], color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20,
                solid_capstyle='butt')
        ax.plot([xlim[0] + i * seg_w, xlim[0] + (i + 1) * seg_w],
                [ylim[1], ylim[1]], color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20,
                solid_capstyle='butt')
        ax.plot([xlim[0], xlim[0]],
                [ylim[0] + i * seg_h, ylim[0] + (i + 1) * seg_h],
                color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20,
                solid_capstyle='butt')
        ax.plot([xlim[1], xlim[1]],
                [ylim[0] + i * seg_h, ylim[0] + (i + 1) * seg_h],
                color=color, linewidth=linewidth,
                transform=ax.transData, clip_on=False, zorder=20,
                solid_capstyle='butt')


# ─── Main map ────────────────────────────────────────────────────────────────

def make_bathy_map():
    """Create the Bransfield Strait bathymetry map."""
    print("=" * 60)
    print("Bransfield Strait Bathymetric Map")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Layer 1: IBCSO v2 base (500 m, full coverage) ---
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    print("Loading IBCSO v2 base bathymetry...")
    ds_ibcso = xr.open_dataset(BATHY_IBCSO_PATH)
    sub = ds_ibcso.sel(
        lat=slice(MAP_LAT_MIN - 0.5, MAP_LAT_MAX + 0.5),
        lon=slice(MAP_LON_MIN - 0.5, MAP_LON_MAX + 0.5),
    )
    ibcso_lat = sub["lat"].values
    ibcso_lon = sub["lon"].values
    ibcso_z = sub["z"].values.astype(np.float64)
    ds_ibcso.close()
    print(f"  IBCSO grid: {ibcso_z.shape[0]} x {ibcso_z.shape[1]}")

    # Build output grid at ~0.004° spacing
    grid_spacing = 0.004
    lon_grid_1d = np.arange(MAP_LON_MIN - 0.5, MAP_LON_MAX + 0.5, grid_spacing)
    lat_grid_1d = np.arange(MAP_LAT_MIN - 0.5, MAP_LAT_MAX + 0.5, grid_spacing)
    lon_grid, lat_grid = np.meshgrid(lon_grid_1d, lat_grid_1d)
    print(f"  Output grid: {len(lon_grid_1d)} x {len(lat_grid_1d)}")

    ibcso_interp = RegularGridInterpolator(
        (ibcso_lat, ibcso_lon), ibcso_z,
        method="linear", bounds_error=False, fill_value=np.nan
    )
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    z_grid = ibcso_interp(pts).reshape(lat_grid.shape)
    print(f"  Depth range: {np.nanmin(z_grid):.0f} to {np.nanmax(z_grid):.0f} m")

    # --- Layer 2: BRAVOSEIS regional multibeam (higher res where available) ---
    print("Overlaying BRAVOSEIS regional multibeam...")
    data = np.loadtxt(BATHY_REGIONAL_PATH)
    lon_raw, lat_raw, z_raw = data[:, 0], data[:, 1], data[:, 2]
    print(f"  Points: {len(z_raw):,}")
    z_mb = griddata(
        (lon_raw, lat_raw), z_raw,
        (lon_grid, lat_grid),
        method='linear'
    )
    valid = ~np.isnan(z_mb)
    z_grid[valid] = z_mb[valid]
    print(f"  Replaced {valid.sum():,} cells with multibeam data")
    del data, lon_raw, lat_raw, z_raw, z_mb

    # --- Layer 3: Orca high-res (highest res in central basin) ---
    if BATHY_ORCA_PATH.exists():
        print("Overlaying MGDS Orca bathymetry (netCDF)...")
        ds_orca = xr.open_dataset(BATHY_ORCA_PATH)
        orca_lat = ds_orca['latitude'].values
        orca_lon = ds_orca['longitude'].values
        orca_z = ds_orca['data'].values
        print(f"  Grid: {orca_z.shape[0]} x {orca_z.shape[1]}")

        interp = RegularGridInterpolator(
            (orca_lat, orca_lon), orca_z,
            method='linear', bounds_error=False, fill_value=np.nan
        )

        lat_mask = (lat_grid_1d >= orca_lat.min()) & (lat_grid_1d <= orca_lat.max())
        lon_mask = (lon_grid_1d >= orca_lon.min()) & (lon_grid_1d <= orca_lon.max())
        lon_mg, lat_mg = np.meshgrid(lon_grid_1d[lon_mask], lat_grid_1d[lat_mask])
        z_interp = interp(np.column_stack([lat_mg.ravel(), lon_mg.ravel()])).reshape(lat_mg.shape)

        valid = ~np.isnan(z_interp)
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        for jj, j in enumerate(lat_idx):
            for ii, i in enumerate(lon_idx):
                if valid[jj, ii]:
                    z_grid[j, i] = z_interp[jj, ii]

        print(f"  Replaced {valid.sum():,} cells with MGDS Orca data")
        ds_orca.close()

    # --- Projection setup ---
    # Use Mercator centered on the study area (avoids UTM zone edge distortion)
    map_proj = ccrs.Mercator(central_longitude=-58.5)
    data_crs = ccrs.PlateCarree()

    # --- Map extent ---
    MAP_LON_MIN, MAP_LON_MAX = -61.5, -55.5
    MAP_LAT_MIN, MAP_LAT_MAX = -63.1, -61.9

    # --- Shaded relief ---
    print("Computing shaded relief...")
    ls = LightSource(azdeg=315, altdeg=45)
    # Only color the ocean (negative depth); mask land
    z_ocean = np.where(z_grid <= 0, z_grid, np.nan)
    z_min, z_max = np.nanpercentile(z_ocean, [2, 98])
    bathy_cmap = cmocean.cm.ice
    # Set land to light gray
    bathy_cmap_copy = bathy_cmap.copy()
    bathy_cmap_copy.set_bad(color='#d9d9d9')
    rgb = ls.shade(z_ocean, cmap=bathy_cmap_copy, blend_mode='soft',
                   vmin=z_min, vmax=z_max)

    # --- Create figure (sized for double-column journal width, 7in) ---
    fig = plt.figure(figsize=(7, 5.2))
    ax = fig.add_axes([0.08, 0.28, 0.76, 0.62], projection=map_proj)

    # Set geographic extent first
    ax.set_extent([MAP_LON_MIN, MAP_LON_MAX, MAP_LAT_MIN, MAP_LAT_MAX],
                  crs=data_crs)

    # Plot shaded relief in geographic coordinates
    ax.imshow(rgb,
              extent=[lon_grid_1d.min(), lon_grid_1d.max(),
                      lat_grid_1d.min(), lat_grid_1d.max()],
              origin='lower', transform=data_crs)

    # --- Coastline / land outlines ---
    import cartopy.feature as cfeature
    ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black',
                   linewidth=0.4, zorder=10)
    ax.coastlines(resolution='10m', linewidth=0.4, color='black', zorder=10)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=bathy_cmap,
                                norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cax = fig.add_axes([0.86, 0.28, 0.015, 0.62])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Depth (m)', fontsize=FS_COLORBAR)
    cbar.ax.tick_params(labelsize=FS_COLORBAR_TICK)

    # --- Gridlines ---
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=0.5, color='white', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': FS_GRIDLINE, 'rotation': 0}
    gl.ylabel_style = {'size': FS_GRIDLINE}
    gl.xpadding = 12
    gl.ypadding = 12

    # --- Scale bar (50 km, neatline-style alternating black/white) ---
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    sb_lat = MAP_LAT_MAX - 0.12
    sb_lon_start = MAP_LON_MIN + 0.4
    # Compute 50 km endpoint
    sb_lon_end, _, _ = geod.fwd(sb_lon_start, sb_lat, 90, 50000)
    # 5 segments of 10 km each
    n_bar_segs = 5
    bar_lw = 4
    sb_lons = np.linspace(sb_lon_start, sb_lon_end, n_bar_segs + 1)

    # Black background outline
    ax.plot([sb_lon_start, sb_lon_end], [sb_lat, sb_lat],
            color='black', linewidth=bar_lw + 2, solid_capstyle='butt',
            transform=data_crs, zorder=14)
    # Alternating segments
    for i in range(n_bar_segs):
        c = 'black' if i % 2 == 0 else 'white'
        ax.plot([sb_lons[i], sb_lons[i + 1]], [sb_lat, sb_lat],
                color=c, linewidth=bar_lw, solid_capstyle='butt',
                transform=data_crs, zorder=15)
    # Labels: 0, 25, 50 km
    label_y = sb_lat + 0.05
    for lon_val, label in [(sb_lon_start, '0'), (sb_lons[n_bar_segs // 2], '25'),
                            (sb_lon_end, '50 km')]:
        ax.text(lon_val, label_y, label,
                ha='center', fontsize=FS_SCALE_BAR, fontweight='bold',
                transform=data_crs, zorder=15,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                          alpha=0.7, edgecolor='none'))

    # --- Neatline (alternating black/white border) ---
    draw_neatline(ax, n_segments=12, linewidth=3)

    # --- North arrow (axes coordinates) ---
    ax.annotate('N', xy=(0.95, 0.97),
                xytext=(0.95, 0.88),
                fontsize=FS_FEATURE_LABEL, fontweight='bold', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                xycoords='axes fraction', textcoords='axes fraction',
                zorder=15,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # --- Title ---
    ax.set_title('Bransfield Strait Bathymetry', fontsize=FS_TITLE,
                 fontweight='bold', pad=10)

    # --- Date stamp ---
    stamp = f"Map updated: {date.today().strftime('%Y-%m-%d')}, WGS84, Mercator"
    ax.text(0.98, 0.02, stamp,
            ha='right', va='bottom', fontsize=FS_DATE_STAMP,
            transform=ax.transAxes, zorder=15,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # --- Mooring locations ---
    for key, info in sorted(MOORINGS.items()):
        ax.plot(info['lon'], info['lat'], marker='^', color='red',
                markersize=7, markeredgecolor='black', markeredgewidth=0.8,
                transform=data_crs, zorder=16)
        ax.text(info['lon'], info['lat'] + 0.04, info['name'],
                fontsize=5, fontweight='bold', ha='center',
                va='bottom', transform=data_crs, zorder=16,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.8, edgecolor='none'))

    # --- Temporary Caption (renderer-based full justification) ---
    caption = (
        "Temporary Caption: Shaded-relief bathymetric map of the Bransfield "
        "Strait, Central Bransfield Basin. Depth shown as colorscale (m). Land "
        "areas (positive elevation) masked in gray. Black outlines show "
        "10m-resolution Natural Earth coastlines. Red triangles show NOAA/PMEL "
        "autonomous hydrophone mooring locations (BRA28\u2013BRA33), deployed "
        "January 2019 \u2013 February 2020. Shaded relief computed with illumination "
        "from 315\u00b0 azimuth, 45\u00b0 altitude. Base bathymetry from IBCSO v2 "
        "(Dorschel et al., 2022; 500 m resolution), overlaid with higher-resolution "
        "multibeam data collected during the BRAVOSEIS experiment, 2019\u20132020. "
        "Highest-resolution bathymetry in the central basin from MGDS gridded model "
        "(DOI: 10.60521/332247). Projection: Mercator (WGS84)."
    )
    add_caption_justified(fig, caption, caption_width=0.85, fontsize=FS_CAPTION,
                          caption_left=0.04, bold_prefix="Temporary Caption:")

    # --- Save ---
    output_file = OUTPUT_DIR / "bransfield_bathy.png"
    print(f"\nSaving to {output_file}...")
    fig.savefig(output_file, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    return output_file


if __name__ == "__main__":
    make_bathy_map()
    print("\nDone!")
