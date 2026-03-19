#!/usr/bin/env python3
"""
crossvalidate_tapaas_locations.py — Compare TAPAAs-located events against
independent earthquake catalogues: Singer (manual), Orca OBS, and USGS/NEIC.

Produces:
  - Console report with match statistics and spatial accuracy
  - outputs/figures/paper/tapaas_crossval_map.png (map of matched events)
  - outputs/data/tapaas_crossval_singer.csv
  - outputs/data/tapaas_crossval_orca.csv

Usage:
    uv run python scripts/crossvalidate_tapaas_locations.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LogNorm
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime, timedelta
from pyproj import Geod
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.patheffects as pe
from scipy.interpolate import griddata, RegularGridInterpolator

from read_dat import MOORINGS

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SINGER_PATH = Path("/home/jovyan/my_data/bravoseis/NOAA/merged_data_amended.txt")
ORCA_PATH = Path("/home/jovyan/my_data/bravoseis/earthquakes/Orca_EQ_data.csv")
USGS_PATH = DATA_DIR / "usgs_eq_catalogue.csv"

BATHY_IBCSO = Path("/home/jovyan/my_data/bravoseis/bathymetry/IBCSO_v2_bed_WGS84.nc")
BATHY_REGIONAL = Path("/home/jovyan/my_data/bravoseis/bathymetry/bransfield.xyz")
BATHY_ORCA = Path("/home/jovyan/my_data/bravoseis/bathymetry/MGDS_Download/BRAVOSEIS/Orca_bathymetry.nc")

GEOD = Geod(ellps="WGS84")
MATCH_TOL_S = 30.0
MAP_LON_MIN, MAP_LON_MAX = -62.5, -54.5
MAP_LAT_MIN, MAP_LAT_MAX = -64.5, -60.7


# ============================================================
# 1. Catalogue parsers
# ============================================================

def parse_singer_catalogue():
    """Parse Singer's fixed-width catalogue, return EQ events with locations."""
    records = []
    with open(SINGER_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            ts_str = parts[0]
            if len(ts_str) < 14:
                continue
            try:
                year = int(ts_str[0:4])
                doy = int(ts_str[4:7])
                hh = int(ts_str[7:9])
                mm = int(ts_str[9:11])
                ss = int(ts_str[11:13])
                ff = int(ts_str[13:])
                dt = datetime(year, 1, 1) + timedelta(
                    days=doy - 1, hours=hh, minutes=mm,
                    seconds=ss, microseconds=ff * 10000
                )
            except (ValueError, IndexError):
                continue
            try:
                n_moorings = int(parts[1])
                lat = float(parts[3])
                lon = float(parts[4])
                residual = float(parts[7])
            except (ValueError, IndexError):
                continue

            singer_class = "other"
            for token in parts[10:]:
                tok_upper = token.upper()
                if tok_upper in ("EQ", "IQ", "IDK", "SS"):
                    singer_class = tok_upper
                    break

            records.append({
                "datetime": pd.Timestamp(dt),
                "n_moorings": n_moorings,
                "lat": lat, "lon": lon,
                "residual": residual,
                "singer_class": singer_class,
            })
    df = pd.DataFrame(records)
    print(f"Singer catalogue: {len(df):,} total events")
    for cls, cnt in df["singer_class"].value_counts().items():
        print(f"  {cls}: {cnt:,}")
    return df


def load_orca_catalogue():
    """Load Orca OBS earthquake catalogue."""
    df = pd.read_csv(ORCA_PATH)
    df = df.drop(columns=["Unnamed: 11"], errors="ignore")
    matlab_epoch = datetime(1, 1, 1)
    df["datetime"] = df["date"].apply(
        lambda d: matlab_epoch + timedelta(days=d - 367)
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"\nOrca OBS catalogue: {len(df):,} events")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  erh median: {df['erh'].median():.1f} km")
    return df


def load_usgs_catalogue():
    """Load USGS teleseismic catalogue."""
    if not USGS_PATH.exists():
        print("\nUSGS catalogue not found.")
        return pd.DataFrame()
    df = pd.read_csv(USGS_PATH)
    df["datetime"] = pd.to_datetime(df["time"])
    print(f"\nUSGS catalogue: {len(df):,} teleseismic events (M{df['mag'].min():.1f}–{df['mag'].max():.1f})")
    return df


# ============================================================
# 2. Build recording windows from full event catalogue
# ============================================================

def build_coverage_windows():
    """Build merged recording windows from full detection catalogue."""
    cat = pd.read_parquet(DATA_DIR / "event_catalogue_full.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])

    segments = cat.groupby(["mooring", "file_number"])["onset_utc"].agg(
        ["min", "max"]).reset_index()
    segments["min"] -= pd.Timedelta(minutes=30)
    segments["max"] += pd.Timedelta(minutes=30)

    windows = sorted(zip(segments["min"], segments["max"]))
    merged = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    total_h = sum((e - s).total_seconds() / 3600 for s, e in merged)
    deploy_h = (cat["onset_utc"].max() - cat["onset_utc"].min()).total_seconds() / 3600
    print(f"\nRecording coverage: {total_h:.0f}h / {deploy_h:.0f}h "
          f"= {100 * total_h / deploy_h:.1f}%")
    print(f"  {len(merged)} merged windows, {cat['onset_utc'].min()} to {cat['onset_utc'].max()}")
    return merged, cat


def in_coverage(dt, windows):
    """Check if a datetime falls within any recording window."""
    ts = pd.Timestamp(dt)
    for s, e in windows:
        if s <= ts <= e:
            return True
    return False


def filter_to_coverage(ref_df, time_col, windows, label):
    """Filter reference catalogue to our recording windows."""
    mask = ref_df[time_col].apply(lambda t: in_coverage(t, windows))
    n_in = mask.sum()
    print(f"  {label}: {n_in:,}/{len(ref_df):,} in coverage ({100*n_in/len(ref_df):.1f}%)")
    return ref_df[mask].copy()


# ============================================================
# 3. Temporal matching + spatial offset
# ============================================================

def match_catalogues(ref_df, ref_time_col, our_df, our_time_col,
                     tol_s=MATCH_TOL_S, spatial_best=True):
    """Match reference events to our TAPAAs locations by time.

    If spatial_best=True, picks the spatially closest match among all
    temporal candidates (reduces false-match noise when event density
    is high). Otherwise picks the temporally closest match.

    Returns DataFrame with one row per reference event including spatial offset.
    """
    our_sorted = our_df.sort_values(our_time_col).reset_index(drop=True)
    our_times = our_sorted[our_time_col].values.astype("datetime64[ns]").astype(np.int64) / 1e9
    our_lons = our_sorted["lon"].values
    our_lats = our_sorted["lat"].values

    results = []
    for _, row in ref_df.iterrows():
        ref_t = row[ref_time_col].timestamp()
        lo = np.searchsorted(our_times, ref_t - tol_s, side="left")
        hi = np.searchsorted(our_times, ref_t + tol_s, side="right")

        if lo >= hi:
            results.append({**row.to_dict(), "matched": False,
                            "our_lat": np.nan, "our_lon": np.nan,
                            "time_offset_s": np.nan, "offset_km": np.nan,
                            "our_tier": None, "our_n_moorings": np.nan,
                            "our_residual_s": np.nan, "our_band": None,
                            "our_assoc_id": None,
                            "n_temporal_candidates": 0})
            continue

        n_cands = hi - lo

        if spatial_best and n_cands > 1 and not np.isnan(row["lat"]):
            # Pick the spatially closest among temporal candidates
            best_dist = np.inf
            best_i = lo
            for j in range(lo, min(hi, lo + 100)):
                _, _, d_m = GEOD.inv(row["lon"], row["lat"],
                                     our_lons[j], our_lats[j])
                if d_m < best_dist:
                    best_dist = d_m
                    best_i = j
            dt_s = abs(our_times[best_i] - ref_t)
        else:
            # Temporally closest
            dts = np.abs(our_times[lo:hi] - ref_t)
            best_i = lo + np.argmin(dts)
            dt_s = dts[np.argmin(dts)]

        best = our_sorted.iloc[best_i]
        _, _, dist_m = GEOD.inv(row["lon"], row["lat"],
                                best["lon"], best["lat"])
        results.append({
            **row.to_dict(),
            "matched": True,
            "our_lat": best["lat"],
            "our_lon": best["lon"],
            "time_offset_s": dt_s,
            "offset_km": dist_m / 1000.0,
            "our_tier": best["quality_tier"],
            "our_n_moorings": best["n_moorings"],
            "our_residual_s": best["residual_s"],
            "our_band": best["detection_band"],
            "our_assoc_id": best["assoc_id"],
            "n_temporal_candidates": n_cands,
        })

    return pd.DataFrame(results)


def print_offset_stats(matched, label):
    """Print spatial offset statistics for matched events."""
    m = matched[matched["matched"]].copy()
    n_ref = len(matched)
    n_matched = len(m)
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Reference events: {n_ref:,}")
    print(f"  Matched (±{MATCH_TOL_S:.0f}s): {n_matched:,} ({100*n_matched/n_ref:.1f}%)")

    if n_matched == 0:
        return m

    o = m["offset_km"]
    print(f"\n  Spatial offset (km):")
    print(f"    Median: {o.median():.1f}")
    print(f"    Mean:   {o.mean():.1f}")
    print(f"    Std:    {o.std():.1f}")
    print(f"    IQR:    [{o.quantile(.25):.1f}, {o.quantile(.75):.1f}]")
    print(f"    <=5 km:  {(o<=5).sum():,} ({100*(o<=5).mean():.1f}%)")
    print(f"    <=10 km: {(o<=10).sum():,} ({100*(o<=10).mean():.1f}%)")
    print(f"    <=20 km: {(o<=20).sum():,} ({100*(o<=20).mean():.1f}%)")
    print(f"    <=50 km: {(o<=50).sum():,} ({100*(o<=50).mean():.1f}%)")

    # By tier
    print(f"\n  By quality tier:")
    for tier in ["A", "B", "C"]:
        sub = m[m["our_tier"] == tier]
        if len(sub) == 0:
            continue
        so = sub["offset_km"]
        print(f"    {tier}: n={len(sub):,}, "
              f"median={so.median():.1f} km, "
              f"<=10km: {100*(so<=10).mean():.0f}%")

    # By n_moorings
    print(f"\n  By mooring count:")
    for nm in sorted(m["our_n_moorings"].unique()):
        sub = m[m["our_n_moorings"] == nm]
        so = sub["offset_km"]
        print(f"    {int(nm)} moorings: n={len(sub):,}, "
              f"median={so.median():.1f} km, "
              f"<=10km: {100*(so<=10).mean():.0f}%")

    # By band
    print(f"\n  By detection band:")
    for band in ["low", "mid", "high"]:
        sub = m[m["our_band"] == band]
        if len(sub) == 0:
            continue
        so = sub["offset_km"]
        print(f"    {band}: n={len(sub):,}, "
              f"median={so.median():.1f} km")

    # Time offset
    print(f"\n  Time offset: median={m['time_offset_s'].median():.1f}s, "
          f"mean={m['time_offset_s'].mean():.1f}s")

    return m


# ============================================================
# 4. Load bathymetry (reused from location map script)
# ============================================================

def load_bathy():
    """Load merged bathymetry."""
    import xarray as xr

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
        data = np.loadtxt(BATHY_REGIONAL)
        z_mb = griddata((data[:, 0], data[:, 1]), data[:, 2],
                        (lon_grid, lat_grid), method="linear")
        valid = ~np.isnan(z_mb)
        z_grid[valid] = z_mb[valid]
        del data

    if BATHY_ORCA.exists():
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


# ============================================================
# 5. Cross-validation map figure
# ============================================================

def make_crossval_map(singer_m, orca_m, usgs_m, lon_grid, lat_grid, z_grid):
    """Create a 3-panel cross-validation map figure."""
    map_proj = ccrs.Mercator(central_longitude=-58.5)
    data_crs = ccrs.PlateCarree()

    ls = LightSource(azdeg=315, altdeg=25)
    z_shade = z_grid.copy()
    z_shade[np.isnan(z_shade)] = 0
    hillshade = ls.hillshade(z_shade, vert_exag=0.005)

    fig = plt.figure(figsize=(24, 8))
    # Two map panels + one regular axes for histogram
    ax1 = fig.add_subplot(1, 3, 1, projection=map_proj)
    ax2 = fig.add_subplot(1, 3, 2, projection=map_proj)
    ax3 = fig.add_subplot(1, 3, 3)  # regular axes for histogram
    axes = [ax1, ax2, ax3]

    def setup_ax(ax, title):
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
        for key, info in sorted(MOORINGS.items()):
            ax.plot(info["lon"], info["lat"], "w^", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.6,
                    transform=data_crs, zorder=10)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5,
                          color="gray", linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        ax.set_extent([-62, -55, -64, -61.2], crs=data_crs)
        ax.set_title(title, fontsize=12, fontweight="bold")

    def plot_offset_arrows(ax, ref_lons, ref_lats, our_lons, our_lats,
                           offsets_km, color, label, max_show=500):
        """Plot reference locations and arrows to our locations, colored by offset."""
        n = len(ref_lons)
        if n > max_show:
            idx = np.random.default_rng(42).choice(n, max_show, replace=False)
        else:
            idx = np.arange(n)

        # Reference positions
        ax.scatter(ref_lons.iloc[idx], ref_lats.iloc[idx], s=8, c="white",
                   edgecolors="black", linewidths=0.3, transform=data_crs,
                   zorder=7, label=f"{label} location")

        # Arrows from reference to our location
        for i in idx:
            ax.annotate("", xy=(our_lons.iloc[i], our_lats.iloc[i]),
                         xytext=(ref_lons.iloc[i], ref_lats.iloc[i]),
                         arrowprops=dict(arrowstyle="-", color=color,
                                         alpha=0.4, linewidth=0.5),
                         transform=data_crs, zorder=6)

        # Our positions
        ax.scatter(our_lons.iloc[idx], our_lats.iloc[idx], s=8, c=color,
                   alpha=0.5, transform=data_crs, zorder=8,
                   label=f"TAPAAs location")

    # --- Panel (a): Singer EQ ---
    ax = axes[0]
    sm = singer_m[singer_m["matched"]].copy()
    n_ref = len(singer_m)
    n_m = len(sm)
    med_off = sm["offset_km"].median() if n_m > 0 else 0
    setup_ax(ax, f"(a) Singer EQ (n={n_m:,}/{n_ref:,} matched)\n"
                 f"median offset = {med_off:.1f} km")
    if n_m > 0:
        plot_offset_arrows(ax, sm["lon"], sm["lat"],
                           sm["our_lon"], sm["our_lat"],
                           sm["offset_km"], "#e74c3c", "Singer")
        ax.legend(fontsize=7, loc="lower right")

    # --- Panel (b): Orca OBS ---
    ax = axes[1]
    om = orca_m[orca_m["matched"]].copy()
    n_ref = len(orca_m)
    n_m = len(om)
    med_off = om["offset_km"].median() if n_m > 0 else 0
    setup_ax(ax, f"(b) Orca OBS (n={n_m:,}/{n_ref:,} matched)\n"
                 f"median offset = {med_off:.1f} km")
    if n_m > 0:
        plot_offset_arrows(ax, om["lon"], om["lat"],
                           om["our_lon"], om["our_lat"],
                           om["offset_km"], "#2ecc71", "Orca")
        ax.legend(fontsize=7, loc="lower right")

    # --- Panel (c): Offset histograms ---
    ax_hist = axes[2]

    bins = np.arange(0, 205, 5)
    if len(sm) > 0:
        ax_hist.hist(sm["offset_km"], bins=bins, alpha=0.6,
                     color="#e74c3c", label=f"Singer EQ (n={len(sm):,})")
    if len(om) > 0:
        ax_hist.hist(om["offset_km"], bins=bins, alpha=0.6,
                     color="#2ecc71", label=f"Orca OBS (n={len(om):,})")
    ax_hist.set_xlabel("Spatial offset (km)", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title("(c) Offset distribution", fontsize=12, fontweight="bold")
    ax_hist.legend(fontsize=9)
    ax_hist.set_xlim(0, 200)
    ax_hist.axvline(10, color="gray", ls="--", alpha=0.5, label="10 km")

    # USGS markers on panel (a) if available
    if len(usgs_m) > 0:
        um = usgs_m[usgs_m["matched"]]
        if len(um) > 0:
            axes[0].scatter(um["lon"], um["lat"],
                            s=100, c="yellow", marker="*",
                            edgecolors="black", linewidths=0.5,
                            transform=data_crs, zorder=12,
                            label=f"USGS M{um['mag'].min():.1f}+")
            axes[0].legend(fontsize=7, loc="lower right")

    fig.suptitle("TAPAAs Location Cross-Validation Against Independent Catalogues",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = FIG_DIR / "tapaas_crossval_map.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {outpath}")


# ============================================================
# 6. Offset summary figure (compact version for methods)
# ============================================================

def make_offset_summary_figure(singer_m, orca_m):
    """Create a compact 2x2 offset summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sm = singer_m[singer_m["matched"]].copy()
    om = orca_m[orca_m["matched"]].copy()

    # (a) CDF of offset
    ax = axes[0, 0]
    for df, label, color in [(sm, "Singer EQ", "#e74c3c"),
                              (om, "Orca OBS", "#2ecc71")]:
        if len(df) == 0:
            continue
        o = np.sort(df["offset_km"].values)
        cdf = np.arange(1, len(o) + 1) / len(o)
        ax.plot(o, cdf, color=color, linewidth=2, label=label)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.axvline(10, color="gray", ls="--", alpha=0.5)
    ax.axvline(50, color="gray", ls="--", alpha=0.3)
    ax.set_xlabel("Spatial offset (km)", fontsize=10)
    ax.set_ylabel("Cumulative fraction", fontsize=10)
    ax.set_title("(a) CDF of spatial offset", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 200)

    # (b) Offset by number of moorings (box plot)
    ax = axes[0, 1]
    data_by_nm = {}
    for nm in range(3, 7):
        vals = []
        for df, label in [(sm, "Singer"), (om, "Orca")]:
            sub = df[df["our_n_moorings"] == nm]
            if len(sub) > 0:
                vals.extend(sub["offset_km"].values)
        if vals:
            data_by_nm[nm] = vals
    if data_by_nm:
        positions = sorted(data_by_nm.keys())
        box_data = [data_by_nm[p] for p in positions]
        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                        showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3498db")
            patch.set_alpha(0.5)
        ax.set_xlabel("Number of moorings", fontsize=10)
        ax.set_ylabel("Spatial offset (km)", fontsize=10)
        ax.set_title("(b) Offset by mooring count", fontsize=12, fontweight="bold")

    # (c) Offset by quality tier
    ax = axes[1, 0]
    combined = pd.concat([sm, om], ignore_index=True)
    if len(combined) > 0:
        for tier, color in [("A", "#27ae60"), ("B", "#f39c12"), ("C", "#e74c3c")]:
            sub = combined[combined["our_tier"] == tier]
            if len(sub) > 0:
                bins = np.arange(0, 205, 5)
                ax.hist(sub["offset_km"], bins=bins, alpha=0.5, color=color,
                        label=f"Tier {tier} (n={len(sub):,})")
        ax.set_xlabel("Spatial offset (km)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("(c) Offset by quality tier", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 200)

    # (d) Offset by detection band
    ax = axes[1, 1]
    if len(combined) > 0:
        for band, color, label in [("low", "#e74c3c", "Low (1–14 Hz)"),
                                    ("mid", "#f39c12", "Mid (14–30 Hz)"),
                                    ("high", "#3498db", "High (>30 Hz)")]:
            sub = combined[combined["our_band"] == band]
            if len(sub) > 0:
                bins = np.arange(0, 205, 5)
                ax.hist(sub["offset_km"], bins=bins, alpha=0.5, color=color,
                        label=f"{label} (n={len(sub):,})")
        ax.set_xlabel("Spatial offset (km)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("(d) Offset by detection band", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 200)

    fig.suptitle("TAPAAs Location Accuracy — Cross-Validation Summary",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = FIG_DIR / "tapaas_crossval_offsets.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("TAPAAs Location Cross-Validation")
    print("=" * 65)

    # --- Load our TAPAAs locations ---
    loc = pd.read_parquet(DATA_DIR / "tapaas_locations.parquet")
    loc["earliest_utc"] = pd.to_datetime(loc["earliest_utc"])
    pub = loc[loc["quality_tier"].isin(["A", "B", "C"])].copy()
    print(f"\nTAPAAs publishable (A+B+C): {len(pub):,}")

    # --- Build coverage windows ---
    windows, cat = build_coverage_windows()

    # --- Load reference catalogues ---
    singer_df = parse_singer_catalogue()
    singer_eq = singer_df[singer_df["singer_class"] == "EQ"].copy()
    print(f"  Singer EQ events: {len(singer_eq):,}")

    orca_df = load_orca_catalogue()
    usgs_df = load_usgs_catalogue()

    # --- Filter to coverage ---
    print("\nFiltering to recording coverage:")
    singer_cov = filter_to_coverage(singer_eq, "datetime", windows, "Singer EQ")
    orca_cov = filter_to_coverage(orca_df, "datetime", windows, "Orca OBS")

    # --- Match Singer EQ → TAPAAs ---
    singer_results = match_catalogues(
        singer_cov, "datetime", pub, "earliest_utc", tol_s=MATCH_TOL_S
    )
    singer_matched = print_offset_stats(singer_results, "Singer EQ vs TAPAAs")

    # Singer class breakdown (EQ matched → by our band)
    print("\n  Singer EQ matches by our detection band:")
    if len(singer_matched) > 0:
        for band, cnt in singer_matched["our_band"].value_counts().items():
            print(f"    {band}: {cnt:,}")

    # --- Match Orca → TAPAAs ---
    orca_results = match_catalogues(
        orca_cov, "datetime", pub, "earliest_utc", tol_s=MATCH_TOL_S
    )
    orca_matched = print_offset_stats(orca_results, "Orca OBS vs TAPAAs")

    # --- Match USGS → TAPAAs ---
    usgs_results = pd.DataFrame()
    if len(usgs_df) > 0:
        # Normalize column names for USGS
        usgs_norm = usgs_df.copy()
        if "longitude" in usgs_norm.columns:
            usgs_norm = usgs_norm.rename(columns={"longitude": "lon",
                                                   "latitude": "lat"})
        # USGS events — match against ALL tiers (these are big events)
        usgs_results = match_catalogues(
            usgs_norm, "datetime", loc, "earliest_utc", tol_s=60.0
        )
        usgs_m = usgs_results[usgs_results["matched"]]
        print(f"\n{'='*65}")
        print(f"  USGS Teleseismic Events")
        print(f"{'='*65}")
        print(f"  Total: {len(usgs_df):,}, Matched: {len(usgs_m):,}")
        for _, row in usgs_results.iterrows():
            status = "MATCHED" if row["matched"] else "MISSED"
            mag = row.get("mag", "?")
            place = row.get("place", "")
            off = f", offset={row['offset_km']:.1f} km" if row["matched"] else ""
            print(f"    M{mag} {place} — {status}{off}")

    # --- Save match tables ---
    singer_results.to_csv(DATA_DIR / "tapaas_crossval_singer.csv", index=False)
    orca_results.to_csv(DATA_DIR / "tapaas_crossval_orca.csv", index=False)
    print(f"\nSaved match tables to outputs/data/")

    # --- Singer EQ: fate of unmatched events ---
    singer_unmatched = singer_results[~singer_results["matched"]].copy()
    n_unmatched = len(singer_unmatched)
    if n_unmatched > 0:
        # Check if unmatched Singer EQs match ANY TAPAAs location (incl tier D)
        singer_vs_all = match_catalogues(
            singer_unmatched, "datetime", loc, "earliest_utc", tol_s=MATCH_TOL_S
        )
        n_in_d = singer_vs_all["matched"].sum()
        print(f"\n  Singer EQ unmatched in A+B+C: {n_unmatched:,}")
        print(f"    Of those, {n_in_d:,} match tier D locations")
        print(f"    Completely unmatched: {n_unmatched - n_in_d:,}")

    # --- Figures ---
    print("\nGenerating cross-validation figures...")
    lon_grid, lat_grid, z_grid = load_bathy()
    make_crossval_map(singer_results, orca_results, usgs_results,
                      lon_grid, lat_grid, z_grid)
    make_offset_summary_figure(singer_results, orca_results)

    # --- Final summary ---
    sm = singer_results[singer_results["matched"]]
    om = orca_results[orca_results["matched"]]
    print(f"\n{'='*65}")
    print("FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"""
TAPAAs publishable locations: {len(pub):,} (A+B+C)

Singer EQ cross-validation:
  Matched: {len(sm):,}/{len(singer_cov):,} ({100*len(sm)/len(singer_cov):.1f}%)
  Median offset: {sm['offset_km'].median():.1f} km
  Within 10 km: {100*(sm['offset_km']<=10).mean():.0f}%
  Within 50 km: {100*(sm['offset_km']<=50).mean():.0f}%

Orca OBS cross-validation:
  Matched: {len(om):,}/{len(orca_cov):,} ({100*len(om)/len(orca_cov):.1f}%)
  Median offset: {om['offset_km'].median():.1f} km
  Within 10 km: {100*(om['offset_km']<=10).mean():.0f}%
  Within 50 km: {100*(om['offset_km']<=50).mean():.0f}%

USGS teleseismic: {len(usgs_df):,} events (M4.6–5.5), {len(usgs_results[usgs_results['matched']]) if len(usgs_results) > 0 else 0} matched
""")


if __name__ == "__main__":
    main()
