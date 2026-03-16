#!/usr/bin/env python3
"""
locate_full.py — Locate events from the full dataset associations.

Uses chunked serial processing with periodic checkpoints for resumability.
Pre-indexes catalogue by event_id for fast lookups.

Usage:
    uv run python scripts/locate_full.py
    uv run python scripts/locate_full.py --resume
    uv run python scripts/locate_full.py --no-jackknife  # faster
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Geod
from multiprocessing import Pool
import time

import sys
sys.path.insert(0, str(Path(__file__).parent))
from read_dat import MOORINGS
MOORING_KEYS = sorted(MOORINGS.keys())
from locate_events import (
    locate_one, refine_location, build_grid, precompute_distances,
    OUTLIER_RESIDUAL_FACTOR, JACKKNIFE_SHIFT_THRESHOLD_KM,
    TIER_A_MAX_RESIDUAL, TIER_B_MAX_RESIDUAL, TIER_C_MAX_RESIDUAL,
    MAX_DIST_FROM_CENTROID_KM,
)

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
CHECKPOINT_DIR = DATA_DIR / "location_checkpoints"

GEOD_WGS84 = Geod(ellps="WGS84")
CHECKPOINT_INTERVAL = 50_000

CENTROID_LAT = np.mean([MOORINGS[m]["lat"] for m in MOORING_KEYS])
CENTROID_LON = np.mean([MOORINGS[m]["lon"] for m in MOORING_KEYS])


def load_travel_times():
    with open(DATA_DIR / "travel_times.json") as f:
        data = json.load(f)
    pair_speeds = {}
    for pk, info in data["pairs"].items():
        k1, k2 = pk.split("-")
        pair_speeds[(k1, k2)] = info["c_eff_ms"]
        pair_speeds[(k2, k1)] = info["c_eff_ms"]
    c_eff = data["effective_speed_mean_ms"]
    return c_eff, pair_speeds


def locate_fast(assoc_row, eid_to_onset, dist_km_grids, lon_grid, lat_grid,
                c_eff, pair_speeds, do_jackknife=False):
    """Fast location for one association using pre-indexed onset lookup."""
    event_ids = assoc_row["event_ids"].split(",")

    # Get onset times per mooring
    mooring_onsets = {}
    for eid in event_ids:
        if eid not in eid_to_onset:
            continue
        mk, t_s = eid_to_onset[eid]
        if mk not in mooring_onsets or t_s < mooring_onsets[mk]:
            mooring_onsets[mk] = t_s

    if len(mooring_onsets) < 3:
        return None

    # Coarse location
    result = locate_one(mooring_onsets, dist_km_grids, c_eff, pair_speeds)
    if result is None:
        return None

    lat_loc = float(lat_grid[result["grid_idx"]])
    lon_loc = float(lon_grid[result["grid_idx"]])
    residual = result["residual_s"]
    moorings_used = sorted(mooring_onsets.keys())
    n_moorings = len(moorings_used)

    # Outlier mooring check (4+ moorings)
    dropped_mooring = None
    if n_moorings >= 4:
        i_row, i_col = result["grid_idx"]
        pred_arrivals = {m: dist_km_grids[m][i_row, i_col] * 1000.0 / c_eff
                         for m in moorings_used}
        t_earliest = min(mooring_onsets.values())
        t_pred_earliest = min(pred_arrivals.values())
        per_mooring_resid = {}
        for mk in moorings_used:
            t_obs_rel = mooring_onsets[mk] - t_earliest
            t_pred_rel = pred_arrivals[mk] - t_pred_earliest
            per_mooring_resid[mk] = t_obs_rel - t_pred_rel

        abs_resids = {mk: abs(r) for mk, r in per_mooring_resid.items()}
        median_resid = np.median(list(abs_resids.values()))
        if median_resid > 0:
            for mk, ar in abs_resids.items():
                if ar > OUTLIER_RESIDUAL_FACTOR * median_resid and ar > 1.0:
                    reduced_onsets = {m: t for m, t in mooring_onsets.items() if m != mk}
                    if len(reduced_onsets) >= 3:
                        r2 = locate_one(reduced_onsets, dist_km_grids, c_eff, pair_speeds)
                        if r2 and r2["residual_s"] < residual * 0.7:
                            lat_loc = float(lat_grid[r2["grid_idx"]])
                            lon_loc = float(lon_grid[r2["grid_idx"]])
                            residual = r2["residual_s"]
                            dropped_mooring = mk
                            moorings_used = sorted(reduced_onsets.keys())
                            n_moorings = len(moorings_used)
                    break

    # Fine refinement
    onsets_for_refine = {m: mooring_onsets[m] for m in moorings_used}
    refined = refine_location(lat_loc, lon_loc, onsets_for_refine, c_eff,
                              pair_speeds=pair_speeds)
    if refined is not None:
        lat_loc = refined["lat"]
        lon_loc = refined["lon"]
        residual = refined["residual_s"]

    # Jackknife
    jackknife_shift_km = 0.0
    jackknife_stable = True
    if do_jackknife and n_moorings >= 4:
        jk_locs = []
        for mk_drop in moorings_used:
            reduced = {m: t for m, t in mooring_onsets.items()
                       if m != mk_drop and m in moorings_used}
            if len(reduced) < 3:
                continue
            r_jk = locate_one(reduced, dist_km_grids, c_eff, pair_speeds)
            if r_jk:
                jk_lat = float(lat_grid[r_jk["grid_idx"]])
                jk_lon = float(lon_grid[r_jk["grid_idx"]])
                _, _, shift_m = GEOD_WGS84.inv(lon_loc, lat_loc, jk_lon, jk_lat)
                jk_locs.append(shift_m / 1000.0)
        if jk_locs:
            jackknife_shift_km = max(jk_locs)
            if jackknife_shift_km > JACKKNIFE_SHIFT_THRESHOLD_KM:
                jackknife_stable = False

    # Distance from centroid
    _, _, dist_m = GEOD_WGS84.inv(lon_loc, lat_loc, CENTROID_LON, CENTROID_LAT)
    dist_km = dist_m / 1000.0

    # Quality tier
    if n_moorings < 3:
        tier = "D"
    elif dist_km > MAX_DIST_FROM_CENTROID_KM:
        tier = "D"
    elif residual <= TIER_A_MAX_RESIDUAL and jackknife_stable and n_moorings >= 4:
        tier = "A"
    elif residual <= TIER_B_MAX_RESIDUAL:
        tier = "B"
    elif residual <= TIER_C_MAX_RESIDUAL:
        tier = "C"
    else:
        tier = "D"

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


# === Worker globals (initialized once per process) ===
_w_dist_grids = None
_w_lon_grid = None
_w_lat_grid = None
_w_c_eff = None
_w_pair_speeds = None
_w_do_jk = None


def _init_worker(c_eff, pair_speeds, do_jk):
    """Each worker builds its own grid (lightweight, ~10MB)."""
    global _w_dist_grids, _w_lon_grid, _w_lat_grid, _w_c_eff, _w_pair_speeds, _w_do_jk
    _w_lon_grid, _w_lat_grid = build_grid(0.01)
    _w_dist_grids = precompute_distances(_w_lon_grid, _w_lat_grid)
    _w_c_eff = c_eff
    _w_pair_speeds = pair_speeds
    _w_do_jk = do_jk


def _worker_locate(item):
    """Worker function: locate one association from pre-extracted mooring_onsets."""
    assoc_id, mooring_onsets, earliest_utc, detection_band = item

    if len(mooring_onsets) < 3:
        return None

    moorings_used = sorted(mooring_onsets.keys())
    n_moorings = len(moorings_used)
    c_eff = _w_c_eff
    pair_speeds = _w_pair_speeds
    dist_km_grids = _w_dist_grids
    lon_grid = _w_lon_grid
    lat_grid = _w_lat_grid

    # Coarse location
    result = locate_one(mooring_onsets, dist_km_grids, c_eff, pair_speeds)
    if result is None:
        return None

    lat_loc = float(lat_grid[result["grid_idx"]])
    lon_loc = float(lon_grid[result["grid_idx"]])
    residual = result["residual_s"]

    # Outlier mooring check (4+ moorings)
    dropped_mooring = None
    if n_moorings >= 4:
        i_row, i_col = result["grid_idx"]
        pred_arrivals = {m: dist_km_grids[m][i_row, i_col] * 1000.0 / c_eff
                         for m in moorings_used}
        t_earliest = min(mooring_onsets.values())
        t_pred_earliest = min(pred_arrivals.values())
        per_mooring_resid = {}
        for mk in moorings_used:
            t_obs_rel = mooring_onsets[mk] - t_earliest
            t_pred_rel = pred_arrivals[mk] - t_pred_earliest
            per_mooring_resid[mk] = t_obs_rel - t_pred_rel

        abs_resids = {mk: abs(r) for mk, r in per_mooring_resid.items()}
        median_resid = np.median(list(abs_resids.values()))
        if median_resid > 0:
            for mk, ar in abs_resids.items():
                if ar > OUTLIER_RESIDUAL_FACTOR * median_resid and ar > 1.0:
                    reduced_onsets = {m: t for m, t in mooring_onsets.items() if m != mk}
                    if len(reduced_onsets) >= 3:
                        r2 = locate_one(reduced_onsets, dist_km_grids, c_eff, pair_speeds)
                        if r2 and r2["residual_s"] < residual * 0.7:
                            lat_loc = float(lat_grid[r2["grid_idx"]])
                            lon_loc = float(lon_grid[r2["grid_idx"]])
                            residual = r2["residual_s"]
                            dropped_mooring = mk
                            moorings_used = sorted(reduced_onsets.keys())
                            n_moorings = len(moorings_used)
                    break

    # Fine refinement
    onsets_for_refine = {m: mooring_onsets[m] for m in moorings_used}
    refined = refine_location(lat_loc, lon_loc, onsets_for_refine, c_eff,
                              pair_speeds=pair_speeds)
    if refined is not None:
        lat_loc = refined["lat"]
        lon_loc = refined["lon"]
        residual = refined["residual_s"]

    # Jackknife
    jackknife_shift_km = 0.0
    jackknife_stable = True
    if _w_do_jk and n_moorings >= 4:
        jk_locs = []
        for mk_drop in moorings_used:
            reduced = {m: t for m, t in mooring_onsets.items()
                       if m != mk_drop and m in moorings_used}
            if len(reduced) < 3:
                continue
            r_jk = locate_one(reduced, dist_km_grids, c_eff, pair_speeds)
            if r_jk:
                jk_lat = float(lat_grid[r_jk["grid_idx"]])
                jk_lon = float(lon_grid[r_jk["grid_idx"]])
                _, _, shift_m = GEOD_WGS84.inv(lon_loc, lat_loc, jk_lon, jk_lat)
                jk_locs.append(shift_m / 1000.0)
        if jk_locs:
            jackknife_shift_km = max(jk_locs)
            if jackknife_shift_km > JACKKNIFE_SHIFT_THRESHOLD_KM:
                jackknife_stable = False

    # Distance from centroid
    _, _, dist_m = GEOD_WGS84.inv(lon_loc, lat_loc, CENTROID_LON, CENTROID_LAT)
    dist_km = dist_m / 1000.0

    # Quality tier
    if n_moorings < 3:
        tier = "D"
    elif dist_km > MAX_DIST_FROM_CENTROID_KM:
        tier = "D"
    elif residual <= TIER_A_MAX_RESIDUAL and jackknife_stable and n_moorings >= 4:
        tier = "A"
    elif residual <= TIER_B_MAX_RESIDUAL:
        tier = "B"
    elif residual <= TIER_C_MAX_RESIDUAL:
        tier = "C"
    else:
        tier = "D"

    if not jackknife_stable and tier in ("A", "B"):
        tier = "C"

    return {
        "assoc_id": assoc_id,
        "lat": lat_loc,
        "lon": lon_loc,
        "residual_s": round(residual, 3),
        "n_moorings": n_moorings,
        "moorings": ",".join(moorings_used),
        "quality_tier": tier,
        "jackknife_shift_km": round(jackknife_shift_km, 2),
        "jackknife_stable": jackknife_stable,
        "dropped_mooring": dropped_mooring,
        "earliest_utc": earliest_utc,
        "detection_band": detection_band,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-jackknife", action="store_true")
    args = parser.parse_args()

    do_jk = not args.no_jackknife
    print("=" * 60, flush=True)
    print(f"Full Dataset Event Location — {args.workers} workers, "
          f"jackknife={'on' if do_jk else 'off'}", flush=True)
    print("=" * 60, flush=True)

    c_eff, pair_speeds = load_travel_times()

    # Load catalogue and build event_id → (mooring, onset_seconds) index
    print("\nLoading catalogue and building index...", flush=True)
    cat = pd.read_parquet(DATA_DIR / "event_catalogue_full.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    eids = cat["event_id"].values
    onsets = cat["onset_utc"].values.astype("datetime64[s]").astype("int64").astype(float)
    mooring_vals = cat["mooring"].values
    eid_to_onset = {}
    for i in range(len(cat)):
        eid_to_onset[eids[i]] = (mooring_vals[i], onsets[i])
    print(f"  Indexed {len(eid_to_onset):,} events", flush=True)
    del cat, eids, onsets, mooring_vals

    # Load associations
    assoc_df = pd.read_parquet(DATA_DIR / "cross_mooring_associations_full.parquet")
    assoc_df["earliest_utc"] = pd.to_datetime(assoc_df["earliest_utc"])
    locatable = assoc_df[assoc_df["n_moorings"] >= 3].copy()
    print(f"  Locatable: {len(locatable):,}", flush=True)

    # Resume
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    already_located = set()
    existing_results = []

    if args.resume:
        partial_path = CHECKPOINT_DIR / "locations_full_partial.parquet"
        if partial_path.exists():
            prev = pd.read_parquet(partial_path)
            existing_results = prev.to_dict("records")
            already_located = set(prev["assoc_id"])
            print(f"  Resuming: {len(existing_results):,} already located", flush=True)

    to_locate = locatable[~locatable["assoc_id"].isin(already_located)]
    print(f"  To locate: {len(to_locate):,}", flush=True)

    if len(to_locate) == 0 and not existing_results:
        print("Nothing to locate.")
        return

    # Pre-extract mooring_onsets for each association (main process only)
    print("\nPre-extracting mooring onsets...", flush=True)
    work_items = []
    for _, row in to_locate.iterrows():
        event_ids = row["event_ids"].split(",")
        mooring_onsets = {}
        for eid in event_ids:
            if eid not in eid_to_onset:
                continue
            mk, t_s = eid_to_onset[eid]
            if mk not in mooring_onsets or t_s < mooring_onsets[mk]:
                mooring_onsets[mk] = t_s
        work_items.append((
            row["assoc_id"],
            mooring_onsets,
            row["earliest_utc"],
            row["detection_band"],
        ))
    print(f"  Prepared {len(work_items):,} work items", flush=True)
    del eid_to_onset  # free ~1GB

    # Parallel location
    print(f"\nLocating with {args.workers} workers...", flush=True)
    t0 = time.time()
    results = list(existing_results)
    n_total = len(existing_results) + len(work_items)
    last_checkpoint_n = len(existing_results)

    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(c_eff, pair_speeds, do_jk),
    ) as pool:
        for result in pool.imap_unordered(_worker_locate, work_items, chunksize=500):
            if result is not None:
                results.append(result)

            n_done = len(results)
            if (n_done - last_checkpoint_n) >= CHECKPOINT_INTERVAL:
                elapsed = time.time() - t0
                n_new = n_done - len(existing_results)
                rate = n_new / elapsed if elapsed > 0 else 0
                remaining = (n_total - n_done) / rate / 60 if rate > 0 else 0
                print(f"  {n_done:,}/{n_total:,} located, "
                      f"{elapsed/60:.1f} min elapsed, ~{remaining:.0f} min remaining",
                      flush=True)

                ckpt_df = pd.DataFrame(results)
                ckpt_df.to_parquet(CHECKPOINT_DIR / "locations_full_partial.parquet",
                                   index=False)
                last_checkpoint_n = n_done

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min", flush=True)

    loc_df = pd.DataFrame(results)

    # Add 2-mooring as tier D
    bearings = assoc_df[assoc_df["n_moorings"] == 2]
    tier_d = [{
        "assoc_id": row["assoc_id"],
        "lat": np.nan, "lon": np.nan, "residual_s": np.nan,
        "n_moorings": row["n_moorings"], "moorings": row["moorings"],
        "quality_tier": "D", "jackknife_shift_km": np.nan,
        "jackknife_stable": False, "dropped_mooring": None,
        "earliest_utc": row["earliest_utc"],
        "detection_band": row["detection_band"],
    } for _, row in bearings.iterrows()]

    loc_df = pd.concat([loc_df, pd.DataFrame(tier_d)], ignore_index=True)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("LOCATION SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Total: {len(loc_df):,}")
    for tier in ["A", "B", "C", "D"]:
        n = (loc_df["quality_tier"] == tier).sum()
        print(f"  Tier {tier}: {n:,}")

    pub = loc_df[loc_df["quality_tier"].isin(["A", "B", "C"])]
    print(f"\nPublishable (A+B+C): {len(pub):,}")
    for n_m in sorted(pub["n_moorings"].unique()):
        sub = pub[pub["n_moorings"] == n_m]
        print(f"  {n_m} moorings: {len(sub):,}")

    # Save
    out_path = DATA_DIR / "event_locations_full.parquet"
    loc_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")

    for f in CHECKPOINT_DIR.glob("locations_full_*"):
        f.unlink()
    print("Checkpoints cleaned up.")


if __name__ == "__main__":
    main()
