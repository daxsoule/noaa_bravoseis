#!/usr/bin/env python3
"""
locate_tapaas.py — Parallel grid-search TDOA location for TAPAAs associations.

Applies the same location pipeline as locate_events.py (coarse grid search,
fine-grid refinement, outlier mooring dropping, jackknife validation, quality
tiering, uncertainty estimation) to the TAPAAs spatial-pruning associations.

Designed for large-scale runs (~1.35M associations):
  - Multiprocessing with configurable worker count
  - Periodic checkpointing (resume after interruption)
  - Progress logging with ETA

Usage:
    uv run python locate_tapaas.py
    uv run python locate_tapaas.py --workers 32
    uv run python locate_tapaas.py --workers 16 --batch-size 5000
    uv run python locate_tapaas.py --resume          # resume from checkpoint
    uv run python locate_tapaas.py --skip-postprocess  # location only, no QC

Output:
    outputs/data/tapaas_locations.parquet
    outputs/data/tapaas_location_checkpoints/batch_*.parquet
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from read_dat import MOORINGS

# Reuse core functions from locate_events
from locate_events import (
    build_grid,
    precompute_distances,
    locate_one,
    refine_location,
    load_travel_times,
    filter_icequake_locations,
    compute_swarm_coherence,
    MOORING_KEYS,
    GEOD,
    TIER_A_MAX_RESIDUAL,
    TIER_B_MAX_RESIDUAL,
    TIER_C_MAX_RESIDUAL,
    JACKKNIFE_SHIFT_THRESHOLD_KM,
    MAX_DIST_FROM_CENTROID_KM,
    OUTLIER_RESIDUAL_FACTOR,
)

# === Paths ===
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
CHECKPOINT_DIR = DATA_DIR / "tapaas_location_checkpoints"


# ─── Worker function (runs in subprocess) ───────────────────────────────

# These globals are set once per worker via the initializer.
_worker_cat_df = None
_worker_dist_km = None
_worker_lon_grid = None
_worker_lat_grid = None
_worker_c_eff = None
_worker_pair_speeds = None
_worker_do_jk = None


def _init_worker(cat_path, dist_km_shm_info, lon_grid_shape, lat_grid_shape,
                 grid_spacing, c_eff, pair_speeds, do_jk):
    """Initialize each worker process with shared data."""
    global _worker_cat_df, _worker_dist_km, _worker_lon_grid
    global _worker_lat_grid, _worker_c_eff, _worker_pair_speeds, _worker_do_jk

    _worker_cat_df = pd.read_parquet(cat_path)
    _worker_cat_df["onset_utc"] = pd.to_datetime(_worker_cat_df["onset_utc"])

    _worker_lon_grid, _worker_lat_grid = build_grid(grid_spacing)
    _worker_dist_km = precompute_distances(_worker_lon_grid, _worker_lat_grid)

    _worker_c_eff = c_eff
    _worker_pair_speeds = pair_speeds
    _worker_do_jk = do_jk


def _locate_one_association(assoc_dict):
    """Locate a single TAPAAs association (called in worker process).

    Takes a plain dict (not a pandas row) for pickling efficiency.
    Returns a result dict or None.
    """
    cat_df = _worker_cat_df
    dist_km_grids = _worker_dist_km
    lon_grid = _worker_lon_grid
    lat_grid = _worker_lat_grid
    c_eff = _worker_c_eff
    pair_speeds = _worker_pair_speeds
    do_jackknife = _worker_do_jk

    event_ids = assoc_dict["event_ids"].split(",")
    n_moorings = assoc_dict["n_moorings"]

    # Get onset times for each mooring
    events = cat_df[cat_df["event_id"].isin(event_ids)]
    if len(events) == 0:
        return None

    mooring_onsets = {}
    for _, ev in events.iterrows():
        mk = ev["mooring"]
        t = ev["onset_utc"].timestamp()
        if mk not in mooring_onsets or t < mooring_onsets[mk]:
            mooring_onsets[mk] = t

    if len(mooring_onsets) < 3:
        return None

    # --- Main location (coarse grid) ---
    result = locate_one(mooring_onsets, dist_km_grids, c_eff, pair_speeds)
    if result is None:
        return None

    lat_loc = float(lat_grid[result["grid_idx"]])
    lon_loc = float(lon_grid[result["grid_idx"]])
    residual = result["residual_s"]
    moorings_used = sorted(mooring_onsets.keys())

    # --- Per-mooring residuals for outlier detection ---
    per_mooring_resid = {}
    for mk in moorings_used:
        i_row, i_col = result["grid_idx"]
        pred_arrivals = {m: dist_km_grids[m][i_row, i_col] * 1000.0 / c_eff
                         for m in moorings_used}
        t_earliest = min(mooring_onsets.values())
        t_pred_earliest = min(pred_arrivals.values())
        t_obs_rel = mooring_onsets[mk] - t_earliest
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
                    reduced_onsets = {m: t for m, t in mooring_onsets.items()
                                      if m != mk}
                    if len(reduced_onsets) >= 3:
                        r2 = locate_one(reduced_onsets, dist_km_grids,
                                        c_eff, pair_speeds)
                        if r2 and r2["residual_s"] < residual * 0.7:
                            lat_loc = float(lat_grid[r2["grid_idx"]])
                            lon_loc = float(lon_grid[r2["grid_idx"]])
                            residual = r2["residual_s"]
                            dropped_mooring = mk
                            result = r2
                            moorings_used = sorted(reduced_onsets.keys())
                            n_moorings = len(moorings_used)
                    break

    # --- Fine grid refinement ---
    onsets_for_refine = {m: mooring_onsets[m] for m in moorings_used}
    refined = refine_location(lat_loc, lon_loc, onsets_for_refine, c_eff,
                              pair_speeds=pair_speeds)
    if refined is not None:
        lat_loc = refined["lat"]
        lon_loc = refined["lon"]
        residual = refined["residual_s"]
        result["rms_grid"] = refined["rms_grid"]
        result["grid_idx"] = np.unravel_index(
            np.argmin(refined["rms_grid"]), refined["rms_grid"].shape)
        result["_fine_lat_grid"] = refined["fine_lat_grid"]
        result["_fine_lon_grid"] = refined["fine_lon_grid"]

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
            r_jk = locate_one(reduced, dist_km_grids, c_eff, pair_speeds)
            if r_jk:
                jk_lat = float(lat_grid[r_jk["grid_idx"]])
                jk_lon = float(lon_grid[r_jk["grid_idx"]])
                _, _, shift_m = GEOD.inv(lon_loc, lat_loc, jk_lon, jk_lat)
                jk_locs.append(shift_m / 1000.0)
        if jk_locs:
            jackknife_shift_km = max(jk_locs)
            if jackknife_shift_km > JACKKNIFE_SHIFT_THRESHOLD_KM:
                jackknife_stable = False

    # --- Distance from array centroid ---
    centroid_lat = np.mean([MOORINGS[m]["lat"] for m in MOORING_KEYS])
    centroid_lon = np.mean([MOORINGS[m]["lon"] for m in MOORING_KEYS])
    _, _, dist_from_centroid_m = GEOD.inv(
        lon_loc, lat_loc, centroid_lon, centroid_lat)
    dist_from_centroid_km = dist_from_centroid_m / 1000.0

    # --- Quality tier ---
    if n_moorings < 3:
        tier = "D"
    elif dist_from_centroid_km > MAX_DIST_FROM_CENTROID_KM:
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

    # --- Location uncertainty ---
    uncertainty_km = np.nan
    rms_grid = result.get("rms_grid")
    if rms_grid is not None:
        ir, ic = result["grid_idx"]
        ny, nx = rms_grid.shape
        if 1 <= ir < ny - 1 and 1 <= ic < nx - 1:
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
            if eigvals[0] > 0 and eigvals[1] > 0:
                deg2km_lat = 111.0
                deg2km_lon = 111.0 * np.cos(np.radians(lat_loc))
                fine_lat = result.get("_fine_lat_grid")
                if fine_lat is not None:
                    grid_spacing_deg = abs(
                        float(fine_lat[1, 0]) - float(fine_lat[0, 0]))
                else:
                    grid_spacing_deg = 0.01
                dy_km = grid_spacing_deg * deg2km_lat
                dx_km = grid_spacing_deg * deg2km_lon
                rms_min = max(residual, 1e-6)
                semi_y = np.sqrt(rms_min / eigvals[0]) * dy_km
                semi_x = np.sqrt(rms_min / eigvals[1]) * dx_km
                uncertainty_km = round(float(np.sqrt(semi_x * semi_y)), 2)

    return {
        "assoc_id": assoc_dict["assoc_id"],
        "lat": lat_loc,
        "lon": lon_loc,
        "residual_s": round(residual, 3),
        "n_moorings": n_moorings,
        "moorings": ",".join(moorings_used),
        "quality_tier": tier,
        "jackknife_shift_km": round(jackknife_shift_km, 2),
        "jackknife_stable": jackknife_stable,
        "dropped_mooring": dropped_mooring,
        "earliest_utc": assoc_dict["earliest_utc"],
        "detection_band": assoc_dict["detection_band"],
        "uncertainty_km": uncertainty_km,
    }


# ─── Classification for TAPAAs ──────────────────────────────────────────

def classify_tapaas_locations(loc_df, cat_df):
    """Classify TAPAAs-located events using Phase 3 catalogue labels.

    Uses the Phase 3 catalogue (frequency-band reclassification with
    gold-standard review) rather than Phase 1/2 labels.
    """
    phase3_path = DATA_DIR / "phase3_catalogue.parquet"
    if not phase3_path.exists():
        print("  WARNING: phase3_catalogue.parquet not found — "
              "using detection_band as proxy")
        band_map = {"low": "tphase", "mid": "unclassified", "high": "icequake"}
        loc_df["event_class"] = loc_df["detection_band"].map(band_map)
        loc_df["event_class"] = loc_df["event_class"].fillna("unclassified")
        return loc_df

    phase3 = pd.read_parquet(phase3_path)
    # Build event_id → class map from Phase 3
    p3_map = dict(zip(phase3["event_id"], phase3["phase3_class"]))

    # Read TAPAAs associations to get event_ids
    tapaas_df = pd.read_parquet(DATA_DIR / "tapaas_associations_full.parquet")
    tapaas_map = dict(zip(tapaas_df["assoc_id"], tapaas_df["event_ids"]))

    assoc_classes = {}
    for assoc_id in loc_df["assoc_id"]:
        eids_str = tapaas_map.get(assoc_id, "")
        if not eids_str:
            assoc_classes[assoc_id] = "unclassified"
            continue
        eids = eids_str.split(",")
        classes = [p3_map[eid] for eid in eids if eid in p3_map]
        if not classes:
            assoc_classes[assoc_id] = "unclassified"
        elif "tphase" in classes or "seismic" in classes:
            assoc_classes[assoc_id] = "tphase"
        elif "icequake" in classes or "cryogenic" in classes:
            assoc_classes[assoc_id] = "icequake"
        elif "vessel" in classes:
            assoc_classes[assoc_id] = "vessel"
        else:
            assoc_classes[assoc_id] = "unclassified"

    loc_df["event_class"] = loc_df["assoc_id"].map(assoc_classes)
    loc_df["event_class"] = loc_df["event_class"].fillna("unclassified")
    return loc_df


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel grid-search TDOA location for TAPAAs associations")
    parser.add_argument("--workers", type=int,
                        default=max(1, mp.cpu_count() - 1),
                        help="Number of parallel workers (default: ncpu-1)")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Checkpoint every N associations (default: 5000)")
    parser.add_argument("--grid-spacing", type=float, default=0.01,
                        help="Coarse grid spacing in degrees (default: 0.01)")
    parser.add_argument("--no-jackknife", action="store_true",
                        help="Skip jackknife validation (faster)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--skip-postprocess", action="store_true",
                        help="Skip classification/QC (location only)")
    parser.add_argument("--band", type=str, default=None,
                        choices=["low", "mid", "high"],
                        help="Process only one detection band")
    args = parser.parse_args()

    print("=" * 60)
    print("BRAVOSEIS TAPAAs Location — Parallel Grid-Search TDOA")
    print("=" * 60)
    print(f"  Workers: {args.workers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grid spacing: {args.grid_spacing}°")
    print(f"  Jackknife: {'disabled' if args.no_jackknife else 'enabled'}")

    # --- Load data ---
    print("\nLoading data...")
    tapaas_path = DATA_DIR / "tapaas_associations_full.parquet"
    assoc_df = pd.read_parquet(tapaas_path)
    assoc_df["earliest_utc"] = pd.to_datetime(assoc_df["earliest_utc"])
    c_eff, pair_speeds = load_travel_times()

    cat_path = DATA_DIR / "event_catalogue_full.parquet"
    if not cat_path.exists():
        cat_path = DATA_DIR / "event_catalogue.parquet"
        print(f"  NOTE: Using {cat_path.name} (full catalogue not found)")
    cat_df = pd.read_parquet(cat_path)
    cat_df["onset_utc"] = pd.to_datetime(cat_df["onset_utc"])

    print(f"  Event catalogue: {len(cat_df):,} events")
    print(f"  TAPAAs associations: {len(assoc_df):,}")
    print(f"  Effective sound speed: {c_eff:.1f} m/s (mean)")
    print(f"  Per-pair speeds: {len(pair_speeds)//2} pairs")

    # Filter by band if requested
    if args.band:
        assoc_df = assoc_df[assoc_df["detection_band"] == args.band].copy()
        print(f"  Filtered to band={args.band}: {len(assoc_df):,}")

    # All TAPAAs associations have >=3 moorings by construction
    locatable = assoc_df.copy()
    print(f"  Locatable: {len(locatable):,}")

    # --- Resume from checkpoint ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    completed_ids = set()
    if args.resume:
        checkpoint_files = sorted(CHECKPOINT_DIR.glob("batch_*.parquet"))
        for cp in checkpoint_files:
            cp_df = pd.read_parquet(cp)
            completed_ids.update(cp_df["assoc_id"].tolist())
        if completed_ids:
            print(f"  Resuming: {len(completed_ids):,} already completed")
            locatable = locatable[~locatable["assoc_id"].isin(completed_ids)]
            print(f"  Remaining: {len(locatable):,}")

    if len(locatable) == 0:
        print("\nAll associations already processed.")
        if not args.skip_postprocess:
            _assemble_and_postprocess(cat_df, args)
        return

    # Convert to list of dicts for pickling
    assoc_dicts = locatable.to_dict("records")
    total = len(assoc_dicts)
    do_jk = not args.no_jackknife

    # --- Parallel location ---
    print(f"\nLocating {total:,} associations with {args.workers} workers...")
    t_start = time.time()

    pool = mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(
            str(cat_path), None, None, None,
            args.grid_spacing, c_eff, pair_speeds, do_jk,
        ),
    )

    results_buffer = []
    n_located = 0
    n_processed = 0
    batch_num = len(list(CHECKPOINT_DIR.glob("batch_*.parquet")))

    try:
        for result in pool.imap_unordered(_locate_one_association,
                                           assoc_dicts, chunksize=50):
            n_processed += 1
            if result is not None:
                results_buffer.append(result)
                n_located += 1

            # Progress update
            if n_processed % 1000 == 0:
                elapsed = time.time() - t_start
                rate = n_processed / elapsed
                eta_s = (total - n_processed) / rate if rate > 0 else 0
                eta_h = eta_s / 3600
                print(f"  {n_processed:,}/{total:,} "
                      f"({n_located:,} located, "
                      f"{rate:.0f}/s, "
                      f"ETA {eta_h:.1f}h)")

            # Checkpoint
            if len(results_buffer) >= args.batch_size:
                batch_df = pd.DataFrame(results_buffer)
                batch_path = CHECKPOINT_DIR / f"batch_{batch_num:04d}.parquet"
                batch_df.to_parquet(batch_path, index=False)
                print(f"  Checkpoint saved: {batch_path.name} "
                      f"({len(batch_df):,} rows)")
                results_buffer = []
                batch_num += 1

    except KeyboardInterrupt:
        print("\n  Interrupted! Saving remaining results...")
    finally:
        pool.close()
        pool.join()

    # Save any remaining results
    if results_buffer:
        batch_df = pd.DataFrame(results_buffer)
        batch_path = CHECKPOINT_DIR / f"batch_{batch_num:04d}.parquet"
        batch_df.to_parquet(batch_path, index=False)
        print(f"  Final checkpoint: {batch_path.name} ({len(batch_df):,} rows)")

    elapsed = time.time() - t_start
    print(f"\nLocation complete: {n_processed:,} processed, "
          f"{n_located:,} located in {elapsed/3600:.1f}h")

    # --- Assemble and postprocess ---
    if not args.skip_postprocess:
        _assemble_and_postprocess(cat_df, args)


def _assemble_and_postprocess(cat_df, args):
    """Merge checkpoints and run classification + QC."""
    print("\n" + "=" * 60)
    print("Assembling results from checkpoints...")

    checkpoint_files = sorted(CHECKPOINT_DIR.glob("batch_*.parquet"))
    if not checkpoint_files:
        print("  No checkpoint files found!")
        return

    dfs = [pd.read_parquet(f) for f in checkpoint_files]
    loc_df = pd.concat(dfs, ignore_index=True)

    # Deduplicate (in case of overlapping checkpoints)
    loc_df = loc_df.drop_duplicates(subset="assoc_id", keep="last")
    print(f"  Total located: {len(loc_df):,}")

    # --- Quality summary ---
    print("\nQuality tier breakdown:")
    for tier in ["A", "B", "C", "D"]:
        n = (loc_df["quality_tier"] == tier).sum()
        if tier != "D" and n > 0:
            med_r = loc_df[loc_df["quality_tier"] == tier]["residual_s"].median()
            print(f"  {tier}: {n:,} (median residual {med_r:.2f} s)")
        else:
            print(f"  {tier}: {n:,}")

    located = loc_df[loc_df["quality_tier"].isin(["A", "B", "C"])]
    print(f"\nTotal located (A+B+C): {len(located):,}")
    if len(located) > 0:
        print(f"  Median residual: {located['residual_s'].median():.2f} s")

    # --- Classification ---
    print("\nClassifying located events...")
    loc_df = classify_tapaas_locations(loc_df, cat_df)

    class_counts = loc_df[loc_df["quality_tier"].isin(["A", "B", "C"])][
        "event_class"].value_counts()
    print("  Located events by class:")
    for cls, n in class_counts.items():
        print(f"    {cls}: {n:,}")

    # --- Icequake filter ---
    print("\nFiltering icequake locations by distance to coast/ice...")
    try:
        loc_df = filter_icequake_locations(loc_df)
    except Exception as e:
        print(f"  WARNING: Icequake filter failed: {e}")

    # --- Swarm coherence ---
    print("\nRunning swarm coherence QC on T-phases...")
    try:
        loc_df = compute_swarm_coherence(loc_df)
    except Exception as e:
        print(f"  WARNING: Swarm coherence failed: {e}")

    # --- Uncertainty summary ---
    unc = loc_df[loc_df["quality_tier"].isin(["A", "B", "C"])]["uncertainty_km"]
    valid_unc = unc.dropna()
    if len(valid_unc) > 0:
        print(f"\nLocation uncertainty:")
        print(f"  Median: {valid_unc.median():.1f} km")
        for tier in ["A", "B", "C"]:
            tu = loc_df[loc_df["quality_tier"] == tier]["uncertainty_km"].dropna()
            if len(tu) > 0:
                print(f"  Tier {tier}: median {tu.median():.1f} km (n={len(tu):,})")

    # --- Save final ---
    out_path = DATA_DIR / "tapaas_locations.parquet"
    loc_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
