#!/usr/bin/env python3
"""
associate_tapaas.py — TAPAAs-style association for the full BRAVOSEIS dataset.

Implements spatial-pruning association inspired by Raumer et al. (2025):
for each anchor detection, candidate detections on other moorings are
added only if there exists at least one grid cell consistent with ALL
observed TDOAs within tolerance. This rejects physically implausible
combinations before they are ever "located."

Processes each detection band independently (low/mid/high).
Checkpoints every 200K anchors for nohup resilience.

Usage:
    nohup uv run python scripts/associate_tapaas.py > outputs/tapaas_run.log 2>&1 &
    # Resume after interruption:
    nohup uv run python scripts/associate_tapaas.py --resume >> outputs/tapaas_run.log 2>&1 &
"""

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from read_dat import MOORINGS

# ── Paths ───────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "outputs" / "data"
CHECKPOINT_DIR = DATA_DIR / "tapaas_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MOORING_KEYS = sorted(MOORINGS.keys())  # m1..m6
N_MOORINGS = len(MOORING_KEYS)
CHECKPOINT_INTERVAL = 200_000

# ── Grid and TDOA parameters ───────────────────────────────────────────────
GRID_SPACING = 0.01  # degrees
GRID_PAD = 1.0       # degrees beyond mooring extent
DELTA_PICK = 2.0     # pick uncertainty (s) — AIC has ~1s uncertainty, be generous
DELTA_CLOCK = 0.0    # no clock drift correction needed
MIN_ASSOC_SIZE = 3   # minimum moorings for a valid association


def load_travel_times():
    with open(DATA_DIR / "travel_times.json") as f:
        data = json.load(f)
    c_eff = data["effective_speed_mean_ms"]
    pair_speeds = {}
    pair_max_dt = {}
    for pk, info in data["pairs"].items():
        mi, mj = pk.split("-")
        pair_speeds[(mi, mj)] = info["c_eff_ms"]
        pair_speeds[(mj, mi)] = info["c_eff_ms"]
        pair_max_dt[(mi, mj)] = info["max_travel_time_s"]
        pair_max_dt[(mj, mi)] = info["max_travel_time_s"]
    global_max = data["global_max_travel_time_s"]
    return c_eff, pair_speeds, pair_max_dt, global_max


def build_tdoa_grids(pair_speeds, c_eff):
    """Pre-compute expected TDOA at each grid cell for each mooring pair.

    Returns
    -------
    tdoa_grids : dict
        {(mi, mj): 2D array of expected TDOA in seconds}
    delta_geom : dict
        {(mi, mj): 2D array of geometric tolerance per cell}
    lat_grid, lon_grid : 2D arrays
    """
    lats_m = [MOORINGS[m]["lat"] for m in MOORING_KEYS]
    lons_m = [MOORINGS[m]["lon"] for m in MOORING_KEYS]

    lat_min = min(lats_m) - GRID_PAD
    lat_max = max(lats_m) + GRID_PAD
    lon_min = min(lons_m) - GRID_PAD
    lon_max = max(lons_m) + GRID_PAD

    lats = np.arange(lat_min, lat_max, GRID_SPACING)
    lons = np.arange(lon_min, lon_max, GRID_SPACING)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Flat-earth distances (km) — valid at regional scale near -62.5°
    deg2km_lat = 111.0
    deg2km_lon = 111.0 * np.cos(np.radians(-62.5))

    dist_km = {}
    for mk in MOORING_KEYS:
        dlat = (lat_grid - MOORINGS[mk]["lat"]) * deg2km_lat
        dlon = (lon_grid - MOORINGS[mk]["lon"]) * deg2km_lon
        dist_km[mk] = np.sqrt(dlat**2 + dlon**2)

    # TDOA grids for each pair
    tdoa_grids = {}
    delta_geom = {}
    for mi, mj in combinations(MOORING_KEYS, 2):
        c = pair_speeds.get((mi, mj), c_eff)
        # Expected TDOA = (dist_j - dist_i) * 1000 / c
        tdoa = (dist_km[mj] - dist_km[mi]) * 1000.0 / c
        tdoa_grids[(mi, mj)] = tdoa
        tdoa_grids[(mj, mi)] = -tdoa  # antisymmetric

        # Geometric tolerance: max TDOA variation within one cell
        # Approximate as gradient * cell_size
        # For a 0.01° cell at ~62.5°S: ~0.52 km E-W, ~1.11 km N-S
        # Conservative estimate: use half-diagonal of cell
        cell_diag_km = np.sqrt((GRID_SPACING * deg2km_lat)**2 +
                               (GRID_SPACING * deg2km_lon)**2) / 2
        # TDOA gradient magnitude ≈ 1000/c * |grad(dist_j - dist_i)|
        # Conservative: use cell_diag_km * 1000 / c as upper bound
        dg = cell_diag_km * 1000.0 / c
        delta_geom[(mi, mj)] = dg
        delta_geom[(mj, mi)] = dg

    print(f"  Grid: {lat_grid.shape[0]} × {lat_grid.shape[1]} = "
          f"{lat_grid.size:,} cells")
    print(f"  TDOA grids: {len(tdoa_grids)} (including antisymmetric)")
    print(f"  Geometric tolerance: ~{cell_diag_km * 1000.0 / c_eff:.2f} s per cell")

    return tdoa_grids, delta_geom, lat_grid, lon_grid


def find_valid_cells(valid_mask, obs_tdoa, pair_key, tdoa_grids, delta_geom):
    """Intersect valid_mask with cells consistent with an observed TDOA.

    Returns updated valid_mask (boolean 2D array).
    """
    expected = tdoa_grids[pair_key]
    tol = delta_geom[pair_key] + 2 * DELTA_PICK + 2 * DELTA_CLOCK
    consistent = np.abs(expected - obs_tdoa) <= tol
    return valid_mask & consistent


def locate_from_valid_cells(valid_mask, mooring_times, tdoa_grids,
                             pair_speeds, c_eff, lat_grid, lon_grid):
    """Find best location among valid cells using RMS residual.

    Returns dict with lat, lon, residual_s or None.
    """
    if not valid_mask.any():
        return None

    moorings = sorted(mooring_times.keys())
    pairs = list(combinations(moorings, 2))
    obs_tdoa = []
    for mi, mj in pairs:
        obs_tdoa.append(mooring_times[mj] - mooring_times[mi])
    obs_tdoa = np.array(obs_tdoa)

    # RMS residual only on valid cells
    valid_idx = np.where(valid_mask)
    n_valid = len(valid_idx[0])

    rms_vals = np.zeros(n_valid)
    for k, (mi, mj) in enumerate(pairs):
        expected = tdoa_grids[(mi, mj)][valid_idx]
        rms_vals += (expected - obs_tdoa[k]) ** 2
    rms_vals = np.sqrt(rms_vals / len(pairs))

    best = np.argmin(rms_vals)
    best_row = valid_idx[0][best]
    best_col = valid_idx[1][best]

    return {
        "lat": float(lat_grid[best_row, best_col]),
        "lon": float(lon_grid[best_row, best_col]),
        "residual_s": float(rms_vals[best]),
    }


def process_band(band_name, cat_band, tdoa_grids, delta_geom,
                  pair_max_dt, global_max, pair_speeds, c_eff,
                  lat_grid, lon_grid, resume_from=0, existing_assocs=None):
    """Run TAPAAs association for one detection band."""

    onset_s = cat_band["onset_utc"].values.astype("datetime64[ms]").astype(np.int64) / 1000
    moorings_arr = cat_band["mooring"].values
    snr_arr = cat_band["snr"].values
    event_ids = cat_band["event_id"].values
    n = len(cat_band)

    assigned = np.zeros(n, dtype=bool)
    associations = existing_assocs if existing_assocs else []

    # Mark existing assignments
    if existing_assocs:
        eid_set = set()
        for a in existing_assocs:
            eid_set.update(a["event_ids"].split(","))
        eid_to_idx = {eid: i for i, eid in enumerate(event_ids)}
        for eid in eid_set:
            if eid in eid_to_idx:
                assigned[eid_to_idx[eid]] = True
        print(f"  Resumed: {len(existing_assocs):,} existing, "
              f"{assigned.sum():,} assigned")

    t0 = time.time()
    last_ckpt = resume_from
    n_tried = 0
    n_pruned = 0

    for i in range(resume_from, n):
        if assigned[i]:
            continue

        anchor_mk = moorings_arr[i]
        anchor_t = onset_s[i]

        # Find all candidates within global_max forward window
        j_end = np.searchsorted(onset_s, anchor_t + global_max, side="right")

        # Gather candidates per mooring (excluding anchor's mooring)
        # Each candidate: (index, mooring, onset_time, snr)
        candidates_by_mooring = {}
        for j in range(i + 1, j_end):
            if assigned[j]:
                continue
            mk_j = moorings_arr[j]
            if mk_j == anchor_mk:
                continue
            dt = onset_s[j] - anchor_t
            max_dt = pair_max_dt.get((anchor_mk, mk_j), global_max)
            if dt > max_dt:
                continue
            if mk_j not in candidates_by_mooring:
                candidates_by_mooring[mk_j] = []
            candidates_by_mooring[mk_j].append(j)

        if len(candidates_by_mooring) < (MIN_ASSOC_SIZE - 1):
            continue

        n_tried += 1

        # Build association greedily with spatial pruning:
        # Start with anchor, add best candidate per mooring if spatially consistent
        best_members = {anchor_mk: i}
        valid_mask = np.ones(lat_grid.shape, dtype=bool)

        # Sort moorings by number of candidates (fewest first = most pruning)
        mooring_order = sorted(candidates_by_mooring.keys(),
                               key=lambda m: len(candidates_by_mooring[m]))

        for mk in mooring_order:
            cands = candidates_by_mooring[mk]

            best_j = None
            best_snr = -1
            best_valid = None

            for j in cands:
                # Check spatial consistency with ALL existing members
                obs_tdoa_j = onset_s[j] - anchor_t
                pair_key = (anchor_mk, mk)

                # Quick check against anchor first
                test_mask = find_valid_cells(
                    valid_mask, obs_tdoa_j, pair_key,
                    tdoa_grids, delta_geom)

                # Also check against all other already-selected members
                ok = test_mask.any()
                if ok:
                    for mk_existing, idx_existing in best_members.items():
                        if mk_existing == anchor_mk:
                            continue
                        obs_tdoa_existing = onset_s[j] - onset_s[idx_existing]
                        pk = (mk_existing, mk) if mk_existing < mk else (mk, mk_existing)
                        if mk_existing < mk:
                            obs_val = obs_tdoa_existing
                        else:
                            obs_val = -obs_tdoa_existing
                        test_mask = find_valid_cells(
                            test_mask, obs_val, pk,
                            tdoa_grids, delta_geom)
                        if not test_mask.any():
                            ok = False
                            break

                if ok and snr_arr[j] > best_snr:
                    best_j = j
                    best_snr = snr_arr[j]
                    best_valid = test_mask

            if best_j is not None:
                best_members[mk] = best_j
                valid_mask = best_valid
            # else: this mooring has no spatially-consistent candidate

        if len(best_members) < MIN_ASSOC_SIZE:
            n_pruned += 1
            continue

        # Mark assigned
        for idx in best_members.values():
            assigned[idx] = True

        # Locate using valid cells
        mooring_times = {mk: onset_s[idx] for mk, idx in best_members.items()}
        loc = locate_from_valid_cells(
            valid_mask, mooring_times, tdoa_grids,
            pair_speeds, c_eff, lat_grid, lon_grid)

        member_eids = [event_ids[idx] for idx in best_members.values()]
        member_onsets = [cat_band.iloc[idx]["onset_utc"] for idx in best_members.values()]
        earliest = min(member_onsets)
        latest = max(member_onsets)

        assoc = {
            "assoc_id": f"T{band_name[0].upper()}{len(associations):07d}",
            "n_moorings": len(best_members),
            "moorings": ",".join(sorted(best_members.keys())),
            "n_events": len(best_members),
            "event_ids": ",".join(member_eids),
            "earliest_utc": earliest,
            "latest_utc": latest,
            "dt_s": round((latest - earliest).total_seconds(), 3),
            "detection_band": band_name,
        }
        if loc:
            assoc["lat"] = loc["lat"]
            assoc["lon"] = loc["lon"]
            assoc["residual_s"] = loc["residual_s"]
        else:
            assoc["lat"] = np.nan
            assoc["lon"] = np.nan
            assoc["residual_s"] = np.nan

        associations.append(assoc)

        # Checkpoint
        if (i - last_ckpt) >= CHECKPOINT_INTERVAL:
            elapsed = time.time() - t0
            rate = max(i - resume_from, 1) / elapsed
            eta_min = (n - i) / rate / 60 if rate > 0 else 0
            print(f"  [{band_name}] {i:,}/{n:,} ({100*i/n:.1f}%) | "
                  f"{len(associations):,} assocs | "
                  f"{n_pruned:,} pruned | "
                  f"{elapsed/60:.1f}m elapsed | ~{eta_min:.0f}m remaining",
                  flush=True)

            # Save checkpoint
            ckpt_data = {
                "band": band_name,
                "resume_from": i,
                "n_associations": len(associations),
            }
            ckpt_path = CHECKPOINT_DIR / f"tapaas_{band_name}_checkpoint.json"
            with open(ckpt_path, "w") as f:
                json.dump(ckpt_data, f)

            assoc_df = pd.DataFrame(associations)
            assoc_df.to_parquet(
                CHECKPOINT_DIR / f"tapaas_{band_name}_partial.parquet",
                index=False)
            last_ckpt = i

    elapsed = time.time() - t0
    print(f"  [{band_name}] Done: {len(associations):,} associations, "
          f"{n_pruned:,} pruned, {elapsed/60:.1f} min", flush=True)

    return pd.DataFrame(associations)


def main():
    parser = argparse.ArgumentParser(
        description="TAPAAs spatial-pruning association for BRAVOSEIS")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--band", choices=["low", "mid", "high"],
                        help="Process only one band (default: all)")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("TAPAAs SPATIAL-PRUNING ASSOCIATION", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)

    # Load travel times
    c_eff, pair_speeds, pair_max_dt, global_max = load_travel_times()
    print(f"Sound speed: {c_eff:.1f} m/s, global max dt: {global_max:.1f} s")
    print(f"Pick tolerance: {DELTA_PICK} s, clock tolerance: {DELTA_CLOCK} s")

    # Build TDOA grids
    print("\nPre-computing TDOA grids...")
    tdoa_grids, delta_geom, lat_grid, lon_grid = build_tdoa_grids(
        pair_speeds, c_eff)

    # Load catalogue
    print("\nLoading full catalogue...")
    cat = pd.read_parquet(DATA_DIR / "event_catalogue_full.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])
    cat = cat.sort_values("onset_utc").reset_index(drop=True)
    print(f"  {len(cat):,} events total")

    bands = [args.band] if args.band else ["low", "mid", "high"]
    all_results = []

    for band in bands:
        print(f"\n{'=' * 50}", flush=True)
        print(f"Processing band: {band}", flush=True)
        print(f"{'=' * 50}", flush=True)

        cat_band = cat[cat["detection_band"] == band].copy()
        cat_band = cat_band.sort_values("onset_utc").reset_index(drop=True)
        print(f"  {len(cat_band):,} events in {band} band")

        # Resume logic
        resume_from = 0
        existing_assocs = None
        if args.resume:
            ckpt_path = CHECKPOINT_DIR / f"tapaas_{band}_checkpoint.json"
            partial_path = CHECKPOINT_DIR / f"tapaas_{band}_partial.parquet"
            if ckpt_path.exists() and partial_path.exists():
                with open(ckpt_path) as f:
                    ckpt = json.load(f)
                resume_from = ckpt["resume_from"]
                partial_df = pd.read_parquet(partial_path)
                existing_assocs = partial_df.to_dict("records")
                print(f"  Resuming from event {resume_from:,} "
                      f"({len(existing_assocs):,} existing)")

        band_result = process_band(
            band, cat_band, tdoa_grids, delta_geom,
            pair_max_dt, global_max, pair_speeds, c_eff,
            lat_grid, lon_grid,
            resume_from=resume_from,
            existing_assocs=existing_assocs)

        all_results.append(band_result)

        # Save per-band result
        band_path = DATA_DIR / f"tapaas_associations_{band}.parquet"
        band_result.to_parquet(band_path, index=False)
        print(f"  Saved: {band_path}")

    # Combine all bands
    if len(all_results) == 3:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values("earliest_utc").reset_index(drop=True)
        # Re-number assoc IDs
        combined["assoc_id"] = [f"T{i:07d}" for i in range(len(combined))]

        out_path = DATA_DIR / "tapaas_associations_full.parquet"
        combined.to_parquet(out_path, index=False)

        print(f"\n{'=' * 70}", flush=True)
        print(f"COMBINED RESULTS", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"Total associations: {len(combined):,}")
        print(f"\nBy band:")
        for band, cnt in combined["detection_band"].value_counts().items():
            print(f"  {band}: {cnt:,}")
        print(f"\nBy mooring count:")
        for nm, cnt in combined["n_moorings"].value_counts().sort_index().items():
            print(f"  {nm} moorings: {cnt:,}")
        if "residual_s" in combined.columns:
            valid = combined["residual_s"].notna()
            print(f"\nResidual (located): median={combined.loc[valid, 'residual_s'].median():.3f} s, "
                  f"mean={combined.loc[valid, 'residual_s'].mean():.3f} s")
        print(f"\nSaved: {out_path}")

    # Clean up checkpoints
    for band in bands:
        for f in CHECKPOINT_DIR.glob(f"tapaas_{band}_*"):
            f.unlink()

    print(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
