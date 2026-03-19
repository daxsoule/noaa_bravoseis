#!/usr/bin/env python3
"""
calibrate_mooring_bias.py — Correct hydroacoustic locations using
mooring-combination-dependent bias from GT seismic locations.

For mooring combos with a GT event, applies the measured correction directly.
For others, transfers the correction from the most similar GT combo (Jaccard),
attenuated by sqrt(similarity).

Validates against S-P distance constraints.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Geod

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "outputs" / "data"
GEOD = Geod(ellps="WGS84")

STATIONS = {
    "AST": (-63.3273, -58.7027), "BYE": (-62.6665, -61.0992),
    "DCP": (-62.9775, -60.6782), "ERJ": (-62.02436, -57.64911),
    "FER": (-62.08976, -58.40655), "FRE": (-62.2068, -58.9607),
    "GUR": (-62.30753, -59.19597), "HMI": (-62.5958, -59.90387),
    "LVN": (-62.6627, -60.3875), "OHI": (-63.3221, -57.8973),
    "PEN": (-62.09932, -57.93673), "ROB": (-62.37935, -59.70353),
    "SNW": (-62.72787, -61.2003), "TOW": (-63.5921, -59.7828),
    "JUBA": (-62.237301, -58.662701), "ESPZ": (-63.398102, -56.996399),
}


def geodesic_dist_km(lat1, lon1, lat2, lon2):
    _, _, d = GEOD.inv(lon1, lat1, lon2, lat2)
    return d / 1000.0


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def main():
    print("=" * 60)
    print("Mooring-Combination Bias Calibration")
    print("=" * 60)

    # Load data
    seis = pd.read_parquet(DATA_DIR / "seismic_locations.parquet")
    resid = pd.read_parquet(DATA_DIR / "location_residuals.parquet")
    hydro = pd.read_parquet(DATA_DIR / "tapaas_locations.parquet")
    picks = pd.read_parquet(DATA_DIR / "land_station_picks.parquet")

    # Filter to GOOD GT events (RMS < 1s)
    good_ids = seis[seis["rms_residual_s"] < 1.0]["assoc_id"].values
    gt_resid = resid[resid["assoc_id"].isin(good_ids)].copy()
    print(f"\nGT events (RMS < 1s): {len(gt_resid)}")

    # Get mooring combination for each GT event
    gt_resid = gt_resid.merge(hydro[["assoc_id", "moorings"]], on="assoc_id")
    print("\nGT corrections:")
    gt_corrections = {}
    for _, row in gt_resid.iterrows():
        combo = row["moorings"]
        # correction = seismic - hydro (subtract from hydro to get seismic)
        dlat = row["lat_seis"] - row["lat_hydro"]
        dlon = row["lon_seis"] - row["lon_hydro"]
        gt_corrections[combo] = (dlat, dlon)
        print(f"  {row['assoc_id']}  {combo:20s}  dlat={dlat:+.4f}°  dlon={dlon:+.4f}°  "
              f"offset={row['offset_km']:.1f}km")

    # All unique mooring combos in catalogue
    all_combos = hydro["moorings"].unique()
    print(f"\nUnique mooring combinations in catalogue: {len(all_combos)}")
    print(f"GT covers {len(gt_corrections)}/{len(all_combos)} combos directly")

    # Build correction lookup for all combos
    correction_table = {}
    for combo in all_combos:
        if combo in gt_corrections:
            correction_table[combo] = {
                "dlat": gt_corrections[combo][0],
                "dlon": gt_corrections[combo][1],
                "method": "direct",
                "similarity": 1.0,
            }
        else:
            # Find most similar GT combo
            combo_set = set(combo.split(","))
            best_sim = 0.0
            best_gt = None
            for gt_combo in gt_corrections:
                gt_set = set(gt_combo.split(","))
                sim = jaccard(combo_set, gt_set)
                if sim > best_sim:
                    best_sim = sim
                    best_gt = gt_combo

            if best_gt and best_sim > 0:
                atten = np.sqrt(best_sim)
                correction_table[combo] = {
                    "dlat": gt_corrections[best_gt][0] * atten,
                    "dlon": gt_corrections[best_gt][1] * atten,
                    "method": f"transfer({best_gt})",
                    "similarity": best_sim,
                }
            else:
                # No overlap — use global mean
                mean_dlat = np.mean([c[0] for c in gt_corrections.values()])
                mean_dlon = np.mean([c[1] for c in gt_corrections.values()])
                correction_table[combo] = {
                    "dlat": mean_dlat * 0.5,  # attenuate global mean
                    "dlon": mean_dlon * 0.5,
                    "method": "global_mean",
                    "similarity": 0.0,
                }

    # Apply corrections
    print("\n--- Applying corrections ---")
    corrected = hydro.copy()
    corr_dlat = np.zeros(len(corrected))
    corr_dlon = np.zeros(len(corrected))
    methods = []

    for i, (_, row) in enumerate(corrected.iterrows()):
        ct = correction_table[row["moorings"]]
        corr_dlat[i] = ct["dlat"]
        corr_dlon[i] = ct["dlon"]
        methods.append(ct["method"])

    corrected["lat_corrected"] = corrected["lat"] + corr_dlat
    corrected["lon_corrected"] = corrected["lon"] + corr_dlon
    corrected["correction_method"] = methods
    corrected["correction_km"] = np.sqrt(
        (corr_dlat * 111.0)**2 +
        (corr_dlon * 111.0 * np.cos(np.radians(corrected["lat"])))**2
    )

    # Summary
    method_counts = corrected["correction_method"].apply(
        lambda x: "direct" if x == "direct" else
        ("transfer" if x.startswith("transfer") else "global_mean")
    ).value_counts()
    print(f"  Direct GT correction: {method_counts.get('direct', 0):,} events")
    print(f"  Transferred correction: {method_counts.get('transfer', 0):,} events")
    print(f"  Global mean fallback: {method_counts.get('global_mean', 0):,} events")
    print(f"  Mean correction: {corrected['correction_km'].mean():.1f} km")

    # Save
    corrected.to_parquet(DATA_DIR / "catalogue_mooring_corrected.parquet", index=False)
    print(f"\n  Saved: {DATA_DIR / 'catalogue_mooring_corrected.parquet'}")

    # ---- Validate against S-P distances ----
    print("\n--- S-P Distance Validation ---")
    vp, vs = 6.0, 3.47

    # Find S-P pairs
    sp_pairs = picks.groupby(["assoc_id", "station"])["phase"].apply(set)
    sp_pairs = sp_pairs[sp_pairs.apply(lambda x: "P" in x and "S" in x)]
    sp_events = sp_pairs.reset_index()[["assoc_id", "station"]]

    results = []
    for _, row in sp_events.iterrows():
        aid, sta = row["assoc_id"], row["station"]
        ev_picks = picks[(picks.assoc_id == aid) & (picks.station == sta)]
        p = ev_picks[ev_picks.phase == "P"].sort_values("probability", ascending=False).iloc[0]
        s = ev_picks[ev_picks.phase == "S"].sort_values("probability", ascending=False).iloc[0]
        sp_time = (pd.to_datetime(s.pick_time) - pd.to_datetime(p.pick_time)).total_seconds()
        if sp_time <= 0:
            continue
        sp_dist = sp_time * vp * vs / (vp - vs)

        ev = corrected[corrected.assoc_id == aid]
        if len(ev) == 0:
            continue
        ev = ev.iloc[0]
        sta_lat, sta_lon = STATIONS[sta]

        orig_dist = geodesic_dist_km(ev["lat"], ev["lon"], sta_lat, sta_lon)
        corr_dist = geodesic_dist_km(ev["lat_corrected"], ev["lon_corrected"], sta_lat, sta_lon)

        results.append({
            "assoc_id": aid, "station": sta,
            "sp_dist_km": sp_dist,
            "original_error_km": abs(orig_dist - sp_dist),
            "mooring_corr_error_km": abs(corr_dist - sp_dist),
        })

    val = pd.DataFrame(results)
    if len(val) > 0:
        print(f"  S-P validation pairs: {len(val)}")
        print(f"  Original error:         mean={val['original_error_km'].mean():.1f} km  "
              f"median={val['original_error_km'].median():.1f} km")
        print(f"  Mooring-corrected error: mean={val['mooring_corr_error_km'].mean():.1f} km  "
              f"median={val['mooring_corr_error_km'].median():.1f} km")
        improved = (val["mooring_corr_error_km"] < val["original_error_km"]).sum()
        print(f"  Improved: {improved}/{len(val)} ({100*improved/len(val):.0f}%)")

        # Compare with spatial IDW correction if available
        try:
            spatial = pd.read_parquet(DATA_DIR / "catalogue_corrected.parquet")
            sp_results = []
            for _, row in sp_events.iterrows():
                aid, sta = row["assoc_id"], row["station"]
                ev_picks = picks[(picks.assoc_id == aid) & (picks.station == sta)]
                p = ev_picks[ev_picks.phase == "P"].sort_values("probability", ascending=False).iloc[0]
                s = ev_picks[ev_picks.phase == "S"].sort_values("probability", ascending=False).iloc[0]
                sp_time = (pd.to_datetime(s.pick_time) - pd.to_datetime(p.pick_time)).total_seconds()
                if sp_time <= 0:
                    continue
                sp_dist = sp_time * vp * vs / (vp - vs)
                ev = spatial[spatial.assoc_id == aid]
                if len(ev) == 0:
                    continue
                ev = ev.iloc[0]
                sta_lat, sta_lon = STATIONS[sta]
                idw_dist = geodesic_dist_km(ev["lat_corrected"], ev["lon_corrected"], sta_lat, sta_lon)
                sp_results.append(abs(idw_dist - sp_dist))
            if sp_results:
                print(f"  Spatial IDW error:       mean={np.mean(sp_results):.1f} km  "
                      f"median={np.median(sp_results):.1f} km")
        except Exception:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
