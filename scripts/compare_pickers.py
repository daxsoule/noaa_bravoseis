#!/usr/bin/env python3
"""
compare_pickers.py — Compare PhaseNet pretrained models for BRAVOSEIS
land station phase picking. Tests stead, instance, ethz, geofon weights.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import seisbench
seisbench.use_backup_repository()
import seisbench.models as sbm
from obspy import read
from pyproj import Geod

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "outputs" / "data"
WAVE_DIR = Path("/home/jovyan/my_data/bravoseis/earthquakes/5m_waveforms")
GEOD = Geod(ellps="WGS84")

GT_IDS = ["T0229023", "T0230172", "T0230339", "T0397542", "T0765914",
          "T0901212", "T1008226", "T1292690", "T1298696", "T1299585",
          "T1320013", "T1330783"]

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

VP, VS = 6.0, 3.47


def pick_event(model, mseed_path, p_thresh=0.2, s_thresh=0.2):
    """Run PhaseNet on one event, return list of pick dicts."""
    st = read(str(mseed_path))
    result = model.classify(st, P_threshold=p_thresh, S_threshold=s_thresh)
    picks = []
    for pick in result.picks:
        station = pick.trace_id.split(".")[1]
        if station not in STATIONS:
            continue
        picks.append({
            "assoc_id": mseed_path.stem,
            "station": station,
            "phase": pick.phase,
            "pick_time": str(pick.peak_time),
            "probability": float(pick.peak_value),
        })
    return picks


def locate_gridsearch(picks_list, center_lat, center_lon):
    """Quick grid search location. Returns RMS or None."""
    df = pd.DataFrame(picks_list)
    df["pick_time"] = pd.to_datetime(df["pick_time"])
    best = df.sort_values("probability", ascending=False).drop_duplicates(
        ["station", "phase"])

    if len(best) < 3:
        return None

    stations = best["station"].values
    phases = best["phase"].values
    times = best["pick_time"].values
    probs = best["probability"].values

    ref_time = times.min()
    obs_times = (times - ref_time) / np.timedelta64(1, "s")
    sta_lats = np.array([STATIONS[s][0] for s in stations])
    sta_lons = np.array([STATIONS[s][1] for s in stations])
    vels = np.array([VP if p == "P" else VS for p in phases])
    weights = probs / probs.sum()
    n = len(best)

    def _search(lat_min, lat_max, lon_min, lon_max, spacing):
        best_rms = np.inf
        best_loc = (center_lat, center_lon)
        for lat in np.arange(lat_min, lat_max + spacing, spacing):
            for lon in np.arange(lon_min, lon_max + spacing, spacing):
                pred_tt = np.zeros(n)
                for k in range(n):
                    _, _, d = GEOD.inv(lon, lat, sta_lons[k], sta_lats[k])
                    dist = d / 1000.0
                    pred_tt[k] = np.sqrt(dist**2 + 100) / vels[k]
                resid_no_t0 = obs_times - pred_tt
                t0 = np.average(resid_no_t0, weights=weights)
                resid = obs_times - (t0 + pred_tt)
                wrms = np.sqrt(np.average(resid**2, weights=weights))
                if wrms < best_rms:
                    best_rms = wrms
                    best_loc = (lat, lon)
        return best_loc, best_rms

    # Coarse
    (lat1, lon1), rms1 = _search(
        center_lat - 2, center_lat + 2,
        center_lon - 2, center_lon + 2, 0.02)
    # Fine
    (lat2, lon2), rms2 = _search(
        lat1 - 0.1, lat1 + 0.1,
        lon1 - 0.1, lon1 + 0.1, 0.002)
    return rms2


def main():
    print("=" * 60)
    print("PhaseNet Model Comparison")
    print("=" * 60)

    hydro = pd.read_parquet(DATA_DIR / "tapaas_locations.parquet")
    model_names = ["stead", "instance", "ethz", "geofon"]

    # Get random sample of 200 events for yield test
    all_files = sorted(WAVE_DIR.glob("*.mseed"))
    np.random.seed(42)
    sample_idx = np.random.choice(len(all_files), 200, replace=False)
    sample_files = [all_files[i] for i in sample_idx]

    results = {}

    for mname in model_names:
        print(f"\n--- Testing: PhaseNet({mname}) ---")
        try:
            model = sbm.PhaseNet.from_pretrained(mname)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        # Pick GT events
        gt_picks_all = []
        gt_rms = []
        for aid in GT_IDS:
            fpath = WAVE_DIR / f"{aid}.mseed"
            if not fpath.exists():
                gt_rms.append(None)
                continue
            picks = pick_event(model, fpath)
            gt_picks_all.extend(picks)

            # Locate
            hloc = hydro[hydro.assoc_id == aid]
            if len(hloc) and len(picks) >= 3:
                rms = locate_gridsearch(picks, hloc.iloc[0].lat, hloc.iloc[0].lon)
                gt_rms.append(rms)
            else:
                gt_rms.append(None)

        n_good = sum(1 for r in gt_rms if r is not None and r < 1.0)
        n_fair = sum(1 for r in gt_rms if r is not None and r < 5.0)
        n_locatable = sum(1 for r in gt_rms if r is not None)

        # Unique stations per GT event
        gt_df = pd.DataFrame(gt_picks_all) if gt_picks_all else pd.DataFrame()
        gt_multi = 0
        if len(gt_df):
            sta_per_ev = gt_df.groupby("assoc_id")["station"].nunique()
            gt_multi = (sta_per_ev >= 3).sum()

        # Pick yield on sample
        n_with_picks = 0
        n_multi_sta = 0
        total_sample_picks = 0
        for fpath in sample_files:
            picks = pick_event(model, fpath)
            total_sample_picks += len(picks)
            if picks:
                n_with_picks += 1
                stas = set(p["station"] for p in picks)
                if len(stas) >= 2:
                    n_multi_sta += 1

        results[mname] = {
            "gt_picks": len(gt_picks_all),
            "gt_locatable": n_locatable,
            "gt_rms_lt1": n_good,
            "gt_rms_lt5": n_fair,
            "gt_multi_sta": gt_multi,
            "sample_yield": n_with_picks / len(sample_files) * 100,
            "sample_multi": n_multi_sta,
            "sample_picks": total_sample_picks,
        }

        print(f"  GT: {len(gt_picks_all)} picks, {n_locatable} locatable, "
              f"{n_good} RMS<1s, {n_fair} RMS<5s")
        print(f"  Sample: {n_with_picks}/200 ({n_with_picks/2:.0f}%) with picks, "
              f"{n_multi_sta} multi-station")

    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':>10s}  {'GT picks':>9s}  {'Locatable':>9s}  {'RMS<1s':>6s}  "
          f"{'RMS<5s':>6s}  {'Yield%':>6s}  {'Multi':>5s}")
    print("-" * 60)
    for mname, r in results.items():
        print(f"{mname:>10s}  {r['gt_picks']:9d}  {r['gt_locatable']:9d}  "
              f"{r['gt_rms_lt1']:6d}  {r['gt_rms_lt5']:6d}  "
              f"{r['sample_yield']:6.1f}  {r['sample_multi']:5d}")

    # Select best model
    best_model = max(results, key=lambda m: (
        results[m]["gt_rms_lt1"], results[m]["gt_rms_lt5"],
        results[m]["sample_yield"]))
    print(f"\nBest model: {best_model}")

    if best_model == "stead":
        # Just copy existing picks
        import shutil
        src = DATA_DIR / "land_station_picks.parquet"
        dst = DATA_DIR / "land_station_picks_best.parquet"
        shutil.copy2(src, dst)
        print(f"  Copied existing stead picks to {dst}")
    else:
        # Re-pick all events with best model
        print(f"  Re-picking all events with {best_model}...")
        model = sbm.PhaseNet.from_pretrained(best_model)
        all_picks = []
        for i, fpath in enumerate(all_files):
            picks = pick_event(model, fpath)
            all_picks.extend(picks)
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(all_files)}...")
        df = pd.DataFrame(all_picks)
        df["pick_time"] = pd.to_datetime(df["pick_time"])
        df["station_lat"] = df["station"].map(lambda s: STATIONS[s][0])
        df["station_lon"] = df["station"].map(lambda s: STATIONS[s][1])
        df["network"] = df["station"].apply(
            lambda s: "AI" if s in ("JUBA", "ESPZ") else "5M")
        df["trace_id"] = df.apply(
            lambda r: f"{r['network']}.{r['station']}..HHZ", axis=1)
        df.to_parquet(DATA_DIR / "land_station_picks_best.parquet", index=False)
        print(f"  Saved {len(df)} picks to land_station_picks_best.parquet")

    print("\nDone.")


if __name__ == "__main__":
    main()
