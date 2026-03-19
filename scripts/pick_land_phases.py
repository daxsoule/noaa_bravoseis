#!/usr/bin/env python3
"""
pick_land_phases.py — Pick P and S wave arrivals on BRAVOSEIS 5M land stations
using PhaseNet (via SeisBench).

Processes all downloaded 60-second waveform snippets, runs PhaseNet for
phase detection, and outputs a pick catalogue with arrival times, probabilities,
and station metadata.

Usage:
    uv run python pick_land_phases.py                     # all events
    uv run python pick_land_phases.py --resume            # skip already processed
    uv run python pick_land_phases.py --p-threshold 0.2   # lower P threshold

Output:
    outputs/data/land_station_picks.parquet
    outputs/data/land_station_picks_summary.parquet  (per-event summary)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import seisbench
import seisbench.models as sbm
from obspy import read

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "outputs" / "data"
WAVE_DIR = Path("/home/jovyan/my_data/bravoseis/earthquakes/5m_waveforms")
LOG_PATH = WAVE_DIR / "download_log.csv"
OUT_PICKS = DATA_DIR / "land_station_picks.parquet"
OUT_SUMMARY = DATA_DIR / "land_station_picks_summary.parquet"

# === Station coordinates ===
STATION_COORDS = {
    # 5M network
    "AST": (-63.3273, -58.7027),
    "BYE": (-62.6665, -61.0992),
    "DCP": (-62.9775, -60.6782),
    "ERJ": (-62.02436, -57.64911),
    "FER": (-62.08976, -58.40655),
    "FRE": (-62.2068, -58.9607),
    "GUR": (-62.30753, -59.19597),
    "HMI": (-62.5958, -59.90387),
    "LVN": (-62.6627, -60.3875),
    "OHI": (-63.3221, -57.8973),
    "PEN": (-62.09932, -57.93673),
    "ROB": (-62.37935, -59.70353),
    "SNW": (-62.72787, -61.2003),
    "TOW": (-63.5921, -59.7828),
    # Permanent stations
    "JUBA": (-62.237301, -58.662701),
    "ESPZ": (-63.398102, -56.996399),
}


def pick_event(model, mseed_path, p_thresh, s_thresh):
    """Run PhaseNet on a single event file.

    Returns list of pick dicts.
    """
    st = read(str(mseed_path))
    assoc_id = mseed_path.stem

    # Run PhaseNet classify
    result = model.classify(
        st,
        P_threshold=p_thresh,
        S_threshold=s_thresh,
    )

    picks = []
    for pick in result.picks:
        station = pick.trace_id.split(".")[1]
        lat, lon = STATION_COORDS.get(station, (np.nan, np.nan))
        picks.append({
            "assoc_id": assoc_id,
            "trace_id": pick.trace_id,
            "station": station,
            "network": pick.trace_id.split(".")[0],
            "phase": pick.phase,
            "pick_time": str(pick.peak_time),
            "probability": round(float(pick.peak_value), 4),
            "station_lat": lat,
            "station_lon": lon,
        })

    return picks


def main():
    parser = argparse.ArgumentParser(
        description="Pick P/S phases on 5M land station waveforms using PhaseNet")
    parser.add_argument("--p-threshold", type=float, default=0.3,
                        help="P-wave pick threshold (default: 0.3)")
    parser.add_argument("--s-threshold", type=float, default=0.3,
                        help="S-wave pick threshold (default: 0.3)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip events already in output file")
    parser.add_argument("--model", type=str, default="stead",
                        choices=["stead", "instance", "ethz", "geofon", "scedc",
                                 "neic", "original"],
                        help="PhaseNet pretrained model (default: stead)")
    args = parser.parse_args()

    print("=" * 60)
    print("PhaseNet Phase Picking — BRAVOSEIS 5M Land Stations")
    print("=" * 60)
    print(f"P threshold: {args.p_threshold}")
    print(f"S threshold: {args.s_threshold}")
    print(f"Model: PhaseNet ({args.model})")

    # Find all waveform files
    mseed_files = sorted(WAVE_DIR.glob("*.mseed"))
    print(f"\nTotal waveform files: {len(mseed_files):,}")

    # Resume logic
    done_ids = set()
    if args.resume and OUT_PICKS.exists():
        existing = pd.read_parquet(OUT_PICKS)
        done_ids = set(existing["assoc_id"].unique())
        print(f"Already processed: {len(done_ids):,}")
        mseed_files = [f for f in mseed_files if f.stem not in done_ids]
        print(f"Remaining: {len(mseed_files):,}")

    if not mseed_files:
        print("Nothing to process.")
        return

    # Load model
    print("\nLoading PhaseNet model...")
    seisbench.use_backup_repository()
    model = sbm.PhaseNet.from_pretrained(args.model)
    print("  Model loaded.")

    # Process events
    print(f"\nProcessing {len(mseed_files):,} events...")
    t_start = time.time()

    all_picks = []
    n_with_picks = 0
    n_no_picks = 0

    for i, fpath in enumerate(mseed_files):
        try:
            picks = pick_event(model, fpath, args.p_threshold, args.s_threshold)
            all_picks.extend(picks)
            if picks:
                n_with_picks += 1
            else:
                n_no_picks += 1
        except Exception as e:
            print(f"  ERROR {fpath.name}: {e}")
            n_no_picks += 1

        # Progress
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_m = (len(mseed_files) - i - 1) / rate / 60 if rate > 0 else 0
            n_p = sum(1 for p in all_picks if p["phase"] == "P")
            n_s = sum(1 for p in all_picks if p["phase"] == "S")
            print(f"  {i+1:,}/{len(mseed_files):,}  "
                  f"{n_with_picks} w/picks, {n_no_picks} empty  "
                  f"({n_p} P, {n_s} S picks)  "
                  f"{rate:.1f}/s  ETA {eta_m:.0f}m")

        # Checkpoint every 1000 events
        if (i + 1) % 1000 == 0 and all_picks:
            _save_picks(all_picks, done_ids, args)

    # Final save
    if all_picks:
        _save_picks(all_picks, done_ids, args)

    elapsed = time.time() - t_start
    n_p = sum(1 for p in all_picks if p["phase"] == "P")
    n_s = sum(1 for p in all_picks if p["phase"] == "S")

    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed/60:.1f} minutes")
    print(f"Events with picks: {n_with_picks:,} / {len(mseed_files):,} "
          f"({100*n_with_picks/len(mseed_files):.1f}%)")
    print(f"Total picks: {len(all_picks):,} ({n_p:,} P, {n_s:,} S)")
    print(f"Output: {OUT_PICKS}")


def _save_picks(all_picks, done_ids, args):
    """Save picks to parquet, merging with existing if resuming."""
    df = pd.DataFrame(all_picks)
    df["pick_time"] = pd.to_datetime(df["pick_time"])

    if done_ids and OUT_PICKS.exists():
        existing = pd.read_parquet(OUT_PICKS)
        df = pd.concat([existing, df], ignore_index=True)

    # Sort by event then pick time
    df = df.sort_values(["assoc_id", "pick_time"]).reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PICKS, index=False)

    # Build per-event summary
    summary = df.groupby("assoc_id").agg(
        n_p_picks=("phase", lambda x: (x == "P").sum()),
        n_s_picks=("phase", lambda x: (x == "S").sum()),
        n_stations_p=("station", lambda x: x[df.loc[x.index, "phase"] == "P"].nunique()),
        n_stations_s=("station", lambda x: x[df.loc[x.index, "phase"] == "S"].nunique()),
        max_p_prob=("probability", lambda x: x[df.loc[x.index, "phase"] == "P"].max()
                    if (df.loc[x.index, "phase"] == "P").any() else 0),
        max_s_prob=("probability", lambda x: x[df.loc[x.index, "phase"] == "S"].max()
                    if (df.loc[x.index, "phase"] == "S").any() else 0),
    ).reset_index()
    summary.to_parquet(OUT_SUMMARY, index=False)
    print(f"  [checkpoint] Saved {len(df):,} picks from "
          f"{df['assoc_id'].nunique():,} events")


if __name__ == "__main__":
    main()
