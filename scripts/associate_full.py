#!/usr/bin/env python3
"""
associate_full.py — Cross-mooring association for the full 6.76M event catalogue.

Resumable: saves checkpoint every 500K events. If interrupted, re-run with --resume
to continue from the last checkpoint.

Usage:
    uv run python scripts/associate_full.py
    uv run python scripts/associate_full.py --resume
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).parent))
from read_dat import MOORINGS

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
CHECKPOINT_DIR = DATA_DIR / "association_checkpoints"

MOORING_KEYS = sorted(MOORINGS.keys())
CHECKPOINT_INTERVAL = 500_000  # save every N events processed


def load_pair_travel_times():
    """Load pair-specific max travel times from JSON."""
    json_path = DATA_DIR / "travel_times.json"
    with open(json_path) as f:
        data = json.load(f)
    pair_max_dt = {}
    for pair_key, info in data["pairs"].items():
        k1, k2 = pair_key.split("-")
        pair_max_dt[(k1, k2)] = info["max_travel_time_s"]
        pair_max_dt[(k2, k1)] = info["max_travel_time_s"]
    global_max = data["global_max_travel_time_s"]
    return pair_max_dt, global_max


def find_associations_resumable(df, pair_max_dt, global_max, resume_from=0,
                                 existing_assocs=None):
    """Find associations with checkpoint support.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted by onset_utc, reset index.
    resume_from : int
        Index to resume scanning from.
    existing_assocs : list or None
        Previously found associations to append to.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    onset = df["onset_utc"].values.astype("datetime64[ms]").astype(np.int64) / 1000
    moorings = df["mooring"].values
    snr = df["snr"].values

    n = len(df)
    assigned = np.zeros(n, dtype=bool)
    associations = existing_assocs if existing_assocs else []

    # If resuming, mark events from existing associations as assigned
    if existing_assocs:
        existing_eids = set()
        for a in existing_assocs:
            existing_eids.update(a["event_ids"].split(","))
        eid_to_idx = {eid: i for i, eid in enumerate(df["event_id"].values)}
        for eid in existing_eids:
            if eid in eid_to_idx:
                assigned[eid_to_idx[eid]] = True
        print(f"  Resumed: {len(existing_assocs):,} existing associations, "
              f"{assigned.sum():,} events marked assigned")

    t0 = time.time()
    last_checkpoint = resume_from

    for i in range(resume_from, n):
        if assigned[i]:
            continue

        anchor_mooring = moorings[i]
        members = [i]
        member_moorings = {anchor_mooring}

        j = i + 1
        while j < n and (onset[j] - onset[i]) <= global_max:
            if not assigned[j] and moorings[j] != anchor_mooring:
                dt = onset[j] - onset[i]
                pair_key = (anchor_mooring, moorings[j])
                max_dt = pair_max_dt.get(pair_key, global_max)
                if dt <= max_dt:
                    members.append(j)
                    member_moorings.add(moorings[j])
            j += 1

        if len(member_moorings) < 2:
            continue

        # Best SNR per mooring
        best_per_mooring = {}
        for m_idx in members:
            mk = moorings[m_idx]
            if mk not in best_per_mooring or snr[m_idx] > snr[best_per_mooring[mk]]:
                best_per_mooring[mk] = m_idx

        final_members = list(best_per_mooring.values())
        final_moorings = set(best_per_mooring.keys())

        if len(final_moorings) < 2:
            continue

        for m_idx in final_members:
            assigned[m_idx] = True

        event_ids = [df.iloc[m]["event_id"] for m in final_members]
        onsets_utc = [df.iloc[m]["onset_utc"] for m in final_members]
        earliest = min(onsets_utc)
        latest = max(onsets_utc)
        dt_s = (latest - earliest).total_seconds()

        associations.append({
            "assoc_id": f"F{len(associations):07d}",
            "n_moorings": len(final_moorings),
            "moorings": ",".join(sorted(final_moorings)),
            "n_events": len(final_members),
            "event_ids": ",".join(event_ids),
            "earliest_utc": earliest,
            "latest_utc": latest,
            "dt_s": round(dt_s, 3),
            "detection_band": df.iloc[final_members[0]]["detection_band"],
        })

        # Progress + checkpoint
        if (i - last_checkpoint) >= CHECKPOINT_INTERVAL:
            elapsed = time.time() - t0
            rate = (i - resume_from) / elapsed
            eta = (n - i) / rate / 60 if rate > 0 else 0
            print(f"  {i:,}/{n:,} events ({100*i/n:.1f}%), "
                  f"{len(associations):,} assocs, "
                  f"{elapsed/60:.1f} min elapsed, ~{eta:.0f} min remaining")

            # Save checkpoint
            ckpt = {
                "resume_from": i,
                "n_associations": len(associations),
            }
            ckpt_path = CHECKPOINT_DIR / "assoc_full_checkpoint.json"
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f)

            assoc_df = pd.DataFrame(associations)
            assoc_df.to_parquet(CHECKPOINT_DIR / "assoc_full_partial.parquet", index=False)
            last_checkpoint = i

    return pd.DataFrame(associations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("Full Dataset Cross-Mooring Association")
    print("=" * 60)

    pair_max_dt, global_max = load_pair_travel_times()
    print(f"Travel times: XBT-derived (global max {global_max:.1f} s)")

    # Load full catalogue
    cat_path = DATA_DIR / "event_catalogue_full.parquet"
    print(f"\nLoading {cat_path.name}...")
    df = pd.read_parquet(cat_path)
    df["onset_utc"] = pd.to_datetime(df["onset_utc"])
    df = df.sort_values("onset_utc").reset_index(drop=True)
    print(f"  {len(df):,} events")

    # Resume logic
    resume_from = 0
    existing_assocs = None
    if args.resume:
        ckpt_path = CHECKPOINT_DIR / "assoc_full_checkpoint.json"
        partial_path = CHECKPOINT_DIR / "assoc_full_partial.parquet"
        if ckpt_path.exists() and partial_path.exists():
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            resume_from = ckpt["resume_from"]
            partial_df = pd.read_parquet(partial_path)
            existing_assocs = partial_df.to_dict("records")
            print(f"\n  Resuming from event {resume_from:,} "
                  f"({len(existing_assocs):,} existing associations)")
        else:
            print("  No checkpoint found — starting fresh")

    # Run association
    print(f"\nFinding associations (starting at event {resume_from:,})...")
    t0 = time.time()
    assoc_df = find_associations_resumable(
        df, pair_max_dt, global_max,
        resume_from=resume_from,
        existing_assocs=existing_assocs,
    )
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")

    # Summary
    print(f"\nTotal associations: {len(assoc_df):,}")
    print(f"\nBy number of moorings:")
    for n_m, count in assoc_df["n_moorings"].value_counts().sort_index().items():
        print(f"  {n_m} moorings: {count:,}")

    all_eids = set()
    for eids in assoc_df["event_ids"]:
        all_eids.update(eids.split(","))
    print(f"\nEvents associated: {len(all_eids):,} / {len(df):,} "
          f"({100*len(all_eids)/len(df):.1f}%)")

    # Save
    out_path = DATA_DIR / "cross_mooring_associations_full.parquet"
    assoc_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Clean up checkpoints
    for f in CHECKPOINT_DIR.glob("assoc_full_*"):
        f.unlink()
    print("Checkpoints cleaned up.")


if __name__ == "__main__":
    main()
