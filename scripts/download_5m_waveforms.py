#!/usr/bin/env python3
"""
download_5m_waveforms.py — Download targeted waveform snippets from the
BRAVOSEIS 5M onshore seismic network for well-located hydroacoustic events.

Downloads 60-second windows (10s before to 50s after event origin time)
from all available 5M stations via GEOFON FDSN web services. Also fetches
JUBA and ESPZ permanent stations from IRIS.

Saves miniSEED files organized by event, with a summary CSV.

Usage:
    uv run python download_5m_waveforms.py                  # pilot: 100 events
    uv run python download_5m_waveforms.py --n-events 500   # larger batch
    uv run python download_5m_waveforms.py --all-tier-a     # all tier A low-band
    uv run python download_5m_waveforms.py --resume          # resume interrupted run

Output:
    my_data/bravoseis/earthquakes/5m_waveforms/<assoc_id>.mseed
    my_data/bravoseis/earthquakes/5m_waveforms/download_log.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# === Paths ===
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "outputs" / "data"
WAVE_DIR = Path("/home/jovyan/my_data/bravoseis/earthquakes/5m_waveforms")
LOG_PATH = WAVE_DIR / "download_log.csv"

# === FDSN parameters ===
PRE_EVENT_S = 10    # seconds before event time
POST_EVENT_S = 50   # seconds after event time
NETWORKS_CHANNELS = [
    # (client, network, station_pattern, channel_pattern)
    ("GFZ", "5M", "*", "HH*"),       # BRAVOSEIS onshore (14 stations)
    ("IRIS", "AI", "JUBA", "HH*"),   # Carlini base, King George Island
    ("IRIS", "AI", "ESPZ", "HH*"),   # Esperanza, Antarctic Peninsula
]


def load_events(args):
    """Load and filter events for download."""
    loc = pd.read_parquet(DATA_DIR / "tapaas_locations.parquet")
    loc["earliest_utc"] = pd.to_datetime(loc["earliest_utc"])

    # Default: low-band tier A with >=5 moorings (best seismic events)
    if args.all_tier_a:
        events = loc[
            (loc["quality_tier"] == "A") &
            (loc["detection_band"] == "low")
        ].copy()
    else:
        events = loc[
            (loc["quality_tier"] == "A") &
            (loc["detection_band"] == "low") &
            (loc["n_moorings"] >= 5)
        ].copy()

    events = events.sort_values("earliest_utc").reset_index(drop=True)
    print(f"Candidate events: {len(events):,}")

    # Limit number
    if not args.all_tier_a and args.n_events < len(events):
        # Sample evenly across time
        idx = np.linspace(0, len(events) - 1, args.n_events, dtype=int)
        events = events.iloc[idx].reset_index(drop=True)
        print(f"Selected {len(events):,} events (evenly spaced)")

    return events


def download_event(clients, assoc_id, event_time, out_dir):
    """Download waveforms for a single event from all networks.

    Returns (n_traces, n_stations, status_str).
    """
    t = UTCDateTime(str(event_time))
    t_start = t - PRE_EVENT_S
    t_end = t + POST_EVENT_S

    all_traces = None
    n_stations = 0

    for client_name, net, sta, cha in NETWORKS_CHANNELS:
        client = clients[client_name]
        try:
            st = client.get_waveforms(net, sta, "*", cha, t_start, t_end)
            if all_traces is None:
                all_traces = st
            else:
                all_traces += st
            # Count unique stations
            stas = set(tr.stats.station for tr in st)
            n_stations += len(stas)
        except Exception:
            pass  # Station may not have data for this time

    if all_traces is None or len(all_traces) == 0:
        return 0, 0, "no_data"

    # Save as miniSEED
    out_path = out_dir / f"{assoc_id}.mseed"
    all_traces.write(str(out_path), format="MSEED")
    return len(all_traces), n_stations, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Download 5M waveform snippets for located events")
    parser.add_argument("--n-events", type=int, default=100,
                        help="Number of events to download (default: 100)")
    parser.add_argument("--all-tier-a", action="store_true",
                        help="Download all tier-A low-band events")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-downloaded events")
    args = parser.parse_args()

    print("=" * 60)
    print("BRAVOSEIS 5M Waveform Download")
    print("=" * 60)

    # Create output directory
    WAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Load events
    events = load_events(args)

    # Check for resume
    done_ids = set()
    if args.resume and LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        done_ids = set(log[log["status"] == "ok"]["assoc_id"])
        print(f"Resuming: {len(done_ids):,} already downloaded")
        events = events[~events["assoc_id"].isin(done_ids)].reset_index(drop=True)
        print(f"Remaining: {len(events):,}")

    if len(events) == 0:
        print("Nothing to download.")
        return

    # Connect to FDSN services
    print("\nConnecting to FDSN services...")
    clients = {
        "GFZ": Client("GFZ"),
        "IRIS": Client("IRIS"),
    }
    print("  GFZ (GEOFON) — 5M network")
    print("  IRIS — JUBA, ESPZ permanent stations")

    # Download loop
    print(f"\nDownloading {len(events):,} events "
          f"({PRE_EVENT_S}s before to {POST_EVENT_S}s after)...")
    t_start = time.time()

    log_rows = []
    n_ok = 0
    n_fail = 0

    for i, (_, ev) in enumerate(events.iterrows()):
        assoc_id = ev["assoc_id"]
        event_time = ev["earliest_utc"]

        n_traces, n_stations, status = download_event(
            clients, assoc_id, event_time, WAVE_DIR)

        log_rows.append({
            "assoc_id": assoc_id,
            "earliest_utc": event_time,
            "lat": ev["lat"],
            "lon": ev["lon"],
            "n_moorings": ev["n_moorings"],
            "n_traces": n_traces,
            "n_5m_stations": n_stations,
            "status": status,
        })

        if status == "ok":
            n_ok += 1
        else:
            n_fail += 1

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_m = (len(events) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  {i+1:,}/{len(events):,} "
                  f"({n_ok} ok, {n_fail} no data, "
                  f"{rate:.1f}/s, ETA {eta_m:.0f}m)")

        # Rate limit — be polite to FDSN servers
        time.sleep(0.5)

    # Save log
    log_df = pd.DataFrame(log_rows)
    if args.resume and LOG_PATH.exists():
        old_log = pd.read_csv(LOG_PATH)
        log_df = pd.concat([old_log, log_df], ignore_index=True)
    log_df.to_csv(LOG_PATH, index=False)

    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} downloaded, {n_fail} no data, "
          f"{elapsed/60:.1f} minutes")
    print(f"Output: {WAVE_DIR}")
    print(f"Log: {LOG_PATH}")

    # Summary stats
    ok_log = log_df[log_df["status"] == "ok"]
    if len(ok_log) > 0:
        print(f"\nStation coverage:")
        print(f"  Median stations per event: {ok_log['n_5m_stations'].median():.0f}")
        print(f"  Median traces per event: {ok_log['n_traces'].median():.0f}")
        total_mb = sum(f.stat().st_size for f in WAVE_DIR.glob("*.mseed")) / 1e6
        print(f"  Total data: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
