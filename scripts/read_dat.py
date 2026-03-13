"""
read_dat.py — Reader for NOAA/PMEL autonomous hydrophone .DAT files.

Binary format (reverse-engineered from BRAVOSEIS 2019 deployment):
    - 256-byte header (big-endian)
    - 14,400,000 samples of unsigned 16-bit big-endian audio
    - 24-bit ADC, stored as top 16 bits
    - 1 kHz sample rate, single channel
    - Each file = one 4-hour recording segment

Duty cycle: two consecutive 4-hour files (~8 hours) every ~2 days.

Usage:
    from read_dat import read_dat, read_header, list_mooring_files

    ts, data, meta = read_dat(filepath)
    header = read_header(filepath)
    catalog = list_mooring_files("/path/to/mooring_dir")
"""

import struct
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


HEADER_SIZE = 256
SAMPLE_RATE = 1000  # Hz
SAMPLES_PER_FILE = 14_400_000  # 4 hours at 1 kHz
DURATION_SEC = SAMPLES_PER_FILE / SAMPLE_RATE  # 14400 s = 4 h


def _parse_timestamp(header: bytes) -> datetime:
    """Parse the header timestamp string at offset 0x58.

    Format: "YYY DDD:HH:MM:SS:mmm"
        YYY  = year since 1900
        DDD  = day of year (1-based)
        HH:MM:SS:mmm = time with milliseconds
    """
    ts_raw = header[0x58:0x78].replace(b"\x00", b"").decode("ascii").strip()
    parts = ts_raw.split()
    year = 1900 + int(parts[0])
    tok = parts[1].split(":")
    doy = int(tok[0])
    hour, minute, sec, ms = int(tok[1]), int(tok[2]), int(tok[3]), int(tok[4])
    return datetime(year, 1, 1) + timedelta(
        days=doy - 1, hours=hour, minutes=minute, seconds=sec, milliseconds=ms
    )


def read_header(filepath) -> dict:
    """Read only the 256-byte header and return parsed metadata.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    dict with keys: timestamp, instrument_id, project, firmware,
        sample_rate, adc_bytes, lowpass_hz, active_seconds,
        duty_cycle_seconds, daq_serial, hardware_serial, file_number
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        header = f.read(HEADER_SIZE)

    return {
        "timestamp": _parse_timestamp(header),
        "instrument_id": header[0x40:0x48].rstrip(b"\x00").decode("ascii"),
        "project": header[0x88:0x98].rstrip(b"\x00").decode("ascii"),
        "firmware": header[0x98:0xA5].rstrip(b"\x00").decode("ascii"),
        "sample_rate": struct.unpack(">I", header[0xC4:0xC8])[0],
        "adc_bytes": header[0xC9],
        "lowpass_hz": struct.unpack(">H", header[0xCC:0xCE])[0],
        "active_seconds": struct.unpack(">I", header[0xD0:0xD4])[0],
        "duty_cycle_seconds": struct.unpack(">I", header[0xD4:0xD8])[0],
        "hardware_serial": header[0xD9:0xE3].rstrip(b"\x00").decode("ascii"),
        "daq_serial": header[0xE8:0xF6].rstrip(b"\x00").decode("ascii"),
        "file_number": int(filepath.stem),
    }


def read_dat(filepath):
    """Read a NOAA/PMEL autonomous hydrophone .DAT file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    timestamp : datetime
        Recording start time (UTC).
    data : np.ndarray
        Signed acoustic waveform (float64), centered on zero.
        14,400,000 samples at 1000 Hz (4 hours).
    metadata : dict
        Parsed header fields.
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        header = f.read(HEADER_SIZE)
        raw = f.read()

    metadata = read_header(filepath)
    data = np.frombuffer(raw, dtype=">u2").astype(np.float64) - 32768.0
    return metadata["timestamp"], data, metadata


def list_mooring_files(mooring_dir, sort_by="timestamp"):
    """List all .DAT files in a mooring directory with timestamps.

    Parameters
    ----------
    mooring_dir : str or Path
    sort_by : str
        "timestamp" (default) or "filename"

    Returns
    -------
    list of dict
        Each dict has: path, file_number, timestamp, instrument_id, daq_serial
    """
    mooring_dir = Path(mooring_dir)
    catalog = []
    for f in mooring_dir.glob("*.DAT"):
        meta = read_header(f)
        catalog.append({
            "path": f,
            "file_number": meta["file_number"],
            "timestamp": meta["timestamp"],
            "instrument_id": meta["instrument_id"],
            "daq_serial": meta["daq_serial"],
        })

    if sort_by == "timestamp":
        catalog.sort(key=lambda x: x["timestamp"])
    else:
        catalog.sort(key=lambda x: x["file_number"])

    return catalog


# --- Mooring metadata ---

MOORINGS = {
    "m1": {
        "name": "BRA28", "hydrophone": "H17C",
        "lat": -(62 + 54.9082 / 60), "lon": -(60 + 11.9788 / 60),
        "bottom_depth_m": 1028.7, "hydrophone_depth_m": 454.7,
        "deployed": datetime(2019, 1, 12, 13, 27),
        "recovered": datetime(2020, 2, 20),
        "data_dir": "m1-h17c-bra28",
        "data_dir_full": "H17C_M1_BRA28",
    },
    "m2": {
        "name": "BRA29", "hydrophone": "H36",
        "lat": -(62 + 50.9967 / 60), "lon": -(59 + 27.0145 / 60),
        "bottom_depth_m": 1245.7, "hydrophone_depth_m": 421.7,
        "deployed": datetime(2019, 1, 12, 19, 24),
        "recovered": datetime(2020, 2, 20),
        "data_dir": "m2-h36-bra29",
        "data_dir_full": "H36_M2_BRA29",
    },
    "m3": {
        "name": "BRA30", "hydrophone": "H13",
        "lat": -(62 + 30.9512 / 60), "lon": -(58 + 53.9875 / 60),
        "bottom_depth_m": 1537.7, "hydrophone_depth_m": 413.7,
        "deployed": datetime(2019, 1, 13, 13, 23),
        "recovered": datetime(2020, 2, 18),
        "data_dir": "m3-h13-bra30",
        "data_dir_full": "H13_M3_BRA30",
    },
    "m4": {
        "name": "BRA31", "hydrophone": "H21",
        "lat": -(62 + 32.0033 / 60), "lon": -(58 + 0.0466 / 60),
        "bottom_depth_m": 1764.5, "hydrophone_depth_m": 465.5,
        "deployed": datetime(2019, 1, 13, 14, 7),
        "recovered": datetime(2020, 2, 15),
        "data_dir": "m4-h21-bra31",
        "data_dir_full": "H21_M4_BRA31",
    },
    "m5": {
        "name": "BRA32", "hydrophone": "H24",
        "lat": -(62 + 17.846 / 60), "lon": -(57 + 53.671 / 60),
        "bottom_depth_m": 1928.0, "hydrophone_depth_m": 479.0,
        "deployed": datetime(2019, 1, 10, 17, 31),
        "recovered": datetime(2020, 2, 17),
        "data_dir": "m5-h24-bra32",
        "data_dir_full": "H24_M5_BRA32",
    },
    "m6": {
        "name": "BRA33", "hydrophone": "H41",
        "lat": -(62 + 14.9886 / 60), "lon": -(57 + 5.9865 / 60),
        "bottom_depth_m": 1717.1, "hydrophone_depth_m": 468.1,
        "deployed": datetime(2019, 1, 13, 21, 26),
        "recovered": datetime(2020, 2, 17),
        "data_dir": "m6-h41-bra33",
        "data_dir_full": "H41_M6_BRA33",
    },
}


def get_data_dir(mooring_info, data_root):
    """Return the correct data_dir key for a given data root.

    Checks for the full dataset directory name first, then falls back
    to the subset directory name.
    """
    full_dir = data_root / mooring_info["data_dir_full"]
    if full_dir.exists():
        return mooring_info["data_dir_full"]
    return mooring_info["data_dir"]


if __name__ == "__main__":
    import sys

    data_root = Path("/home/jovyan/my_data/bravoseis/NOAA")

    if len(sys.argv) > 1:
        # Read a specific file
        ts, data, meta = read_dat(sys.argv[1])
        print(f"File:       {sys.argv[1]}")
        print(f"Timestamp:  {ts} UTC")
        print(f"Instrument: {meta['instrument_id']}")
        print(f"Samples:    {len(data):,}")
        print(f"Duration:   {len(data)/SAMPLE_RATE:.0f} s ({len(data)/SAMPLE_RATE/3600:.1f} h)")
        print(f"Range:      {data.min():.0f} to {data.max():.0f}")
        print(f"Mean:       {data.mean():.1f}, Std: {data.std():.1f}")
    else:
        # Print catalog for all moorings
        for key, info in sorted(MOORINGS.items()):
            mdir = data_root / info["data_dir"]
            if not mdir.exists():
                print(f"{key}: directory not found")
                continue
            catalog = list_mooring_files(mdir)
            print(f"=== {key} / {info['name']} ({info['hydrophone']}) — {len(catalog)} files ===")
            if catalog:
                print(f"  First: {catalog[0]['timestamp']}")
                print(f"  Last:  {catalog[-1]['timestamp']}")
                span = (catalog[-1]["timestamp"] - catalog[0]["timestamp"]).days
                print(f"  Span:  {span} days")
            print()
