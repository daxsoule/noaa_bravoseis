"""Parse Singer's acoustic analysis notes from Excel into structured Parquet.

Reads the 'AnalysisNotes' sheet from Soule_SeaSickAcousticNotes.xlsx and
extracts daily event counts (EQ, IQ, IDK), whale/boat noise flags, and
non-locatable event counts into a tidy Parquet dataset.

Usage:
    uv run python scripts/parse_singer_notes.py
"""

import re
from datetime import datetime
from pathlib import Path

import openpyxl
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
XLSX_PATH = Path(
    "/home/jovyan/my_data/bravoseis/NOAA/Soule_SeaSickAcousticNotes.xlsx"
)
SHEET_NAME = "AnalysisNotes"
FIRST_DATA_ROW = 2  # row 1 is header
LAST_DATA_ROW = 396
OUTPUT_PATH = Path(
    "/home/jovyan/repos/specKitScience/noaa_bravoseis/outputs/data/"
    "singer_daily_notes.parquet"
)

# Columns E through Z for non-locatable events (indices 4..25 in 0-based)
NONLOC_START_COL = 4  # column E
NONLOC_END_COL = 25   # column Z (inclusive)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_doy_date(doy_str: str) -> datetime | None:
    """Convert '2019-012' (year-DOY) to a datetime.date."""
    try:
        return datetime.strptime(doy_str.strip(), "%Y-%j").date()
    except (ValueError, AttributeError):
        return None


def parse_counts(summary: str | None) -> dict:
    """Extract EQ, IQ, IDK counts from a summary string.

    Handles patterns like:
        '3 EQs, 49 IQs, 17 IDKs'
        '1 maybe EQ, 42 IQs, 9 IDKs'
        '32 IQs, 10 EQs; 5 Screenshots, 3 IDKs'
    """
    result = {"n_eq": 0, "n_iq": 0, "n_idk": 0}
    if not summary:
        return result

    text = str(summary)

    # EQ count: match "N EQ" or "N EQs" or "N maybe EQ"
    eq_matches = re.findall(r"(\d+)\s+(?:maybe\s+)?EQs?\b", text)
    result["n_eq"] = sum(int(x) for x in eq_matches)

    # IQ count
    iq_matches = re.findall(r"(\d+)\s+IQs?\b", text)
    result["n_iq"] = sum(int(x) for x in iq_matches)

    # IDK count
    idk_matches = re.findall(r"(\d+)\s+IDKs?\b", text)
    result["n_idk"] = sum(int(x) for x in idk_matches)

    return result


def check_whale(summary: str | None) -> bool:
    """Check if summary mentions whale calls."""
    if not summary:
        return False
    text = str(summary).lower()
    return any(w in text for w in ("whale", "humpback", "cetacean"))


def check_boat_noise(summary: str | None) -> bool:
    """Check if summary mentions boat or ship noise."""
    if not summary:
        return False
    text = str(summary).lower()
    return any(w in text for w in ("boat", "ship"))


def count_nonlocatable(row_cells) -> int:
    """Count non-empty cells in columns E-Z."""
    count = 0
    for idx in range(NONLOC_START_COL, min(NONLOC_END_COL + 1, len(row_cells))):
        val = row_cells[idx].value
        if val is not None and str(val).strip():
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Reading {XLSX_PATH.name}, sheet '{SHEET_NAME}'...")
    wb = openpyxl.load_workbook(str(XLSX_PATH), read_only=True)
    ws = wb[SHEET_NAME]

    records = []
    skipped = 0

    for i, row in enumerate(
        ws.iter_rows(min_row=FIRST_DATA_ROW, max_row=LAST_DATA_ROW),
        start=FIRST_DATA_ROW,
    ):
        col_a = row[0].value  # Date (DOY string)

        # Skip rows where column A is empty or doesn't start with 2019/2020
        if not col_a or not str(col_a).strip().startswith(("2019", "2020")):
            skipped += 1
            continue

        doy_str = str(col_a).strip()
        date = parse_doy_date(doy_str)
        day_of_week = (
            str(row[1].value).strip() if len(row) > 1 and row[1].value else None
        )
        analyst = (
            str(row[2].value).strip() if len(row) > 2 and row[2].value else None
        )
        summary = row[3].value if len(row) > 3 else None
        summary_text = str(summary).strip() if summary else None

        counts = parse_counts(summary_text)
        n_nonloc = count_nonlocatable(row)

        records.append(
            {
                "date": date,
                "doy": doy_str,
                "day_of_week": day_of_week,
                "analyst": analyst,
                "n_eq": counts["n_eq"],
                "n_iq": counts["n_iq"],
                "n_idk": counts["n_idk"],
                "has_whale": check_whale(summary_text),
                "has_boat_noise": check_boat_noise(summary_text),
                "summary_text": summary_text,
                "n_nonlocatable": n_nonloc,
            }
        )

    wb.close()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")

    # Summary
    print(f"\n{'='*60}")
    print(f"PARSED SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows parsed:  {len(df)}")
    print(f"Rows skipped:       {skipped}")
    print(f"Date range:         {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique analysts:    {df['analyst'].dropna().unique().tolist()}")
    print(f"\nEvent count totals:")
    print(f"  EQ total:         {df['n_eq'].sum()}")
    print(f"  IQ total:         {df['n_iq'].sum()}")
    print(f"  IDK total:        {df['n_idk'].sum()}")
    print(f"\nDays with whale calls:  {df['has_whale'].sum()}")
    print(f"Days with boat noise:   {df['has_boat_noise'].sum()}")
    print(f"Days with non-locatable events: {(df['n_nonlocatable'] > 0).sum()}")
    print(f"Total non-locatable event cells: {df['n_nonlocatable'].sum()}")
    print(f"\nMonthly EQ counts:")
    monthly = df.set_index("date").resample("MS")["n_eq"].sum()
    for dt, count in monthly.items():
        if count > 0:
            print(f"  {dt.strftime('%Y-%m')}: {count}")

    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print(f"\nLast 5 rows:")
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    main()
