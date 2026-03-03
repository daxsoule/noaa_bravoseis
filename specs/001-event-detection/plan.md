# Analysis Plan: Acoustic Event Detection

**Spec**: `specs/001-event-detection/spec.md`
**Created**: 2026-03-02
**Status**: Draft

## Summary

Detect discrete acoustic events in 717 DAT files from 6 BRAVOSEIS hydrophone
moorings using an STA/LTA energy detector operating in 4 frequency bands.
Characterize each event (onset, duration, peak frequency, bandwidth, SNR),
associate events across moorings, and compile a Parquet catalogue. This
catalogue is the foundation for spec 002 (event discrimination/classification).

## Analysis Environment

**Language/Version**: Python 3.12
**Key Packages**:
- Existing: numpy, scipy, matplotlib
- New: pandas, pyarrow (for Parquet I/O), obspy (for STA/LTA — provides
  optimized `classic_sta_lta` and `recursive_sta_lta` implementations)
**Environment File**: `pyproject.toml` (managed by uv)

## Compute Environment

- [x] Shared server (JupyterHub, CPU instance)

**Data scale**: ~80 GB raw DAT files (717 × 115 MB). Processed sequentially,
one file at a time (~115 MB in memory). Output catalogue: small (< 100 MB).

**Timeline pressure**: None. Exploratory/publication-track.

**Known bottlenecks**:
- Reading + computing spectrograms for 717 files: ~30–60 min
- STA/LTA on bandpassed energy envelopes: fast (rolling averages on 1D arrays)
- Total pipeline: estimate 1–2 hours for full run

## Constitution Check

- [x] Data sources match those defined in constitution
- [x] Coordinate systems/units are consistent (UTC, Hz, meters)
- [x] Figure standards will be followed (paper tier, 300 DPI, justified captions)
- [x] Quality checks are incorporated (cross-mooring consistency, frequency
      validation, false positive inspection)

**Issues to resolve**: None.

## Research Decisions

### R1: Bandpass pre-filtering

**Decision**: Apply bandpass filter before detection, not after.

**Rationale**: The instrument band is 1–250 Hz (400 Hz lowpass in hardware,
but science band is 1–250 Hz per constitution). Filtering before STA/LTA
removes out-of-band noise that could trigger false detections. Use a 4th-order
Butterworth bandpass for each detection band.

**Alternatives considered**: Post-detection filtering would keep all data for
characterization, but at the cost of more false triggers. Since we characterize
events from the raw spectrogram anyway, pre-filtering the STA/LTA input is
cleaner.

### R2: STA/LTA parameters (starting values)

**Decision**: Start with standard seismoacoustic values, tune on the 3 existing
spectrogram windows.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| STA window | 2 s | Short enough for impulsive events, long enough to average |
| LTA window | 60 s | Captures ~1 min of background noise |
| Trigger threshold | 3.0 | Standard first pass; will tune down if too few detections |
| Detrigger threshold | 1.5 | Event ends when STA/LTA returns to ~background |
| Min event duration | 0.5 s | Reject very short transients (likely noise spikes) |
| Min inter-event gap | 2.0 s | Prevent splitting one event into fragments |

**Tuning strategy**: Run detector on the 3 spectrogram windows already
generated. Compare detections against visually identified events. Adjust
thresholds to minimize false positives while catching known signals. Document
final parameters.

### R3: Maximum inter-mooring travel time

**Decision**: 200 s maximum association window.

**Rationale**: Maximum mooring separation is ~300 km (M1 to M6). At ~1480 m/s
sound speed, max travel time is ~203 s. Round to 200 s. Events on different
moorings within this window are candidate associations.

### R4: obspy vs. hand-rolled STA/LTA

**Decision**: Use obspy's `recursive_sta_lta()` — it's the standard
implementation, C-optimized, and handles edge cases.

**Alternatives considered**: Hand-rolling with numpy cumsum would avoid the
obspy dependency but provides no advantage. obspy is the standard tool for
this exact problem. If obspy is too heavy a dependency, fall back to a minimal
numpy implementation.

## Project Structure

```text
specs/001-event-detection/
├── spec.md
├── plan.md              # This file
└── tasks.md             # Task breakdown (via /speckit.tasks)

# New scripts (in project root, matching existing convention)
detect_events.py         # STA/LTA detection across all files
associate_events.py      # Cross-mooring association
make_detection_figures.py # All 4 output figures

# Existing (reused)
read_dat.py              # DAT file reader
make_spectrogram.py      # Array spectrogram layout (reused for example figs)
make_bathy_map.py        # add_caption_justified() helper

outputs/
├── data/
│   ├── event_catalogue.parquet          # All detections, all moorings
│   └── cross_mooring_associations.parquet
├── figures/journal/
│   ├── detection_rate_timeline.png
│   ├── duration_vs_peak_freq.png
│   ├── example_detections_*.png         # 3–5 array spectrograms with overlays
│   └── cross_mooring_statistics.png
└── tables/
    ├── catalogue_summary.csv
    └── cross_mooring_counts.csv
```

## Data Pipeline

### Stage 1: Event Detection (`detect_events.py`)

- **Input**: Raw DAT files (717 files across 6 moorings)
- **Processing**:
  1. Iterate over all moorings and their DAT files (sequential)
  2. For each file:
     a. Load waveform via `read_dat()`
     b. For each of 4 frequency bands:
        - Bandpass filter (4th-order Butterworth)
        - Compute energy envelope (squared amplitude, smoothed)
        - Run `recursive_sta_lta(envelope, nsta, nlta)`
        - Trigger/detrigger to extract event windows
     c. For each detection, characterize from the raw spectrogram:
        - Compute spectrogram of event segment (± 2 s padding)
        - Extract: onset_utc, duration_s, end_utc, peak_freq_hz,
          bandwidth_hz, peak_db, snr, detection_band
     d. Append to running list
  3. De-duplicate: if same event detected in multiple bands within
     the same mooring, keep the band with highest SNR, note all bands
  4. Assign event_id (mooring + file_number + sequential index)
  5. Save as `outputs/data/event_catalogue.parquet`
- **Output**: `outputs/data/event_catalogue.parquet`
- **Key parameters**: STA=2s, LTA=60s, trigger=3.0, detrigger=1.5,
  min_duration=0.5s, min_gap=2.0s
- **Runtime estimate**: ~1–2 hours for all 717 files

### Stage 2: Cross-Mooring Association (`associate_events.py`)

- **Input**: `outputs/data/event_catalogue.parquet`
- **Processing**:
  1. Load catalogue
  2. For each pair of recording windows that overlap in time across
     moorings, find events within 200 s of each other
  3. Group into associations (connected components: if event A on M1
     matches event B on M2, and event B matches event C on M4, all three
     are one association)
  4. For each association: record number of moorings, arrival time
     differences, member event IDs
  5. Compute summary statistics
- **Output**:
  - `outputs/data/cross_mooring_associations.parquet`
  - `outputs/tables/cross_mooring_counts.csv`
  - `outputs/tables/catalogue_summary.csv`

### Stage 3: Figures (`make_detection_figures.py`)

- **Input**: Event catalogue + cross-mooring associations + raw DAT files
- **Processing**:
  1. **Detection rate timeline**: Bin events by recording window, stack by
     detection band. X-axis: date. Y-axis: events per recording window.
     Paper tier styling.
  2. **Duration vs. peak frequency**: Scatter plot, colored by detection
     band. Log-scale duration axis if range is large.
  3. **Example detections**: Select 3–5 events detected on multiple
     moorings. For each, generate an array spectrogram (reusing
     `make_spectrogram.py` layout) with detection trigger points overlaid
     as horizontal markers or colored bands.
  4. **Cross-mooring statistics**: Bar chart showing number of events
     detected on 1, 2, 3, ... 6 moorings, grouped by detection band.
  5. All figures: paper tier (14pt title, 10pt axis/caption, 300 DPI),
     justified captions via `add_caption_justified()`.
- **Output**: 4+ PNG files in `outputs/figures/journal/`

## Script Plan

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `detect_events.py` | STA/LTA detection in 4 frequency bands across all 717 DAT files. Characterize each event. Save catalogue. | Raw DAT files | `event_catalogue.parquet` |
| `associate_events.py` | Find events on 2+ moorings within 200 s travel time window. Group into associations. | `event_catalogue.parquet` | `cross_mooring_associations.parquet`, summary tables |
| `make_detection_figures.py` | Generate all 4 spec'd figures: timeline, scatter, examples, cross-mooring stats. | Catalogue, associations, raw DAT files | 4+ PNG figures |

## Dependencies

```text
detect_events.py
       ↓
associate_events.py ─────────────────┐
       ↓                              │
make_detection_figures.py ←──────────┘
```

**Parallel opportunities**: `associate_events.py` and the non-example figures
in `make_detection_figures.py` could run in parallel once the catalogue
exists. But the example detection figures need association data to pick
multi-mooring events. Simplest to run sequentially.

## Tuning Workflow

Before processing all 717 files, tune the detector on a small subset:

1. **Select tuning files**: The 3 DAT files already used for spectrograms
   (00001282, 00001283, 00002166) across available moorings = ~15 files
2. **Run detector** with starting parameters on these files only
3. **Compare against visual inspection** of the 3 existing spectrogram
   figures — manually count events visible in each 10-min window
4. **Adjust thresholds**:
   - Too many detections → raise trigger threshold
   - Missing visible events → lower trigger threshold or adjust STA/LTA
     window lengths
   - Events being split → increase min_gap
5. **Document final parameters** in the script and spec
6. **Run full pipeline** on all 717 files

This avoids waiting 1–2 hours per iteration during tuning.

## Open Questions

- [x] Bandpass pre-filtering → Yes, before STA/LTA (R1)
- [x] STA/LTA starting parameters → Defined (R2)
- [x] Max travel time → 200 s (R3)
- [x] STA/LTA implementation → obspy recursive_sta_lta (R4)
- [ ] Within-mooring band de-duplication strategy → Keep highest-SNR band,
  record all bands. May need refinement after seeing data.
- [ ] Exact event characterization from spectrogram — bandwidth definition
  (90% energy contour) may need tuning.

Both remaining questions are implementation details, not blocking.

## Notes

- **Package installation**: Before starting:
  ```bash
  uv add obspy pandas pyarrow
  ```
- **obspy note**: obspy is a large package. If installation is problematic,
  the STA/LTA can be implemented in ~20 lines of numpy (cumulative sum
  trick). The algorithm is simple; obspy just provides an optimized version.
- **Downstream dependency**: Spec 002 (event discrimination) consumes
  `outputs/data/event_catalogue.parquet`. Column names and types defined
  in this spec must match what spec 002 expects.
- **Existing code reuse**:
  - `read_dat.py`: file reader, mooring metadata, SAMPLE_RATE constant
  - `make_spectrogram.py`: spectrogram computation, array layout
  - `make_bathy_map.py`: `add_caption_justified()` for figure captions
- **Memory**: One DAT file = ~115 MB as float64. Spectrogram of a 4-hour
  file with nperseg=1024, noverlap=512 produces ~28,000 time steps × 513
  freq bins = ~115 MB. Total per-file memory ~250 MB. Fine for any machine.
