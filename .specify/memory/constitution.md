# NOAA BRAVOSEIS Research Constitution

## Research Context

The BRAVOSEIS (BRAnsfield VOlcanic SEISmology) experiment is a collaborative
international experiment involving scientists from Spain, Germany, and the
United States studying the seismicity and volcanic structure of the Central
Bransfield Basin. The experiment objective is to characterize the distribution
of active extension across the basin and determine whether the volcanic
structure and deformation of the rift are consistent with a back-arc basin
transitioning from rifting to seafloor spreading.

This project focuses on the US hydroacoustic component: a regional network of
6 autonomous hydrophone moorings deployed in the Bransfield Strait. The
hydrophones record signals in the 1–250 Hz band containing three primary
signal types: **earthquakes** (predominantly < 50 Hz), **ice quakes**
(1–200 Hz), and **whale calls**. The analysis approach uses machine learning
to classify these signals in the hydroacoustic catalogue, followed by source
location within the study area. Once the catalogue is confidently sorted,
downstream analyses include mapping and characterization of the classified
events.

The outputs of this project are intended for peer-reviewed publication.

## Core Principles

### I. Reproducibility

Analysis should be fully reproducible from raw data to final outputs.
Scripts run without manual intervention. Random seeds are fixed and
documented. Environment dependencies are explicit (pyproject.toml / uv.lock).

### II. Data Integrity

Raw data is immutable — all transformations produce new files, never
overwrite sources. Data lineage is traceable through the analysis chain.
Missing or suspect values are flagged, not silently dropped or filled.

### III. Provenance

Every output links back to: the code that produced it, the input data,
and key parameter choices. Figures and tables can be regenerated from
tracked artifacts. If you can't trace how a number was made, it doesn't
belong in the paper.

## Data Sources

### 1. Hydroacoustic Moorings (NOAA/PMEL)

Six autonomous hydrophone moorings deployed in the Bransfield Strait during
the 2019–2020 field season. Each mooring consists of an anchor, acoustic
release, and line suspending the hydrophone at ~500 m depth below the sea
surface. Frequency band: 1–250 Hz.

**Deployment and recovery:**

| Mooring | Hydrophone | Deployed (UTC) | Recovered (UTC) | Lat (S) | Lon (W) | Bottom Depth (m) | Hydrophone Depth (m) |
|---------|-----------|----------------|-----------------|---------|---------|-------------------|---------------------|
| BRA28 / M1 | H17C | 2019-01-12 13:27 | 2020-02-20 | 62° 54.908' | 60° 11.979' | 1028.7 | 454.7 |
| BRA29 / M2 | H36 | 2019-01-12 19:24 | 2020-02-20 | 62° 50.997' | 59° 27.015' | 1245.7 | 421.7 |
| BRA30 / M3 | H13 | 2019-01-13 13:23 | 2020-02-18 | 62° 30.951' | 58° 53.988' | 1537.7 | 413.7 |
| BRA31 / M4 | H21 | 2019-01-13 14:07 | 2020-02-15 | 62° 32.003' | 58° 00.047' | 1764.5 | 465.5 |
| BRA32 / M5 | H24 | 2019-01-10 17:31 | 2020-02-17 | 62° 17.846' | 57° 53.671' | 1928.0 | 479.0 |
| BRA33 / M6 | H41 | 2019-01-13 21:26 | 2020-02-17/18 | 62° 14.989' | 57° 05.987' | 1717.1 | 468.1 |

- **Deployment span**: ~13 months per mooring (398–405 days)
- **Duty cycle**: 8 hours on (~2 consecutive 4-hour files), ~40 hours off,
  repeating every ~2 days. ~5% temporal coverage. Recording hour drifts
  across the deployment (not locked to a fixed UTC time).
- **Files per mooring**: 104–125 `.DAT` files (717 total across all 6)
- **Local path**: `/home/jovyan/my_data/bravoseis/NOAA/`
- **Reader**: `read_dat.py` (project module)
- **File format**: Binary `.DAT`, CSAC DAQ system, firmware `CFxLgSP3i2_3`
  - 256-byte big-endian header (timestamp, instrument ID, acquisition params)
  - Data: unsigned 16-bit big-endian (`>u2`), subtract 32768 to center
  - 24-bit ADC, stored as top 16 bits
  - 14,400,000 samples per file = 4 hours at 1 kHz
  - File size: 28,800,256 bytes (uniform)
  - File numbers are a global counter (non-sequential per mooring)
- **Sample rate**: 1000 Hz
- **ADC resolution**: 24-bit (stored as 16-bit)
- **Lowpass filter**: 400 Hz
- **Gain**: 1 (unity)
- **Calibration**: No hydrophone sensitivity or ADC voltage range in headers.
  Data is usable for relative analyses (spectrograms, event detection,
  cross-correlation) but not absolute SPL without calibration documents.
- **Processing heritage**: NOAA/PMEL IDL-based software library (Fox et al., 2001)

### 2. Bathymetry (BRAVOSEIS experiment)

Multibeam bathymetry collected during the BRAVOSEIS cruise.

- **Local path**: `/home/jovyan/my_data/bravoseis/bathymetry/bransfield.xyz`
- **Format**: ASCII XYZ (lon, lat, depth in meters)
- **Coverage**: Bransfield Strait regional
- **Coordinate system**: WGS84 geographic (lon/lat)

### 3. Sound Speed Profiles (XBT)

In-situ expendable bathythermograph (XBT) profiles collected during the
BRAVOSEIS deployment cruise (January 2019). These provide direct measurements
of the sound speed structure through which hydroacoustic signals propagate
between moorings.

- **Local path**: `/home/jovyan/my_data/bravoseis/XBT/XBT/`
- **Format**: `.asvp` files (3 variants: CSV, SVP_VERSION_2, SoundVelocity)
- **Profiles**: 11 files, depths from surface to 1022–1425 m
- **Primary profile**: `T5_18_01_19.asvp` — 1563 points, 0–1022 m,
  62.7°S 59.15°W (mid-array, between M1 and M2). Deepest clean profile.
- **Sound speed range**: ~1453 m/s at surface → ~1481 m/s at 1022 m
- **Profile shape**: Surface-limited — speed increases monotonically with
  depth (no SOFAR channel). Acoustic phases refract upward and reflect off
  the sea surface before propagating laterally.
- **Effective horizontal speed**: ~1456 m/s (harmonic mean, 0–450 m
  hydrophone depth). Significantly lower than the 1480 m/s commonly assumed.
- **Processing script**: `compute_travel_times.py`
- **Output**: `outputs/data/travel_times.json`
- **Previous reference**: Dziak et al. (2010) — U.S. Navy sound speed database.
  The in-situ XBT profiles supersede this climatological estimate.

## Technical Environment

- **Language**: Python ≥ 3.12
- **Package manager**: uv (always use `uv`, not `pip`)
- **Key packages**: cartopy, matplotlib, numpy, scipy, xarray, netcdf4, h5py,
  pygmt, pyproj, rasterio
- **Dependencies**: Managed via `pyproject.toml` + `uv.lock`
- **Compute environment**: JupyterHub (jovyan user)
- **Data storage**: `/home/jovyan/my_data/bravoseis/`
- **Version control**: Git

## Coordinate Systems & Units

- **Spatial reference**: WGS84 geographic (EPSG:4326) for data storage and
  exchange. UTM Zone 21S (EPSG:32721) for projected maps.
- **Time**: UTC throughout. Deployment logs use day-of-year format
  (e.g., `119 013:01:36:49:642` = year 2019, day 013).
- **Depth**: Meters, positive downward from sea surface.
- **Frequency**: Hz (hydroacoustic signals 1–250 Hz).
- **Missing data**: NaN for floating-point data.

## Figure Standards

Figures are for peer-reviewed **journal publication**.

- **Color palette**: Colorblind-safe (Okabe-Ito or equivalent)
- **Resolution**: 600 DPI minimum for raster, vector (PDF/SVG) preferred
- **Format**: PNG at 600 DPI for drafts, PDF/SVG for submission
- **Font**: Helvetica / Arial (sans-serif)
- **Required map elements**: Scale bar, north arrow, gridlines, colorbar,
  neatline, coordinate labels, date stamp, projection info
- **Dimensions**: Sized for single-column (3.5 in) or double-column (7 in)
  journal widths

### Caption Protocol

**Phase 1 — Temp Caption** (auto-generated on every new figure):
Describe what is plotted, axes, data sources, and processing steps applied.
Prefixed with "Temp Caption:" to distinguish from final captions.

**Phase 2 — Caption Workshop** (triggered on request):
Seven-question guided review:
1. What does this figure show?
2. What are the axes / spatial extent?
3. What trends or patterns are visible?
4. Are there inflection points or regime changes?
5. Are there outliers or anomalies?
6. What are the data sources and processing?
7. Why is this figure worth including — what does it tell the reader?

## Figure Evaluation

Both shared rubrics are adopted at **Paper** sizing tier:

- **Map rubric** (`specs/rubrics/map-evaluation-rubric.md`): 14 criteria.
  Scorecard: `specs/map-scorecard.md`.
  Paper tier: Title >= 14pt, Axis/Caption >= 10pt, Feature Labels >= 8pt,
  Min DPI 300. Neatline recommended but not required.

- **Time series rubric** (`specs/rubrics/timeseries-evaluation-rubric.md`):
  19 criteria. Scorecard: `specs/timeseries-scorecard.md`.
  Paper tier: Title >= 14pt, Axis/Caption >= 10pt, Tick Labels >= 8pt,
  Line Weight >= 1pt, Min DPI 300.

Rubrics may be updated as the project evolves.

**Mandatory**: Every new figure must include a Phase 1 Temp Caption
(see Caption Protocol above). No exceptions — even exploratory figures get
a Temp Caption so the context is never lost.

## Quality Checks

- **Temporal continuity**: Verify no unexpected gaps between consecutive
  `.DAT` files within each mooring
- **Cross-mooring consistency**: Events detected on multiple moorings should
  have physically plausible arrival time differences
- **Frequency validation**: Confirm spectral content falls within expected
  1–250 Hz instrument band
- **Location validation**: Source locations should fall within or near the
  study area (Bransfield Strait)
- **Classification validation**: Spot-check ML classifications against
  manual picks; report confusion matrix
- **Thresholds**: To be refined as analysis progresses. Document each
  threshold decision in the relevant spec when it is determined.

## Methods Notes

Key observations and derived quantities useful for the methods section of the
paper. Each entry is confirmed from data inspection or figure generation.

### Recording Duty Cycle

The hydrophones operated on a duty cycle of approximately **8 hours of
recording followed by ~40 hours off**, repeating every ~2 days. This yields
roughly **5% temporal coverage** over each mooring's 13-month deployment.
Each recording window consists of **two consecutive 4-hour DAT files**.

The recording start time **drifts slowly across the deployment** — it is not
locked to a fixed UTC hour. This is visible in the recording timeline figure
(`recording_timeline.png`), where the bars shift gradually over the 13-month
span.

Across all 6 moorings, a total of **717 DAT files** were recorded
(104–125 per mooring). M3 (BRA30/H13) has the fewest files (104), while
M5 (BRA32/H24) has the most (125). M3's last recorded file extends to
2020-02-22, slightly past the other moorings' final recordings on 2020-02-19.

**Implication for analysis**: The ~5% duty cycle means transient events
(earthquakes, ice quakes) may be missed if they occur during off periods.
However, the duty cycle is sufficient for statistical characterization of
event rates and spectral properties over weeks-to-months timescales.
Cross-mooring detections are possible only when recording windows overlap,
which they typically do since all moorings follow a similar schedule.

### Event Detection Approach (spec 001)

Events are detected using an **STA/LTA (Short-Term Average / Long-Term
Average) energy detector** — the standard first-pass method for seismo-
acoustic data. Detection runs independently in **4 frequency bands**:

| Band | Range | Target signals |
|------|-------|----------------|
| Low | 1–50 Hz | Earthquakes, T-phases |
| Mid | 10–200 Hz | Ice quakes |
| High | 50–250 Hz | Biological (whale calls) |
| Broadband | 1–250 Hz | All signal types |

**Starting STA/LTA parameters** (subject to tuning):
- STA window: 2 s, LTA window: 60 s
- Trigger: STA/LTA >= 3.0, Detrigger: STA/LTA <= 1.5
- Min event duration: 0.5 s, Min inter-event gap: 2.0 s
- Bandpass: 4th-order Butterworth, applied before STA/LTA

**Cross-mooring association**: Events on different moorings are associated
using **pair-specific travel time windows** derived from in-situ XBT sound
speed profiles (`compute_travel_times.py`). Effective horizontal speed is
computed as the harmonic mean of the sound speed profile from the surface to
the deeper hydrophone of each pair:

    c_eff = z_max / ∫(1/c(z) dz, 0, z_max)

A **15% safety factor** is applied. Resulting windows range from **21 s**
(M4–M5, 27 km) to **139 s** (M1–M6, 176 km). This replaces the previous
constant 120 s / 1480 m/s assumption, which was 6× too wide for close pairs
and slightly too narrow for the most distant pair.

| Pair | Distance | c_eff | Max window |
|------|----------|-------|------------|
| M4–M5 | 27 km | 1456 m/s | 21.2 s |
| M1–M2 | 39 km | 1456 m/s | 30.7 s |
| M3–M4 | 46 km | 1456 m/s | 36.6 s |
| M1–M6 | 176 km | 1456 m/s | 138.8 s |

Each detected event is characterized by: onset time (UTC), duration, peak
frequency, bandwidth (90% energy), peak amplitude (relative dB), and SNR
(peak STA/LTA ratio). The catalogue is saved as Parquet.

**Implementation**: obspy `recursive_sta_lta()` (C-optimized). Numpy
fallback available if obspy installation is problematic.

### Event Discrimination Approach (spec 002)

Classification proceeds in **two phases**:

**Phase 1 — Unsupervised Discovery**: Extract ~20 handcrafted spectral
features per event (band powers, duration, rise/decay time, peak frequency,
bandwidth, spectral slope, frequency modulation). Project into 2D via
**UMAP** and cluster with **HDBSCAN** to discover natural signal groupings
without imposing predefined categories. Clusters are visually inspected via
spectrogram montages and labeled by a single reviewer into broad classes
(earthquake, ice_quake, whale_call, noise, unknown) and subclasses where
the data supports it. Minimum class size: **100 events**.

**Gate**: Phase 1 deliverables (UMAP plot, montages, labeled dataset) must
be reviewed and approved before Phase 2 begins.

**Phase 2 — Supervised CNN**: Train a lightweight convolutional neural
network (~100K–500K parameters, 3–4 conv blocks) on labeled spectrogram
patches using PyTorch on GPU. Weighted cross-entropy loss, AdamW optimizer,
early stopping. Target: **>=80% macro F1** on broad classes. Apply to full
catalogue with confidence scores.

**Future extension**: Convolutional autoencoder for learned feature
extraction, potentially improving separation of ambiguous event types.
Deferred until handcrafted features are validated.

### Array Spectrograms

Ten-minute spectrogram arrays across all 6 moorings enable visual
identification of signals propagating across the network. Three reference
windows are generated (`make_spectrogram.py`):

| Window | UTC start | DAT file | Moorings | Notes |
|--------|-----------|----------|----------|-------|
| 1 | 2019-08-14 17:38 | 00001282 | M1,M2,M4,M5,M6 | M3 off-duty |
| 2 | 2019-08-14 21:16 | 00001283 | M1,M2,M4,M5,M6 | M3 off-duty |
| 3 | 2020-01-09 06:24 | 00002166 | All 6 | All moorings recording |

Spectrogram parameters: `nperseg=1024` (1.024 s), 50% overlap, fs=1000 Hz,
0–250 Hz display range. Shared colorscale (2nd–98th percentile) across all
panels per figure. These windows serve as the **tuning reference** for
STA/LTA parameter calibration.

## Project Notes

- **International collaboration**: Spain, Germany, and United States.
  The European component operates broadband land seismometers on the South
  Shetland Islands and Antarctic Peninsula. The US contributes the
  hydroacoustic network and short-period OBS network around Orca volcano.
- **Cruise**: 44-day cruise departing Ushuaia 2019-12-30, returning Punta
  Arenas 2020-02-10. Moorings recovered during this cruise.
- **References**:
  - Fox et al. (2001) — PMEL hydroacoustic processing library (IDL)
  - Dziak et al. (2010) — Sound speed propagation models
- **Data sharing**: No restrictions or embargo.
- **Target journals**: JASA (Journal of the Acoustical Society of America)
  or Antarctic Science.
