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

### 3. Sound Speed Profiles

Travel times and velocities determined by applying propagation models to the
U.S. Navy ocean sound speed database (Dziak et al., 2010). The Bransfield
Strait sound velocity profile is surface-limited: velocity decreases linearly
from the seafloor to the sea surface, causing acoustic phases to refract
upward and reflect off the sea-surface-air interface before propagating
laterally through the basin.

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
