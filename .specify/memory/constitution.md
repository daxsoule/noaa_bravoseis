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

### 3. SEASICK Manual Detection Catalogue (NOAA/PMEL)

Manual detections and classifications from NOAA's SEASICK processing
package. Fixed-width text format with locations and event types.

- **Local path**: `/home/jovyan/my_data/bravoseis/NOAA/merged_data_amended.txt`
- **Format**: Fixed-width text, 18,763 events
- **Event types**: IQ (ice quake, 13,795), EQ (earthquake, 2,256),
  IDK (unknown, 2,010), SS* (specific subtypes, ~430)
- **Fields**: Timestamp (YYYYDDDHHMMSSF), n_moorings, mooring_order,
  lat, lon, errors, location parameters, event type, in/out network, notes
- **Date format**: Packed timestamp: year (4), day-of-year (3), HHMMSS (6),
  fraction (1+)
- **Known issue**: EQ events show seasonal pattern correlated with IQ events,
  suggesting misclassifications in the manual picks. Use only high-confidence
  events for validation (e.g., EQ events with low location error, or
  multi-mooring IQ events with typical ice-quake spectral characteristics).
- **Use**: Comparison with automated STA/LTA detections; benchmark for
  the discrimination pipeline. Not ground truth — treat as a reference
  catalogue with known limitations.

### 4. Earthquake Catalogue (Orca Seismic Network)

Located earthquake catalogue from the BRAVOSEIS OBS/land seismometer network
for validation of hydroacoustic detections.

- **Local path**: `/home/jovyan/my_data/bravoseis/earthquakes/Orca_EQ_data.csv`
- **Format**: CSV, 5,790 events
- **Columns**: x, y, z, zn, elevation, complete, erz, erh, date, lon, lat
- **Date format**: MATLAB datenum (days since 0000-01-00)
- **Coverage**: Bransfield Strait / Orca volcano region
- **Use**: Cross-validation of hydroacoustic T-phase detections against
  independently located seismicity. Validated: 89% of covered Orca EQs
  matched hydrophone detections; 43% matched T-phase-labeled events

### 5. USGS Global Earthquake Catalogue

Teleseismic earthquake catalogue retrieved from the USGS FDSN event service
for validation of large events.

- **Retrieved via**: `https://earthquake.usgs.gov/fdsnws/event/1/query`
- **Local path**: `outputs/data/usgs_eq_catalogue.csv`
- **Query**: −66° to −58°N, −65° to −53°W, 2019-01-01 to 2020-03-01, M≥0
- **Events**: 4 (M4.6–5.5) — confirms Bransfield seismicity is predominantly
  small/local, below global network detection threshold
- **Use**: Context only; Orca catalogue is the primary seismic reference

### 6. Sound Speed Profiles (XBT)

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
acoustic data.

**STA/LTA parameters**:
- STA window: 2 s, LTA window: 60 s
- Trigger: STA/LTA >= 3.0, Detrigger: STA/LTA <= 1.5
- Min event duration: 0.5 s, Min inter-event gap: 2.0 s
- All filters: 4th-order Butterworth

**Three-pass detection strategy**: Initial detection used 4 overlapping
frequency bands (1–50, 10–200, 50–250, 1–250 Hz) with cross-band
deduplication. QC revealed two problems: (1) low-frequency energy (< 15 Hz,
primarily T-phases and earthquake coda) dominated the LTA window,
suppressing STA/LTA ratios for concurrent higher-frequency transients; and
(2) the deduplication step masked mid/high events by relabeling them under
the highest-SNR band (almost always low). Average spectral profiles across
moorings confirmed dominant energy below 15–20 Hz.

The solution separates the data into **three non-overlapping frequency
bands** before computing STA/LTA, so each pass's LTA reflects only its own
frequency regime:

| Pass | Filter | Range | Target signals |
|------|--------|-------|----------------|
| Pass 1 | Lowpass 15 Hz | 1–15 Hz | Earthquakes, T-phases, ice quakes (low-freq component) |
| Pass 2 | Bandpass 15–30 Hz | 15–30 Hz | Fin whale calls (~20 Hz), ice quakes, mixed seismicity |
| Pass 3 | Highpass 30 Hz | 30–250 Hz | Ice quakes (high-freq component), other whale calls, biological |

**Breakpoint rationale**: 15 Hz separates the dominant seismic energy from
the mid band; 30 Hz separates the fin whale / low ice-quake regime from the
higher biological and cryogenic signals. These are energy-regime boundaries,
not signal-type boundaries — a single broadband ice quake may trigger in all
three passes.

No cross-pass deduplication is applied. A physical event appearing in
multiple passes is resolved during the downstream discrimination phase.

**Results** (717 files, 6 moorings):

| Band | Events |
|------|--------|
| Low (1–15 Hz) | 84,698 |
| Mid (15–30 Hz) | 132,494 |
| High (30–250 Hz) | 79,978 |
| **Total** | **297,170** |

For comparison, the original 4-band approach detected 152,040 events with
only 17,781 in the mid band — the three-pass strategy recovered 7× more
mid-band events.

Each detected event is characterized by: onset time (UTC), duration, peak
frequency, bandwidth (90% energy), peak amplitude (relative dB), SNR
(peak STA/LTA ratio), and detection pass. The catalogue is saved as Parquet.

**Implementation**: Custom vectorized `classic_sta_lta()` using numpy
cumulative sums.

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

### Onset Refinement (spec 001)

STA/LTA onset picks are biased late — FP validation showed only 11% hit the
true first arrival, 68% fell in the event coda. A second-pass **AIC picker**
(Maeda 1985) with **kurtosis fallback** refines onsets for source location.

**Algorithm**: For each event, extract a 7 s window (5 s pre-trigger + 2 s
post-trigger), apply the same band-specific filter used for detection, then
run AIC on the squared envelope. If AIC quality < 0.4, fall back to a
kurtosis-based picker (0.5 s sliding window). If both fail, keep the original
onset.

**Constraint**: Positive shifts (refined onset later than STA/LTA trigger) are
rejected — the STA/LTA already triggers late, so any forward shift is treated
as a picker error. Rejected events revert to the original onset with
downgraded quality.

**Quality grading**:
- Grade A (quality >= 0.7): high confidence — use for source location
- Grade B (quality 0.4–0.7): moderate confidence — use for source location
- Grade C (quality < 0.4): low confidence — **exclude from location**, retain
  in catalogue for classification and statistical analysis

**Results** (297,170 events):

| Method | Events | Fraction |
|--------|--------|----------|
| AIC | 284,425 | 95.7% |
| Kurtosis | 686 | 0.2% |
| Original (kept) | 12,059 | 4.1% |

| Grade | Events | Fraction |
|-------|--------|----------|
| A | 233,751 | 78.7% |
| B | 51,007 | 17.2% |
| C | 12,412 | 4.2% |

Median onset shift: **−0.83 s** (IQR: [−1.46, −0.39] s). Shifts are
consistently negative (earlier), confirming the picker moves onsets backward
from the coda toward the true first arrival.

**Known limitation**: Low-frequency events (< 5 Hz) with emergent, gradual
onsets are poorly served by AIC, which assumes a sharp noise-to-signal
transition. These events tend to receive grade C picks. A frequency-domain
onset method or manual picks may be needed for this subset.

**QC montage review** (50 events, overweighted grade C): 34/50 acceptable.
Most issues were late picks on emergent low-frequency signals. The grade
system reliably flags uncertain picks.

**New catalogue columns**: `onset_utc_refined`, `onset_shift_s`,
`onset_method` (aic/kurtosis/original), `onset_quality` (0–1),
`onset_grade` (A/B/C).

**Implementation**: `refine_onsets.py` (vectorized AIC using cumulative sums).
QC: `validate_onsets.py`.

### Seismic Onset Refinement (spec 001)

The AIC picker picks late on emergent T-phases — the 4.3% grade C picks are
concentrated in 6 seismic clusters (11,958 events). A **seismic-tuned dual
onset picker** runs two independent methods and uses an **AIC-first rescue
strategy**: keep the AIC pick when it works (grade A/B), switch to the seismic
pick only when AIC is struggling (grade C or quality < 0.4).

**Target clusters** (identified during Phase 1c cluster labeling):

| Cluster | Events | Type | Filter | Pre-window |
|---------|--------|------|--------|------------|
| low_0 | 4,686 | emergent T-phase | 1–15 Hz | 10 s |
| low_1 | 3,666 | mixed seismic | 1–15 Hz | 8 s |
| low_2_2 | 1,134 | impulsive broadband | 1–15 Hz | 5 s |
| mid_0 | 1,056 | mixed seismic | 5–30 Hz | 8 s |
| mid_3_1 | 1,067 | emergent broadband | 5–30 Hz | 8 s |
| mid_3_3 | 349 | impulsive | 5–30 Hz | 5 s |

Mid-band filter extends to 5 Hz (below the 15 Hz detection passband) to
capture leading-edge T-phase energy that arrives before the mid-band onset.

**Picker 1 — Envelope STA/LTA**: Hilbert transform → 50 ms boxcar smooth →
energy (squared envelope) → STA/LTA (0.5 s / 5.0 s) via vectorized cumsum →
trigger at ratio 2.0 → backtrack to ratio 1.2 (the "creep onset" where energy
first rises above background). Quality from peak STA/LTA and rise steadiness.

**Picker 2 — Kurtosis onset**: Sliding-window excess kurtosis (0.25 s window,
cumsum method) → noise kurtosis from first 25% of window → z-score relative to
noise stats → trigger at 3σ with 0.1 s sustained exceedance → backtrack to
z > 1.0 for true onset. Quality from peak z-score and sustainment.

**Combining logic (AIC-first rescue)**:
1. If existing AIC pick is grade A or B (quality >= 0.4): **keep AIC pick**
2. If AIC is grade C: run both seismic pickers, reject picks with quality
   < 0.15, positive shift, or shift > 95% of pre-window
3. Take the earlier valid seismic pick (more negative shift = earlier onset)
4. If both seismic pickers fail: keep AIC pick as-is (no downgrade)

**Results** (11,958 seismic events):

| Method | Events | Fraction |
|--------|--------|----------|
| AIC kept | 11,525 | 96.4% |
| Envelope rescue | 354 | 3.0% |
| Kurtosis rescue | 79 | 0.7% |

| Grade | Old (AIC) | New (dual) |
|-------|-----------|------------|
| A | 9,311 (77.9%) | 9,315 (77.9%) |
| B | 2,137 (17.9%) | 2,324 (19.4%) |
| C | 510 (4.3%) | 319 (2.7%) |

Grade C reduced by **37%** (510 → 319). Of the 510 old grade C events, 191
were rescued to grade A (4) or B (187). No events were downgraded — the rescue
is strictly additive.

**Output**: `outputs/data/seismic_onsets.parquet` — per-event columns:
event_id, cluster_id, seis_onset_utc, seis_onset_shift_s, seis_onset_method,
seis_onset_quality, seis_onset_grade, env_pick_shift_s, env_pick_quality,
kurt_pick_shift_s, kurt_pick_quality, pre_window_s, filter_low_hz,
filter_high_hz.

**Implementation**: `pick_seismic_onsets.py`. QC figures in
`outputs/figures/exploratory/seismic_onsets/`.

### Event Discrimination Approach (spec 002)

Classification proceeds in **two phases**:

**Phase 1 — Unsupervised Discovery**: Extract ~20 handcrafted spectral
features per event (band powers, duration, rise/decay time, peak frequency,
bandwidth, spectral slope, frequency modulation). Cluster **each detection
band independently** (low <15 Hz, mid 15–30 Hz, high >30 Hz) using UMAP
projection into 2D followed by HDBSCAN clustering. Clusters are visually
inspected via spectrogram montages and labeled by a single reviewer into
broad classes (earthquake, ice_quake, whale_call, noise, unknown) and
subclasses where the data supports it. Minimum class size: **100 events**.

**Per-band clustering rationale**: An initial all-band clustering pass
(297,170 events, 19 features, UMAP + HDBSCAN) produced a single
mega-cluster containing 99% of events. Visual inspection of the UMAP
embedding revealed clear internal structure (arms, ridges, density
gradients) organized primarily by detection band — low-band events
dominated the left side, mid-band the center, and high-band the right.
Feature-colored maps confirmed strong frequency-driven gradients (peak
frequency, spectral centroid, spectral slope) as the dominant axes of
variation. The 11 splinter clusters (totaling <1% of events) contained
recognizable signals — fin whale 20 Hz calls, tonal high-frequency whale
calls, T-phases, broadband impulsive events — but were scattered across
clusters by outlier status rather than grouped by signal type.

Clustering per band removes the dominant frequency-regime axis (which is
already known from detection) and allows subtler within-band structure to
emerge: T-phases vs. local earthquakes in the low band, fin whale pulse
trains vs. background in the mid band, tonal whale calls vs. ice cracking
in the high band.

**Phase 1 classification results** (feature-based, expert-reviewed):

Per-band HDBSCAN clustering produced 3/4/5 clusters for low/mid/high bands
respectively, plus noise. The small distinct clusters (totaling ~6,500 events)
were identified as T-phases via spectrogram montage review (confirmed by Bob
Dziak, NOAA/PMEL). However, feature-based filtering recovered a much larger
T-phase population hiding in the bulk clusters:

| Label | Feature criteria | Detections | Unique events (est) |
|-------|-----------------|-----------|-------------------|
| **T-phase (earthquake)** | peak_freq <30 Hz, power >48 dB, slope <−0.5, duration ≤3 s | ~52,800 | ~42,000 |
| **Cryogenic (icequake)** | duration >3 s, power >48 dB, peak_freq <30 Hz, slope <−0.2 | ~23,900 | ~22,000 |
| **Type A (broadband transient)** | Positive spectral slope, ~200 Hz peak, 41 dB, high freq modulation | ~13,700 | TBD |
| **Bulk (unresolved)** | Remainder — heterogeneous, low SNR | ~206,000 | TBD |

Key distinguishing features between T-phases and icequakes:
- **Duration**: T-phases ≤3 s (impulsive), icequakes >3 s (sustained)
- **Spectral slope**: T-phases steeper (< −0.5), icequakes moderate (−0.2 to −0.5)
- **Temporal pattern**: T-phases are episodic/bursty (swarm-correlated),
  icequakes show austral summer peaks (Jan–Mar) consistent with seasonal ice dynamics
- **Rise time**: Icequakes have slower rise (mean 1.6 s) vs T-phases (fast onset)

**Type A broadband transients identified as vessel noise** based on:
- Spectral character: positive spectral slope, peak ~188 Hz, 41 dB, broadband
  (211 Hz bandwidth), high frequency modulation — classic propeller
  cavitation / ship machinery noise
- Temporal pattern: 85% of events concentrate in burst days (>100 events/day)
  with quiet gaps between. Each burst lasts 1–4 days, consistent with vessel
  transits through the array
- Multi-mooring simultaneity: 47% of 200 s time bins have 2+ moorings
  detecting simultaneously; 11% have 4+ moorings — consistent with a single
  moving broadband source
- Seasonal correlation: peak activity May–Sep (austral winter), matching the
  seasonal westward shift of krill fishing trawlers into the Bransfield Strait
  (CCAMLR Subarea 48.1). Lower activity Nov–Mar when tourism (smaller,
  quieter vessels) dominates Antarctic Peninsula traffic
- **~24 distinct vessel passages** estimated over 13 months, producing ~11,900
  unique physical events across 40 burst days
- Passage 1 (Jan 13, 2019) aligns with the R/V Sarmiento de Gamboa deploying
  the BRAVOSEIS hydrophone array (Jan 4–17 cruise)
- Cross-band deduplication: 12,596 detections → 11,887 unique events (only
  688 multi-band); 10,458 after removing overlap with T-phase/icequake labels

Literature context: Ship visits to the Antarctic Peninsula/South Shetland
Islands account for 88% of all Southern Ocean vessel traffic (McCarthy et al.
2022, PNAS). Traffic composition: tourism 67%, research 21%, fishing 7%,
supply 5%. Krill fishing in CCAMLR Subarea 48.1 (Bransfield Strait) accounts
for up to 57% of the regional krill harvest, with the fleet shifting westward
in winter months.

**Validation against Orca EQ catalogue** (5,789 located earthquakes):
- Hydrophone coverage overlaps only ~11% of Orca catalogue (636 events)
  due to ~5% duty cycle
- Of covered Orca EQs: **89% matched at least one hydrophone detection**
  (within 5 min), confirming good detector sensitivity
- **43% matched T-phase-labeled events** specifically; remainder detected
  but below the 48 dB power threshold (weaker events in the bulk population)
- Median T-phase arrival delay: **28.2 s** after EQ origin — consistent with
  ~40 km propagation at ~1.45 km/s water speed
- USGS catalogue contains only 4 events ≥M4.6 in the region/period

**USGS catalogue**: Retrieved via FDSN API for −66° to −58°N, −65° to −53°W,
2019-01-01 to 2020-03-01, M≥0. Only 4 events (M4.6–5.5), confirming the
Bransfield Strait seismicity is predominantly small/local, well below the
global network detection threshold.

**Classification completeness and threshold rationale**:

The feature-based T-phase filter (power >48 dB, slope < −0.5) was calibrated
against the distinct HDBSCAN clusters confirmed by expert montage review.
Cross-validation against the Orca EQ catalogue shows this filter captures
the **high-confidence** T-phase population but not the full earthquake
detection set:

- Of 636 Orca EQs with hydrophone coverage, 89% produced at least one
  hydrophone detection within 5 minutes of the origin time
- 43% (275 events) matched our T-phase-labeled events specifically
- The remaining 57% (362 events) were detected but fell below classification
  thresholds — median power 41 dB (vs 48 dB cutoff), median spectral slope
  −0.22 (vs −0.5 cutoff). Spectrogram montage review confirms these are
  weaker T-phases that blend into the bulk population in feature space.

This is expected: smaller-magnitude earthquakes produce weaker hydroacoustic
signatures that are harder to distinguish from non-earthquake transients
using handcrafted features alone. The Phase 1 unsupervised approach is
designed to identify what separates cleanly without supervision. Recovery
of the weaker earthquake tail is deferred to Phase 2 (supervised CNN),
where Orca-matched examples can serve as training labels for events near
the decision boundary.

**Gate**: Phase 1 deliverables (UMAP plot, montages, labeled dataset) must
be reviewed and approved before Phase 2 begins.

**Phase 2 — Hybrid CNN+MLP Classifier**: A dual-branch neural network
combining spectrogram patches (CNN branch, 4 conv blocks) with handcrafted
features (MLP branch, 2 FC layers), fused before a classification head.
Total: 258,627 parameters. PyTorch on NVIDIA L40S GPU.

Architecture rationale: Pure CNN (spectrogram-only) achieved only 39% macro
F1 — spectrograms alone are insufficiently discriminative because Phase 1
labels were defined by summary statistics (peak frequency, spectral slope,
power, duration), not visual appearance. The hybrid model feeds both raw
spectrograms and the handcrafted features, allowing the CNN branch to learn
complementary visual patterns while the MLP branch captures the statistics
the labels were based on.

Spectrogram parameters: 8 s window (2 s pre-event pad), nperseg=256,
overlap=87.5%, 0–100 Hz, resized to 64×128 pixels. Per-sample normalization
(zero-mean, unit-variance) so the model learns spectral shape, not absolute
power level.

Training: 70/15/15 train/val/test split (62,683 / 13,432 / 13,432 events).
Weighted random sampling for class balance. Cross-entropy loss (unweighted —
sampler handles balance). AdamW optimizer (lr=1e-3, weight_decay=1e-4),
cosine annealing LR schedule. SpecAugment-style augmentation (time/freq
masking, time shift). Early stopping (patience=8 epochs on val macro F1).

**Test set results**: 95.1% accuracy, **93.7% macro F1** (target: >=80%).

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| T-phase | 1.00 | 0.92 | 0.96 | 8,402 |
| Icequake | 0.93 | 1.00 | 0.96 | 3,435 |
| Vessel | 0.80 | 1.00 | 0.89 | 1,595 |

Main confusion: 400 T-phases predicted as vessel (4.8%), 249 T-phases
predicted as icequake (3.0%). Icequake and vessel classes have near-perfect
recall.

**Bulk population predictions** (207,528 events):

| CNN Prediction | Count | % | High-conf (>=0.8) |
|---------------|-------|---|-------------------|
| T-phase | 123,866 | 59.7% | 117,077 |
| Icequake | 49,282 | 23.7% | 44,989 |
| Vessel | 34,380 | 16.6% | 29,280 |

Confidence distributions are heavily right-skewed (median >0.98 for all
classes), indicating the model is decisive even on previously unclassified
events.

**Implementation**: `train_cnn.py` (extraction, training, inference).
Cached spectrograms in `outputs/data/spectrograms/`. Model checkpoint in
`outputs/data/cnn_model.pt`. Predictions in `outputs/data/cnn_predictions.parquet`.

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

## Preliminary Results

### Catalogue Summary

The three-pass STA/LTA detector produced **297,170 detections** across 717
four-hour DAT files from 6 moorings over ~13 months (Jan 2019 – Feb 2020).
Due to the ~5% duty cycle, this represents a sparse but consistent sample of
the Bransfield Strait acoustic environment.

### Phase 1 Event Classification

Feature-based classification (UMAP + HDBSCAN clustering, supplemented by
feature filtering and expert montage review) identified four populations:

| Class | Detections | Unique events (est) | % of catalogue |
|-------|-----------|-------------------|---------------|
| T-phase (earthquake) | 55,783 | ~42,000 | 18.8% |
| Cryogenic (icequake) | 23,331 | ~22,000 | 7.8% |
| Vessel noise (Type A) | 10,458 | ~11,900 | 3.5% |
| Unresolved (bulk) | 207,598 | — | 69.9% |

T-phase classification confirmed by Bob Dziak (NOAA/PMEL) via spectrogram
montage review. Icequake classification supported by seasonal temporal
pattern (austral summer peaks). Vessel noise identification supported by
burst pattern, multi-mooring simultaneity, and seasonal correlation with
krill fishing fleet activity (see below). The classified populations are
mutually exclusive in feature space (zero overlap between T-phases and
vessel noise during seismic swarm periods).

### Phase 2 CNN Classification

The hybrid CNN+MLP classifier (93.7% macro F1 on test set) was applied to
the 207,528 previously unclassified bulk events, recovering the full
catalogue:

| Class | Phase 1 | Phase 2 (high-conf) | Combined |
|-------|---------|-------------------|----------|
| T-phase | 55,783 | 117,077 | **172,860** |
| Icequake | 23,331 | 44,989 | **68,320** |
| Vessel noise | 10,458 | 29,280 | **39,738** |
| Low-confidence | — | 16,182 | 16,182 |

The Phase 2 model tripled the classified T-phase population and nearly
tripled icequakes, consistent with the Phase 1 finding that ~50% of bulk
events had features consistent with weak T-phases.

### Cross-Validation Against Orca Seismic Network

The co-deployed Orca OBS/land seismometer network located 5,789 earthquakes
in the Bransfield Strait during the same period. Due to the hydrophone duty
cycle, only **636 Orca events (11%)** fall within our recording windows.

Of these 636 covered earthquakes:
- **89% (567)** produced at least one hydrophone detection within 5 minutes
  of the seismic origin time — confirming high detector sensitivity
- **43% (275)** matched T-phase-labeled events specifically
- **57% (362)** were detected but below the feature classification threshold
  (median power 41 dB vs 48 dB cutoff) — consistent with smaller-magnitude
  events producing weaker hydroacoustic signatures
- **Median T-phase arrival delay: 28.2 s** — physically consistent with
  ~40 km propagation at ~1.45 km/s effective water speed

The USGS global catalogue contains only 4 events ≥M4.6 in the study region
during this period, confirming that Bransfield Strait seismicity is
predominantly local and small-magnitude.

### Hydroacoustic Detection of Seismicity Beyond the OBS Network

The hydrophone array detected several major T-phase swarms that the Orca
seismic network largely missed:

**Feb 11, 2019**: 4,976 T-phase detections in a single day across 5 moorings
(M3 off-duty), sustained over ~8 hours at rates up to 200 events/hour.
Spectrogram montage confirms classic T-phase signatures (impulsive broadband
bursts with dominant energy below 50 Hz). The Orca catalogue contains only
117 events in the surrounding week (Feb 8–14), with the Orca activity peaking
2–3 days later (Feb 13–14) rather than on Feb 11.

**Apr 22–24, 2019**: A three-day swarm building to 3,495 T-phases on the
peak day (Apr 24), visible across 5 moorings. Orca recorded only 21 events
in the same week — background-level activity with no corresponding swarm.

These discrepancies suggest the hydrophone array is detecting earthquake
swarms **outside the Orca network's location capability** — either beyond
its geographic footprint (the Orca network is focused on the Orca volcano
area near 58.4°W, 62.4°S) or below its location threshold (events arriving
at too few OBS stations to locate). The hydrophones, with a much larger
detection radius due to efficient T-phase propagation through the water
column, capture seismicity that the local seismic network cannot.

**Implication**: Source location of the hydroacoustic T-phase catalogue
(using cross-mooring arrival time differences) is needed to determine
whether these swarms originate from known fault structures, from along
the Bransfield rift axis, or from more distant sources. This is the
critical next analysis step.

### Vessel Noise (Type A Broadband Transients)

The ~10,500 Type A detections (11,900 unique physical events after cross-band
deduplication) are identified as vessel traffic based on spectral, temporal,
and spatial evidence:

- **Spectral signature**: Positive spectral slope, peak ~188 Hz, 41 dB median
  power, broadband (211 Hz bandwidth) — characteristic of propeller cavitation
  and ship machinery noise
- **Burst pattern**: 85% of events concentrate in burst days (>100 events/day);
  each burst lasts 1–4 days consistent with vessel transits through the array
- **Multi-mooring detection**: 47% of 200 s time bins have 2+ moorings
  detecting simultaneously; 11% have 4+ — a single moving broadband source
  illuminating the full array
- **~24 distinct vessel passages** over 13 months, with the largest bursts
  generating 500–1,100 events over 2–3 days
- **Seasonal pattern**: Peak activity May–Sep (austral winter), correlating
  with the seasonal westward shift of krill fishing trawlers into the
  Bransfield Strait (CCAMLR Subarea 48.1). Lower activity Nov–Mar when
  tourism vessels (smaller, quieter) dominate Antarctic Peninsula traffic
- **Deployment cruise detection**: Passage 1 (Jan 13, 110 events) aligns with
  the R/V Sarmiento de Gamboa deploying the hydrophone array (Jan 4–17, 2019)

**No contamination of seismic results**: During both T-phase swarm periods
(Feb 11, Apr 22–24), Type A vessel noise counts were 0–9 events/day
(background level), with zero overlap between T-phase and vessel noise
classifications. The populations occupy completely separate regions of
feature space (negative vs positive spectral slope; <30 Hz vs >100 Hz
peak frequency).

### Bulk Population (Unresolved)

The 207,598 unresolved events (70% of catalogue) represent the detector's
ambient detection floor — events that triggered STA/LTA but lack distinctive
features for confident classification:

- **11 dB weaker** than classified events (median power 42.4 dB vs 53.7 dB)
- **Spectral slope centered near zero** (median −0.33) — no strong spectral
  character in either direction
- **Three dominant clusters** account for 96%: mid_3 (90,669), high_4
  (56,106), low_2 (53,352) — the large "background" clusters in each band
- **36% have power <40 dB** (near noise floor); 46% have power 40–48 dB
  (below classification thresholds); 18% have power ≥48 dB
- **49.5% pass relaxed T-phase criteria** (peak_freq <30 Hz, slope <−0.2,
  duration ≤5 s) — likely weak or distant earthquakes below the conservative
  48 dB power threshold
- Spectrogram montage review shows a mix of: faint noise-floor triggers,
  weak T-phases, possible icequakes below threshold, and some higher-frequency
  signals (possible whale calls, reserved for separate study)

The bulk population is the primary target for **Phase 2 supervised CNN**
classification, where high-confidence Phase 1 labels and Orca-matched
examples can serve as training data to recover weaker events near the
decision boundary.

### Icequake Seasonality

The 22,000 identified icequakes show a clear seasonal pattern:
- **Austral summer peaks**: Jan–Mar 2019 and Jan 2020
- **Minimum**: Sep–Dec 2019 (late winter / early spring)
- Secondary peak in Jul 2019 (austral winter — possibly thermal cracking
  or pressure ridging of sea ice)

This seasonal modulation is distinct from the episodic/bursty temporal
pattern of T-phases and provides independent support for the cryogenic
classification. The pattern is consistent with increased calving, ice
breakup, and glacial activity during warmer months.

### Source Location (Grid-Search TDOA)

Event locations are computed by **grid-search TDOA** (time-difference-of-arrival)
minimization over a geographic grid covering the Bransfield Strait. For each
cross-mooring association with >=3 moorings, the algorithm finds the grid point
that minimizes the RMS residual between observed and predicted inter-station
travel time differences.

**Grid**: 0.01° spacing (~1 km) covering the study area ±0.5° padding. Total
154,700 grid points. Geodesic distances precomputed from each grid point to
all 6 moorings using WGS84 ellipsoid. Effective horizontal sound speed:
**1455.5 m/s** (XBT-derived, harmonic mean over water column).

**Multipath protection**: Three mechanisms address multipath-contaminated onsets:

1. **Per-mooring outlier detection** (>=4 moorings): If one mooring's individual
   residual exceeds 3× the median residual AND >1 s, relocate without that
   mooring. If residual improves by >30%, accept the reduced solution. Applied
   to 403 events.

2. **Jackknife (leave-one-out) validation** (>=4 moorings): Relocate N times
   dropping each mooring in turn. If max location shift >15 km, flag as
   jackknife-unstable and downgrade quality tier. 281 events (1.9%) flagged.

3. **Distance constraint**: Locations >150 km from the array centroid are
   assigned tier D (unreliable), since the array geometry cannot constrain
   locations far outside its footprint.

**Quality tiers**:

| Tier | Criteria | Count | Median residual |
|------|----------|-------|-----------------|
| A | >=4 moorings, residual <1 s, jackknife-stable | 4,304 | 0.00 s |
| B | >=3 moorings, residual <2 s | 6,979 | 0.00 s |
| C | >=3 moorings, residual 2–5 s or jackknife-unstable | 926 | 3.29 s |
| D | 2 moorings, >150 km from array, or residual >5 s | 28,575 | — |

**Note**: Tier A/B residuals of 0.00 s are expected for 3-mooring events —
with 2 TDOAs and 2 unknowns (lat, lon), the system is fully determined and
the grid search finds an exact-fit solution. Residual-based quality assessment
is only meaningful for >=4 mooring events (over-determined system). Tier A
requires >=4 moorings for this reason.

**Results**: 12,209 events located (tiers A+B+C):

| Class | Located | Tier A | Tier B | Tier C |
|-------|---------|--------|--------|--------|
| T-phase | 9,466 | 3,494 | 5,224 | 748 |
| Icequake | 1,030 | — | — | — |
| Vessel | 377 | — | — | — |
| Unclassified | 1,336 | — | — | — |

T-phase locations concentrate along the **central Bransfield Rift axis**
between BRA30–BRA33, consistent with known extensional tectonics. Temporal
panels (2-month bins) show clear swarm clustering, with the Feb 11 and
Apr 22-24 swarms spatially concentrated in the central basin.

**Known issue**: Icequake locations scatter into open water (Drake Passage),
which is physically implausible — icequakes should originate near coastlines
and glaciers. This likely reflects: (1) misclassification by the CNN of
distant T-phases or other signals as icequakes, (2) poor 3-mooring locations
with no residual-based quality check, and/or (3) incorrect associations
linking unrelated events. Icequake locations require additional filtering
(e.g., restricting to proximity of known ice sources) before publication.

**Implementation**: `scripts/locate_events.py`. Outputs:
`outputs/data/event_locations.parquet`,
`outputs/figures/exploratory/location/` (overview map, 6-panel T-phase and
icequake maps with time-colored, size-scaled dots).

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
  - McCarthy et al. (2022) — Ship traffic connects Antarctica's fragile
    coasts to worldwide ecosystems (PNAS). AIS vessel traffic 2014–2018.
- **Data sharing**: No restrictions or embargo.
- **Target journals**: JASA (Journal of the Acoustical Society of America)
  or Antarctic Science.
