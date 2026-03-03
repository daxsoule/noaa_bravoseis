# Analysis Specification: Acoustic Event Detection

**Directory**: `specs/001-event-detection`
**Created**: 2026-03-02
**Status**: Draft
**Branch**: `001-event-detection`

## Research Question(s)

1. What discrete acoustic events are present in the BRAVOSEIS hydrophone
   recordings, and how many occur per recording window across the 13-month
   deployment?
2. Can events be reliably detected across multiple moorings simultaneously,
   enabling downstream arrival-time-based source location?
3. What is the distribution of detected events in time, frequency, and
   duration — and do these distributions suggest distinct signal populations
   (e.g., earthquakes vs. ice quakes vs. whale calls)?

**Hypothesis**: The Bransfield Strait hydroacoustic record contains a rich
mixture of tectonic, cryogenic, and biological signals. A broadband energy
detector operating on short-time spectrograms should yield a catalogue of
discrete events that cluster into distinct spectral families, with event
rates varying seasonally (ice quakes peaking in austral summer, whale calls
peaking in austral autumn/winter).

## Data Description

### Primary Data

- **Source**: NOAA/PMEL autonomous hydrophone moorings (BRA28–BRA33 / M1–M6)
- **Coverage**: 2019-01-12 to 2020-02-22 UTC, ~13 months per mooring
- **Temporal sampling**: 8 hours on / ~40 hours off duty cycle (~5% coverage).
  Two consecutive 4-hour DAT files per recording window.
- **Format**: Binary `.DAT` files, 256-byte header + 14,400,000 unsigned
  16-bit big-endian samples at 1 kHz. Reader: `read_dat.py`
- **Access**: `/home/jovyan/my_data/bravoseis/NOAA/m{1-6}-*/`
- **Total files**: 717 across 6 moorings (104–125 per mooring)
- **Known issues**:
  - No absolute calibration (hydrophone sensitivity, ADC voltage range not
    in headers). Detection thresholds must be relative, not absolute SPL.
  - M3 has fewest files (104). Recording start times drift across deployment.
  - ~5% temporal coverage means transient events during off-periods are missed.

### Secondary Data

- **Source**: Bathymetry (Bransfield Strait multibeam XYZ + MGDS Orca gridded)
- **Purpose**: Not directly used in detection, but needed for downstream
  source location validation. Already loaded in `make_bathy_map.py`.

## Methods Overview

### 1. Data preparation

- Load each DAT file via `read_dat()`, yielding centered float64 waveforms
- Compute short-time spectrograms per file: `scipy.signal.spectrogram()`
  with `nperseg=1024` (1.024 s window), `noverlap=512` (50% overlap),
  `fs=1000`. This produces ~1 Hz frequency resolution and ~0.5 s time steps.
- Convert power to dB: `10 * log10(Sxx + 1e-20)`
- [TODO: Decide whether to apply bandpass pre-filtering (e.g., 1–250 Hz)
  or work with the full 0–500 Hz band and filter post-detection]

### 2. Detection algorithm

- **Approach**: STA/LTA (Short-Term Average / Long-Term Average) energy
  detector applied to broadband energy or per-frequency-band energy envelopes.
  - Compute energy in sliding windows (STA: ~1–5 s, LTA: ~30–120 s)
  - Trigger when STA/LTA ratio exceeds threshold; detrigger when it falls
    below a lower threshold
  - Record event onset time, duration, and peak STA/LTA ratio
- **Frequency bands**: Run detection independently in multiple bands to
  capture different signal types:
  - Low-frequency band (1–50 Hz): earthquakes, T-phases
  - Mid-frequency band (10–200 Hz): ice quakes
  - High-frequency band (50–250 Hz): biological signals
  - Broadband (1–250 Hz): catches everything
- **Per-mooring detection**: Each mooring is processed independently. Cross-
  mooring association is a separate downstream step.
- [TODO: STA/LTA window lengths and trigger/detrigger thresholds — need
  empirical tuning on a subset of data. See Validation Approach.]
- [TODO: Minimum event duration and inter-event gap to avoid splitting
  single events into fragments]

### 3. Event characterization (per detection)

For each detected event, extract:
- Onset time (UTC), duration, end time
- Peak frequency, bandwidth (frequency range containing 90% of energy)
- Peak amplitude (relative dB), SNR (peak STA/LTA ratio)
- Mooring ID, DAT file number

### 4. Cross-mooring association

- For each recording window where multiple moorings are recording
  simultaneously, find events detected on 2+ moorings within a plausible
  travel-time window
- [TODO: Maximum inter-mooring travel time. At ~1480 m/s sound speed and
  ~300 km maximum mooring separation, max travel time is ~200 s. Events on
  different moorings within this window are candidate associations.]

### 5. Catalogue assembly

- Compile all detections into a single Parquet catalogue with columns:
  event_id, mooring, file_number, onset_utc, duration_s, end_utc,
  peak_freq_hz, bandwidth_hz, peak_db, snr, detection_band
- Compile cross-mooring associations as a separate table

**Justification**: STA/LTA is the standard first-pass detector for seismo-
acoustic data. It requires no training data (unlike ML), produces a complete
catalogue that can be subsequently classified, and its parameters are
physically interpretable. The multi-band approach ensures sensitivity to the
three target signal types (earthquakes, ice, biology) which occupy different
frequency ranges.

## Expected Outputs

### Figures

- **Figure: Detection rate timeline** — Events per recording window over
  the 13-month deployment, stacked by frequency band. Shows seasonal
  patterns and recording coverage.
- **Figure: Event duration vs. peak frequency** — Scatter plot of all
  detections, colored by detection band. Should reveal clustering by
  signal type.
- **Figure: Example detections** — Array spectrograms (reusing
  `make_spectrogram.py` style) for 3–5 representative events showing
  the detection trigger points overlaid.
- **Figure: Cross-mooring detection statistics** — How many events are
  seen on 1, 2, 3, ... 6 moorings simultaneously.

### Tables/Statistics

- **Table: Catalogue summary** — Total detections per mooring, per band,
  and overall. Mean/median event duration and SNR.
- **Table: Cross-mooring association counts** — Number of events detected
  on N moorings, by frequency band.

### Key Metrics

- Total number of detected events (all moorings, all bands)
- Fraction of events detected on 2+ moorings (candidates for source location)
- Event rate per hour of recording, by band
- Median and interquartile range of event duration and peak frequency

## Validation Approach

- **Threshold tuning on known examples**: Use the three spectrogram windows
  already generated (2019-08-14 17:38, 2019-08-14 21:16, 2020-01-09 06:24)
  as a visual reference. Manually identify 10–20 events visible in these
  spectrograms and confirm the detector finds them.
- **False positive check**: Randomly sample 50 detections and visually
  inspect their spectrograms. Target: <20% false positive rate.
- **Cross-mooring consistency**: For events detected on multiple moorings,
  verify arrival time differences are physically plausible (< 200 s, consistent
  with sound propagation at ~1480 m/s).
- **Frequency validation**: Confirm detected events have spectral content
  within the 1–250 Hz instrument band (no aliased or spurious detections
  above Nyquist).
- **Comparison with known seismicity**: If earthquake catalogues exist for
  the Bransfield Strait during 2019–2020, compare detection times with
  known events.

## Completion Criteria

- [ ] STA/LTA parameters tuned and documented
- [ ] All 717 DAT files processed across all 6 moorings
- [ ] Event catalogue saved as Parquet with all characterization fields
- [ ] Cross-mooring association table generated
- [ ] Detection rate timeline figure produced
- [ ] Event duration vs. peak frequency figure produced
- [ ] Example detection figure produced (with trigger overlays)
- [ ] Validation: manual inspection of 50 random detections (<20% false positive)
- [ ] Validation: cross-mooring travel times physically plausible
- [ ] Results reproducible from raw DAT files via a single script invocation

## Assumptions & Limitations

**Assumptions**:
- STA/LTA is appropriate as a first-pass detector. It will catch impulsive
  and semi-continuous signals but may miss very long-duration, gradually
  emerging signals (e.g., slow tremor). This is acceptable for a first catalogue.
- The 5% duty cycle provides representative sampling of event rates. Absolute
  counts will underestimate true rates by ~20x, but relative rates (between
  months, between moorings) are valid.
- Without absolute calibration, detection thresholds are relative to
  background noise at each mooring. Cross-mooring amplitude comparisons
  are not meaningful.
- Sound speed of ~1480 m/s is a reasonable estimate for cross-mooring
  travel time bounds. Actual propagation depends on the sound velocity
  profile (surface-limited in Bransfield Strait).

**Limitations**:
- This analysis detects events but does not classify them. Classification
  is a separate downstream spec.
- No source location is attempted here. Cross-mooring association provides
  candidates for subsequent location analysis.
- The detector will likely produce duplicate detections for a single physical
  event recorded in multiple frequency bands. De-duplication across bands
  is deferred to the classification stage.
- Very low-SNR events near the noise floor will be missed. The false
  negative rate is not estimated.

## Notes

- The existing `make_spectrogram.py` array spectrogram pipeline provides a
  ready-made visualization tool for inspecting candidate detections. The
  example detection figures should reuse this code.
- Processing 717 files x 14.4M samples = ~10 billion samples total. At
  1 kHz, this is ~10,000 hours of data. The STA/LTA computation should
  be efficient (rolling averages on 1D energy arrays), but processing all
  files will take on the order of minutes to tens of minutes.
- The `read_dat.py` reader loads full 4-hour files into memory (~115 MB per
  file as float64). This is fine for single-file processing but a batch
  pipeline should process files sequentially, not load all into memory.
