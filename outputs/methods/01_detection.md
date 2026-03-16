# Methods — Event Detection

*BRAVOSEIS Hydroacoustic Analysis Pipeline — Step 1 of 6*

*This document was generated with AI assistance (Claude, Anthropic) and is intended for internal review. All parameters, counts, and figures are derived from version-controlled scripts and data.*

---

## 1. Overview

This document describes the event detection stage of the BRAVOSEIS hydroacoustic analysis pipeline. The detection stage identifies transient acoustic events in continuous recordings from 6 autonomous moorings (M1–M6) deployed in the Bransfield Strait during 2019–2020.

The detection algorithm was developed and validated on a 717-file subset of the full archive (~8.7% temporal coverage). After validation, the same algorithm was applied without modification to the full 14,663-file archive. Both sets of results are reported below.

---

## 2. Instrument and Data Format

Key specifications as read from the DAT file binary headers (*[TODO: confirm with Bob Dziak]*):

- **Sample rate:** 1 kHz (from header offset `0xC4`)
- **ADC resolution:** 24-bit acquisition, stored as 16-bit
- **File format:** 4-hour segments (14,400,000 samples per file)
- **Recording mode:** Continuous (back-to-back 4-hour files, no duty cycle gaps)
- **Total archive:** 14,663 files across 6 moorings (~423 GB), ~2,440–2,452 files per mooring
- **Temporal coverage:** ~408 days (January 2019 – February 2020), 99.9% per mooring
- **Development subset:** 717 files (~24 GB), a sparse sample of the full archive (~120 files per mooring)

Note: The 717-file development subset appeared to have a duty cycle (~8 hours on / ~40 hours off) because it was a sparse sample drawn from the full archive. Analysis of the complete 14,663-file dataset confirms continuous recording with no scheduled off-duty periods.

---

## 3. STA/LTA Detection Algorithm

Events are detected using a Short-Term Average / Long-Term Average (STA/LTA) energy ratio detector applied independently to each mooring and each frequency band.

### 3.1 Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| STA window | 2 s | Resolves impulsive T-phase onsets (typical onset < 1 s) |
| LTA window | 60 s | Spans several event durations to represent ambient noise floor |
| Trigger threshold | 3.0 | Standard for regional seismo-acoustic networks (Fox et al. 2001) |
| Detrigger threshold | 1.5 | Event ends when energy returns near background |
| Min event duration | 0.5 s | Rejects sub-second glitches (ADC artifacts, single-sample spikes) |
| Min inter-event gap | 2.0 s | Prevents fragmentation of events with brief amplitude dips |
| Filter order | 4th-order Butterworth | Standard for seismo-acoustic processing |

### 3.2 Parameter Provenance

The STA/LTA parameters used here follow from prior work on the same class of instruments and study regions. The table below compares BRAVOSEIS parameters against three related studies from the same research group.

| Parameter | BRAVOSEIS | Soule & Wilcock (2013) | Wilcock (2012) | Weekly et al. (2013) |
|-----------|-----------|----------------------|----------------|---------------------|
| Target signals | T-phases, icequakes, vessel | Fin whale 20 Hz calls | Fin whale 20 Hz calls | Microearthquakes |
| Detector type | STA/LTA energy ratio | RMS amplitude ratio | RMS amplitude ratio | RMS amplitude ratio |
| STA window | 2.0 s | 0.25 s | 0.25 s | (supplement) |
| LTA window | 60 s | 60 s | 60 s | (supplement) |
| Trigger threshold | 3.0 | 3.0 | 3.0 | (supplement) |
| Bandpass filter | 1–15, 15–30, 30–250 Hz | 10–35 Hz | 10–35 Hz | (supplement) |
| Min stations | 2 moorings | 4 stations (8 ch) | 4 stations (8 ch) | 6 picks (≥2P, ≥2S) |
| Association window | pair-specific (21–139 s) | 2.5 s fixed | 2.5 s fixed | — |
| Discrimination | Frequency-band clustering | Spectral ratio (15–35 vs 5–15 Hz) | Spectral ratio (15–35 vs 5–15 Hz) | Frequency content |
| Arrival picking | AIC + kurtosis | Hilbert envelope, 2× noise | Hilbert envelope, 2× noise | Autoregressive + RMS |
| Array aperture | ~175 km, 6 hydrophones | ~6 km, 8 OBS | ~6 km, 8 OBS | ~6 km, 8 OBS |

The LTA window (60 s) and trigger threshold (3.0) are identical across all studies. The STA window differs: 0.25 s for the Endeavour OBS studies (targeting ~1 s fin whale pulses) vs. 2.0 s for BRAVOSEIS (targeting longer-duration T-phases and icequakes). The longer STA window averages over cycle-to-cycle noise while remaining short enough to resolve impulsive onsets. The three-band frequency strategy replaces the single narrowband filter used in prior work because BRAVOSEIS targets multiple signal types spanning 1–250 Hz.

### 3.3 Implementation

- Custom vectorized `classic_sta_lta()` using numpy cumulative sums (no ObsPy dependency)
- Script: `scripts/detect_events.py`

---

## 4. Three-Pass Frequency Band Strategy

### 4.1 Motivation

A single broadband detector fails to detect low-amplitude events in bands dominated by high-energy signals. An initial 4-band overlapping approach (1–50, 10–200, 50–250, 1–250 Hz) with cross-band deduplication produced only 152,040 events due to two problems:

1. **LTA contamination:** Low-frequency energy (< 15 Hz, primarily T-phases) dominated the LTA window across all bands, suppressing STA/LTA ratios for concurrent higher-frequency transients
2. **Deduplication masking:** Cross-band deduplication relabeled events under the highest-SNR band (almost always low), destroying band attribution

### 4.2 Solution

Three non-overlapping frequency bands with independent STA/LTA computation per band. No cross-band deduplication is applied; events appearing in multiple bands are resolved during downstream classification.

| Pass | Filter | Frequency Range | Target Signals |
|------|--------|-----------------|----------------|
| 1 | Lowpass 15 Hz | 1–15 Hz | Earthquakes, T-phases |
| 2 | Bandpass 15–30 Hz | 15–30 Hz | Fin whale calls (~20 Hz), icequakes |
| 3 | Highpass 30 Hz | 30–250 Hz | Icequakes (high-freq), whale calls, vessel noise |

### 4.3 Band Boundary Rationale

- **15 Hz:** Separates dominant seismic/T-phase energy from mid band. Average spectral profiles across all moorings confirmed dominant energy peak below 15–20 Hz.
- **30 Hz:** Separates fin whale and low-frequency icequake regime from higher biological and cryogenic signals.
- These are energy-regime boundaries, not signal-type boundaries. A single broadband icequake may trigger in all three passes.

### 4.4 Alternative Considered

Per-mooring adaptive LTA based on running noise estimates was rejected. The non-overlapping band approach is simpler, deterministic, and recovered 7x more mid-band events (132,494 vs 17,781).

---

## 5. Detection Results

### 5.1 Development Subset (717 files)

| Band | Events |
|------|--------|
| Low (1–15 Hz) | 84,698 |
| Mid (15–30 Hz) | 132,494 |
| High (30–250 Hz) | 79,978 |
| **Total** | **297,170** |

### 5.2 Full Archive (14,663 files)

| Band | Events |
|------|--------|
| Low (1–15 Hz) | 1,941,141 |
| Mid (15–30 Hz) | 2,952,251 |
| High (30–250 Hz) | 1,867,678 |
| **Total** | **6,761,070** |

The scaling factor from subset to full is approximately 22.7x, consistent with the ~20x increase in file count.

### 5.3 Temporal Structure

The daily detection rate over the deployment shows clear temporal structure, including two prominent T-phase swarms and intermittent vessel traffic.

![Detection Rate Timeline](../figures/exploratory/detection/detection_rate_timeline.png)

*Daily event counts across all six moorings, stacked by detection frequency band (orange: 1–15 Hz, blue: 15–30 Hz, green: 30–250 Hz). STA/LTA parameters: STA = 2 s, LTA = 60 s, trigger = 3.0, detrigger = 1.5. Two prominent T-phase swarms are visible: February 11, 2019 (~5,000 events) and April 22–24, 2019 (~3,500 events on peak day). Vessel traffic passages appear as broadband bursts lasting 1–4 days.*

### 5.4 Duration–Frequency Distribution

Detected events occupy distinct regions of duration–frequency space, confirming that the three-band strategy captures different source populations.

![Duration vs Peak Frequency](../figures/exploratory/detection/duration_vs_peak_freq.png)

*Scatter plot of event duration versus peak frequency for all 297,170 detected events (subset), colored by detection band. Duration on logarithmic scale. Two distinct populations are visible in the low band: short-duration (< 3 s) events (T-phases) and longer-duration (> 3 s) events (icequakes and coda).*

### 5.5 Example Cross-Mooring Detection

A representative multi-mooring detection illustrates how the same event propagates across the array with measurable moveout.

![Example Cross-Mooring Detection](../figures/exploratory/detection/example_detection_20190417_0919.png)

*Two-minute spectrogram array centered on association A007517, detected on all six moorings. Each panel shows one mooring (M1 at top, M6 at bottom). Red shading marks the detected event window; red vertical lines mark the refined onset time. Spectrogram parameters: nperseg = 1024, 50% overlap, 0–250 Hz.*

---

## 6. Event Features

Each detected event is characterized by the following attributes:

| Feature | Description |
|---------|-------------|
| `onset_utc` | STA/LTA trigger time (UTC) |
| `duration_s` | Time from trigger to detrigger |
| `peak_freq_hz` | Frequency with maximum mean power in event spectrogram |
| `bandwidth_hz` | Frequency range containing 90% of event energy |
| `peak_db` | Maximum power in event spectrogram (relative dB) |
| `snr` | Peak STA/LTA ratio during event |
| `detection_band` | Frequency band in which event was detected |

---

## 7. QC Verification

The detection stage is verified by automated QC checks (script: `scripts/qc_verification.py`, steps D0–D1):

- **D0:** Mooring metadata — 6 moorings, coordinates within Bransfield Strait bounds, sample rate = 1000 Hz, hydrophone depths 413.7–479.0 m
- **D1.1–D1.2:** Detection parameters match specification (STA, LTA, trigger, detrigger, min duration, min gap, three bands)
- **D1.3:** No NaN values in critical columns (event_id, onset_utc, duration_s, mooring, snr, detection_band)
- **D1.4:** No duplicate event IDs
- **D1.5:** All onset times within deployment window (2019-01-10 to 2020-02-22)
- **D1.6:** All durations ≥ 0.5 s (minimum duration threshold)
- **D1.7:** Detection band values are exactly {low, mid, high}
- **D1.8:** Mooring values are exactly {m1, m2, m3, m4, m5, m6}
- **D1.9:** All SNR values ≥ 3.0 (trigger threshold)

All 18 detection QC checks pass.

---

## 8. Reproducibility

An interactive Jupyter notebook that walks through each step of this detection pipeline on a single file is available in the companion methods repository:

**[TODO: insert repo URL when published]**

Notebook: `01_event_detection.ipynb` — loads a raw DAT file, applies bandpass filtering, computes STA/LTA, extracts events, characterizes spectral features, and compares results against the production catalogue. All adjustable parameters are documented inline.
