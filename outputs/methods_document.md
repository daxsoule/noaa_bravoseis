---
title: "BRAVOSEIS Hydroacoustic Event Detection: Methods & Decisions"
author: "BRAVOSEIS Working Group"
date: "March 2026"
geometry: margin=1in
fontsize: 11pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{graphicx}
  - \usepackage{caption}
  - \captionsetup{font=small,labelfont=bf}
  - \usepackage{hyperref}
  - \hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{\small BRAVOSEIS Methods}
  - \fancyhead[R]{\small Draft — March 2026}
  - \fancyfoot[C]{\thepage}
---

# 1. Overview

This document summarizes the methods and key decisions for the BRAVOSEIS
hydroacoustic event detection pipeline. It is intended as a working
reference for peer review of the analysis approach prior to manuscript
preparation.

The BRAVOSEIS experiment deployed 6 autonomous hydrophone moorings in
Bransfield Strait (62°S, 57–60°W) from January 2019 to February 2020.
The hydrophones record in the 1–250 Hz band, capturing three primary
signal types: **earthquakes/T-phases** (predominantly < 50 Hz),
**ice quakes** (1–200 Hz), and **whale calls** (50–250 Hz).

The detection pipeline produces an event catalogue from the raw recordings,
characterizing each event by onset time, duration, peak frequency,
bandwidth, and signal-to-noise ratio. Cross-mooring associations identify
events detected on multiple hydrophones for downstream source location.

# 2. Data Description

## 2.1 Hydrophone Array

Table 1 summarizes the mooring network.

| Mooring | Hydrophone | Lat (°S) | Lon (°W) | Bottom (m) | Hydro. Depth (m) |
|---------|-----------|----------|----------|-----------|-------------------|
| M1 / BRA28 | H17C | 62° 54.9' | 60° 12.0' | 1029 | 455 |
| M2 / BRA29 | H36  | 62° 51.0' | 59° 27.0' | 1246 | 422 |
| M3 / BRA30 | H13  | 62° 31.0' | 58° 54.0' | 1538 | 414 |
| M4 / BRA31 | H21  | 62° 32.0' | 58° 00.0' | 1765 | 466 |
| M5 / BRA32 | H24  | 62° 17.8' | 57° 53.7' | 1928 | 479 |
| M6 / BRA33 | H41  | 62° 15.0' | 57° 06.0' | 1717 | 468 |

: **Table 1.** BRAVOSEIS mooring network. Hydrophone depths are approximate
(measured from sea surface). All moorings deployed January 2019, recovered
February 2020.

**Recording format:** Binary `.DAT` files, 1 kHz sample rate, 16-bit
unsigned integers (24-bit ADC), 4 hours per file (14.4 M samples).
717 total files across all moorings.

**Duty cycle:** ~8 hours recording / ~40 hours off, repeating every ~2 days
(~5% temporal coverage). Recording start time drifts slowly across the
deployment.

![Recording timeline showing the duty cycle across all 6 moorings. Bars
indicate recording windows; gaps correspond to off-duty periods. The
recording hour drifts across the 13-month deployment.](figures/exploratory/recording_timeline.png){width=100%}

## 2.2 Sound Speed Profiles

In-situ XBT profiles from the deployment cruise provide the sound speed
structure for travel-time calculations. The primary profile (T5_18_01_19,
mid-array between M1–M2) shows a surface-limited profile with speed
increasing monotonically from ~1453 m/s at the surface to ~1481 m/s at
1022 m depth. No SOFAR channel is present.

![XBT-derived sound speed profile used for travel-time calculations.
Effective horizontal speed (~1456 m/s) is computed as the harmonic mean
from the surface to hydrophone depth.](figures/exploratory/sound_speed_profile.png){width=60%}

# 3. Event Detection Method

## 3.1 STA/LTA Detector

Events are detected using a classic **Short-Term Average / Long-Term
Average (STA/LTA)** energy detector — the standard first-pass method for
seismoacoustic data. The algorithm computes the ratio of short-term to
long-term average energy on the squared (envelope) signal. An event trigger
occurs when STA/LTA exceeds a threshold and ends when it falls below a
detrigger level.

**Decision: STA/LTA parameters.** These values were chosen based on
standard practice for 1 kHz hydroacoustic data and tuned on 3 reference
spectrogram windows containing known events:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| STA window | 2 s | Short enough to capture onset of impulsive events (< 1 s rise time) |
| LTA window | 60 s | Long enough to average over background noise; avoids contamination by clustered events |
| Trigger threshold | 3.0 | Standard starting value; balances detection rate vs. false positives |
| Detrigger threshold | 1.5 | Standard; defines event end when energy returns near background |
| Min event duration | 0.5 s | Excludes sub-second glitches while retaining short impulsive events |
| Min inter-event gap | 2.0 s | Prevents retriggering on the same event's coda |
| Bandpass filter | 4th-order Butterworth | Applied before STA/LTA to isolate frequency bands |

: **Table 2.** STA/LTA detection parameters.

## 3.2 Three-Pass Detection Strategy

**Decision: Non-overlapping frequency passes.** Initial detection used 4
overlapping frequency bands (1–50, 10–200, 50–250, 1–250 Hz) with
cross-band deduplication. QC inspection revealed two problems:

1. **LTA suppression**: Low-frequency energy (< 15 Hz, primarily T-phases
   and earthquake coda) dominated the LTA window, suppressing the STA/LTA
   ratio for concurrent higher-frequency transients. A loud earthquake at
   5 Hz could prevent detection of a simultaneous whale call at 80 Hz.

2. **Deduplication masking**: When detections from multiple bands overlapped
   in time, only the highest-SNR band was kept. Since low-frequency energy
   almost always has the highest SNR, 52% of events detected in 3+ bands
   were labeled "low," and 32% of mid/high detections were discarded entirely.

Average spectral profiles across moorings confirmed dominant energy below
15–20 Hz (up to 15 dB above the 50–250 Hz floor).

The solution separates the data into **three non-overlapping frequency
bands** before computing STA/LTA, so each pass's LTA reflects only its
own frequency regime:

| Pass | Filter | Range | Target Signals |
|------|--------|-------|----------------|
| Pass 1 | Lowpass 15 Hz | 1–15 Hz | Earthquakes, T-phases, ice quakes (low-freq component) |
| Pass 2 | Bandpass 15–30 Hz | 15–30 Hz | Fin whale calls (~20 Hz), ice quakes, mixed seismicity |
| Pass 3 | Highpass 30 Hz | 30–250 Hz | Ice quakes (high-freq), other whale calls, biological |

: **Table 3.** Three-pass detection configuration. All filters are 4th-order
Butterworth.

**Why 15 and 30 Hz?** Spectral analysis across multiple moorings showed
dominant seismic energy concentrating below 15 Hz. The 15 Hz breakpoint
separates T-phase/earthquake energy from the mid band. The 30 Hz breakpoint
separates the fin whale (~20 Hz) and lower ice-quake regime from higher
biological and cryogenic signals. These are energy-regime boundaries, not
signal-type boundaries — a single broadband ice quake may trigger in all
three passes, and that is intentional.

No cross-pass deduplication is applied. Events appearing in multiple passes
are resolved during the downstream discrimination phase.

**Results (717 files, 6 moorings):**

| Band | Events | Comparison to original 4-band |
|------|--------|-------------------------------|
| Low (1–15 Hz) | 84,698 | ~81k (similar) |
| Mid (15–30 Hz) | 132,494 | 17,781 → 132k (**7.4× increase**) |
| High (30–250 Hz) | 79,978 | 52,505 → 80k (1.5× increase) |
| **Total** | **297,170** | 152,040 → 297k |

: **Table 4.** Event counts by detection band.

The mid band saw the largest improvement — 7× more events — because the
15–30 Hz range was previously dominated by spectral leakage from the strong
low-frequency energy.

![Detection QC — busy window (M1, 14 Jul 2019). 57 events in 10 minutes.
Colored lines mark event onsets by frequency band (orange: low 1–15 Hz,
blue: mid 15–30 Hz, green: high 30–250 Hz). The spectrogram (top) and
waveform (bottom) share a common time axis for direct
comparison.](figures/exploratory/detection_qc_m1_20190714_1300.png){width=100%}

![Detection QC — moderate window (M4, 21 Aug 2019). 25 events in 10 minutes.
All three bands are well represented (6 low, 7 mid, 12 high), compared to
the original 4-band approach which found only 13 events total in this
window.](figures/exploratory/detection_qc_m4_20190821_0620.png){width=100%}

# 4. Cross-Mooring Association

Events detected on multiple moorings are associated using **pair-specific
travel-time windows** derived from the XBT sound speed profile. For each
mooring pair, the maximum allowed arrival-time difference is:

$$\Delta t_{\max} = \frac{d_{ij}}{c_{\text{eff}}} \times 1.15$$

where $d_{ij}$ is the great-circle distance between moorings $i$ and $j$,
$c_{\text{eff}} \approx 1456$ m/s is the effective horizontal sound speed
(harmonic mean of the XBT profile), and the 1.15 factor is a 15% safety
margin to account for non-straight ray paths and sound speed variability.

**Decision: Pair-specific windows replace constant window.** The original
approach used a constant 120 s window based on an assumed 1480 m/s speed.
This was 6× too wide for close pairs (M4–M5: 27 km → 21 s actual) and
slightly too narrow for the most distant pair (M1–M6: 176 km → 139 s).

| Pair | Distance (km) | $c_{\text{eff}}$ (m/s) | Max $\Delta t$ (s) |
|------|--------------|----------------------|-------------------|
| M4–M5 | 26.9 | 1456.1 | 21.2 |
| M1–M2 | 38.8 | 1455.6 | 30.7 |
| M5–M6 | 41.6 | 1456.1 | 32.9 |
| M3–M4 | 46.3 | 1455.8 | 36.6 |
| M2–M3 | 46.7 | 1454.8 | 36.9 |
| M3–M5 | 57.4 | 1456.1 | 45.3 |
| M4–M6 | 56.3 | 1455.9 | 44.5 |
| M1–M3 | 80.0 | 1455.6 | 63.2 |
| M2–M4 | 82.2 | 1455.8 | 64.9 |
| M3–M6 | 97.7 | 1455.9 | 77.2 |
| M2–M5 | 100.9 | 1456.1 | 79.7 |
| M1–M4 | 120.2 | 1455.8 | 95.0 |
| M1–M5 | 136.9 | 1456.1 | 108.1 |
| M2–M6 | 138.2 | 1455.9 | 109.2 |
| M1–M6 | 175.7 | 1455.9 | 138.8 |

: **Table 6.** Mooring pair distances and maximum travel-time windows.
Association uses a greedy windowed clustering algorithm with connected
components.

# 5. Event Characterization

Each detected event is characterized by:

| Feature | Description | Method |
|---------|-------------|--------|
| `onset_utc` | Event start time | STA/LTA trigger sample → UTC |
| `duration_s` | Event duration | Trigger to detrigger |
| `peak_freq_hz` | Frequency of maximum power | Mean power spectrum over event window |
| `bandwidth_hz` | Frequency range (90% energy) | 5th–95th percentile of cumulative spectral energy |
| `peak_db` | Maximum power (relative dB) | Peak of event spectrogram |
| `snr` | Signal-to-noise ratio | Peak STA/LTA value during event |
| `detection_band` | Frequency band (low/mid/high) | Band label from the detection pass |
| `detection_pass` | Which pass detected it | 1 (LP 15), 2 (BP 15–30), or 3 (HP 30) |

: **Table 6.** Event characterization features.

# 6. Detection Results Overview

The following figures were generated from the original 4-band catalogue
(152,040 events) and will be regenerated from the three-pass catalogue
(297,170 events) in a subsequent update.

![Detection rate timeline — daily event counts stacked by frequency band.
Gaps correspond to the ~40-hour off-duty periods.](figures/exploratory/detection_rate_timeline.png){width=100%}

![Duration vs. peak frequency for all detected events. Events cluster by
band as expected.](figures/exploratory/duration_vs_peak_freq.png){width=85%}

![Cross-mooring statistics — distribution of events by the number of
moorings on which they were detected. Gray: single-mooring events. Blue:
multi-mooring associations.](figures/exploratory/cross_mooring_statistics.png){width=85%}

# 7. False Positive Validation (Preliminary)

A stratified random sample of 50 events (17 low, 17 mid, 16 high;
seed=42) was drawn from the three-pass catalogue. For each event, a
10-second window centered on the detection was plotted as a spectrogram
with the band-filtered waveform below it. Event onset and end times are
marked on both panels.

![False positive validation montage — 50 randomly sampled events with
spectrograms (top) and band-filtered waveforms (bottom) for each. Vertical
lines mark event onset (solid) and end (dashed), colored by detection band
(orange: low, blue: mid, green: high).](figures/exploratory/fp_validation_montage.png){width=100%}

**Preliminary observations:** Manual inspection of the first several
events reveals a systematic onset-timing issue. In multiple cases, the
STA/LTA trigger fires on a **coda or multipath arrival** rather than the
initial event onset. The true first arrival is visible in the spectrogram
and filtered waveform before the pick, but the strong initial energy
floods the 60 s LTA window, suppressing the STA/LTA ratio during the
actual onset. By the time the ratio exceeds 3.0, the detector picks a
later arrival.

This is a known limitation of STA/LTA detectors with long LTA windows
when applied to emergent-onset or multipath-rich signals. Potential
mitigations include:

- An **onset refinement** step (e.g., AIC or kurtosis picker) applied
  within a window around each STA/LTA trigger
- **Shorter LTA** windows (trading sensitivity for onset accuracy)
- **Pre-trigger padding** to search backward from each trigger for the
  true first break

Full FP rate quantification is pending completion of the 50-event review.

# 8. Open Questions

1. **Onset timing accuracy**: STA/LTA triggers appear to land on coda or
   multipath arrivals rather than true first arrivals in some cases (see
   Section 7). An onset refinement step is needed before the catalogue
   can be used for source location.

2. **Multipath**: No explicit multipath handling is implemented. Strong
   bottom/surface reflections arriving >2 s after the direct path may
   trigger as separate events.

3. **Cross-mooring association**: The association step has not yet been
   re-run on the three-pass catalogue (297k events). The increased event
   density may increase coincidental associations.

4. **Duplicate resolution**: Events appearing in multiple passes (e.g., a
   broadband ice quake triggering all three) need to be reconciled during
   the discrimination phase.

5. **Per-pass threshold tuning**: The current trigger=3.0 was chosen for
   the original broadband detection. Each pass's noise characteristics
   differ and may benefit from pass-specific thresholds.

6. **Summary figures**: The detection rate timeline, duration vs. frequency,
   and cross-mooring figures need regeneration from the three-pass catalogue.
