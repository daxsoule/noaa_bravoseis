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

**Results:** Of 50 events reviewed, 46 (92%) were judged to be true
positive detections and 4 (8%) were false positives — well below the
20% FP threshold. However, a systematic onset-timing issue was identified:
only 11% of picks landed on the true first arrival, 68% fell in the event
coda, and 8% landed at peak energy. This motivated the onset refinement
step described below.

# 8. Onset Refinement

## 8.1 Motivation

STA/LTA onset picks are biased late. The 60 s LTA window becomes
contaminated by early event energy, suppressing the STA/LTA ratio during
the true onset. By the time the ratio exceeds 3.0, the pick lands in
the coda rather than at the first arrival. Since source location requires
accurate first-arrival times, a second-pass refinement is needed.

## 8.2 AIC Picker with Kurtosis Fallback

For each event, the onset is refined using an **Akaike Information
Criterion (AIC) picker** (Maeda, 1985) with a **kurtosis-based fallback**:

1. Extract a 7 s window: 5 s before the STA/LTA trigger + 2 s after
2. Apply the same band-specific filter used for detection
3. Pad 1 s on each side to mitigate filter edge effects, then trim
4. Compute the squared envelope
5. Run the AIC picker — fits a two-segment noise/signal variance model
   and finds the minimum AIC as the onset estimate
6. If AIC quality < 0.4, fall back to a kurtosis picker (0.5 s sliding
   window) that detects the first statistical departure from Gaussian noise
7. If both fail, keep the original onset with low confidence

**Decision: Reject positive shifts.** Any refined onset that falls *after*
the original STA/LTA trigger is rejected and reverted to the original. The
STA/LTA already triggers late, so a forward shift indicates a picker error
(typically the AIC latching onto a noise variance change rather than the
true signal onset). Rejected events receive downgraded quality scores.

**Decision: AIC quality threshold = 0.4.** Initial threshold of 0.3 allowed
marginal picks that visual inspection showed were not true onsets. Raising
to 0.4 eliminated these while retaining 95.7% of picks.

## 8.3 Quality Grading

Each refined onset receives a quality score (0–1) based on the sharpness
of the AIC trough, mapped to letter grades:

| Grade | Quality Range | Use | Events | Fraction |
|-------|--------------|-----|--------|----------|
| A | ≥ 0.7 | Source location | 233,751 | 78.7% |
| B | 0.4–0.7 | Source location | 51,007 | 17.2% |
| C | < 0.4 | **Exclude from location** | 12,412 | 4.2% |

: **Table 7.** Onset quality grading. Grade A/B events (95.8%) are used for
source location; grade C events are retained for classification and
statistical analysis but excluded from travel-time-based location.

## 8.4 Results

| Method | Events | Fraction |
|--------|--------|----------|
| AIC | 284,425 | 95.7% |
| Kurtosis | 686 | 0.2% |
| Original (kept) | 12,059 | 4.1% |

: **Table 8.** Onset refinement method distribution.

Median onset shift: **−0.83 s** (IQR: [−1.46, −0.39] s). Shifts are
consistently negative (earlier), confirming the picker moves onsets
backward from the coda toward the true first arrival.

![Onset shift histogram by detection band. Most shifts are between −0.5
and −3 s, consistent with correcting coda picks to first arrivals. The
red line indicates the median shift for refined events in each
band.](figures/exploratory/onset_shift_histogram.png){width=100%}

## 8.5 QC Validation

A montage of 50 events (overweighted toward grade C: 24 C, 14 B, 12 A)
was visually inspected. Of 50 events, 34 (68%) had acceptable refined
picks. Most issues were late picks on emergent low-frequency signals
(< 5 Hz) where the AIC assumption of a sharp noise-to-signal transition
breaks down.

![Onset refinement QC montage — 50 events showing original STA/LTA onset
(dashed line) and refined AIC/kurtosis onset (solid line). Titles show
method, grade, quality score, and shift
amount.](figures/exploratory/onset_refinement_montage.png){width=100%}

**Known limitation:** Low-frequency events (< 5 Hz) with emergent, gradual
onsets are poorly served by the AIC picker. These events tend to receive
grade C picks. A frequency-domain onset method or manual picks may be
needed for this subset.

# 9. Event Discrimination (In Progress)

## 9.1 Approach

Classification proceeds in two phases:

1. **Phase 1 — Unsupervised Discovery**: Extract handcrafted spectral
   features, project to 2D with UMAP, cluster with HDBSCAN
2. **Phase 2 — Supervised CNN**: Train a lightweight CNN on spectrogram
   patches using Phase 1 labels (gated on Phase 1 review)

## 9.2 Feature Extraction

For each of the 297,170 events, 19 spectral features were extracted from
spectrogram patches (nperseg=1024, noverlap=512, 0–250 Hz):

- 10 band powers (25 Hz bands: 0–25, 25–50, ..., 225–250 Hz)
- Peak frequency, peak power (dB), bandwidth (90% energy)
- Duration, rise time, decay time
- Spectral slope, frequency modulation, spectral centroid

All 297,170 events yielded complete feature vectors (100%).

## 9.3 Clustering (Preliminary)

UMAP (n_neighbors=15, min_dist=0.1) + HDBSCAN (min_cluster_size=50)
identified 12 clusters, but with a highly skewed distribution: cluster 8
contains 294,399 events (99.1%) while the 11 remaining clusters range
from 51 to 642 events. Silhouette score = 0.259.

This indicates the handcrafted features do not adequately separate the
main signal types in the current form. Options under evaluation include:
improving the feature set, clustering by detection band separately, or
proceeding directly to CNN classification on spectrogram patches.

# 10. Open Questions

1. **Clustering**: The mega-cluster problem (Section 9.3) needs resolution
   before Phase 1 labeling can proceed meaningfully.

2. **Multipath**: No explicit multipath handling is implemented. Strong
   bottom/surface reflections arriving >2 s after the direct path may
   trigger as separate events.

3. **Cross-mooring association**: The association step has not yet been
   re-run on the three-pass catalogue (297k events). The increased event
   density may increase coincidental associations. Event discrimination
   should precede association to reduce false associations.

4. **Duplicate resolution**: Events appearing in multiple passes (e.g., a
   broadband ice quake triggering all three) need to be reconciled during
   the discrimination phase.
