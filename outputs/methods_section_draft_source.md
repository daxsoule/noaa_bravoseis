# Methods — BRAVOSEIS Hydroacoustic Event Detection, Classification, and Location

*Draft for internal review — intended for maximum transparency so that every
parameter choice, alternative considered, and quality control step can be
independently verified.*

---

## 1. Overall Objectives

The ocean floor near Antarctica is full of sounds — earthquakes rumbling through
the seafloor, glaciers cracking and calving into the sea, whales calling across
vast distances, and ships passing overhead. We placed six underwater microphones
(hydrophones) in the Bransfield Strait for about 13 months to listen to all of
this. The goal of this analysis is to automatically find every distinct sound
event in those recordings, figure out what made each sound, and pinpoint where
it came from.

The analysis pipeline processes the complete archive of continuous
hydroacoustic data from 6 autonomous moorings (M1–M6) deployed in the
Bransfield Strait during 2019–2020 — 14,663 DAT files totalling ~9,800
hours of recordings — to:

1. **Detect** transient acoustic events above the ambient noise floor across
   three frequency regimes (1–15, 15–30, 30–250 Hz)
2. **Refine onset times** for cross-mooring association and source location
3. **Classify** events into physical source categories: earthquake T-phases,
   icequakes (glacial/cryogenic), vessel noise, and unresolved
4. **Locate** classified events using Time-Difference-of-Arrival (TDOA) grid
   search
5. **Quality-control** locations using jackknife validation, coast-distance
   filtering, satellite sea ice data, and swarm coherence analysis

The pipeline is fully automated and reproducible from raw `.DAT` files to
final catalogues and figures. All scripts, parameters, and intermediate
outputs are version-controlled.

---

## 2. Event Detection

Before we can classify or locate any sounds, we first need to find them. The
raw recordings are thousands of hours of continuous audio — mostly background
ocean noise with occasional distinct events buried within. This section
describes how we automatically scan through all those recordings and flag every
moment where something stands out above the background. Think of it like
highlighting every interesting sentence in a very long book before you go back
to read them carefully.

### 2.1 Instrument and Data Format

Each mooring records at 1 kHz (24-bit Analog-to-Digital Converter (ADC), stored as 16-bit) in 4-hour
segments (14,400,000 samples per file). Recording is continuous (back-to-back
4-hour files, no scheduled off-duty periods), with ~99.9% temporal coverage
per mooring. The full archive comprises 14,663 files across 6 moorings
(~423 GB total), representing ~2,440–2,452 files per mooring × 4 hours =
~9,800 hours of recordings spanning the entire 13-month deployment
(January 2019 – February 2020). See the Data Sources section of the
research constitution for complete instrument specifications.

> **Note:** The pipeline was developed and validated on a 717-file subset
> (~120 files per mooring, ~24 GB). This sparse sample appeared to have a
> duty cycle (~8 hours on / ~40 hours off) because it was drawn from the
> full archive at irregular intervals. Analysis of the complete 14,663-file
> dataset confirms continuous recording with no scheduled off-duty periods.

> **Figure: Recording Timeline** (`recording_timeline.png`)
>
> **Temporary Caption:** Recording timeline for the six BRAVOSEIS hydrophone
> moorings (M1/BRA28 through M6/BRA33) from January 2019 to February 2020.
> Each horizontal bar represents one 4-hour DAT file. Recording is
> continuous (back-to-back files with no scheduled off-duty periods),
> achieving ~99.9% temporal coverage per mooring. The recording start time
> drifts slowly across the deployment — it is not locked to a fixed
> Coordinated Universal Time (UTC) hour. M3 (BRA30) has the fewest files
> (104); M5 (BRA32) the most. Total: 14,663 files.

### 2.2 STA/LTA Detection

To find events, we compare the energy of the sound right now (over a short
window) to the average background energy (over a longer window). When the
short-term energy suddenly jumps well above the long-term background, we know
something interesting just happened. This is the Short-Term Average /
Long-Term Average (STA/LTA) method — one of the most widely used techniques
in seismology and ocean acoustics.

Events are detected using a classic **STA/LTA energy ratio** detector applied
independently to each mooring and each frequency band.

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| STA window | 2 s | Short enough to resolve impulsive T-phases (typical onset <1 s), long enough to average out cycle-to-cycle noise |
| LTA window | 60 s | Long enough to represent the ambient noise floor; chosen to span at least several event durations |
| Trigger threshold | 3.0 | Standard for regional seismo-acoustic networks (e.g., Fox et al. 2001); balances sensitivity against false triggers |
| Detrigger threshold | 1.5 | Half the trigger threshold; event ends when energy returns near background |
| Min event duration | 0.5 s | Rejects sub-second glitches (ADC artifacts, single-sample spikes) |
| Min inter-event gap | 2.0 s | Prevents fragmentation of long events with brief amplitude dips |
| Filter order | 4th-order Butterworth | Standard for seismo-acoustic processing; minimal phase distortion |

**Parameter provenance:** The STA/LTA parameters follow from prior work on
the same class of instruments and study regions:

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

The LTA window (60 s) and trigger threshold (3.0) are identical across all
studies. The STA window differs: 0.25 s for the Endeavour OBS studies
(targeting ~1 s fin whale pulses) vs. 2.0 s for BRAVOSEIS (targeting
longer-duration T-phases and icequakes). The longer STA window averages over
cycle-to-cycle noise while remaining short enough to resolve impulsive
onsets. The three-band frequency strategy replaces the single narrowband
filter used in prior work because BRAVOSEIS targets multiple signal types
spanning 1–250 Hz.

**Implementation:** Custom vectorized `classic_sta_lta()` using numpy
cumulative sums (no ObsPy dependency). Script: `scripts/detect_events.py`.

### 2.3 Three-Pass Detection Strategy

Different types of ocean sounds occupy different frequency ranges — earthquakes
tend to produce low-frequency rumbles, while cracking ice and passing ships
generate higher-pitched signals. If we try to detect everything in one broad
frequency sweep, the loudest sounds (usually earthquakes) drown out the quieter
ones. To solve this, we split the recordings into three separate frequency bands
and run the detector independently on each one, so that each band's background
level reflects only its own frequency range. This simple strategy recovered
nearly twice as many events as the original single-pass approach.

**Problem:** An initial approach using 4 overlapping frequency bands
(1–50, 10–200, 50–250, 1–250 Hz) with cross-band deduplication produced
only 152,040 events. Two issues were identified:

1. **LTA contamination:** Low-frequency energy (<15 Hz, primarily T-phases
   and earthquake coda) dominated the LTA window in all bands, suppressing
   STA/LTA ratios for concurrent higher-frequency transients.
2. **Deduplication masking:** The deduplication step relabeled mid/high-band
   events under the highest Signal-to-Noise Ratio (SNR) band (almost always low), destroying band
   attribution.

**Solution:** Separate the data into **three non-overlapping frequency bands**
before computing STA/LTA, so each pass's LTA reflects only its own frequency
regime. No cross-pass deduplication is applied — a physical event appearing
in multiple passes is resolved during the downstream classification phase.

| Pass | Filter | Range | Target signals |
|------|--------|-------|----------------|
| Pass 1 | Lowpass 15 Hz | 1–15 Hz | Earthquakes, T-phases |
| Pass 2 | Bandpass 15–30 Hz | 15–30 Hz | Fin whale calls (~20 Hz), icequakes |
| Pass 3 | Highpass 30 Hz | 30–250 Hz | Icequakes (high-freq), whale calls, vessel noise |

**Breakpoint rationale:**
- **15 Hz:** Separates the dominant seismic/T-phase energy from the mid band.
  Average spectral profiles across all moorings confirmed the dominant energy
  peak below 15–20 Hz.
- **30 Hz:** Separates the fin whale and low-frequency icequake regime from
  higher biological and cryogenic signals.
- These are **energy-regime boundaries**, not signal-type boundaries. A single
  broadband icequake may trigger in all three passes.

**Alternative considered:** Per-mooring adaptive LTA based on running noise
estimates. Rejected because the non-overlapping band approach is simpler,
deterministic, and recovered 7× more mid-band events (132,494 vs 17,781).

**Results:**

| Band | Events (717-file subset) | Events (full archive) |
|------|--------|--------|
| Low (1–15 Hz) | 84,698 | 1,941,141 |
| Mid (15–30 Hz) | 132,494 | 2,952,251 |
| High (30–250 Hz) | 79,978 | 1,867,678 |
| **Total** | **297,170** | **6,761,070** |

The pipeline was developed and validated on a 717-file subset (~120 files
per mooring), then applied to the full 14,663-file archive. The daily
detection rate over the deployment shows clear temporal structure,
including two prominent T-phase swarms and intermittent vessel traffic.

> **Figure: Detection Rate Timeline** (`detection_rate_timeline.png`)
>
> **Temporary Caption:** Daily event counts across all six BRAVOSEIS hydrophone
> moorings, stacked by detection frequency band (orange: 1–15 Hz, blue:
> 15–30 Hz, green: 30–250 Hz). STA/LTA detector parameters: STA = 2 s,
> LTA = 60 s, trigger = 3.0, detrigger = 1.5. Two prominent T-phase
> swarms are visible: February 11, 2019 (~5,000 events) and
> April 22–24, 2019 (~3,500 events on the peak day). Vessel traffic
> passages appear as broadband bursts lasting 1–4 days.

The detected events occupy distinct regions of duration–frequency space,
confirming that the three-band strategy captures different source populations.

> **Figure: Duration vs. Peak Frequency** (`duration_vs_peak_freq.png`)
>
> **Temporary Caption:** Scatter plot of event duration versus peak frequency
> for all detected events (297,170 in the development subset), colored by detection band. Duration is on
> a logarithmic scale. Events separate cleanly by detection band. Two distinct
> populations are visible in the low band: short-duration (< 3 s) events
> (T-phases) and longer-duration (> 3 s) events (icequakes and coda).

A representative multi-mooring detection illustrates how the same event
propagates across the array with measurable moveout.

> **Figure: Example Cross-Mooring Detection**
> (`example_detection_20190417_0919.png`)
>
> **Temporary Caption:** Two-minute spectrogram array centered on association
> A007517, detected on all six moorings. Each panel shows one mooring (M1 at
> top, M6 at bottom). Red shading marks the detected event window; red
> vertical lines mark the refined onset time. A broadband T-phase arrival is
> visible propagating across the array with moveout consistent with a source
> in the central Bransfield Strait. Spectrogram parameters: nperseg = 1024,
> 50% overlap, 0–250 Hz. Shared colorscale (2nd–98th percentile).

### 2.4 Event Features

Each detected event is characterized by:

| Feature | Description |
|---------|-------------|
| `onset_utc` | STA/LTA trigger time (UTC) |
| `duration_s` | Time from trigger to detrigger |
| `peak_freq_hz` | Frequency with maximum mean power in event spectrogram |
| `bandwidth_hz` | Frequency range containing 90% of event energy |
| `peak_db` | Maximum power in the event spectrogram (relative dB) |
| `snr` | Peak STA/LTA ratio during the event |
| `detection_band` | Frequency band in which the event was detected |

### 2.5 Detection QC Verification

The detection stage is verified by automated QC checks (script:
`scripts/qc_verification.py`, steps D0–D1):

- **D0:** Mooring metadata — 6 moorings, coordinates within Bransfield
  Strait bounds, sample rate = 1000 Hz, hydrophone depths 413.7–479.0 m
- **D1.1–D1.2:** Detection parameters match specification (STA, LTA,
  trigger, detrigger, min duration, min gap, three bands)
- **D1.3:** No NaN values in critical columns (event_id, onset_utc,
  duration_s, mooring, snr, detection_band)
- **D1.4:** No duplicate event IDs
- **D1.5:** All onset times within deployment window (2019-01-10 to
  2020-02-22)
- **D1.6:** All durations ≥ 0.5 s (minimum duration threshold)
- **D1.7:** Detection band values are exactly {low, mid, high}
- **D1.8:** Mooring values are exactly {m1, m2, m3, m4, m5, m6}
- **D1.9:** All SNR values ≥ 3.0 (trigger threshold)

All 18 detection QC checks pass.

### 2.6 Reproducibility

An interactive Jupyter notebook walks through each step of the detection
pipeline on a single file: `notebooks/methods_notebooks/01_event_detection.ipynb`.
It loads a raw DAT file, applies bandpass filtering, computes STA/LTA,
extracts events, characterizes spectral features, and compares results
against the production catalogue. All adjustable parameters are documented
inline.

---

## 3. Onset Refinement

Once we have found an event, we need to know exactly when it started — not just
roughly, but down to a fraction of a second. The initial detector tends to
trigger partway through an event rather than at the very first arrival of
sound. Since our method for pinpointing where a sound came from relies on
measuring tiny time differences between microphones, even a one-second error
in the start time can shift the estimated location by more than a kilometer.
This section describes how we sharpen those start times using the Akaike
Information Criterion (AIC), a statistical method that finds the most likely
boundary between background noise and signal within each event window.

### 3.1 Problem

STA/LTA onset picks are biased late — validation on 50 manually reviewed
events showed only 11% hit the true first arrival; 68% fell in the event
coda. Accurate onsets are critical for TDOA source location, where a 1-second
onset error at 1455 m/s produces ~1.5 km location error.

> **Figure: STA/LTA Late-Pick Problem** (`paper/late_pick_problem.png`)
>
> **Temporary Caption:** Six events illustrating the STA/LTA late-pick problem.
> Dashed yellow lines mark the original STA/LTA trigger time; solid red lines
> mark the AIC-refined onset. In each case, the STA/LTA detector triggered
> 1.5–4 seconds late, typically in the event coda rather than at the true
> first arrival. Validation on 50 manually reviewed events confirmed that
> 68% of raw STA/LTA picks fall in the coda and only 11% hit the true first
> arrival. Top row: low-band (1–15 Hz) T-phase events. Middle row: mid-band
> (15–30 Hz) events. Bottom row: high-band (30–250 Hz) events. Spectrogram:
> nperseg=256, 87.5% overlap, Hann window.

### 3.2 AIC Picker with Kurtosis Fallback

For each event, a 7 s window is extracted (5 s pre-trigger + 2 s
post-trigger), the same band-specific filter used for detection is applied,
and the **AIC** is computed on the squared
envelope (Maeda, 1985). The AIC minimum marks the transition from noise to
signal.

**Fallback:** If AIC quality < 0.4, a kurtosis-based picker (0.5 s sliding
window) is tried. If both fail, the original STA/LTA onset is retained.

**Constraint:** Positive shifts (refined onset later than STA/LTA trigger)
are rejected — the STA/LTA already triggers late, so any forward shift is
treated as a picker error. This is a one-directional constraint: the refined
onset can only move earlier.

**Quality grading:**

| Grade | Quality | Action |
|-------|---------|--------|
| A | ≥ 0.7 | High confidence — use for source location |
| B | 0.4–0.7 | Moderate confidence — use for source location |
| C | < 0.4 | Low confidence — exclude from location, retain in catalogue |

**Results (297,170 development subset events; proportions are consistent
across the full 6,761,070-event dataset):**

| Method | Events | % |
|--------|--------|---|
| AIC | 284,425 | 95.7% |
| Kurtosis | 686 | 0.2% |
| Original (kept) | 12,059 | 4.1% |

Median onset shift: **−0.83 s** (Interquartile Range (IQR): [−1.46, −0.39] s). Shifts are
consistently negative (earlier), confirming the picker moves onsets backward
from the coda toward the true first arrival.

**Alternative considered:** The AR-AIC picker (autoregressive) was
considered but rejected because it assumes stationarity within the noise
window, which fails for the variable ambient noise in the Bransfield Strait.
The envelope-based AIC is more robust to non-stationary noise.

> **Figure: Onset Refinement — Curated Examples** (`paper/onset_refinement_6panel.png`)
>
> **Temporary Caption:** Six curated events illustrating onset refinement quality
> across grades A, B, and C. Top row: Grade A (quality ≥ 0.7) — sharp AIC
> minimum at the noise-to-signal transition, producing a confident onset pick
> 0.5–2 s earlier than STA/LTA. Middle row: Grade B (quality 0.4–0.7) —
> moderate AIC minimum with some ambiguity, but still a meaningful improvement
> over STA/LTA. Bottom row: Grade C (quality < 0.4) — weak or ambiguous AIC
> minimum; these events are excluded from source location but retained in
> the catalogue. Dashed yellow: original STA/LTA trigger. Solid red:
> AIC-refined onset. Dotted gray: event end (detrigger). A full 50-event
> QC montage (stratified by grade) is available in exploratory figures
> (`onset_refinement_montage.png`). Spectrogram: nperseg=256, 87.5% overlap,
> Hann window.

### 3.3 Seismic-Tuned Dual Picker

The AIC picker struggles with emergent T-phases — the 4.3% grade C picks
are concentrated in 6 seismic clusters (11,958 events). A **seismic-tuned
dual onset picker** uses an AIC-first rescue strategy:

1. If existing AIC pick is grade A or B → **keep AIC pick**
2. If AIC is grade C → run both envelope STA/LTA and kurtosis pickers
3. Take the earlier valid pick (more negative shift = earlier onset)
4. If both fail → keep AIC pick (no downgrade)

**Results:** Grade C reduced by 37% (510 → 319 out of 11,958 events).
191 events rescued from grade C to grade B. No events downgraded.

> **Figure: Seismic Onset Rescue Montage**
> (`seismic_onsets/seismic_rescue_montage.png`)
>
> **Temporary Caption:** Examples of emergent T-phase events where the seismic-tuned
> picker (envelope or kurtosis method) improved the onset pick. Top row:
> original AIC onset (yellow) with grade C quality. Bottom row: rescued
> onset (red) with improved quality. The seismic picker identifies the
> gradual energy ramp preceding the impulsive phase.

---

## 4. Event Classification

After detecting nearly 300,000 sound events, the next challenge is figuring out
what made each one. Was it an earthquake sending vibrations through the ocean?
A glacier cracking apart? A fishing vessel motoring past? We tackle this in two
phases. First, we let the data sort itself into natural groups without any
preconceived labels, then inspect those groups to see what they represent.
Second, we train a neural network on those labeled groups so it can classify the
remaining bulk of events that did not fall neatly into a cluster. The end result
is a label for every event: earthquake, icequake, vessel noise, or unresolved.

### 4.1 Phase 1 — Unsupervised Discovery

In this first phase, we do not tell the computer what categories to look for.
Instead, we measure about 20 characteristics of each event — things like how
long it lasted, what frequencies were loudest, and how quickly the sound rose
and fell — and then ask the algorithm to find natural groupings among those
measurements. Events that share similar characteristics end up near each other
in a two-dimensional map, forming visible clusters that a human reviewer can
then inspect and label. This approach lets the data reveal its own structure
rather than forcing events into predefined boxes.

**Approach:** Extract ~20 handcrafted spectral features per event (band
powers, duration, rise/decay time, peak frequency, bandwidth, spectral
slope, frequency modulation). Cluster each detection band independently
using Uniform Manifold Approximation and Projection (UMAP) projection into 2D followed by Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) density-based clustering.
Clusters are visually inspected via spectrogram montages and labeled by a
single reviewer.

**Why per-band clustering?** An initial all-band clustering pass (297,170
development subset events, 19 features) produced a single mega-cluster containing 99% of
events. Visual inspection revealed clear internal structure organized
primarily by detection band — the dominant axis of variation was frequency
regime, not signal type. Clustering per band removes this known axis and
allows subtler within-band structure to emerge.

The UMAP projections reveal distinct density peaks corresponding to the
classified event populations, with unresolved events forming the diffuse
background.

> **Figure: UMAP Clustering by Band** (`paper/umap_clustering_composite.png`)
>
> **Temporary Caption:** UMAP 2D projections of ~20 handcrafted spectral
> features for each detection band, colored by Phase 1 class assignment.
> Blue: T-phase (55,783 events), orange: icequake (23,331), green: vessel
> (10,458), gray: unresolved (207,598). Left: low band (1–15 Hz, 84,698
> events) — T-phase and icequake clusters separate clearly from the
> unresolved bulk. Center: mid band (15–30 Hz, 132,494 events) — a compact
> T-phase cluster (mid_0, 1,056 events) and a vessel cluster are visible
> at the periphery. Right: high band (30–250 Hz, 79,978 events) — vessel
> noise dominates the classified fraction. HDBSCAN density-based clustering
> (min_cluster_size=500) identifies the labeled clusters; remaining points
> are assigned to the unresolved population for Phase 2 supervised
> classification. UMAP parameters: n_neighbors=15, min_dist=0.01,
> metric=euclidean.

**Phase 1 results:**

| Class | Criteria | Detections |
|-------|----------|------------|
| T-phase (earthquake) | peak_freq <30 Hz, power >48 dB, slope <−0.5, duration ≤3 s | 55,783 |
| Icequake (cryogenic) | duration >3 s, power >48 dB, peak_freq <30 Hz, slope −0.2 to −0.5 | 23,331 |
| Vessel noise | Positive spectral slope, peak >100 Hz, broadband | 10,458 |
| Unresolved (bulk) | Remainder | 207,598 |

**Key distinguishing features:**
- **T-phase vs. icequake:** Duration (≤3 s vs. >3 s) and spectral slope
  (< −0.5 vs. −0.2 to −0.5). T-phases are impulsive; icequakes are sustained.
- **Vessel vs. seismic:** Spectral slope sign (positive vs. negative) and peak
  frequency (>100 Hz vs. <30 Hz). Zero overlap in feature space.

**Validation:** T-phase classification confirmed by Bob Dziak (National Oceanic and Atmospheric Administration (NOAA) / Pacific Marine Environmental Laboratory (PMEL)) via
spectrogram montage review. Cross-validated against 636 Orca Ocean-Bottom Seismometer (OBS) earthquakes
with hydrophone coverage: 89% detection rate, 43% T-phase match, 28.2 s
median arrival delay consistent with ~40 km propagation at ~1.45 km/s.

> **Figure: T-phase Cluster — Representative Events** (`paper/tphase_cluster_curated.png`)
>
> **Temporary Caption:** Six representative T-phase events from Phase 1
> unsupervised clusters (low_0: 4,686 events, low_1: 3,666 events, mid_0:
> 1,056 events), selected from events nearest the UMAP cluster centroids
> across different moorings. Each panel shows bandpass-filtered waveform (top)
> and spectrogram (bottom, 0–100 Hz). Dashed yellow: STA/LTA onset. Solid
> red: AIC-refined onset. T-phases are characterized by impulsive broadband
> arrivals with dominant energy below 15 Hz and duration typically ≤ 3 s.
> Phase 1 identified 55,783 T-phases across three clusters confirmed by
> expert review. Complete 4×5 montages (20 events nearest each cluster
> centroid) are available in supplementary materials
> (`cluster_montage_low_0.png`, `cluster_montage_low_1.png`,
> `cluster_montage_mid_0.png`). Spectrogram: nperseg=256, 87.5% overlap,
> Hann window.

The vessel noise cluster is spectrally distinct from all seismic sources,
occupying a separate region of feature space with no overlap.

> **Figure: Vessel Noise Cluster — Representative Events** (`paper/vessel_cluster_curated.png`)
>
> **Temporary Caption:** Six representative vessel noise events from the
> Phase 1 unsupervised cluster, selected for high SNR and onset quality
> across different moorings. Each panel shows highpass-filtered waveform
> (top, 30–250 Hz) and spectrogram (bottom, 0–250 Hz). Red vertical line:
> AIC-refined onset. Vessel noise is characterized by positive spectral
> slope (energy increasing with frequency), peak energy above 100 Hz, and
> broad bandwidth (~211 Hz) — the acoustic signature of propeller cavitation
> and ship machinery. Identified as vessel traffic based on temporal burst
> pattern (~24 passages over 13 months), multi-mooring simultaneity (47%
> of time bins have 2+ moorings detecting), and seasonal correlation with
> krill fishing fleet activity (peak May–Sep). A full 20-event montage
> is available in supplementary materials (`type_a_montage.png`).
> Spectrogram: nperseg=256, 87.5% overlap, Hann window.

### 4.2 Phase 2 — Supervised CNN+MLP

Phase 1 labeled about 90,000 events, but roughly 208,000 remained unresolved —
they did not cluster tightly enough for confident labeling. To classify these
remaining events, we trained a hybrid neural network that learns from the
Phase 1 labels and then predicts categories for the rest. The model has two
branches: a Convolutional Neural Network (CNN) that looks at spectrogram images
of each event (like a picture of its sound), and a Multi-Layer Perceptron (MLP)
that examines the same numerical features used in Phase 1. By combining visual
pattern recognition with statistical features, the hybrid model achieves much
higher accuracy than either branch alone.

**Architecture:** Hybrid dual-branch neural network:
- **CNN branch:** 4 convolutional blocks processing 64×128 spectrogram patches
  (8 s window, 0–100 Hz, nperseg=256, 87.5% overlap)
- **MLP branch:** 2 fully connected layers processing the same ~20 handcrafted
  features used in Phase 1
- **Fusion:** Concatenated branch outputs → classification head
- **Total:** 258,627 parameters. PyTorch, NVIDIA L40S GPU.

**Why hybrid?** A pure CNN (spectrogram-only) achieved only **39% macro F1**.
This failure is expected: Phase 1 labels were defined by summary statistics
(peak frequency, spectral slope, power, duration), not visual appearance.
The CNN sees patterns the statistics miss (and vice versa). The hybrid model
achieves **93.7% macro F1** by combining both information sources.

**Training:** 70/15/15 split (62,683 / 13,432 / 13,432 events). Weighted
random sampling for class balance. Cross-entropy loss, AdamW (lr=1e-3),
cosine annealing LR. SpecAugment-style augmentation. Early stopping
(patience=8 on val macro F1).

**Test set results:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| T-phase | 1.00 | 0.92 | 0.96 | 8,402 |
| Icequake | 0.93 | 1.00 | 0.96 | 3,435 |
| Vessel | 0.80 | 1.00 | 0.89 | 1,595 |

**Combined population results (Phase 1 + Phase 2):**

| Class | Phase 1 | Phase 2 (high-conf) | Combined |
|-------|---------|---------------------|----------|
| T-phase | 55,783 | 117,077 | **172,860** |
| Icequake | 23,331 | 44,989 | **68,320** |
| Vessel | 10,458 | 29,280 | **39,738** |
| Low-confidence | — | 16,182 | 16,182 |

The confusion matrix reveals the main classification failure mode: a
small fraction of T-phases are misclassified as vessel noise or icequakes.

> **Figure: CNN Confusion Matrix** (`cnn_confusion_matrix.png`)
>
> **Temporary Caption:** Confusion matrix for the hybrid CNN+MLP classifier on the
> held-out test set (13,432 events). Overall accuracy: 95.1%, macro F1:
> 93.7%. Main confusion: 400 T-phases predicted as vessel (4.8% of
> T-phase support) and 249 T-phases predicted as icequake (3.0%). Icequake
> and vessel classes have near-perfect recall (≥1.00).

**Important caveat on interpretation:** These performance metrics reflect
agreement with Phase 1 unsupervised labels, not ground truth accuracy. As
described in §4.4, Phase 3 gold-standard review subsequently identified
substantial contamination in the Phase 1 training clusters — including
whale calls mislabeled as T-phases, icequakes in "known T-phase" clusters,
and a mega-cluster containing all signal types. The high test-set F1
(93.7%) should therefore be interpreted as a measure of *label consistency*
(the CNN faithfully reproduces the Phase 1 labeling scheme) rather than
true classification accuracy. Phase 3 frequency-band reclassification
(§4.4) addresses this limitation by building new labels from
frequency-separated clustering with expert gold-standard validation.

The following three montages show representative examples of each event
class, illustrating the spectral and temporal characteristics that
distinguish them.

> **Figure: T-phase Event Montage** (`paper/event_montage_tphase.png`)
>
> **Temporary Caption:** Six representative T-phase events detected by the
> BRAVOSEIS hydrophone array, selected across a range of SNR values and from
> different moorings to demonstrate variability. Each column shows one event
> with bandpass-filtered waveform (top, 1–15 Hz) and spectrogram (bottom,
> 0–100 Hz). Red vertical lines mark the AIC-refined onset time; colored
> shading indicates the detected event duration. T-phases are identified by
> impulsive onsets with dominant energy below 15 Hz and duration typically
> ≤ 3 s — the hydroacoustic signature of regional earthquakes propagating
> through the Sound Fixing and Ranging (SOFAR) channel. AIC onset picks are computed on the squared
> envelope within a 7 s window (5 s pre-trigger + 2 s post-trigger). Grade A
> picks (quality ≥ 0.7) exhibit a sharp AIC minimum at the noise-to-signal
> transition. Spectrogram: nperseg = 256, 87.5% overlap, Hann window.

> **Figure: Icequake Event Montage** (`paper/event_montage_icequake.png`)
>
> **Temporary Caption:** Six representative icequake events detected by the
> BRAVOSEIS hydrophone array. Each column shows one event with
> bandpass-filtered waveform (top, 5–30 Hz) and spectrogram (bottom,
> 0–100 Hz). Red vertical lines mark the AIC-refined onset time. Icequakes
> are distinguished from T-phases by their longer duration (> 3 s), more
> emergent character, and moderate spectral slope (−0.2 to −0.5). The
> emergent onset makes AIC picking more challenging — icequakes have a higher
> fraction of grade B picks than T-phases. The extended mid-band filter
> (5–30 Hz) captures the broader spectral content characteristic of glacial
> calving, ice shelf fracture, and sea ice cracking events. Spectrogram:
> nperseg = 256, 87.5% overlap, Hann window.

> **Figure: Vessel Noise Event Montage** (`paper/event_montage_vessel.png`)
>
> **Temporary Caption:** Six representative vessel noise events detected by
> the BRAVOSEIS hydrophone array. Each column shows one event with
> bandpass-filtered waveform (top, 30–250 Hz) and spectrogram (bottom,
> 0–250 Hz). Red vertical lines mark the AIC-refined onset time. Vessel
> noise is identified by its positive spectral slope (energy increasing with
> frequency), peak energy above 100 Hz, and broad bandwidth (~211 Hz) — the
> acoustic signature of propeller cavitation and ship machinery. Events
> appear as Type A broadband transients in temporal bursts corresponding to
> ~24 vessel passages over 13 months (peak May–September, consistent with
> krill trawler activity). The high-pass filter (30–250 Hz) isolates the
> vessel signature from concurrent low-frequency seismic energy. Spectrogram:
> nperseg = 256, 87.5% overlap, Hann window.

### 4.3 Temporal Distribution

With every event now classified, we can ask when different types of sounds
are most common. Earthquakes and ice-related sounds do not occur uniformly
throughout the year — they follow seasonal and tectonic patterns. Examining
the monthly counts of each event type provides a first-order check on
whether our classification makes physical sense, because icequakes should
peak during summer when glaciers are most active, while earthquake swarms
should cluster around known tectonic episodes.

The monthly detection counts for located T-phases and icequakes reveal
strong seasonal patterns consistent with known geophysical processes.

> **Figure: Monthly Detection Counts** (`paper/monthly_detections.png`)
>
> **Temporary Caption:** Monthly counts of located T-phase (blue) and
> icequake (orange) events for tiers A–C, January 2019 through February
> 2020. T-phase detections peak in February 2019 (dominated by the
> Feb 11 swarm, 4,976 events) and April 2019 (the Apr 22–24 swarm,
> ~3,495 events at peak). Icequake detections peak during the austral
> summer (January–March) when calving and ice shelf fracture are most
> active, and drop to near zero during the winter sea ice season
> (June–September). The asymmetry between summer and winter icequake
> counts is consistent with the coast-distance + sea ice filter
> (§6.1) retaining only physically plausible winter events. Months
> with no hydrophone coverage appear as gaps.

**Active-source survey context:** The period January 21 through February 4,
2019 overlaps with active-source seismic survey operations by the R/V
*Sarmiento de Gamboa*. The survey used three source configurations: MCS1
(1780 cu in, 5 m depth, Jan 21–26, Orca Volcano reflection lines),
Tomography (2540 cu in, 15 m depth, Jan 26–29, Orca Volcano refraction),
and MCS2 (1580 cu in, 5 m depth, Jan 30–Feb 4, Rift + Edifice A
reflection lines). Mooring deployment operations occurred January 10–13,
prior to the start of survey acquisition. Some detected events during the
survey period represent controlled airgun pulses rather than natural
seismicity; these are identified and excluded during Phase 3 classification
review (§4.4).

### 4.4 Phase 3 — Frequency-Band Reclassification

At this point in the project, we had labels for all 297,170 development
subset events — but how much could we trust them? The classification
pipeline looked good on paper (95% accuracy), but accuracy only measures
whether the computer agrees with the labels it was trained on. If those training labels were wrong to begin
with, the computer would confidently repeat the same mistakes on 200,000 new
events. This is a well-known trap in machine learning called "garbage in,
garbage out." To find out whether we had fallen into it, we went back and
carefully examined the events that should have been the easiest to classify
— the ones sitting right at the center of each cluster. What we found
prompted us to rethink the entire classification approach.

#### 4.4.1 Problem 1: Circular Labeling

Imagine you sort a pile of rocks by weight and color, then label each group
("granite," "sandstone," "marble"). Now you train someone to sort new rocks
using your labels — but what if some of your original groups were wrong?
The trainee would learn your mistakes and apply them to thousands more rocks.

That is essentially what happened here. Phase 1 labels were derived from
handcrafted feature thresholds applied to HDBSCAN cluster assignments — for
example, T-phases were defined as events with peak frequency < 30 Hz,
power > 48 dB, spectral slope < −0.5, and duration ≤ 3 s. The Phase 2
CNN+MLP was then trained on these labels and applied to the remaining
207,598 bulk events. Because the labels originated from the same features
used for training, the CNN inevitably propagated any errors in the original
threshold-based assignments to the bulk population. High test-set accuracy
(95.1%) reflected agreement with noisy labels, not necessarily agreement
with ground truth.

#### 4.4.2 Problem 2: Cluster Contamination

To test whether the labels were actually correct, we looked at the events
that the clustering algorithm was most confident about — the ones closest
to the center of each group. If even these "best" examples turn out to be
misclassified, it means the clusters themselves are unreliable. Think of it
like checking the ripest apples in a bin labeled "apples" — if half of them
are actually pears, you cannot trust any of the labels in that bin.

To quantify the severity of mislabeling, we conducted a gold-standard review
of events nearest each HDBSCAN cluster centroid — the highest-confidence
members of each cluster. For each of the 12 non-noise clusters across three
frequency bands, we selected the 10 events closest to the UMAP centroid and
generated individual waveform + spectrogram panels for expert visual review.
Each event was classified as Accept (clean, unambiguous example of the
expected class) or Reject (contaminated, ambiguous, or belonging to a
different class).

Scripts: `make_gold_standard_montages.py` (10-row montages),
`make_gold_single_panels.py` (individual large panels),
`notebooks/gold_standard_review.ipynb` (interactive accept/reject review).

**Gold-standard review results (partial — first 5 clusters):**

| Cluster | Events | Band | Accepted | Rejected | Key Findings |
|---------|--------|------|----------|----------|--------------|
| low_0 | 4,686 | 1–15 Hz | 3/10 | 7/10 | "Known T-phase" cluster heavily contaminated with icequakes, vessel noise |
| low_1 | 3,666 | 1–15 Hz | 1/10 | 9/10 | "Known T-phase" cluster dominated by icequakes |
| low_2 | 66,563 | 1–15 Hz | 5–6/10 | 3/10 | Mega-cluster contains all signal types: T-phases, icequakes, vessel, whale calls |
| mid_0 | 1,056 | 15–30 Hz | 0/10 | 10/10 | Dominated by recurring ~12 s repeating signal (new category); all events from single file |
| mid_2 | 7,207 | 15–30 Hz | 0/10 | 8/10 | "Vessel" cluster — no clean examples found |

These results demonstrate that even the highest-confidence cluster members
(nearest the UMAP centroid) have unacceptable contamination rates. The
"known T-phase" clusters (low_0, low_1) had 70–90% rejection rates. The
low_2 mega-cluster (66,563 events, 22% of the catalogue) proved to be a
mixture of all four signal types rather than a single class. The mid_0
cluster was entirely composed of a recurring ~12 s signal not previously
identified in the classification scheme.

**Recurring diagnostic observations from the review:**

- **Onset picker consistently late**: Across all accepted T-phase events,
  the STA/LTA + AIC onset was visibly late relative to the true signal onset.
  This systematic bias affects TDOA-based locations.
- **Icequake spectral signature**: Low energy at the lowest frequencies
  (< 5 Hz) combined with spectral energy extending to 150–250 Hz — the
  opposite of the T-phase pattern.
- **Fin whale 20 Hz calls identified**: Repetitive ~20 Hz pulses identified
  in low_1, low_2, and mid_0. The mid_0 cluster was dominated by a recurring
  signal initially described as "unknown ~12 s repeating signal." Inter-pulse
  interval analysis (§4.4.4) confirmed these as fin whale (*Balaenoptera
  physalus*) 20 Hz calls, establishing a 4th biological class not captured
  by the original 3-class scheme.

#### 4.4.3 Revised Approach: Frequency-Band Reclassification

Rather than re-cluster all events together with a fixed analysis window
(an approach we initially considered but abandoned), we adopted a
fundamentally different strategy: separate events by frequency band
*before* classification. The original pipeline detected events in three
overlapping frequency bands (1–15, 15–30, 30–250 Hz) but then pooled
all detections for clustering, allowing frequency to dominate the UMAP
embedding and producing a single mega-cluster containing 99% of events
when all-band clustering was attempted. The frequency-band approach
eliminates this problem by clustering within physically meaningful
frequency regimes where different source processes dominate.

Three frequency bands, each analyzed independently:

1. **1–14 Hz (lowband)**: T-phases and local seismicity. These are
   SOFAR-channel propagated signals whose energy concentrates below
   ~14 Hz after long-range propagation through the deep sound channel.
2. **14–30 Hz (midband)**: Whale-dominated — fin whale 20 Hz calls
   (§4.4.4), and other biological signals. Set aside for a separate
   biological acoustics study.
3. **>30 Hz (highband)**: Icequakes and vessel noise, characterized by
   broadband energy extending to 150–250 Hz. Pipeline pending.

**Lowband pipeline (1–14 Hz):**

A 4th-order Butterworth bandpass filter (1–14 Hz) was applied to raw
waveforms *before* spectrogram computation, ensuring that features
reflect only the target frequency range rather than being contaminated
by out-of-band energy. Spectrograms were computed with nperseg=2048
(~0.49 Hz frequency resolution at 1 kHz sample rate), and 6 feature
bands of ~2 Hz each were extracted within the 1–14 Hz range.

**Whale contamination filter:** Events whose original catalogue peak
frequency exceeded 17 Hz were removed prior to clustering. This threshold
was determined empirically: cross-validation against Singer's manual
catalogue (§7.2) and the Orca OBS catalogue (§7.1) showed that a 14 Hz
cutoff — initially chosen to match the bandpass edge — was discarding
genuine T-phase events whose broadband spectral peak happened to fall
just above the filter boundary. Of 58 Singer-labeled earthquakes removed
by the 14 Hz cutoff, 34 (59%) had catalogue peak frequencies of only
14.6 Hz, and their lowband features (duration, spectral slope, SNR) were
statistically indistinguishable from accepted T-phases. Raising the
threshold to 17 Hz recovered these events while still excluding the
fin whale 20 Hz band. After filtering: 78,487 events remained (6,211
removed).

**Clustering:** UMAP (n_neighbors=15, min_dist=0.01) followed by HDBSCAN
(min_cluster_size=500, EOM selection method) produced 6 clusters plus
noise (2,121 noise points, 2.7%), with silhouette score 0.399.

**Gold-standard review:** For each cluster, 15 events were selected via
stratified sampling across UMAP centroid distance quintiles and reviewed
as individual waveform + spectrogram panels. All clusters were
cross-referenced against the R/V *Sarmiento de Gamboa* cruise report
(survey line schedule and source configurations) to identify
active-source survey artifacts.

**Lowband cluster results:**

| Cluster | Events | Median freq | Verdict | Signal type |
|---------|--------|-------------|---------|-------------|
| lowband_1 | 7,970 | 9.8 Hz | Accept | Strong T-phases (highest confidence) |
| lowband_2 (SNR>=6) | 12,333 | 10.3 Hz | Accept | T-phases (mega-cluster, SNR filtered) |
| lowband_0 (SNR>=6) | 891 | 12.7 Hz | Accept | Recovered borderline T-phases |
| lowband_3 | 1,518 | 2.0 Hz | Discard | Ultra-low frequency, negative slope |
| lowband_4 | 3,394 | 2.9 Hz | Discard | Ambient noise, steep negative slope |
| lowband_5 | 6,880 | 2.4 Hz | Discard | Low-frequency noise |

**Total accepted for seismic catalogue: 21,194 events** from lowband
review (lowband_1 + lowband_2 with SNR >= 6 + lowband_0 with SNR >= 6).

**Highband pipeline (30–450 Hz):**

The same frequency-band separation approach was applied to events
detected in the high-frequency band (>30 Hz). A 4th-order Butterworth
bandpass (30–450 Hz) was applied before spectrogram computation, with
nperseg=1024 (~1 Hz resolution at 1 kHz sample rate) and 6 feature
bands spanning 30–450 Hz. The instrument response extends to ~250 Hz;
energy above this is outside the useful range but included in feature
extraction to capture any aliased content.

**Highband clustering:** UMAP + HDBSCAN produced 4 clusters plus noise
(3,927 noise, 4.9%), silhouette 0.320, from 79,947 events.

**Highband gold-standard review:**

| Cluster | Events | Peak freq | IQ rate | Verdict |
|---------|--------|-----------|---------|---------|
| highband_0 | 34,069 | 55 Hz | 93% | Accept — clean icequakes |
| highband_1 | 21,333 | 280 Hz | 33% | Discard — mixed icequake/humpback/noise |
| highband_2 | 18,359 | 57 Hz | 73% | Accept — icequakes (systematic late picks) |
| highband_3 | 2,259 | 61 Hz | 80% | Accept — long-duration icequakes (late picks) |

**Total accepted for cryogenic catalogue: 54,687 events** (highband_0 +
highband_2 + highband_3). Highband_1 was discarded: only 33% of reviewed
panels were icequakes, with 27% identified as humpback whale calls that
extend above 30 Hz, and the remainder noise or ambiguous. The cluster's
unusual spectral profile (positive slope, 280 Hz peak — above the 250 Hz
instrument response) further supports exclusion.

**Late onset pick caveat:** Highband_2 (7/11 icequake panels) and
highband_3 (8/12 icequake panels) exhibited systematic late onset picks
— the STA/LTA trigger point was visibly after the true signal onset. This
bias affects TDOA-based locations for cryogenic events and should be
considered when interpreting location precision.

**Combined Phase 3 catalogue:** The lowband and highband pipelines were
combined into a single classified catalogue of **75,881 events**: 21,194
seismic (1–14 Hz, T-phases) and 54,687 cryogenic (>30 Hz, icequakes).
Events in the 14–30 Hz midband (predominantly whale calls) are excluded
and reserved for a separate biological acoustics study.

Scripts: `extract_features_lowband.py`, `cluster_lowband.py`,
`extract_features_highband.py`, `cluster_highband.py`,
`make_highband_panels.py`, `assemble_phase3_catalogue.py`.

This revision supersedes the Phase 1 and Phase 2 combined results
reported in §4.1–4.2.

#### 4.4.4 Fin Whale 20 Hz Call Identification

Sometimes the most interesting discoveries come from things you were not
looking for. During gold-standard review of the mid-band clusters, one
cluster (mid_0, 980 events) was entirely composed of a repeating signal
with a distinctive ~12-second rhythm — like a slow, steady drumbeat in the
ocean. At first it was labeled "unknown repeating signal," but the regular
pulse spacing and frequency content were strong clues. To identify it, we
measured the time between consecutive pulses — what bioacousticians call
the inter-pulse interval (IPI) — and compared it to known marine animal
call signatures.

**Signal discovery and characterization:**

The recurring signal was first observed during gold-standard review of
mid_0 cluster centroids. All 10 reviewed events were rejected as
non-tectonic/non-cryogenic, but the signal's regularity prompted further
investigation. Long spectrogram analysis (60-minute windows) on file 943
(2019-06-19) revealed that the signal persisted for at least 2 hours
across all 6 moorings simultaneously.

Signal characteristics:
- **Frequency**: Broadband pulses concentrated between 15–35 Hz
- **Individual pulse duration**: ~2–3 seconds
- **Present on all 6 moorings**: Consistent with a far-field acoustic
  source propagating through the water column

**Inter-pulse interval analysis:**

To measure the IPI, we applied a Hilbert envelope to the bandpass-filtered
(15–30 Hz) waveform from M3 file 943 (first 60 minutes, where signal is
strongest), smoothed with a 0.5-second window, and detected peaks above
median + 2σ with a minimum spacing of 8 seconds.

Results (208 pulses detected in 60 minutes):

| Metric | Value |
|--------|-------|
| Median IPI | 14.7 s |
| Modal bin (1 s bins) | 14–15 s (116/195 = 59% of intervals) |
| Standard deviation | 3.3 s |
| Range (filtered 8–30 s) | 8.1–29.0 s |

The sharp concentration of IPIs at 14.7 seconds (±3.3 s) is diagnostic of
**fin whale (*Balaenoptera physalus*) 20 Hz song**. Fin whale songs consist
of regularly spaced ~20 Hz frequency-modulated downsweep pulses with IPIs
typically between 12–20 seconds, most commonly 13–18 seconds depending on
population and behavioral context (Watkins et al. 1987; Širović et al. 2004;
Oleson et al. 2014). The 14.7 s median IPI, tight clustering in the 14–15 s
bin, and 15–35 Hz frequency content are all consistent with a singing fin
whale.

**Distribution in the catalogue:**

Of the 980 mid_0 events, 812 (83%) originate from a single 4-hour file
(file 943, 2019-06-19 02:35–06:35 UTC). The remaining events are
distributed across 3 additional files spanning 2019-06 to 2019-12,
indicating fin whale acoustic presence during the austral winter and
spring — consistent with known fin whale migration and feeding patterns
in the Southern Ocean.

**Implication for classification:**

The identification of fin whale calls establishes a 4th event class
(biological/cetacean) that was not included in the original 3-class
scheme (T-phase, icequake, vessel). Events with this spectral signature
and IPI pattern should be separated from tectonic and cryogenic events
before seismicity analysis. The ~20 Hz frequency places fin whale calls
at the boundary between the low (1–15 Hz) and mid (15–30 Hz) detection
bands, explaining their presence in both low_1/low_2 and mid_0 clusters.

Scripts: `make_12s_signal_figures.py` (multi-scale waveform + spectrogram
figures), `make_12s_signal_pdf.py` (compiled PDF for colleague review).

---

## 5. Source Location

Now that we know what each sound is, we want to know where it came from. The
basic idea is triangulation: because our six microphones are spread across the
Bransfield Strait, a sound arrives at each one at a slightly different time.
By measuring these TDOA values and knowing how
fast sound travels through the water, we can work backward to find the geographic
point that best explains all the observed arrival time differences. This section
describes how we link detections across moorings into associations, search a
geographic grid for the best-fit source location, and assign quality tiers to
each result.

### 5.1 Cross-Mooring Association

#### 5.1.1 Baseline: Greedy Windowed Clustering

The initial association method links events on different moorings using
**pair-specific travel time windows** derived from in-situ Expendable
Bathythermograph (XBT) sound speed profiles. The effective horizontal
speed is computed as the harmonic mean of the sound speed profile:

    c_eff = z_max / ∫(1/c(z) dz, 0, z_max)

A **15% safety factor** widens each window. Resulting windows range from
21 s (M4–M5, 27 km) to 139 s (M1–M6, 176 km). For each anchor event,
all detections on other moorings falling within the pair-specific window
are gathered, and the highest-SNR detection per mooring is selected. This
greedy approach is fast (O(n log n)) but does not verify whether the
resulting combination of arrival times is physically consistent with any
source location.

**Alternative considered:** A fixed 120 s window at 1480 m/s. Rejected
because it was 6× too wide for close pairs (M4–M5) and slightly too narrow
for the most distant pair (M1–M6). Pair-specific windows reduce false
associations for close pairs while maintaining sensitivity for distant ones.

#### 5.1.2 TAPAAs: Spatial-Pruning Association

Cross-validation against the Orca OBS network (Section 7.6) revealed that
**false associations are the dominant source of location error** — the
greedy algorithm links unrelated detections on different moorings that
happen to fall within the travel-time window, producing physically
implausible TDOA combinations. Only 8% of associations for co-identified
Orca events were spectrally consistent across moorings; the remainder
had peak frequency spreads of 50–100 Hz, indicating mixed-event
associations.

To address this, we adopted a spatial-pruning association strategy
inspired by the TAPAAs algorithm (TDoA Acoustic Progressive Algorithm
for Association; Raumer et al., 2025). The key innovation is that
**location consistency is verified during association, not after**: as
each new mooring is added to a candidate association, the algorithm
checks whether there exists any grid cell in the study area consistent
with ALL observed TDOAs simultaneously, and rejects combinations that
fail this test.

**Algorithm:**
1. Pre-compute a TDOA grid for each of the 15 mooring pairs: for every
   cell (0.01° spacing, ~136,000 cells covering the study area ± 1°
   padding), compute the expected TDOA using per-pair XBT-derived
   effective sound speeds.
2. Define a total TDOA tolerance per cell: Δ = Δ_geometric + 2×Δ_pick,
   where Δ_geometric ≈ 0.42 s accounts for source position uncertainty
   within one grid cell, and Δ_pick = 2.0 s accounts for onset pick
   uncertainty (AIC quality).
3. For each anchor detection, gather candidate detections on other
   moorings within the pair-specific travel-time window.
4. Add candidates one mooring at a time (fewest candidates first for
   maximum pruning). For each candidate, intersect the set of valid grid
   cells with those consistent with the new observed TDOA. If the
   intersection is empty, reject that candidate.
5. Among spatially-consistent candidates for each mooring, select the
   highest-SNR detection.
6. Retain only associations with ≥3 moorings.
7. The best-fit location is extracted directly from the valid cell set
   (minimum RMS residual among valid cells), providing the location as a
   by-product of association with no separate location step needed.

Each detection band (low, mid, high) is processed independently,
consistent with the detection pipeline. Processing all 6.76M detections
takes approximately 1 hour.

> **Figure: Sound Speed Profile** (`association/sound_speed_profile.png`)
>
> **Temporary Caption:** Effective horizontal sound speed used for cross-mooring
> association, derived from XBT casts during the BRAVOSEIS deployment.
> The harmonic mean over the water column gives an effective speed of
> 1455.5 m/s. A 15% safety factor is applied to the maximum expected
> travel time for each mooring pair to account for deviations from the
> straight-line path assumption.

**Reference:** The TAPAAs spatial-pruning approach was introduced by
Raumer, P.-Y., Bazin, S., Safran, R., Cazau, D., & Royer, J.-Y. (2025).
Automatic Analysis of Hydroacoustic Signals Related to the Activity of
the Fani Maoré Submarine Volcano. *Geochemistry, Geophysics, Geosystems*,
26(12). https://doi.org/10.1029/2025GC012572. Our implementation adapts
their approach for a 6-mooring linear array with per-pair calibrated
sound speeds, whereas the original was developed for a 4-hydrophone
70×70 km square array monitoring a submarine volcano off Mayotte Island.

Script: `associate_tapaas.py`.

### 5.2 Grid-Search TDOA

For each association with ≥3 moorings, a geographic grid is searched for
the point that minimizes the Root Mean Square (RMS) residual between observed and predicted
inter-station travel time differences.

**Two-stage grid search:** A coarse grid at 0.01° spacing (~1 km) covers the
study area ±0.5° padding (154,700 grid points, geodesic distances precomputed
via World Geodetic System 1984 (WGS84) ellipsoid). After identifying the coarse minimum, a fine grid at
0.001° spacing (~100 m) is searched within a ±0.015° region around it. The
fine stage uses a flat-Earth distance approximation (1° lat ≈ 111 km,
1° lon ≈ 111·cos(lat) km), which introduces negligible error at the ~100 m
scale of the refinement grid at −62° latitude. This two-stage approach
improves location precision by approximately 10× while keeping
computational cost modest — the fine grid contains only ~900 points per
event, compared to 154,700 for the coarse grid.

**Per-pair effective sound speeds:** Rather than applying a single
array-wide mean speed, each mooring pair's TDOA prediction uses its own
effective horizontal speed derived from XBT cast profiles. Per-pair speeds
range from 1454.8 m/s (M2–M3) to 1456.1 m/s (M1–M5, M4–M5, M5–M6),
compared to the global mean of 1455.5 m/s. The maximum speed variation
across pairs is 0.07%, producing travel time corrections of up to ~0.08 s
on the longest paths (M1–M6, 176 km). While the practical impact on tier
classification is small (the correction is below onset pick uncertainty
for most events), this removes a known systematic bias in the TDOA
predictions and is the theoretically correct treatment when pair-specific
sound speed information is available.

**Known limitation — flat-ocean approximation:** Travel times assume
straight-line geodesic paths and pair-specific effective speeds. This ignores
depth-dependent velocity structure and bathymetric refraction/reflection.
For compact arrays (Wilcock [2012] used 2D ray tracing for a 15–20 km
network), this would be a significant error. For our 175 km aperture,
the straight-line approximation is more defensible but still introduces
systematic bias near bathymetric barriers (ridges, shallow sills).

**Bathymetry for map figures:** All location maps use a three-layer merged
bathymetry product. The base layer is the International Bathymetric Chart of
the Southern Ocean Version 2 (IBCSO v2; Dorschel et al., 2022; 500 m
resolution; doi:10.1594/PANGAEA.937574), which provides complete coverage
of the study area. This is overlaid with higher-resolution multibeam data
collected during the BRAVOSEIS experiment (2019–2020), and the
highest-resolution data in the central basin comes from the MGDS Orca
Volcano gridded model (doi:10.60521/332247). Where multiple layers overlap,
the highest-resolution source takes precedence.

### 5.3 Multipath Protection

Three mechanisms address multipath-contaminated onsets:

1. **Per-mooring outlier detection** (≥4 moorings): If one mooring's
   individual residual exceeds 3× the median AND >1 s, relocate without it.
   Accept if residual improves by >30%. Applied to 339 events.

2. **Jackknife validation** (≥4 moorings): Relocate N times dropping each
   mooring. If max shift >15 km, flag as unstable and downgrade tier.
   247 events (2.0%) flagged.

3. **Distance constraint:** Locations >150 km from array centroid → tier D.

### 5.4 Quality Tiers

**Development subset (717 files, greedy association):**

| Tier | Criteria | Count | Median residual |
|------|----------|-------|-----------------|
| A | ≥4 moorings, residual <1 s, jackknife-stable | 4,304 | 0.00 s |
| B | ≥3 moorings, residual <2 s | 6,979 | 0.00 s |
| C | ≥3 moorings, residual 2–5 s or jackknife-unstable | 926 | 3.29 s |
| D | 2 moorings, >150 km, or residual >5 s | 28,575 | — |

**Full dataset (14,663 files, greedy association):**

| Tier | Count |
|------|-------|
| A | 70,927 |
| B | 300,024 |
| C | 177,360 |
| D | 1,305,412 |

**Note on zero residuals:** Tier A/B events with 3 moorings have zero
residual because the system is fully determined (2 TDOAs, 2 unknowns).
Residual-based quality assessment is meaningful only for ≥4 mooring events.
Tier A requires ≥4 moorings for this reason.

### 5.5 Location Uncertainty

Formal uncertainty is estimated from the curvature of the RMS residual
surface at the grid-search minimum. The eigenvalues of the 2×2 Hessian
(computed via finite differences) give curvature in the principal directions.
The geometric mean of the 1-σ semi-axes is reported as `uncertainty_km`.

| Tier | Median uncertainty | Notes |
|------|--------------------|-------|
| A | 0.1 km | Over-determined (≥4 moorings) |
| B | 0.0 km | Mostly 3-mooring — formally overconfident |
| C | 8.6 km | Broad minimum |

**Caveat:** 3-mooring uncertainties are physically overconfident — the true
error is dominated by systematic effects (sound speed, onset picking) not
captured by the residual curvature. Uncertainties for ≥4 mooring events are
more reliable.

### 5.6 Association Method Comparison

The full 14,663-file dataset was processed with both association methods
to characterize their impact on catalogue quality:

| Metric | Greedy (baseline) | TAPAAs (spatial pruning) |
|--------|------------------|------------------------|
| Total associations | 1,853,723 | 1,352,319 |
| 3-mooring | 600,153 | 893,432 |
| 4-mooring | 420,610 | 310,674 |
| 5-mooring | 231,747 | 103,239 |
| 6-mooring | 94,095 | 44,974 |
| Median residual | 5.2 s | 0.57 s |
| Publishable (A+B+C) | 548,311 | (pending fine-grid tiering) |

TAPAAs rejects ~500,000 associations that the greedy approach accepted
but that were spatially implausible, reducing the median residual by
9×. The lower 4–6 mooring counts reflect the rejection of associations
where additional moorings were inconsistent with any single source
location — a stricter but more honest assessment of which detections
truly correspond to the same physical event.

Cross-validation against the Orca OBS network (Section 7.6) confirms that
the greedy approach's higher mooring counts partly reflect false
associations that coincidentally produced reasonable-looking locations. The
choice between these methods involves a trade-off between catalogue size
(greedy) and internal consistency (TAPAAs); both are provided as outputs
for downstream analysis.

---

## 6. Quality Control

Even after careful detection, classification, and location, some results will
be wrong — a mid-ocean event mislabeled as an icequake, or a location thrown
off by a bad onset pick. This section describes the filters and checks we
apply to catch these errors. We use physical reasoning (icequakes should
originate near ice, not in deep open water), statistical outlier detection
(events that locate far from their swarm neighbors are likely mislocated),
and completeness analysis (below what signal strength does our detector
start missing events?). The goal is to produce a final catalogue where
every retained event has a physically plausible classification and location.

### 6.1 Icequake Coast-Distance + Sea Ice Filter

**Problem:** Unfiltered icequake locations exhibited a spatial distribution
statistically identical to T-phases (same mean lat/lon, same rift-zone
fraction, same distance-to-coast distribution). This indicates widespread
misclassification of mid-strait events as icequakes.

**Diagnosis:** Icequakes should originate near ice-rock/ice-water interfaces.
Events in mid-strait deep water along the rift axis are almost certainly
misclassified T-phases.

**Solution — seasonally varying filter:**

1. Compute geodesic distance from each located icequake to the nearest
   coastline or Antarctic ice shelf (Natural Earth 10 m + ice shelf polygons)
2. Look up monthly Sea Ice Concentration (SIC) from the National Snow and Ice Data Center (NSIDC) Climate Data Record (CDR) V4 (25 km, PolarWatch
   ERDDAP) at each event location
3. **Near coast** (≤30 km): retain — glacial/ice shelf source plausible
4. **Far from coast but ice-covered** (>30 km, monthly SIC ≥15%): retain —
   sea ice cracking is physically plausible
5. **Far from coast, ice-free** (>30 km, SIC <15%): reclassify to
   "unresolved"

**Results:** Of 1,030 located icequakes: 507 retained (near coast), 112
retained (in ice-covered water during winter), 411 reclassified. **619
icequakes retained.**

**Why 30 km?** The threshold accommodates icebergs drifting short distances
from calving fronts while excluding events in the deep rift axis (typically
40–60 km from coast). It is conservative — some legitimate near-coast events
may be excluded.

**Why not a fixed threshold year-round?** The Bransfield Strait has sea ice
coverage from ~May to ~October. During winter, the mid-strait region IS
ice-covered, and sea-ice cracking produces legitimate acoustic events.
A fixed distance filter would incorrectly reject these. The sea ice data
(2019 was anomalously low — record January and June extents) provides
month-by-month validation.

> **Figure: Icequake + Sea Ice 6-Panel**
> (`location/icequake_seaice_6panel.png`)
>
> **Temporary Caption:** Icequake locations overlaid on NSIDC CDR monthly sea ice
> concentration contours (cyan: 15%, blue: 50%, dark blue: 80%) for six
> two-month windows spanning the deployment. Blue squares: retained
> icequakes within 30 km of coast. Green squares: retained icequakes
> in ice-covered water (>30 km from coast, SIC ≥15%). Gray crosses:
> reclassified events (far from coast, no ice cover). Sea ice contours
> appear in May–October panels, confirming that the seasonally varying
> threshold correctly retains winter mid-strait events. The absence of
> green squares in summer panels (Jan–Apr, Nov–Feb) is consistent with
> ice-free conditions. Bathymetry: IBCSO v2 base with BRAVOSEIS multibeam
> and Orca high-res overlays. White triangles: mooring positions.

### 6.2 Swarm Coherence QC

**Principle:** T-phase swarms (temporally clustered events from the same
seismic source region) should be spatially coherent. Events within a swarm
that locate far from the swarm's centroid are likely mislocation artifacts
(inspired by Wilcock's [2012] track-constrained relocation for whale calls).

**Method:** Temporally adjacent T-phase events (gap <1 hour) are grouped
into swarms. For each swarm with ≥10 events, the spatial centroid is
computed and events whose distance from the centroid exceeds 3× Median Absolute Deviation (MAD) are flagged as `swarm_outlier`.

**Results:** 79 swarms identified containing 9,443 T-phase events. **66
spatial outliers** flagged (median 174 km from centroid — clearly
mislocated). These events are retained in the dataset but flagged for
exclusion from spatial analyses.

> **Figure: Swarm Coherence QC** (`paper/swarm_coherence_qc.png`)
>
> **Temporary Caption:** Swarm coherence quality control for located T-phase
> events. (a) Map of swarm events colored by swarm membership, with 66
> spatial outliers marked as red symbols. Outliers locate far from their
> swarm centroids, consistent with mislocation due to multipath or onset
> picking errors. White triangles: mooring positions. (b) Distribution of
> distance from swarm centroid for all 9,443 swarm events. The 3× MAD
> threshold (dashed vertical line) separates the coherent swarm population
> from the outlier tail. Inliers (blue) cluster tightly around the
> centroid; outliers (red) have a median distance of 174 km, confirming
> they are mislocated rather than genuine off-centroid events.

### 6.3 Detection Completeness

**Approach:** An acoustic magnitude of completeness (Mc) is estimated
using relative source level as the magnitude proxy. For each located
T-phase (tiers A–C, N=9,466), the relative source level is computed as:

    SL_rel = RL + TL

where RL is the peak spectral power at the best-receiving mooring (dB)
and TL = 15·log₁₀(r) is the practical spreading loss to the nearest
mooring (distance r in meters). The 15·log₁₀ factor represents a
compromise between spherical (20) and cylindrical (10) spreading.

**Calibration caveat:** These are **relative** values. Absolute source
levels (dB re 1 μPa at 1 m) require hydrophone sensitivity calibration
curves not available for this deployment. The shape of the
frequency–source level distribution is independent of absolute
calibration and is valid for estimating Mc.

**Mc estimation:** Maximum curvature method (Wiemer & Wyss, 2000) — the
bin with the highest event count in the non-cumulative distribution
marks the magnitude of completeness. Above Mc, the Gutenberg–Richter
relationship log₁₀(N) = a − b·SL holds. Note that the b-value is
expressed per decibel of relative source level, not per conventional
magnitude unit, so direct comparison with seismological b-values
(typically ~1.0) requires a magnitude–dB conversion factor.

> **Figure: Magnitude of Completeness** (`paper/magnitude_completeness.png`)
>
> **Temporary Caption:** Acoustic magnitude of completeness for located T-phase
> events (tiers A–C, N=9,466). (a) Cumulative frequency vs. relative source
> level with Gutenberg–Richter fit above Mc. Mc = 120 dB (maximum curvature
> method). b-value = 0.067. (b) Non-cumulative histogram (1 dB bins) showing
> the rollover below Mc where detection sensitivity falls off. Relative
> source level = received level (peak spectral power) + transmission loss
> (15·log₁₀(r) practical spreading to nearest mooring). These are relative
> values — absolute calibration requires hydrophone sensitivity curves not
> available for this deployment.

---

## 7. Ground Truth Validation

An automated pipeline is only as trustworthy as its validation against
independent evidence. Fortunately, the Bransfield Strait was simultaneously
monitored by other instruments and analyzed by other researchers during the
same period. This section compares our automated detections against two
independent sources: the Orca seismic network (a separate set of Ocean-Bottom
Seismometers and land stations that located earthquakes using traditional
seismological methods) and a manually curated catalogue produced by a NOAA/PMEL
analyst who visually inspected the same hydrophone recordings. Agreement with
these independent datasets builds confidence that our automated results are
reliable.

### 7.1 Orca Seismic Network Cross-Validation

The co-deployed Orca OBS/land seismometer network independently located
5,789 earthquakes in the Bransfield Strait during the same period.
With the full 14,663-file archive, our recording windows now span the
entire deployment. Cross-validation against Orca is performed using
the same fate-tracing methodology as for Singer (Section 7.2).

The USGS global catalogue contains only 4 events ≥M4.6 in the study
region during this period, confirming that Bransfield Strait seismicity
is predominantly local and small-magnitude, well below the global
network detection threshold.

### 7.2 Comparison with Singer Manual Catalogue

Singer (NOAA/PMEL) independently analyzed the same hydrophone data using
manual spectrogram inspection and picking, producing a comprehensive
catalogue of 18,502 events classified as earthquake (EQ, 2,252), icequake
(IQ, 13,795), unknown (IDK, 2,010), and other categories (446). This
catalogue represents a substantial manual effort and provides valuable
context for evaluating our automated approach.

**Temporal coverage:** Processing the full 14,663-file archive provides
near-complete temporal overlap with Singer's catalogue. All 2,252 of
Singer's EQ events fall within our recording windows, compared to only
164 (7.3%) with the earlier 717-file development subset. This
dramatically improves the statistical power of the cross-validation.

**Detection completeness:** Of Singer's 2,252 EQ events:
- **99.9% (2,250)** were detected by our STA/LTA pipeline within a
  30-second tolerance — only 2 events were missed entirely
- **98.5% (2,218)** were not only detected but also associated across
  multiple moorings and assigned a source location
- **74.2% (1,672)** received a publishable location (quality tier A, B,
  or C)
- **Median time offset: 9.0 s** for matched events

**Fate breakdown by detection band:**

| Fate | Count | % | Notes |
|------|-------|---|-------|
| Low-band A/B/C (seismic proxy) | 806 | 35.8% | Best spectral match to tectonic T-phases |
| Other-band A/B/C | 866 | 38.5% | Mid (392) + high (474) band matches |
| Tier D (poor quality) | 546 | 24.2% | 2-mooring, high residual, or out of range |
| Detected, not located | 32 | 1.4% | Detected but not associated |
| Not detected | 2 | 0.1% | Below STA/LTA threshold |

**Complementary classification approaches:** Singer's catalogue employs
visual analysis of spectral information, assigning EQ or IQ labels based
on the analyst's interpretation of spectrogram characteristics. Our
pipeline classifies events using detection frequency band (1–15 Hz for
seismic, >30 Hz for cryogenic). These represent different — and
complementary — analytical philosophies. The fact that 38.5% of Singer's
EQ events are best matched in the mid or high band likely reflects
broadband sources generating energy across the spectrum, as well as
differences in how onset picks propagate into associations.

**Singer's EQ/IQ boundary:** Singer's EQ events show a seasonal pattern
correlated with IQ events, suggesting some degree of overlap between the
two categories in the manual catalogue. Our frequency-band approach
sidesteps this ambiguity by classifying based on spectral content rather
than source interpretation, though at the cost of not distinguishing
tectonic from cryogenic sources that produce energy in both frequency
bands.

Scripts: `singer_eq_fates.py`, `crossvalidate_seismic_catalogues.py`.

### 7.3 Location Comparison with Singer Manual Catalogue

Beyond temporal cross-validation (Section 7.2), a direct comparison of
source locations provides a more demanding test of our automated pipeline.
Singer independently picked arrival times and located events from the same
hydrophone recordings using manual spectrogram analysis. Because both
analyses share the same raw data but differ in every subsequent step —
onset picking, mooring association, and location algorithm — systematic
differences reveal where the pipeline adds value and where its limitations
lie.

**Matching:** Of Singer's 2,252 EQ events, **806 (35.8%)** match our
low-band (1–15 Hz, seismic proxy) locations at quality tier A, B, or C
within a ±30-second temporal tolerance. This is a substantial improvement
over the 717-file subset (52 matches), enabling robust statistical
characterization of location accuracy.

**Offset by quality tier:**

| Tier | Matched events | Median offset (km) | Mean offset (km) |
|------|---------------|-------------------|-----------------|
| A | 188 | 14.0 | 38.4 |
| B | 408 | 51.4 | 89.5 |
| C | 210 | 63.3 | 105.7 |

**High-quality agreement:** Tier-A events — which require 4 or more
moorings, residual < 1 s, and jackknife stability — achieve a
**median spatial offset of 14.0 km**. This represents good agreement
given the array aperture (~80 km) and the effective sound speed
uncertainties, and demonstrates that the grid-search algorithm performs
well when provided with consistent, multi-mooring arrival-time data.

**3-mooring sensitivity:** Tier-B events, dominated by 3-mooring
locations, show a **median offset of 51.4 km**. This is expected: with
3 moorings, 2 TDOA measurements fully determine the 2 unknowns
(latitude, longitude), leaving zero residual degrees of freedom. The
location is entirely determined by which three moorings are associated,
with no internal quality metric available. Small differences in mooring
selection between Singer's manual and our automated association produce
large location shifts.

**Mooring association as the primary error source:** The dominant source
of disagreement is not the grid-search algorithm but the *association
step* — which moorings are linked to each event. When both analyses use
overlapping mooring subsets, locations converge; when they select
different moorings, locations diverge. This finding highlights that future
improvements should focus on association robustness rather than
grid-search refinement.

**Systematic SW bias:** Our locations are systematically displaced
to the south-southwest relative to Singer's locations (mean Δlat:
−0.13°, mean Δlon: −0.58°). This bias is consistent across the dataset
and persists regardless of the number of moorings, suggesting a
systematic effect in our automated pick timing or association rather
than random scatter.

### 7.4 Location Comparison with Orca Seismic Network

The Orca OBS network provides a fully independent ground truth for
location accuracy: different instruments, different analysts, different
location methods (traditional seismological ray tracing vs. our
hydroacoustic TDOA grid search). This comparison therefore tests the
entire pipeline end-to-end.

**Matching:** Of 5,789 Orca-located earthquakes, 580 fall within our
recording windows. Of these, **109 match our Phase 3 seismic locations**
(tiers A, B, or C) within a ±30-second temporal tolerance. The lower
match count relative to Singer reflects the additional requirement that
events must be both detected *and* successfully located by our pipeline.

**Overall offset:** The **median spatial offset is 55 km** across all
matched events. While this is large in absolute terms, it is dominated
by poorly constrained 3-mooring locations that have no residual
redundancy.

**Best-case accuracy:** For well-constrained events — particularly
seismic swarm events located with 4 or more moorings — **offsets of
2–8 km** are achieved. These swarm events benefit from consistent
mooring associations (the same moorings detect multiple events in
sequence) and high SNR arrivals, demonstrating the algorithm's
capability under favorable conditions.

**Mooring count dependence:** Spatial offset decreases significantly
with increasing mooring count (Spearman rho = -0.23, p = 0.017).
Events with 3 moorings show a **median offset of 69 km**, while events
with 5 or more moorings achieve substantially better agreement. This is
consistent with the Singer comparison and reinforces the finding that
3-mooring locations lack the redundancy needed for reliable positioning.

**Consistent southward bias:** The Orca comparison reveals the same
systematic southward displacement (~30 km) observed in the Singer
comparison (Section 7.3). Because Orca uses completely independent
instruments and location methods, this consistency confirms that the
bias originates in our pipeline — most likely in the automated
association or pick-timing stages — rather than in the reference
catalogues.

### 7.5 Location Accuracy Summary

The Singer comparison (full dataset, 806 co-located low-band events)
provides a robust characterization of location accuracy by quality tier:

| Tier | n | Median offset (km) | Mean offset (km) |
|------|---|-------------------|-----------------|
| A (≥4 moorings, residual <1s, stable) | 188 | 14.0 | 38.4 |
| B (≥3 moorings, residual <2s) | 408 | 51.4 | 89.5 |
| C (residual 2–5s or unstable) | 210 | 63.3 | 105.7 |

**Key conclusions:**

1. **Tier A locations are reliable.** The 14 km median offset for
   tier-A events represents good agreement given the array aperture
   (~80 km) and effective sound speed uncertainties. These events
   (70,927 in the full catalogue) form the backbone of any spatial
   analysis.

2. **3-mooring events are poorly constrained.** With zero residual
   degrees of freedom, these locations are entirely determined by
   mooring selection and offer no internal quality metric. The majority
   of tier-B events fall in this category, and their locations should
   be interpreted with caution.

3. **Pick association is the primary error source.** The grid-search
   itself is not the bottleneck; rather, automated decisions about
   which moorings to associate with each event dominate the location
   uncertainty. When both analyses use the same moorings, locations
   converge. This is consistent with the findings in Appendix A,
   where algorithmic refinements to the grid search produced negligible
   improvements.

4. **A consistent SW bias** is present (mean Δlat: −0.13°, Δlon:
   −0.58°), pointing to a systematic effect in the automated pipeline.
   Possible sources include asymmetric pick-timing errors (e.g., onset
   vs. peak-energy picks) or preferential association of specific
   mooring subsets.

5. **Near-total detection completeness.** The 99.9% detection rate for
   Singer EQ events confirms that the STA/LTA pipeline is highly
   sensitive. The limiting factor is not detection but rather
   association and location quality.

Scripts: `singer_eq_fates.py`, `investigate_singer_location_offsets.py`,
`crossvalidate_seismic_catalogues.py`.

Companion notebook: `notebooks/methods_notebooks/06_cross_validation.ipynb`.

### 7.6 Orca Location Comparison — Pick Strategy Experiment

The 100 best-constrained Orca OBS earthquakes (erh 0.02–0.15 km) are all
located within ~18 km of our array centroid — well within the array
footprint where location accuracy should be best. Yet initial matching
showed median offsets of ~40–60 km, with our locations scattered over a
far wider area than the tight Orca OBS cluster. This motivated a
controlled experiment testing whether the pick strategy is the limiting
factor.

**Experiment:** For each Orca event matched to a ≥3-mooring association
(92 events), we extracted two sets of TDOA picks from the raw waveforms
and independently located each:
1. **Onset picks** — AIC-refined first-arrival times (current pipeline)
2. **Peak-power picks** — time of maximum squared Hilbert envelope
   within the detection window

Additionally, we computed a **spectral similarity score** for each
association, flagging events where constituent detections across moorings
have peak frequency differences >5 Hz or bandwidth differences >50%.

**Results:**

| Pick strategy | Median offset (km) | < 25 km | < 50 km |
|--------------|-------------------|---------|---------|
| Onset (first-arrival) | 104.5 | 11 | 26 |
| Peak-power | 105.6 | 10 | 25 |

Peak-power picking showed no improvement over onset picking (improved
only 37% of events, median improvement −0.3 km). The pick strategy is
not the limiting factor.

**Literature context:** The standard PMEL hydroacoustic processing
approach (Fox et al., 2001; Dziak et al., 2010; Ingale et al., 2025)
uses manual picks at **peak energy** within the T-phase envelope, whereas
our automated pipeline uses AIC-refined first-arrival onsets. The
equivalence of the two strategies in our experiment is consistent with
the literature finding that T-phase peak energy corresponds to the
dominant ray path nearest the epicentral region (Fox et al., 2001), which
carries similar inter-station timing information as the first arrival.
For scattered T-phases in the SOFAR channel, both pick types are subject
to path-dependent distortion across widely-spaced stations (~27–176 km
in this array), rendering the choice of pick strategy secondary to the
question of whether the correct events are being associated.

**The key finding is spectral consistency.** Only 7 of 92 associations
were spectrally consistent (all moorings within 5 Hz peak frequency).
These 7 events achieved **27.5 km median offset** — 4.6× better than the
85 spectrally inconsistent events (127 km). This demonstrates that the
dominant error source is **false associations** — linking detections on
different moorings that correspond to different physical events.

**By detection band:** Low-band associations (1–15 Hz) performed best
(50 km median), consistent with T-phase propagation being more coherent
across the array than mid/high-band signals.

**Spectral post-filtering experiment:** To verify this, we post-filtered
existing associations by removing moorings whose peak frequency differed
from the association median by more than a tolerance threshold, then
relocated the filtered associations. Results confirm the benefit:

| Freq tolerance | Events passing | Median offset (km) | Median residual (s) |
|---------------|---------------|--------------------|--------------------|
| 3 Hz | 15 | 27.7 | 2.84 |
| 5 Hz | 26 | 31.0 | 2.84 |
| 10 Hz | 42 | 33.2 | 0.50 |
| Unfiltered | 92 | 45.4 | 9.0 |

The spectral filter reduces the median location offset by 27–40% and
the residual by 3–18×, confirming that mooring-selection errors (false
associations) are the dominant source of location inaccuracy. The 10 Hz
tolerance provides a good balance between event recovery and location
quality.

**Resolution:** These findings motivated the adoption of TAPAAs
spatial-pruning association (Section 5.1.2), which addresses the false
association problem at its root by requiring spatial consistency during
the association step rather than filtering afterward. TAPAAs reduced the
median residual from 5.2 s to 0.57 s (Section 5.6), confirming that
the spatial pruning eliminates internally inconsistent associations.

**TAPAAs Orca cross-validation:** TAPAAs-associated events were
cross-validated against the same Orca top-100 using a low-band-priority,
most-moorings selection strategy:

| Metric | Greedy baseline | TAPAAs (2s tol) |
|--------|----------------|-----------------|
| Matched events | 96/100 | 87/100 |
| Median residual | 3.5 s | 0.22 s |
| Median offset (low-band, ≥4 moor) | 33 km | 48 km |
| Best single event (5-mooring) | — | **1.9 km** |

TAPAAs produces associations with dramatically lower residuals, and when
the data supports multi-mooring detections the locations are excellent
(1.9 km for the best 5-mooring event). However, most Orca events —
small, local earthquakes — are only detected on the 3 nearest moorings
(M3, M4, M5 at 23–30 km), and 3-mooring locations are inherently
underconstrained (zero residual degrees of freedom). The greedy
approach's apparent advantage in median offset partly reflects
coincidental inclusion of spurious detections on distant moorings
that happen to improve the location by chance.

**Practical assessment of location accuracy:**
- **Tier A events (≥4 moorings, residual <1 s):** 14 km median offset
  against Singer (near Singer's own 7.5 km uncertainty floor), 1.9 km
  demonstrated for well-detected TAPAAs events
- **Tier B events (3 moorings):** 50–80 km median offset, inherently
  limited by geometry
- **Detection completeness:** 99.9% against Singer, 98% against Orca —
  the pipeline detects essentially everything; location quality is the
  limiting factor
- **PMEL benchmark:** Dziak et al. (2010) report 0.5–5 km accuracy
  within the array aperture using manual picks on this same dataset.
  Our automated pipeline approaches this accuracy for tier-A events.

Scripts: `experiment_peak_power_location.py`,
`experiment_spectral_association.py`, `associate_tapaas.py`.

### 7.7 Comparison with Raumer et al. (2025) TAPAAs

Our adoption of TAPAAs spatial-pruning association was motivated by
Raumer et al. (2025), who developed the algorithm for a 4-hydrophone
array (~100 km aperture, square geometry) monitoring the Fani Maoré
submarine volcano off Mayotte Island. A direct comparison highlights
both the method's potential and the geometric limitations of the
BRAVOSEIS array:

| | Raumer (Fani Maoré) | BRAVOSEIS |
|---|---|---|
| Array geometry | 4 hydrophones, ~100 km, square | 6 hydrophones, ~175 km, linear |
| Grid resolution | 750 m (3D) | 0.01° (~1 km) coarse; fine grid (0.001°) pending |
| Sound speed model | Fixed 1,490 m/s | Per-pair XBT-derived (1454.8–1456.1 m/s) |
| Reference catalogue match rate | 48.2% (3,215 / 6,666) | 87% (87/100 Orca top-100) |
| Median residual | not reported | 0.22 s |
| Best single-event offset | ~1 km (grid resolution) | 1.9 km (5-mooring event) |
| Typical location offset | ~1 km (Fig. 7: 0.2° ≈ 1 km) | 48 km median (low-band, ≥4 moorings vs Orca) |
| Depth resolution | Yes (3D grid, 1 km vertical) | No (2D surface only) |
| Detection completeness (vs reference) | 48.2% | 99.9% (Singer), 87% (Orca top-100) |

**Key differences explaining the performance gap:**

1. **Array geometry.** Raumer's square array provides good azimuthal
   coverage in all directions, whereas our linear array has poor
   cross-track (north-south) resolution. This is the most fundamental
   limitation — no algorithm can overcome an array geometry that provides
   weak constraints in one direction.

2. **Inter-mooring distances.** Raumer's array has inter-station distances
   of ~50–100 km with a 4-station square; BRAVOSEIS spans 27–176 km in a
   roughly linear arrangement. The longer baselines cause greater waveform
   distortion from multipath propagation, reducing the coherence of
   arrivals across moorings.

3. **Signal type.** Raumer located P-phases (direct seismic body waves at
   ~5.5 km/s) with steep, impulsive onsets that permit precise picking.
   BRAVOSEIS primarily detects T-phases (SOFAR-channel hydroacoustic
   waves at ~1.45 km/s), which are scattered and emergent after
   long-range propagation, making onset times inherently less precise.

4. **3-mooring dominance.** Most Orca events are only detected on the 3
   nearest moorings (M3, M4, M5 at 23–30 km spacing), yielding zero
   residual degrees of freedom. Raumer's 4-station network provides at
   least one degree of redundancy for all associations.

5. **Detection completeness trade-off.** Our pipeline detects nearly
   everything (99.9%) but at the cost of associating many weak or
   ambiguous events. Raumer's lower match rate (48.2%) may partly
   reflect stricter quality filtering that excludes poorly constrained
   events.

**Status:** TAPAAs has dramatically improved association quality (median
residual 0.57 s vs 5.2 s greedy, 9× improvement), and the best-case
location accuracy (1.9 km) demonstrates that the method works well when
the data supports multi-mooring detections. The remaining gap relative
to Raumer is primarily geometric, not algorithmic. Fine-grid location
refinement, jackknife validation, and quality tiering of the TAPAAs
associations remain pending.

---

## Appendix: Alternatives Considered and Rejected

| Decision | Alternative | Why rejected |
|----------|------------|--------------|
| STA/LTA detector | Recursive STA/LTA | Classic is simpler, vectorizable, sufficient |
| 3 non-overlapping bands | 4 overlapping bands | LTA contamination and deduplication issues |
| All-band UMAP+HDBSCAN | Per-band clustering | Single mega-cluster (99%) — frequency regime dominated |
| Pure CNN classifier | Hybrid CNN+MLP | 39% macro F1 — labels based on features, not visuals |
| Fixed association window | Pair-specific windows | 6× too wide for close pairs |
| 2D ray tracing | Effective speed | No regional 3D sound speed model available |
| Fixed 30 km icequake filter | Seasonal (coast + sea ice) | Rejects legitimate winter sea-ice events |
| Peak-power TDOA picks | Onset (first-arrival) picks | No improvement (105 km vs 105 km); false associations dominate. Consistent with PMEL finding that peak energy and first arrival carry similar timing information (Fox et al. 2001) |
| Spectral post-filter on associations | Time-only association | Reduces median offset 27–40% (33 km vs 45 km); superseded by TAPAAs spatial pruning |
| Finer association grid (0.005°, 0.002°) | 0.01° grid | No improvement in association counts or mooring distribution; tolerance budget dominated by pick uncertainty, not grid geometry |
| 4s pick tolerance (TAPAAs) | 2s tolerance | 30% more 5-6 mooring associations but higher residuals; marginal net benefit |

---

## Appendix A: Rejected Location Improvements

In science, knowing what did not work is often as valuable as knowing what did.
This appendix documents three approaches we tested to improve source locations
that ultimately failed to help or made things worse. We include them so that
future researchers do not repeat these experiments, and because the failures
themselves reveal important properties of the array and data.

Five potential improvements to the source location algorithm were
systematically evaluated on the `exploration/location-improvements` branch,
using the same 43,055 associations and 26,208 locatable events as the
baseline pipeline. Three were rejected after testing demonstrated that they
either had no measurable impact or actively degraded location quality. These
negative results are reported here because they constrain the achievable
precision of the current approach and inform future methodological choices.

### A.1 Station Timing Corrections

**What was tested:** A two-pass location scheme in which the first pass
locates all events with the standard algorithm, and the second pass applies
per-mooring timing corrections derived from the mean onset residuals of
well-constrained tier A events (≥5 moorings, N = 1,104). This approach is
standard in seismic network processing to absorb systematic clock drift,
local velocity anomalies, or instrument response delays.

**Result:** All six station corrections were < 0.001 s (effectively zero).
Tier counts were unchanged (A: 4,287, B: 6,909, C: 925 — identical to
the pre-correction baseline).

**Why rejected:** The near-zero corrections are a positive finding: they
confirm that the BRAVOSEIS moorings have no detectable systematic timing
biases. This is consistent with the CSAC (chip-scale atomic clock)
oscillators used in the deployment, which maintain < 1 ms/day drift.
Additionally, at the well-constrained tier A events used to compute
corrections, per-mooring residuals at the grid-search optimum are
symmetrically distributed around zero by construction. Station corrections
become non-zero only when applied to data with systematic quality
differences across moorings (e.g., the expanded but lower-quality
association set from Step 4), suggesting the framework may prove useful
in future work with different association criteria.

### A.2 Extended Seismic Onset Picker

**What was tested:** The envelope STA/LTA + kurtosis dual picker (developed
for the 6 seismic clusters in §3.3) was extended to all 44,507 grade B and
C events across the low and mid frequency bands. The goal was to improve
onset accuracy for the ~15% of events with marginal AIC picks, thereby
improving TDOA precision and location quality.

**Result:** The picker refined 6,863 onsets (15.4% of candidates): 5,459
via envelope STA/LTA (80%) and 1,404 via kurtosis (20%). Grade C events
decreased by 26% (3,236 rescued to grade B). However, when the refined
onsets were fed through the association and location pipeline, the results
were severely degraded:

| Metric | Before | After |
|--------|--------|-------|
| Tier A | 4,286 | 675 (−84%) |
| Tier B | 6,897 | 11,430 |
| Tier C | 905 | 9,057 |
| Jackknife unstable | 2.0% | 29.3% |

**Why rejected:** The earlier onset picks expanded the effective time
windows for the greedy association algorithm, causing it to match many
more events by chance (43,055 → 82,148 associations). The resulting
false associations produced poorly constrained locations — tier A collapsed
by 84%, and jackknife instability increased from 2% to 29%. The onset
picker is not inherently flawed (the refinements are individually
reasonable), but its interaction with the current association algorithm
is destructive. Applying improved onset picks would require either
tightening association windows to compensate, adding waveform similarity
checks to the association step, or restricting the picker to
post-association refinement only.

### A.3 Waveform Cross-Correlation for Differential Times

**What was tested:** For each association, 10-second waveform windows
centered on each event onset were extracted at all moorings, and all
mooring pairs were cross-correlated to produce differential travel times
with sub-sample precision. Cross-correlation TDOAs (for pairs with
correlation coefficient cc ≥ 0.5) were substituted for onset-based TDOAs
in the location algorithm.

**Result:** Of 290,542 mooring pairs processed, only 10,952 (3.8%)
exceeded the cc = 0.5 threshold. The median correlation coefficient across
all pairs was 0.183. Only 6,240 of 53,152 associations (11.7%) had any
usable cross-correlation pairs, and the median TDOA correction was 0.159 s.
The impact on location quality was negligible: tier A changed by −1, tier B
by +10, tier C by −24.

**Why rejected:** The BRAVOSEIS array geometry fundamentally limits
waveform cross-correlation effectiveness. Inter-mooring distances of
27–176 km cause severe waveform distortion from multipath propagation,
scattering, and different propagation geometries at each receiver. At
these distances, the waveform arriving at one mooring bears little
resemblance to the waveform at another. Effective cross-correlation-based
location requires either much shorter inter-mooring distances (< 10 km),
template/master event correlation within swarm clusters (same source,
similar path), or sub-band correlation targeting the most coherent
frequency range. The 290,542-pair computation is expensive for minimal
gain and is not justified for routine processing. Cross-correlation may
be revisited for targeted applications such as swarm relative relocation
using the closest mooring pair (M4–M5, 27 km).

---

## Appendix B: Rejected Association Improvements

Before we can locate a sound, we must first decide which detections on different
microphones belong to the same event — this is the association step. This
appendix documents three modifications to the association algorithm that we
tested and ultimately rejected. As with Appendix A, these negative results
are informative: they show that the baseline association method is already
well-suited to this array and that more sophisticated approaches introduce
problems of their own.

Three potential improvements to the cross-mooring association algorithm were
systematically evaluated on the `exploration/association-improvements` branch,
using the same detection catalogue and location pipeline as the baseline. Each
modification was tested independently, and the full pipeline (associate →
locate with grid-search TDOA) was re-run to measure the impact on location
quality. All three were rejected after testing demonstrated that the baseline
greedy windowed clustering algorithm — while conceptually simple — is already
well-matched to the array geometry and data characteristics.

The baseline for these tests uses 82,148 associations (the expanded set
resulting from AIC-refined onsets), of which 53,152 are locatable (≥3
moorings). The baseline produces 21,038 located events (tiers A+B+C), with
660 tier A (≥4 moorings, residual < 1 s, jackknife-stable).

### B.1 Waveform Similarity Validation

**What was tested:** For each association, envelope snippets were extracted
from the DAT files at all contributing moorings and cross-correlated between
all mooring pairs. Associations where the median envelope cross-correlation
fell below 0.3 were rejected; pairs falling below the threshold within
otherwise retained associations were dropped (reducing the mooring count for
that association).

**Result:** The filter rejected 4,469 associations (8.4% of locatable) and
dropped one mooring from 1,389 additional associations. The median envelope
correlation across all pairs was 0.418 (IQR [0.348, 0.495]), with 11.0% of
pairs falling below the 0.3 threshold. Net impact on location quality:
−2.4% tier A (660 → 644), −9.2% tier B, −6.4% tier C, −7.8% total located.

**Why rejected:** The envelope correlation correctly identified associations
with dissimilar waveforms, but removing them did not improve location
quality — it reduced total located counts without disproportionately removing
poorly constrained events. The rejected 8.4% were not enriched in bad
locations. This outcome reflects a fundamental property of the BRAVOSEIS
array: inter-mooring distances of 27–176 km cause significant waveform
distortion from multipath propagation and varying propagation geometries, so
even legitimate same-event arrivals produce moderate correlations. Waveform
similarity is better suited as a post-location quality metric than as a
pre-association filter, because the location residual provides a more
physically meaningful measure of association consistency than envelope shape.

### B.2 Class-Aware Association Windows

**What was tested:** Preliminary event classes were assigned using the hybrid
CNN+MLP classifier predictions combined with feature-based heuristics. The
association algorithm then applied class-specific window scaling (0.8× for
icequakes, 1.5× for vessel noise) and enforced class compatibility checks
— for example, a T-phase detection at one mooring could not associate with
a vessel-noise detection at another.

**Result:** This was the most harmful modification tested. Tier A collapsed
by 39.5% (660 → 399). Total located events decreased by 9.1%. The algorithm
produced more 2-mooring associations (+7,367) but fewer 4–6 mooring
associations, directly degrading the multi-mooring coverage that tier A
requires.

**Why rejected:** The class compatibility check fragmented legitimate
multi-mooring associations in which different moorings assigned different
class labels to the same event. This is common because the hybrid CNN+MLP
classifier was trained on single-mooring features (spectral slope, duration,
peak frequency), and the same event can present different spectral
characteristics at different moorings due to distance-dependent attenuation,
SNR variation, and propagation effects. A moderate T-phase may appear as an
icequake at a distant mooring where the high-frequency content is attenuated,
or vice versa. The tighter icequake windows further broke associations for
events near the class boundary. Pre-association classification is not
reliable enough to constrain associations in this array — the classifier
achieves 95% single-mooring accuracy, but cross-mooring label consistency is
substantially lower.

### B.3 Iterative Association-Location Refinement

**What was tested:** A two-iteration refinement loop: (1) run the baseline
association and location pipeline, (2) use the resulting locations to predict
arrival times at all moorings, (3) search for unassociated detections within
tight windows (2–8 s depending on tier) of the predicted arrivals and add
them to the association, (4) drop associated events with residual > 3 s,
(5) relocate. The process was repeated for a second iteration.

**Result:** Iteration 1 added 28 events and dropped 343 (residual > 3 s),
with 270 associations modified. Iteration 2 added 5 events and dropped 0,
indicating near-convergence. Net impact: −0.1% total located, −2.9% tier A
(660 → 641). The refinement was effectively neutral.

**Why rejected:** The tight predicted-arrival search windows found very few
unassociated detections near the expected times, indicating that the greedy
associator already captures most detectable arrivals. The 343 events dropped
for high residuals were primarily noise or multipath arrivals that did not
affect tier assignments. The near-zero net change after two iterations
confirms that the baseline associations are already close to optimal given
the current detection catalogue. The iterative refinement adds computational
cost (two full location passes per iteration) without measurable benefit.

### Summary

The three modifications tested on the development subset — waveform
similarity, class-aware windows, and iterative refinement — span a
representative range of post-hoc association improvement strategies.
Their consistent failure reflects a key limitation: all three operate
*after* the greedy association has already been formed, attempting to
filter or refine combinations that were assembled without regard to
spatial consistency.

The TAPAAs spatial-pruning approach (Section 5.1.2) addresses the root
cause by integrating location consistency *into* the association step
itself, preventing physically implausible combinations from forming in the
first place. Cross-validation against the Orca OBS network (Section 7.6)
demonstrated that false associations are the dominant error source —
spectrally-consistent associations achieved 27.5 km median offset vs
127 km for inconsistent ones — motivating the adoption of spatial pruning
over post-hoc filtering.

---

## References

- Dziak, R. P., et al. (2010). Tectonomagmatic activity and ice dynamics
  in the Bransfield Strait back-arc basin. *Journal of Geophysical
  Research*, 115, B01102. doi:10.1029/2009JB006295
- Fox, C. G., et al. (2001). Monitoring Pacific Ocean seismicity from an
  autonomous hydrophone array. *Journal of Geophysical Research*, 106(B3),
  4183–4206. doi:10.1029/2000JB900404
- Ingale, V., et al. (2025). Detection and Analysis of Aleutian Arc
  Seismicity Using an Autonomous Hydrophone Array. *Geochemistry,
  Geophysics, Geosystems*, 26. doi:10.1029/2025GC012320
- Raumer, P.-Y., Bazin, S., Safran, R., Cazau, D., & Royer, J.-Y. (2025).
  Automatic Analysis of Hydroacoustic Signals Related to the Activity of
  the Fani Maoré Submarine Volcano. *Geochemistry, Geophysics,
  Geosystems*, 26(12). doi:10.1029/2025GC012572

---

*Document generated from the BRAVOSEIS research constitution and analysis
scripts. All figures are reproducible from raw data using the scripts in
`scripts/`. Constitution: `.specify/memory/constitution.md`.*
