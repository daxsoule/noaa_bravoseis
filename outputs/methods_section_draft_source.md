# Methods — BRAVOSEIS Hydroacoustic Event Detection, Classification, and Location

*Draft for internal review — intended for maximum transparency so that every
parameter choice, alternative considered, and quality control step can be
independently verified.*

---

## 1. Overall Objectives

The analysis pipeline processes ~2,870 hours of continuous hydroacoustic data
from 6 autonomous moorings (M1–M6) deployed in the Bransfield Strait during
2019–2020 to:

1. **Detect** transient acoustic events above the ambient noise floor across
   three frequency regimes (1–15, 15–30, 30–250 Hz)
2. **Refine onset times** for cross-mooring association and source location
3. **Classify** events into physical source categories: earthquake T-phases,
   icequakes (glacial/cryogenic), vessel noise, and unresolved
4. **Locate** classified events using time-difference-of-arrival (TDOA) grid
   search
5. **Quality-control** locations using jackknife validation, coast-distance
   filtering, satellite sea ice data, and swarm coherence analysis

The pipeline is fully automated and reproducible from raw `.DAT` files to
final catalogues and figures. All scripts, parameters, and intermediate
outputs are version-controlled.

---

## 2. Event Detection

### 2.1 Instrument and Data Format

Each mooring records at 1 kHz (24-bit ADC, stored as 16-bit) in 4-hour
segments (14,400,000 samples per file). The duty cycle is ~8 hours on /
~40 hours off, yielding ~5% temporal coverage per mooring over 13 months
(717 files total across 6 moorings). See §Data Sources in the constitution
for complete instrument specifications.

> **Figure: Recording Timeline** (`recording_timeline.png`)
>
> **Temporary Caption:** Recording timeline for the six BRAVOSEIS hydrophone moorings
> (M1/BRA28 through M6/BRA33) from January 2019 to February 2020. Each
> horizontal bar represents one 4-hour DAT file. Bars appear in pairs
> (two consecutive files per recording window) separated by ~40-hour
> off-duty gaps. The recording start time drifts slowly across the
> deployment — it is not locked to a fixed UTC hour. M3 (BRA30) has the
> fewest files (104); M5 (BRA32) the most (125). Total: 717 files.

### 2.2 STA/LTA Detection

Events are detected using a classic **Short-Term Average / Long-Term Average
(STA/LTA) energy ratio** detector applied independently to each mooring and
each frequency band.

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

**Implementation:** Custom vectorized `classic_sta_lta()` using numpy
cumulative sums (no ObsPy dependency). Script: `scripts/detect_events.py`.

### 2.3 Three-Pass Detection Strategy

**Problem:** An initial approach using 4 overlapping frequency bands
(1–50, 10–200, 50–250, 1–250 Hz) with cross-band deduplication produced
only 152,040 events. Two issues were identified:

1. **LTA contamination:** Low-frequency energy (<15 Hz, primarily T-phases
   and earthquake coda) dominated the LTA window in all bands, suppressing
   STA/LTA ratios for concurrent higher-frequency transients.
2. **Deduplication masking:** The deduplication step relabeled mid/high-band
   events under the highest-SNR band (almost always low), destroying band
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
- **30 Hz:** Separates the fin whale / low icequake regime from higher
  biological and cryogenic signals.
- These are **energy-regime boundaries**, not signal-type boundaries. A single
  broadband icequake may trigger in all three passes.

**Alternative considered:** Per-mooring adaptive LTA based on running noise
estimates. Rejected because the non-overlapping band approach is simpler,
deterministic, and recovered 7× more mid-band events (132,494 vs 17,781).

**Results:**

| Band | Events |
|------|--------|
| Low (1–15 Hz) | 84,698 |
| Mid (15–30 Hz) | 132,494 |
| High (30–250 Hz) | 79,978 |
| **Total** | **297,170** |

> **Figure: Detection Rate Timeline** (`detection_rate_timeline.png`)
>
> **Temporary Caption:** Daily event counts across all 6 BRAVOSEIS hydrophone moorings,
> stacked by detection frequency band (orange: 1–15 Hz, blue: 15–30 Hz,
> green: 30–250 Hz). STA/LTA detector parameters: STA=2 s, LTA=60 s,
> trigger=3.0, detrigger=1.5. Gaps correspond to the ~40-hour off-duty
> periods in the recording duty cycle. Two prominent T-phase swarms are
> visible: Feb 11, 2019 (~5,000 events) and Apr 22–24, 2019 (~3,500 events
> on the peak day). Vessel traffic passages appear as broadband bursts
> lasting 1–4 days.

> **Figure: Duration vs. Peak Frequency** (`duration_vs_peak_freq.png`)
>
> **Temporary Caption:** Scatter plot of event duration versus peak frequency for all
> 297,170 detected events, colored by detection band. Duration is on a
> logarithmic scale. Events cluster by band as expected. Two distinct
> populations are visible in the low band: short-duration (<3 s) events
> (T-phases) and longer-duration (>3 s) events (icequakes and coda).

> **Figure: Example Cross-Mooring Detection**
> (`example_detection_20190417_0919.png`)
>
> **Temporary Caption:** Two-minute spectrogram array centered on association A007517,
> detected on all 6 moorings. Each panel shows one mooring (M1 at top,
> M6 at bottom). Red shading marks the detected event window; red vertical
> lines mark the refined onset time. A broadband T-phase arrival is visible
> propagating across the array with moveout consistent with a source in the
> central Bransfield Strait. Spectrogram parameters: nperseg=1024, 50%
> overlap, 0–250 Hz. Shared colorscale (2nd–98th percentile).

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

---

## 3. Onset Refinement

### 3.1 Problem

STA/LTA onset picks are biased late — validation on 50 manually reviewed
events showed only 11% hit the true first arrival; 68% fell in the event
coda. Accurate onsets are critical for TDOA source location, where a 1-second
onset error at 1455 m/s produces ~1.5 km location error.

### 3.2 AIC Picker with Kurtosis Fallback

For each event, a 7 s window is extracted (5 s pre-trigger + 2 s
post-trigger), the same band-specific filter used for detection is applied,
and the **Akaike Information Criterion (AIC)** is computed on the squared
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

**Results (297,170 events):**

| Method | Events | % |
|--------|--------|---|
| AIC | 284,425 | 95.7% |
| Kurtosis | 686 | 0.2% |
| Original (kept) | 12,059 | 4.1% |

Median onset shift: **−0.83 s** (IQR: [−1.46, −0.39] s). Shifts are
consistently negative (earlier), confirming the picker moves onsets backward
from the coda toward the true first arrival.

**Alternative considered:** The AR-AIC picker (autoregressive) was
considered but rejected because it assumes stationarity within the noise
window, which fails for the variable ambient noise in the Bransfield Strait.
The envelope-based AIC is more robust to non-stationary noise.

> **Figure: Onset Refinement Montage** (`onset_refinement_montage.png`)
>
> **Temporary Caption:** Onset refinement quality control montage showing 50 events,
> oversampled toward grade C (low quality) for critical review. Each cell
> shows waveform (top) and spectrogram (bottom). Yellow vertical line: raw
> STA/LTA onset. Red vertical line: AIC-refined onset. Green vertical line:
> kurtosis-refined onset (if applicable). Of 50 reviewed events, 34 had
> acceptable refined onsets. Most issues were late picks on emergent
> low-frequency signals — the grade system reliably flags these.

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

### 4.1 Phase 1 — Unsupervised Discovery

**Approach:** Extract ~20 handcrafted spectral features per event (band
powers, duration, rise/decay time, peak frequency, bandwidth, spectral
slope, frequency modulation). Cluster each detection band independently
using UMAP projection into 2D followed by HDBSCAN density-based clustering.
Clusters are visually inspected via spectrogram montages and labeled by a
single reviewer.

**Why per-band clustering?** An initial all-band clustering pass (297,170
events, 19 features) produced a single mega-cluster containing 99% of
events. Visual inspection revealed clear internal structure organized
primarily by detection band — the dominant axis of variation was frequency
regime, not signal type. Clustering per band removes this known axis and
allows subtler within-band structure to emerge.

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

**Validation:** T-phase classification confirmed by Bob Dziak (NOAA/PMEL) via
spectrogram montage review. Cross-validated against 636 Orca OBS earthquakes
with hydrophone coverage: 89% detection rate, 43% T-phase match, 28.2 s
median arrival delay consistent with ~40 km propagation at ~1.45 km/s.

> **Figure: Cluster Montage — T-phases** (`cluster_montage_low_0.png`,
> `cluster_montage_low_1.png`, `cluster_montage_mid_0.png`)
>
> **Temporary Caption:** Spectrogram montages of the 20 events nearest the UMAP
> cluster centroid for each T-phase cluster. Cluster low_0 (4,686 events):
> emergent T-phases in the 1–15 Hz band with gradual onset ramps. Cluster
> low_1 (3,666 events): mixed seismic events with impulsive onsets.
> Cluster mid_0 (1,056 events): T-phases detected in the 15–30 Hz band.
> These three clusters were confirmed as earthquake T-phases by expert
> review.

> **Figure: Cluster Montage — Vessel Noise** (`type_a_montage.png`)
>
> **Temporary Caption:** Top 20 Type A broadband transients by SNR. Spectrograms show
> characteristic positive spectral slope with peak energy above 100 Hz,
> broad bandwidth (~211 Hz), and high frequency modulation — the acoustic
> signature of propeller cavitation and ship machinery noise. Identified
> as vessel traffic based on temporal burst pattern (~24 passages over
> 13 months), multi-mooring simultaneity (47% of time bins have 2+ moorings
> detecting), and seasonal correlation with krill fishing fleet activity
> (peak May–Sep).

### 4.2 Phase 2 — Supervised CNN+MLP

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

**Bulk population results (207,528 events):**

| Class | Phase 1 | Phase 2 (high-conf) | Combined |
|-------|---------|---------------------|----------|
| T-phase | 55,783 | 117,077 | **172,860** |
| Icequake | 23,331 | 44,989 | **68,320** |
| Vessel | 10,458 | 29,280 | **39,738** |
| Low-confidence | — | 16,182 | 16,182 |

> **Figure: CNN Confusion Matrix** (`cnn_confusion_matrix.png`)
>
> **Temporary Caption:** Confusion matrix for the hybrid CNN+MLP classifier on the
> held-out test set (13,432 events). Overall accuracy: 95.1%, macro F1:
> 93.7%. Main confusion: 400 T-phases predicted as vessel (4.8% of
> T-phase support) and 249 T-phases predicted as icequake (3.0%). Icequake
> and vessel classes have near-perfect recall (≥1.00).

> **Figure: T-phase Event Montage** (`paper/event_montage_tphase.png`)
>
> **Temporary Caption:** Four representative T-phase events detected by the
> BRAVOSEIS hydrophone array, selected across a range of SNR values and from
> different moorings to demonstrate variability. Each column shows one event
> with bandpass-filtered waveform (top, 1–15 Hz) and spectrogram (bottom,
> 0–100 Hz). Red vertical lines mark the AIC-refined onset time; colored
> shading indicates the detected event duration. T-phases are identified by
> impulsive onsets with dominant energy below 15 Hz and duration typically
> ≤3 s — the hydroacoustic signature of regional earthquakes propagating
> through the SOFAR channel. AIC onset picks are computed on the squared
> envelope within a 7 s window (5 s pre-trigger + 2 s post-trigger). Grade A
> picks (quality ≥0.7) exhibit a sharp AIC minimum at the noise-to-signal
> transition. Spectrogram: nperseg=256, 87.5% overlap, Hann window.

> **Figure: Icequake Event Montage** (`paper/event_montage_icequake.png`)
>
> **Temporary Caption:** Four representative icequake events detected by the
> BRAVOSEIS hydrophone array. Each column shows one event with bandpass-filtered
> waveform (top, 5–30 Hz) and spectrogram (bottom, 0–100 Hz). Red vertical
> lines mark the AIC-refined onset time. Icequakes are distinguished from
> T-phases by their longer duration (>3 s), more emergent character, and
> moderate spectral slope (−0.2 to −0.5). The emergent onset makes AIC picking
> more challenging — icequakes have a higher fraction of grade B picks than
> T-phases. The extended mid-band filter (5–30 Hz) captures the broader
> spectral content characteristic of glacial calving, ice shelf fracture, and
> sea ice cracking events. Spectrogram: nperseg=256, 87.5% overlap, Hann window.

> **Figure: Vessel Noise Event Montage** (`paper/event_montage_vessel.png`)
>
> **Temporary Caption:** Four representative vessel noise events detected by the
> BRAVOSEIS hydrophone array. Each column shows one event with bandpass-filtered
> waveform (top, 30–250 Hz) and spectrogram (bottom, 0–250 Hz). Red vertical
> lines mark the AIC-refined onset time. Vessel noise is identified by its
> positive spectral slope (energy increasing with frequency), peak energy above
> 100 Hz, and broad bandwidth (~211 Hz) — the acoustic signature of propeller
> cavitation and ship machinery. Events appear as Type A broadband transients
> in temporal bursts corresponding to ~24 vessel passages over 13 months (peak
> May–Sep, consistent with krill trawler activity). The high-pass filter
> (30–250 Hz) isolates the vessel signature from concurrent low-frequency
> seismic energy. Spectrogram: nperseg=256, 87.5% overlap, Hann window.

---

## 5. Source Location

### 5.1 Cross-Mooring Association

Events on different moorings are linked using **pair-specific travel time
windows** derived from in-situ XBT sound speed profiles. The effective
horizontal speed is computed as the harmonic mean of the sound speed profile:

    c_eff = z_max / ∫(1/c(z) dz, 0, z_max)

A **15% safety factor** widens each window. Resulting windows range from
21 s (M4–M5, 27 km) to 139 s (M1–M6, 176 km).

**Alternative considered:** A fixed 120 s window at 1480 m/s. Rejected
because it was 6× too wide for close pairs (M4–M5) and slightly too narrow
for the most distant pair (M1–M6). Pair-specific windows reduce false
associations for close pairs while maintaining sensitivity for distant ones.

> **Figure: Sound Speed Profile** (`association/sound_speed_profile.png`)
>
> **Temporary Caption:** Effective horizontal sound speed used for cross-mooring
> association, derived from XBT casts during the BRAVOSEIS deployment.
> The harmonic mean over the water column gives an effective speed of
> 1455.5 m/s. A 15% safety factor is applied to the maximum expected
> travel time for each mooring pair to account for deviations from the
> straight-line path assumption.

### 5.2 Grid-Search TDOA

For each association with ≥3 moorings, a geographic grid is searched for
the point that minimizes the RMS residual between observed and predicted
inter-station travel time differences.

**Grid:** 0.01° spacing (~1 km), covering the study area ±0.5° padding.
154,700 grid points. Geodesic distances precomputed from each grid point
to all 6 moorings (WGS84 ellipsoid, pyproj).

**Effective sound speed:** 1455.5 m/s (XBT-derived harmonic mean).

**Known limitation — flat-ocean approximation:** Travel times assume
straight-line geodesic paths and a single effective speed. This ignores
depth-dependent velocity structure and bathymetric refraction/reflection.
For compact arrays (Wilcock, 2012 used 2D ray tracing for a 15–20 km
network), this would be a significant error. For our 175 km aperture,
the straight-line approximation is more defensible but still introduces
systematic bias near bathymetric barriers (ridges, shallow sills).

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

| Tier | Criteria | Count | Median residual |
|------|----------|-------|-----------------|
| A | ≥4 moorings, residual <1 s, jackknife-stable | 4,304 | 0.00 s |
| B | ≥3 moorings, residual <2 s | 6,979 | 0.00 s |
| C | ≥3 moorings, residual 2–5 s or jackknife-unstable | 926 | 3.29 s |
| D | 2 moorings, >150 km, or residual >5 s | 28,575 | — |

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

---

## 6. Quality Control

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
2. Look up monthly sea ice concentration (NSIDC CDR V4, 25 km, PolarWatch
   ERDDAP) at each event location
3. **Near coast** (≤30 km): retain — glacial/ice shelf source plausible
4. **Far from coast but ice-covered** (>30 km, monthly SIC ≥15%): retain —
   sea ice cracking is physically plausible
5. **Far from coast, ice-free** (>30 km, SIC <15%): reclassify to
   "unclassified"

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
> ice-free conditions. Bathymetry: merged BRAVOSEIS regional + Orca
> multibeam. White triangles: mooring positions.

### 6.2 Swarm Coherence QC

**Principle:** T-phase swarms (temporally clustered events from the same
seismic source region) should be spatially coherent. Events within a swarm
that locate far from the swarm's centroid are likely mislocation artifacts
(inspired by Wilcock 2012's track-constrained relocation for whale calls).

**Method:** Temporally adjacent T-phase events (gap <1 hour) are grouped
into swarms. For each swarm with ≥10 events, the spatial centroid is
computed and events whose distance from the centroid exceeds 3× MAD
(Median Absolute Deviation) are flagged as `swarm_outlier`.

**Results:** 79 swarms identified containing 9,443 T-phase events. **66
spatial outliers** flagged (median 174 km from centroid — clearly
mislocated). These events are retained in the dataset but flagged for
exclusion from spatial analyses.

---

## Appendix: Alternatives Considered and Rejected

| Decision | Alternative | Why rejected |
|----------|------------|--------------|
| STA/LTA detector | Recursive STA/LTA | Classic is simpler, vectorizable, sufficient |
| 4 overlapping bands | 3 non-overlapping bands | LTA contamination and deduplication issues |
| All-band UMAP+HDBSCAN | Per-band clustering | Single mega-cluster (99%) — frequency regime dominated |
| Pure CNN classifier | Hybrid CNN+MLP | 39% macro F1 — labels based on features, not visuals |
| Fixed association window | Pair-specific windows | 6× too wide for close pairs |
| 2D ray tracing | Effective speed | No regional 3D sound speed model available |
| Fixed 30 km icequake filter | Seasonal (coast + sea ice) | Rejects legitimate winter sea-ice events |

---

*Document generated from the BRAVOSEIS research constitution and analysis
scripts. All figures are reproducible from raw data using the scripts in
`scripts/`. Constitution: `.specify/memory/constitution.md`.*
