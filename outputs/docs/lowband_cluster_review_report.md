# Lowband (1–14 Hz) Cluster Review Report

**BRAVOSEIS Phase 3 Frequency-Band Reclassification**
Date: 2026-03-15
Reviewers: D. Soule, Claude (AI assistant)

## Overview

The lowband pipeline isolates events with energy in the 1–14 Hz band using a 4th-order
Butterworth bandpass filter applied before feature extraction. A whale contamination filter
removes events with catalogue peak frequency > 14 Hz, which eliminated 32,495 events
(38.4%) — predominantly fin whale 20 Hz calls with sub-14 Hz spectral leakage.

After filtering: **52,175 events** clustered into 8 clusters + noise via UMAP + HDBSCAN
(min_cluster_size=200, silhouette=0.083).

| Cluster | Events | Median freq | Verdict | Signal type |
|---------|--------|-------------|---------|-------------|
| lowband_2 | 312 | 3.4 Hz | Consult | Tonal 2–5 Hz, unknown |
| lowband_5 | 488 | 3.4 Hz | Discard | Deployment noise |
| lowband_1 | 643 | 2.0 Hz | Defer | Low-SNR seismic? |
| lowband_7 | 645 | 11.7 Hz | **Accept** | **T-phase** |
| lowband_4 | 3,076 | 2.0 Hz | Discard | Noise / false triggers |
| lowband_3 | 3,170 | 2.4 Hz | Discard | Ambient noise / mixed triggers |
| lowband_0 | 5,195 | 9.3 Hz | **Accept** | **T-phase / seismic** |
| lowband_6 | 32,368 | 9.8 Hz | Partial | SNR>=6 accepted (8,565); rest deferred |
| lowband_noise | 6,278 | 5.4 Hz | — | HDBSCAN noise |

---

## Reviewed Clusters

### lowband_2 (312 events, median 3.4 Hz) — CONSULT

**Notebook:** `notebooks/review_lowband_2.ipynb`

**Description:** Highly distinctive quasi-monochromatic waveforms at 2–5 Hz. Very regular
sinusoidal oscillations (not impulsive), duration 5–11 s, steep negative spectral slope
(-1.7 to -4.1). Spectrograms show persistent horizontal energy bands at 2–4 Hz with
harmonics at ~5 Hz and ~7 Hz. Dates mostly Oct 2019 – Feb 2020 (austral spring/summer).
Distributed across M1, M2, M3, M5.

**Assessment:** NOT seismic events. The tonal, long-duration, sinusoidal character is
inconsistent with earthquakes or T-phases.

**Hypotheses:**
1. Ocean microseism / infragravity wave signals
2. Ice-related tremor (ice shelf or sea ice processes)
3. Volcanic harmonic tremor
4. Biological source not captured by the 14 Hz whale filter

**Action:** PDF prepared for Bob Dziak review
(`outputs/docs/lowband2_tonal_signals_for_dziak.pdf`). Consultation item #1.

**Verdict:** Pending expert identification. Exclude from seismic catalogue.

---

### lowband_5 (488 events, median 3.4 Hz) — DISCARD

**Notebook:** `notebooks/review_lowband_5.ipynb`

**Description:** Short (1.2–4.1 s), impulsive, low-frequency (2–4 Hz) events. Low SNR
(3.0–5.8). One high-amplitude outlier (panel 5, SNR=22.9). Heavily clustered: 312/488
events on Jan 12–13, 2019, concentrated on M3 and M6.

**Cruise report cross-reference:** Jan 12–13 is the mooring deployment period — BRA28,
BRA29 deployed Jan 12; BRA30, BRA31, BRA33 deployed Jan 13. Events start at 17:33 UTC
Jan 12, coinciding with deployment operations. The seismic survey did not begin until
Jan 21. M3 and M6 dominance matches the moorings being deployed on those days.

**Assessment:** Ship noise artifacts from R/V Sarmiento de Gamboa during mooring deployment
operations (dynamic positioning thrusters, winch, station-keeping). Even if some are real
seismic events, safer to exclude than risk contaminating the catalogue with deployment
noise.

**Verdict:** Discard. Deployment noise artifacts.

---

### lowband_1 (643 events, median 2.0 Hz) — DEFER

**Notebook:** `notebooks/review_lowband_1.ipynb`

**Description:** Low-SNR (3.0–6.4, mostly 3–5), low-frequency (1.5–4.4 Hz) impulsive
events with short coda. Low amplitude, barely above noise in many panels. Bimodal
frequency: some at ~1.5–2.4 Hz, others at ~4.4 Hz. Spectrograms show transient broadband
bursts below 5 Hz. Dates spread across full deployment with peaks Jun–Nov 2019 and
Jan 2020. Well-distributed across all 6 moorings.

**Cruise report cross-reference:** Only 38 events during deployment period (Jan 10–20),
none during shooting. No survey artifact concern.

**Assessment:** Possibly small local earthquakes or distant tectonic events, but low SNR
makes them marginal detections. Not representative of the primary T-phase population
being targeted.

**Verdict:** Defer. Revisit after reviewing the larger, higher-frequency clusters.

---

### lowband_7 (645 events, median 11.7 Hz) — ACCEPT (T-phase)

**Notebook:** `notebooks/review_lowband_7.ipynb`

**Description:** Classic T-phase morphology with emergent onset, spindle-shaped envelope,
and gradual decay. Peak frequency 9.3–13.7 Hz, duration 3.5–7.1 s, good SNR (median
11.4, range 4.3–22.4). Positive spectral slope (+1.1 to +2.9) — energy increases with
frequency. Broadband energy concentrated at 8–14 Hz in spectrograms.

**Temporal pattern:** Seasonal peak Jul–Aug 2019 (austral winter). Only 2 events during
the cruise period. Notable seismic swarm on Aug 21, 2019: 50 T-phases in 4 hours.

**Mooring distribution:** M4 (190), M6 (166), M5 (158) dominant — southern/western
array, consistent with Bransfield Rift axis source region.

**Fin whale check:** NEGATIVE.
- No regular inter-pulse interval (only 3 intervals in 10–20s range out of hundreds)
- Multi-mooring arrivals with seismic propagation delays (0.5–11 s between moorings)
- Swarm behavior with irregular spacing — opposite of metronomic whale calling
- Austral winter peak inconsistent with fin whale calling season

**Notable events:**
- Aug 21, 2019 swarm: 50 events in 4 hours, M4-dominated, multi-mooring arrivals
  confirming seismic origin. Intervals range 30s to 12 min (no periodicity).

**Verdict:** Accept as T-phase. Clean cluster with textbook morphology.

---

### lowband_4 (3,076 events, median 2.0 Hz) — DISCARD

**Notebook:** `notebooks/review_lowband_4.ipynb`

**Description:** Very short (median 1.9 s), very low SNR (3.0–4.5, all near detection
threshold), low-frequency (1.5–5.4 Hz) events with no coherent waveform morphology.
Spectrograms show brief transient blips — single time-frequency cells with no structure.
Low amplitude waveforms barely distinguishable from background noise.

**Mooring distribution:** M5 massively dominant (1,259/3,076 = 41%), suggesting a
mooring-specific noise source or local ambient noise sensitivity.

**Temporal pattern:** Peak Jul 2019, Nov 2019, Jan 2020. 122 events during deployment
period, zero during shooting — not airgun artifacts.

**Assessment:** STA/LTA false triggers on ambient noise fluctuations. The very short
duration, threshold-level SNR, absence of coherent waveform morphology, and strong M5
bias all indicate these are not geologic signals.

**Verdict:** Discard. Detector noise floor in this frequency band.

---

### lowband_3 (3,170 events, median 2.4 Hz) — DISCARD

**Notebook:** `notebooks/review_lowband_3.ipynb`

**Description:** Low-frequency (1.5–4.9 Hz), short-duration (median 2.2 s), low-SNR
(3.0–4.3) events with steep negative spectral slope (-1.9 to -4.2). Contains two
sub-populations: tonal sinusoidal oscillations resembling weaker versions of lowband_2,
and impulsive blips resembling lowband_4 noise triggers. A minority of panels (7, 8, 14,
15) show higher-amplitude broadband events that could be real seismic signals.

**Mooring distribution:** M1 (1,371) and M3 (1,152) massively dominant (80% of events).
Virtually no events on M4 (9). This strong mooring bias suggests site-specific ambient
noise rather than a distributed geologic source.

**Temporal pattern:** Peak Nov–Dec 2019 (austral late spring). 139 events during
deployment period, zero during shooting — not airgun artifacts.

**Assessment:** Grab-bag cluster capturing ambient noise features specific to M1 and M3.
Mix of tonal noise (ocean swell, ice, or flow noise on the hydrophone) and STA/LTA false
triggers. Low SNR, inconsistent morphology, and strong mooring bias make these unreliable
as seismic events.

**Verdict:** Discard. Ambient noise / mixed false triggers.

---

### lowband_0 (5,195 events, median 9.3 Hz) — ACCEPT (T-phase / seismic)

**Notebook:** `notebooks/review_lowband_0.ipynb`

**Description:** T-phase-range events (7.8–12.2 Hz) with longer duration than lowband_7
(median 6.0 s vs 4.4 s) and broader SNR range (3.5–21.4, median 8.7). Positive spectral
slope. Many panels show classic T-phase morphology: emergent onset, broadband 5–14 Hz
energy, spindle envelope. Some weaker events mixed in.

**Mooring distribution:** Well-distributed across all 6 moorings, M6 slightly dominant.

**Temporal pattern:** Big Feb 2019 spike (983 events), then spread across deployment.
463 events during deployment period, zero during shooting.

**Tracing:** 417 events from old low_0/low_1 clusters (where user identified clear
earthquakes) landed in this cluster. Earthquake population split between here and
mega-cluster lowband_6.

**Verdict:** Accept as T-phase / seismic. Second-tier confidence after lowband_7. Good
source of additional training examples (high-SNR events at panels 9, 10, 12).

---

### lowband_6 (32,368 events, median 9.8 Hz) — PARTIAL ACCEPT

**Notebook:** `notebooks/review_lowband_6.ipynb`

**Description:** Mega-cluster containing 62% of all lowband events. Very short duration
(median 2.4 s), low SNR (median 4.4, Q75=6.2). Heterogeneous morphology ranging from
clear earthquakes to noise triggers. 2,571 events trace back to old low_0/low_1 clusters
where user identified clear earthquakes.

**Full cluster review (15 panels):** Mix of marginal noise triggers (panels 2–8, 11–14)
and clear seismic events (panels 9, 10, 15). Too heterogeneous for blanket verdict.

**High-SNR subset (SNR >= 6): 8,565 events (26.5% of cluster)**
- Peak freq: 5.4–13.7 Hz, concentrated 7–14 Hz
- Duration: median 3.1 s (vs 2.4 s for full cluster)
- SNR: 6.2–12.8 — all clearly above noise
- Every panel shows a real seismic signal
- Mix of T-phases (emergent onset) and local earthquakes (impulsive onset)
- Well-distributed across all moorings
- 8% trace to old earthquake-containing clusters

**Low-SNR remainder (SNR < 6): ~23,800 events**
- Mix of weak real events and noise triggers
- Cannot reliably classify at this stage
- **Note:** Many of these may be locatable as secondary arrivals. An event with
  high SNR on one mooring (e.g., SNR 15 on M4) may have low SNR on more distant
  moorings (e.g., SNR 3 on M1/M2). The per-detection SNR >= 6 filter is a
  classification threshold, not a location threshold. After locating the
  high-confidence events, revisit these as candidate secondary picks for
  multi-mooring association.

**Temporal pattern:** Feb–Mar 2019 spike, Sep–Oct 2019 peak. 1,030 events during
deployment, zero during shooting.

**Verdict:** Accept SNR >= 6 subset (8,565 events) as seismic events (T-phases + local
earthquakes). Defer low-SNR remainder.

---

## Review Complete — Summary

**Accepted (seismic catalogue):**
- lowband_7: 645 events — highest-confidence T-phases (synthetic kernel source)
- lowband_0: 5,195 events — T-phase / seismic, second-tier confidence
- lowband_6 (SNR>=6): 8,565 events — T-phases + local earthquakes
- **Total accepted: 14,405 events**

**Discarded:**
- lowband_5: 488 events — deployment noise
- lowband_4: 3,076 events — STA/LTA false triggers
- lowband_3: 3,170 events — ambient noise / mixed triggers
- **Total discarded: 6,734 events**

**Deferred / consultation:**
- lowband_2: 312 events — tonal 2–5 Hz, pending Bob Dziak ID
- lowband_1: 643 events — low-SNR seismic, revisit later
- lowband_6 (SNR<6): ~23,800 events — weak, mixed quality
- lowband_noise: 6,278 events — HDBSCAN noise
- **Total deferred: ~31,033 events**

## Methods Notes

- **Whale filter:** Removed events with catalogue peak_freq > 14 Hz before clustering.
  This eliminated 32,495 events (38.4%), overwhelmingly fin whale 20 Hz calls with
  sub-14 Hz spectral leakage.
- **Clustering:** UMAP (n_neighbors=15, min_dist=0.01) + HDBSCAN (min_cluster_size=200,
  EOM selection). Silhouette score 0.083. 6,278 noise points (12.0%).
- **Panel display fix:** Spectrograms now computed on bandpass-filtered signal (not raw).
  Peak frequency in headers now from lowband features (not unfiltered catalogue).
- **Cruise report cross-reference:** All clusters checked against R/V Sarmiento de Gamboa
  operations schedule (mooring deployment Jan 10–13, MCS1 Jan 21–26, tomography Jan
  26–29, MCS2 Jan 30 – Feb 4).

## Source Configurations (from cruise report)

| Config | Volume | Depth | Dates | Survey |
|--------|--------|-------|-------|--------|
| MCS 1 | 1780 ci | 5 m | Jan 21–26 | Orca Volcano reflection |
| Tomo | 2540 ci | 15 m | Jan 26–29 | Orca Volcano tomography |
| MCS 2 | 1580 ci | 5 m | Jan 30 – Feb 4 | Rift + Edifice A reflection |
