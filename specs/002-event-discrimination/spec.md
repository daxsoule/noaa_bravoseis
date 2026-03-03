# Analysis Specification: Acoustic Event Discrimination

**Directory**: `specs/002-event-discrimination`
**Created**: 2026-03-02
**Status**: Draft
**Branch**: `002-event-discrimination`
**Depends on**: `001-event-detection` (event catalogue as input)

## Research Question(s)

1. What distinct classes of acoustic signals are present in the BRAVOSEIS
   hydrophone recordings, and how many natural groupings emerge from the
   data?
2. Can detected events be reliably classified into earthquake, ice quake,
   whale call, and noise/unknown categories using spectral features?
3. Do finer subcategories emerge within the broad classes — e.g., distinct
   whale call types, T-phases vs. local volcano-tectonic earthquakes — and
   are they separable by an automated classifier?
4. How do classified event rates vary across the 13-month deployment — are
   there seasonal patterns consistent with known biology (whale migration)
   and cryogenic activity (ice quake seasonality)?

**Hypothesis**: Unsupervised clustering of spectral features will reveal
distinct signal populations that map onto the three expected broad classes
(earthquake, ice quake, whale call) plus noise. Within these, subcategories
will emerge naturally — particularly among whale calls (species-specific
spectral signatures) and earthquakes (T-phases vs. local events). A CNN
trained on labeled clusters will achieve >=80% classification accuracy on
held-out data, enabling a catalogued dataset suitable for downstream source
location and scientific interpretation.

## Data Description

### Primary Data

- **Source**: Event catalogue from spec `001-event-detection`
- **Format**: Parquet table with columns: event_id, mooring, file_number,
  onset_utc, duration_s, end_utc, peak_freq_hz, bandwidth_hz, peak_db,
  snr, detection_band
- **Access**: `outputs/data/event_catalogue.parquet` (produced by spec 001)
- **Known issues**: Catalogue may contain false positives (~20% expected).
  Some physical events may be split across multiple detections or duplicated
  across frequency bands. These issues are handled during labeling (Phase 1).

### Secondary Data

- **Source**: Raw DAT files (NOAA/PMEL hydrophone moorings BRA28–BRA33)
- **Purpose**: Extract spectrogram patches for each detected event
- **Access**: `/home/jovyan/my_data/bravoseis/NOAA/m{1-6}-*/`
- **Reader**: `read_dat.py`

## Methods Overview

This analysis proceeds in two phases: unsupervised discovery to understand
what signal types exist and generate training labels, followed by supervised
classification to build a production-quality classifier.

### Phase 1: Unsupervised Discovery

**Goal**: Discover natural groupings in the event catalogue without
imposing predefined categories.

#### 1a. Feature extraction

For each detected event, extract a spectrogram patch from the raw DAT file:
- Load the relevant DAT file segment (event onset − 2 s to event end + 2 s,
  with padding for context)
- Compute spectrogram: `nperseg=1024`, `noverlap=512`, `fs=1000`
  (matching spec 001 parameters)
- Clip to 0–250 Hz
- Convert to dB: `10 * log10(Sxx + 1e-20)`
- Normalize: subtract median background (from LTA window) to produce
  SNR-relative spectrograms

From each spectrogram patch, extract summary features:
- **Spectral shape**: Mean power in each of 10 frequency bands
  (0–25, 25–50, ..., 225–250 Hz)
- **Temporal shape**: Duration, rise time (onset to peak), decay time
  (peak to end)
- **Peak characteristics**: Peak frequency, peak power (dB), bandwidth
  (frequency range containing 90% of energy)
- **Spectral slope**: Linear fit to log-power vs. log-frequency
- **Modulation**: Presence of frequency modulation (variance of
  instantaneous frequency over time)

This yields a feature vector of ~20 dimensions per event.

#### 1b. Dimensionality reduction and clustering

- Standardize features (zero mean, unit variance)
- Apply UMAP (n_neighbors=15, min_dist=0.1, n_components=2) to project
  into 2D for visualization
- Apply HDBSCAN on the UMAP embedding to identify clusters
  (min_cluster_size to be tuned empirically, starting at 50)
- Produce a 2D scatter plot colored by cluster assignment — this is the
  primary discovery figure

#### 1c. Cluster interpretation and labeling

- For each cluster, generate a montage of 20 representative spectrogram
  patches (sorted by distance to cluster centroid)
- Visually inspect each montage and assign a label:
  - Broad class: earthquake, ice_quake, whale_call, noise, unknown
  - Subclass (if visually distinct within a broad class): e.g.,
    fin_whale_20hz, blue_whale_zcall, tphase, local_vt
- Merge clusters that represent the same signal type
- Split clusters that contain mixed signal types (adjust HDBSCAN parameters
  or manually partition)
- Minimum class size: 100 events. Clusters with fewer than 100 events are
  merged into a parent broad class or "unknown".
- Labeling is performed by a single reviewer. Decision criteria for each
  cluster label are documented in the labeled dataset. A second reviewer
  may be added later if inter-rater reliability is needed for publication.

#### 1d. Labeled dataset

- Output: Parquet table extending the event catalogue with columns:
  cluster_id, broad_class, subclass, labeled_by, confidence
  (high/medium/low based on cluster tightness and visual clarity)
- Also save the extracted feature vectors and UMAP coordinates for all
  events

### Gate: Phase 1 Review

Phase 2 does not begin until Phase 1 deliverables (UMAP plot, cluster
montages, labeled dataset) are reviewed and approved. If clustering
produces messy or uninterpretable results, revisit feature extraction
or spec 001 detector parameters before proceeding.

### Phase 2: Supervised Classification (CNN)

**Goal**: Train a classifier that can label new events automatically,
enabling application to the full catalogue and future deployments.

#### 2a. Training data preparation

- From Phase 1 labeled dataset, extract fixed-size spectrogram patches:
  - Resize/pad all patches to uniform dimensions
    [TODO: Determine patch size — likely 128 x 128 or 64 x 128
    (time x frequency) pixels, depending on typical event durations]
  - Normalize to 0–1 range per patch
- Split into train (70%), validation (15%), test (15%), stratified by class
- Apply data augmentation for underrepresented classes:
  - Time shift (±10% of patch width)
  - Additive Gaussian noise
  - Frequency masking (SpecAugment-style)

#### 2b. Model architecture

- Lightweight CNN appropriate for spectrogram classification:
  - 3–4 convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool)
  - Global average pooling
  - Dense layer → softmax output
  - ~100K–500K parameters (small dataset, avoid overfitting)
- Framework: PyTorch
- [TODO: Exact architecture — start simple, increase complexity only if
  validation accuracy plateaus. Consider transfer learning from a
  pretrained audio model (e.g., PANNs) if training data is limited
  (<1000 examples per class).]

#### 2c. Training

- Loss: Cross-entropy (weighted by inverse class frequency to handle
  class imbalance)
- Optimizer: AdamW, learning rate 1e-3 with cosine annealing
- Early stopping on validation loss (patience=10 epochs)
- Track per-class precision, recall, F1 on validation set each epoch

#### 2d. Evaluation

- Report confusion matrix on held-out test set
- Per-class precision, recall, F1-score
- Overall accuracy and macro-averaged F1
- Target: >=80% macro F1 on 4 broad classes (earthquake, ice_quake,
  whale_call, noise)
- Identify failure modes: which classes are most confused with each other?

#### 2e. Full catalogue classification

- Apply trained model to all events in the spec 001 catalogue
- Save predictions with confidence scores
- Flag low-confidence predictions for manual review

## Expected Outputs

### Figures

- **Figure: UMAP cluster map** — 2D scatter of all events colored by
  cluster/class. The primary discovery figure showing how signals separate.
- **Figure: Cluster montages** — Grid of representative spectrogram patches
  per cluster (20 examples each). Shows what each cluster looks like.
- **Figure: Classified event rate timeline** — Events per recording window
  over the 13-month deployment, stacked by broad class. Reveals seasonal
  patterns.
- **Figure: Confusion matrix** — CNN test-set performance, showing which
  classes are well-separated and which are confused.
- **Figure: Feature importance / UMAP colored by feature** — Which spectral
  features drive the separation (e.g., color UMAP by peak frequency,
  duration, spectral slope).

### Tables/Statistics

- **Table: Class distribution** — Number of events per broad class and
  subclass, with percentage of total catalogue.
- **Table: CNN performance** — Per-class precision, recall, F1, and overall
  accuracy.
- **Table: Seasonal event rates** — Events per recording-hour by broad class,
  binned by month.

### Key Metrics

- Number of natural clusters identified by HDBSCAN
- Number of broad classes and subclasses after labeling
- CNN macro-averaged F1-score on test set (target: >=0.80)
- Fraction of catalogue classified with high confidence (>0.9 softmax)
- Seasonal variation: ratio of peak-to-minimum monthly event rate per class

## Validation Approach

- **Phase 1 validation**:
  - Cluster stability: Rerun UMAP + HDBSCAN with different random seeds
    (n=5). Clusters that appear consistently are robust.
  - Silhouette score on UMAP embedding as a global measure of cluster
    quality.
  - Visual inspection of montages — each cluster should contain visually
    similar spectrograms.

- **Phase 2 validation**:
  - Stratified train/val/test split ensures all classes are represented.
  - Confusion matrix on held-out test set (never seen during training).
  - Compare CNN predictions against Phase 1 cluster labels on the test
    set — they should agree at the broad-class level.
  - Spot-check: visually inspect 20 random events where CNN and cluster
    labels disagree.

- **Cross-mooring consistency**: For events detected on multiple moorings,
  the classifier should assign the same broad class to all detections of the
  same physical event.

- **Literature sanity check**: During Phase 1c labeling, conduct a
  literature search for published whale call descriptions in the Bransfield
  Strait / Antarctic Peninsula region (blue, fin, humpback, minke).
  Compare cluster spectral characteristics against known call types.
  Capture references in the spec as they are found.

## Completion Criteria

- [ ] Feature extraction completed for all events in spec 001 catalogue
- [ ] UMAP + HDBSCAN clustering run with stability analysis
- [ ] Cluster montages generated and all clusters labeled
- [ ] Labeled dataset saved as Parquet
- [ ] CNN trained and evaluated on held-out test set
- [ ] Macro F1 >= 0.80 on broad classes (or documented explanation if not)
- [ ] Full catalogue classified with confidence scores
- [ ] Classified event rate timeline figure produced
- [ ] Confusion matrix figure produced
- [ ] Results reproducible from spec 001 catalogue via script invocation

## Assumptions & Limitations

**Assumptions**:
- The event catalogue from spec 001 contains the majority of real signals
  in the data. Events missed by the STA/LTA detector cannot be classified.
- Spectral features are sufficient to discriminate the target signal types.
  This is well-supported in the ocean acoustics literature — earthquakes,
  ice quakes, and whale calls occupy different regions of time-frequency
  space.
- HDBSCAN will find meaningful clusters. If the feature space is too noisy
  or overlapping, clusters may not be clean — in which case, fall back to
  manual labeling of a random sample and skip directly to supervised
  classification.
- A lightweight CNN with ~100K–500K parameters is sufficient. If the dataset
  is very large (>50K events) a larger model may help; if very small
  (<2K events per class) transfer learning may be needed.

**Limitations**:
- Phase 1 labeling is subjective. Different experts might assign different
  labels to ambiguous clusters. Documenting decision criteria mitigates
  this but does not eliminate it.
- The classifier is trained on this specific deployment. Generalization to
  other deployments (different moorings, locations, seasons) is not tested
  and should not be assumed.
- Without absolute calibration, amplitude-based features are relative to
  each mooring's noise floor. Features should be SNR-based, not absolute.
- Low-SNR events near the detection threshold may be unclassifiable. The
  "unknown" class will catch these.
- Subcategory identification depends on what's actually in the data. If
  only one whale species is present, no whale subcategories will emerge.

## Clarifications

### Session 2026-03-02

- Q: Is a GPU available for CNN training? → A: Yes, available on JupyterHub (select GPU instance at launch).
- Q: Who labels the clusters? → A: Single reviewer. Second reviewer deferred to publication stage if needed.
- Q: Minimum class size to keep as separate category? → A: 100 events. Below that, merge into parent class or "unknown".
- Q: Should Phase 2 be gated on Phase 1 review? → A: Yes. Phase 1 deliverables must be reviewed and approved before CNN training begins.
- Q: Whale call literature references available? → A: Not yet. Plan literature search during Phase 1c labeling.

## Notes

- **Software dependencies**: Phase 1 requires umap-learn, hdbscan, and
  scikit-learn. Phase 2 requires PyTorch. Add to `pyproject.toml`.
- **Compute**: Phase 1 (feature extraction + UMAP + HDBSCAN) runs on CPU
  in under an hour. Phase 2 (CNN training) uses GPU (available on
  JupyterHub — select GPU instance at launch).
- **Reuse**: The `make_spectrogram.py` spectrogram computation can be
  refactored into a shared utility for feature extraction. The array
  spectrogram layout can be reused for example detection figures.
- **Iteration**: The two-phase design is intentionally iterative. Phase 1
  results may reveal that the spec 001 detector needs parameter adjustments
  (e.g., too many false positives in a particular band). Feed this back
  to spec 001 before proceeding to Phase 2.
- **Future extension — autoencoder feature extraction**: Once the handcrafted
  feature pipeline is established, a convolutional autoencoder trained on
  spectrogram patches could learn a latent representation that captures
  patterns not anticipated in the handcrafted features. The autoencoder
  latent space would replace (or augment) the ~20 handcrafted features as
  input to UMAP + HDBSCAN, potentially improving separation of ambiguous
  event types. This is deferred because: (1) handcrafted features are more
  interpretable for initial discovery, (2) autoencoder training requires
  thousands of patches to converge, and (3) understanding what's in the
  data first makes autoencoder results easier to validate.
