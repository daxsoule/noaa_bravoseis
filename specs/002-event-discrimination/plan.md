# Analysis Plan: Acoustic Event Discrimination

**Spec**: `specs/002-event-discrimination/spec.md`
**Created**: 2026-03-02
**Status**: Draft
**Depends on**: `001-event-detection` must be implemented first

## Summary

Classify detected acoustic events from the BRAVOSEIS hydrophone array into
earthquakes, ice quakes, whale calls, and noise using a two-phase approach:
(1) extract handcrafted spectral features, cluster with UMAP + HDBSCAN to
discover natural groupings and generate training labels, then (2) train a
lightweight CNN on spectrogram patches to build a production classifier.
Key outputs: classified event catalogue, UMAP cluster map, cluster montages,
confusion matrix, and seasonal event rate timeline.

## Analysis Environment

**Language/Version**: Python 3.12
**Key Packages**:
- Existing: numpy, scipy, matplotlib
- Phase 1 (new): umap-learn, hdbscan, scikit-learn, pandas, pyarrow
- Phase 2 (new): torch, torchvision, torchaudio
**Environment File**: `pyproject.toml` (managed by uv)

## Compute Environment

- [x] Shared server (JupyterHub)

**Data scale**: ~80 GB raw DAT files (717 files × 115 MB). Processed
sequentially — only one file in memory at a time (~115 MB). Feature table
and spectrogram patches: ~1–5 GB depending on event count.

**Timeline pressure**: None stated. Exploratory/publication-track.

**Known bottlenecks**:
- Feature extraction reads all 717 DAT files sequentially (~30–60 min)
- CNN training on GPU: minutes to low hours for a lightweight model
- Phase 1 → Phase 2 gate requires human review (not automated)

## Constitution Check

- [x] Data sources match those defined in constitution
- [x] Coordinate systems/units are consistent (UTC time, Hz frequency)
- [x] Figure standards will be followed (paper tier, 300 DPI, justified captions)
- [x] Quality checks are incorporated (cluster stability, cross-mooring
      consistency, literature sanity check)

**Issues to resolve**: None. Spec aligns with constitution.

## Project Structure

```text
specs/002-event-discrimination/
├── spec.md
├── plan.md              # This file
├── research.md          # Whale call literature + method decisions
└── tasks.md             # Task breakdown (via /speckit.tasks)

scripts/
├── extract_features.py          # Phase 1a: feature extraction
├── cluster_events.py            # Phase 1b: UMAP + HDBSCAN
├── make_cluster_montages.py     # Phase 1c: visualization for labeling
├── label_clusters.py            # Phase 1c: apply labels to dataset
├── train_classifier.py          # Phase 2: CNN training + evaluation
└── classify_catalogue.py        # Phase 2e: apply model to full catalogue

outputs/
├── data/
│   ├── event_features.parquet       # Feature vectors for all events
│   ├── event_patches/               # Spectrogram patch NPZ files (by mooring)
│   ├── umap_coordinates.parquet     # UMAP 2D projection
│   ├── labeled_events.parquet       # Cluster labels + broad/subclass
│   ├── classified_catalogue.parquet # Final CNN predictions
│   └── cluster_labels.json          # Cluster → class mapping
├── figures/exploratory/
│   ├── maps/                        # Bathymetry, station maps
│   ├── spectrograms/                # Array spectrograms, example windows
│   ├── detection/                   # STA/LTA examples, QC, rates, distributions
│   ├── validation/                  # False positive analysis
│   ├── onsets/                      # Onset refinement QC
│   ├── association/                 # Cross-mooring stats, travel times
│   └── clustering/                  # UMAP, montages, feature maps
│       ├── umap_cluster_map.png
│       ├── umap_feature_maps.png
│       ├── cluster_montage_*.png
│       ├── classified_event_timeline.png
│       └── confusion_matrix.png
├── figures/journal/                 # Promoted publication-quality figures only
├── models/
│   ├── classifier.pt                # Trained CNN weights
│   └── training_log.json            # Loss/metrics per epoch
└── tables/
    ├── class_distribution.csv
    ├── cnn_performance.csv
    └── seasonal_event_rates.csv
```

**Structure notes**: Scripts live in the project root (matching existing
`make_*.py` and `read_dat.py` convention). No `scripts/` subdirectory
needed — project is flat. Intermediate data goes in `outputs/data/` to
keep raw data immutable.

## Data Pipeline

### Stage 0: Prerequisite — Event Catalogue (spec 001)

- **Input**: Raw DAT files → STA/LTA detector (spec 001)
- **Output**: `outputs/data/event_catalogue.parquet`
- **Status**: Not yet implemented. **Must complete spec 001 first.**

### Stage 1: Feature Extraction (`extract_features.py`)

- **Input**: `outputs/data/event_catalogue.parquet` + raw DAT files
- **Processing**:
  1. For each event in the catalogue, load the corresponding DAT file
     segment (onset − 2 s to end + 2 s)
  2. Compute spectrogram (nperseg=1024, noverlap=512, fs=1000, 0–250 Hz)
  3. Save spectrogram patch as compressed NPZ (for CNN later)
  4. Extract ~20 handcrafted features per event:
     - 10 band powers (25 Hz bands)
     - Duration, rise time, decay time
     - Peak frequency, peak power, bandwidth
     - Spectral slope
     - Frequency modulation metric
  5. Normalize features: subtract median background per mooring
- **Output**:
  - `outputs/data/event_features.parquet` (feature vectors)
  - `outputs/data/event_patches/` (NPZ spectrogram patches, one per mooring)
- **Notes**: Process DAT files sequentially to limit memory. Group events
  by file number to avoid re-reading the same file.

### Stage 2: Per-Band Clustering (`cluster_events.py`)

- **Input**: `outputs/data/event_features.parquet`
- **Processing** (repeated independently for each detection band: low, mid, high):
  1. Filter events to the target band
  2. Standardize features (zero mean, unit variance), fitted per band
  3. UMAP projection (n_neighbors=15, min_dist=0.1, n_components=2)
  4. HDBSCAN clustering (min_cluster_size=50, tuned empirically)
  5. Stability analysis: rerun 5× with different random seeds
  6. Compute silhouette score
  7. Generate UMAP scatter plot colored by cluster
  8. Generate UMAP plots colored by individual features
  9. Prefix cluster IDs by band (e.g., low_0, mid_3, high_7)
- **Rationale**: Initial all-band clustering produced a single mega-cluster
  (99% of events) organized primarily by detection band in UMAP space.
  Per-band clustering removes this dominant axis and exposes within-band
  structure (e.g., T-phases vs. local earthquakes in the low band, fin
  whale calls vs. background in mid band).
- **Output**:
  - `outputs/data/umap_coordinates.parquet` (all bands, with band-prefixed cluster IDs)
  - `outputs/figures/exploratory/clustering/umap_{band}_cluster_map.png` (one per band)
  - `outputs/figures/exploratory/clustering/umap_{band}_feature_maps.png` (one per band)
  - Cluster statistics printed to stdout

### Stage 3: Cluster Visualization (`make_cluster_montages.py`)

- **Input**: UMAP coordinates + spectrogram patches
- **Processing**:
  1. For each cluster, find 20 events nearest to centroid
  2. Load their spectrogram patches
  3. Arrange in a 4×5 grid with cluster ID, size, and mean features
  4. Add justified caption per montage
- **Output**: `outputs/figures/exploratory/clustering/cluster_montage_*.png`

### Stage 4: Labeling (`label_clusters.py`)

- **Input**: UMAP coordinates + cluster assignments
- **Processing**:
  1. Read `outputs/data/cluster_labels.json` (manually created after
     reviewing montages — maps cluster_id → {broad_class, subclass})
  2. Apply labels to all events
  3. Enforce minimum class size (100 events; merge small clusters)
  4. Compute class distribution statistics
- **Output**:
  - `outputs/data/labeled_events.parquet`
  - `outputs/tables/class_distribution.csv`

**--- GATE: Phase 1 Review ---**

User reviews UMAP plot, montages, and labeled dataset. Approves or
requests iteration before proceeding to Phase 2.

### Stage 5: CNN Training (`train_classifier.py`)

- **Input**: `outputs/data/labeled_events.parquet` + spectrogram patches
- **Processing**:
  1. Load spectrogram patches for all labeled events
  2. Resize/pad to uniform dimensions (determined from median event
     duration — likely 128×128 or 64×128 pixels)
  3. Stratified split: 70% train / 15% val / 15% test
  4. Data augmentation: time shift, Gaussian noise, frequency masking
  5. Train lightweight CNN (3–4 conv blocks, ~100K–500K params):
     - Cross-entropy loss (weighted by inverse class frequency)
     - AdamW optimizer, lr=1e-3, cosine annealing
     - Early stopping on val loss (patience=10)
     - Log per-class precision/recall/F1 each epoch
  6. Evaluate on test set: confusion matrix, per-class metrics
- **Output**:
  - `outputs/models/classifier.pt`
  - `outputs/models/training_log.json`
  - `outputs/figures/exploratory/clustering/confusion_matrix.png`
  - `outputs/tables/cnn_performance.csv`

### Stage 6: Full Catalogue Classification (`classify_catalogue.py`)

- **Input**: Trained model + all spectrogram patches
- **Processing**:
  1. Load trained CNN
  2. Run inference on all events (including those not in training set)
  3. Save predictions with softmax confidence scores
  4. Flag events with confidence < 0.5 for manual review
  5. Generate classified event rate timeline (stacked by broad class)
  6. Compute seasonal event rates table
- **Output**:
  - `outputs/data/classified_catalogue.parquet`
  - `outputs/figures/exploratory/clustering/classified_event_timeline.png`
  - `outputs/tables/seasonal_event_rates.csv`

## Script Plan

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `extract_features.py` | Extract spectrogram patches + handcrafted features from raw DAT files for each detected event | `event_catalogue.parquet`, raw DAT files | `event_features.parquet`, `event_patches/` |
| `cluster_events.py` | UMAP + HDBSCAN clustering with stability analysis and visualization | `event_features.parquet` | `umap_coordinates.parquet`, UMAP figures |
| `make_cluster_montages.py` | Generate spectrogram montages for each cluster for visual inspection | UMAP output, spectrogram patches | `cluster_montage_*.png` |
| `label_clusters.py` | Apply human-assigned labels from JSON mapping to the event catalogue | `cluster_labels.json`, UMAP output | `labeled_events.parquet`, class distribution |
| `train_classifier.py` | Train CNN on labeled spectrogram patches, evaluate on held-out test set | `labeled_events.parquet`, patches | `classifier.pt`, confusion matrix |
| `classify_catalogue.py` | Classify full catalogue, generate timeline and seasonal stats | Trained model, all patches | `classified_catalogue.parquet`, timeline fig |

## Dependencies

```text
[spec 001: event detection]
         ↓
  extract_features.py
         ↓
  cluster_events.py
         ↓
  make_cluster_montages.py
         ↓
  label_clusters.py ← (requires manual cluster_labels.json)
         ↓
  *** GATE: Phase 1 Review ***
         ↓
  train_classifier.py
         ↓
  classify_catalogue.py
```

**Parallel opportunities**:
- `cluster_events.py` and `make_cluster_montages.py` could be combined
  into a single script, but kept separate for clarity.
- `classify_catalogue.py` figure generation is independent of
  `train_classifier.py` evaluation figures — but they're fast enough
  that parallelism doesn't matter.

**Manual steps** (not automated):
1. After `make_cluster_montages.py`: review montages, create
   `outputs/data/cluster_labels.json` mapping cluster IDs to class names
2. After `label_clusters.py`: review labeled dataset, approve for Phase 2

## Open Questions

- [x] GPU availability → Yes, select at JupyterHub launch
- [x] Single vs. dual reviewer → Single reviewer, second deferred
- [x] Minimum class size → 100 events
- [x] Phase 1/2 gate → Yes, explicit gate
- [ ] CNN patch size → Determined empirically from median event duration
  (Stage 5, not blocking)
- [ ] CNN architecture details → Start with 3 conv blocks, iterate
  (Stage 5, not blocking)
- [ ] Transfer learning need → Depends on class sizes after labeling
  (Stage 5, not blocking)

All open questions are deferred to implementation — none are blocking.

## Notes

- **Spec 001 dependency**: This plan cannot execute until spec 001 (event
  detection) produces `outputs/data/event_catalogue.parquet`. Plan spec 001
  first.
- **Package installation**: Before starting, run:
  ```bash
  uv add umap-learn hdbscan scikit-learn pandas pyarrow torch torchvision
  ```
- **Shared utilities**: `read_dat.py` already exists. Spectrogram
  computation parameters (nperseg, noverlap, fs) should be importable
  constants — consider adding to `read_dat.py` or a new `config.py`.
- **Existing code reuse**: `make_spectrogram.py` demonstrates spectrogram
  computation and figure layout. `make_bathy_map.py` provides
  `add_caption_justified()` for figure captions.
- **Literature search**: Conduct during Phase 1c labeling (not a separate
  script). Document findings in `specs/002-event-discrimination/research.md`.
