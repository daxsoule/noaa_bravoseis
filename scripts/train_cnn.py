#!/usr/bin/env python3
"""
train_cnn.py — Phase 2 supervised CNN classifier for BRAVOSEIS events.

Trains a lightweight CNN on spectrogram patches from high-confidence Phase 1
labels (T-phase, icequake, vessel noise), then classifies the ~208k bulk
unclassified events with confidence scores.

Architecture: 4 conv blocks, ~200K parameters, spectrogram input.
Training: weighted cross-entropy, AdamW, cosine LR, early stopping.
Target: >=80% macro F1 on held-out test set.

Usage:
    uv run python train_cnn.py                    # full pipeline
    uv run python train_cnn.py --extract-only      # just build spectrograms
    uv run python train_cnn.py --train-only        # train (spectrograms exist)
    uv run python train_cnn.py --predict-only      # predict bulk (model exists)

Output:
    outputs/data/cnn_predictions.parquet
    outputs/data/cnn_model.pt
    outputs/figures/exploratory/cnn_training_curves.png
    outputs/figures/exploratory/cnn_confusion_matrix.png
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from datetime import timedelta
from scipy.signal import spectrogram as scipy_spectrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
)
from collections import Counter
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from read_dat import read_dat, MOORINGS, SAMPLE_RATE

# === Paths ===
DATA_ROOT = Path("/home/jovyan/my_data/bravoseis/NOAA")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures" / "exploratory"
CACHE_DIR = OUTPUT_DIR / "data" / "spectrograms"

# === Spectrogram parameters ===
WINDOW_SEC = 8.0        # total window around event
PAD_SEC = 2.0           # pre-event padding
NPERSEG = 256           # FFT window (~0.256s at 1kHz)
NOVERLAP = 224          # 87.5% overlap — fine time resolution
FREQ_MAX = 100          # Hz — covers T-phases, icequakes, most vessel energy
IMG_HEIGHT = 64         # frequency bins (after resampling)
IMG_WIDTH = 128         # time bins (after resampling)

# === Training parameters ===
BATCH_SIZE = 256
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8            # early stopping patience
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42

# === Class definitions ===
CLASS_NAMES = ["tphase", "icequake", "vessel"]
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Label Assignment
# ============================================================

def assign_labels():
    """Assign Phase 1 labels to events. Returns (labeled_df, bulk_df)."""
    print("Assigning Phase 1 labels...")

    cat = pd.read_parquet(DATA_DIR / "event_catalogue.parquet")
    features = pd.read_parquet(DATA_DIR / "event_features.parquet")
    umap_df = pd.read_parquet(DATA_DIR / "umap_coordinates.parquet")
    cat["onset_utc"] = pd.to_datetime(cat["onset_utc"])

    # T-phase: cluster-based + feature-based
    tphase_clusters = {"low_0", "low_1", "mid_0"}
    cluster_tphase = set(
        umap_df[umap_df["cluster_id"].isin(tphase_clusters)]["event_id"]
    )
    feat_tphase_mask = (
        (features["spectral_slope"] < -0.5)
        & (features["peak_freq_hz"] < 30)
        & (features["peak_power_db"] > 48)
        & (features["duration_s"] <= 3)
    )
    all_tphase = cluster_tphase | set(features.loc[feat_tphase_mask, "event_id"])

    # Icequake
    feat_ice_mask = (
        (features["duration_s"] > 3)
        & (features["peak_power_db"] > 48)
        & (features["peak_freq_hz"] < 30)
        & (features["spectral_slope"] < -0.2)
    )
    ice_cluster = set(umap_df[umap_df["cluster_id"] == "high_2"]["event_id"])
    all_ice = (set(features.loc[feat_ice_mask, "event_id"]) | ice_cluster) - all_tphase

    # Vessel noise
    type_a_mask = (
        (features["spectral_slope"] > 0)
        & (features["peak_freq_hz"] > 100)
        & (features["bandwidth_hz"] > 150)
        & (features["freq_modulation"] > 30)
    )
    all_vessel = set(features.loc[type_a_mask, "event_id"]) - all_tphase - all_ice

    # Build label map
    label_map = {}
    for eid in all_tphase:
        label_map[eid] = 0  # tphase
    for eid in all_ice:
        label_map[eid] = 1  # icequake
    for eid in all_vessel:
        label_map[eid] = 2  # vessel

    # Split into labeled and bulk
    cat["label"] = cat["event_id"].map(label_map)
    labeled = cat[cat["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype(int)
    bulk = cat[cat["label"].isna()].copy()

    print(f"  Labeled: {len(labeled):,} events")
    for i, name in enumerate(CLASS_NAMES):
        n = (labeled["label"] == i).sum()
        print(f"    {name}: {n:,}")
    print(f"  Bulk (unlabeled): {len(bulk):,} events")

    return labeled, bulk


# ============================================================
# Spectrogram Extraction
# ============================================================

_dat_cache = {}
MAX_CACHE = 3  # sorted by file, so only need a small cache


def get_dat(filepath):
    """Read DAT file with simple LRU cache."""
    key = str(filepath)
    if key not in _dat_cache:
        if len(_dat_cache) >= MAX_CACHE:
            oldest = next(iter(_dat_cache))
            del _dat_cache[oldest]
        ts, data, _ = read_dat(filepath)
        _dat_cache[key] = (ts, data)
    return _dat_cache[key]


def extract_spectrogram(event_row):
    """Extract a fixed-size spectrogram patch for one event.

    Returns (IMG_HEIGHT, IMG_WIDTH) float32 array in dB, or None on failure.
    """
    mooring = event_row["mooring"]
    file_num = event_row["file_number"]
    onset = event_row["onset_utc"]
    duration = event_row["duration_s"]

    info = MOORINGS[mooring]
    dat_path = DATA_ROOT / info["data_dir"] / f"{file_num:08d}.DAT"
    if not dat_path.exists():
        return None

    file_ts, data = get_dat(dat_path)

    # Window centered on event
    t_start = onset - timedelta(seconds=PAD_SEC)
    offset_s = (t_start - file_ts).total_seconds()
    s0 = int(offset_s * SAMPLE_RATE)
    s1 = s0 + int(WINDOW_SEC * SAMPLE_RATE)

    if s0 < 0 or s1 > len(data):
        return None

    segment = data[s0:s1].astype(np.float64)

    # Compute spectrogram
    freqs, times, Sxx = scipy_spectrogram(
        segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP
    )

    # Crop to FREQ_MAX
    freq_mask = freqs <= FREQ_MAX
    Sxx = Sxx[freq_mask, :]

    # Convert to dB
    Sxx_dB = 10 * np.log10(Sxx + 1e-20)

    # Resize to fixed dimensions using simple interpolation
    from scipy.ndimage import zoom

    zoom_f = (IMG_HEIGHT / Sxx_dB.shape[0], IMG_WIDTH / Sxx_dB.shape[1])
    patch = zoom(Sxx_dB, zoom_f, order=1).astype(np.float32)

    return patch


def extract_spectrograms_batch(df, output_path, desc=""):
    """Extract spectrograms for a DataFrame of events, save as .npz.

    Sorts by (mooring, file_number) to maximize DAT cache hits.
    """
    if output_path.exists():
        data = np.load(output_path, allow_pickle=True)
        specs = data["spectrograms"]
        eids = data["event_ids"]
        print(f"  Loaded cached: {output_path.name} ({len(eids):,} events)")
        return specs, eids

    # Sort by mooring + file for sequential I/O
    df_sorted = df.sort_values(["mooring", "file_number"]).reset_index(drop=True)
    n = len(df_sorted)
    specs = np.zeros((n, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)
    event_ids = df_sorted["event_id"].values

    t0 = time.time()
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        patch = extract_spectrogram(row)
        if patch is not None:
            specs[i] = patch
            valid_mask[i] = True

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            n_ok = valid_mask[:i + 1].sum()
            print(
                f"    {desc} {i+1:,}/{n:,} ({n_ok:,} ok) "
                f"[{rate:.0f} evt/s, ETA {eta:.0f}s]"
            )

    # Keep only valid
    specs = specs[valid_mask]
    event_ids = event_ids[valid_mask]
    print(f"  {desc}: {valid_mask.sum():,}/{n:,} valid spectrograms")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, spectrograms=specs, event_ids=event_ids)
    print(f"  Saved: {output_path}")

    return specs, event_ids


# ============================================================
# Dataset
# ============================================================

class HybridDataset(Dataset):
    """Dataset with spectrogram patches + handcrafted features."""

    def __init__(self, spectrograms, features, labels, augment=False):
        self.specs = torch.from_numpy(spectrograms).float()
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.specs[idx].unsqueeze(0)  # (1, H, W)
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)

        feat = self.features[idx]

        if self.augment:
            shift = np.random.randint(-IMG_WIDTH // 10, IMG_WIDTH // 10)
            spec = torch.roll(spec, shift, dims=2)

            if np.random.rand() < 0.3:
                f0 = np.random.randint(0, IMG_HEIGHT - 8)
                fw = np.random.randint(1, 8)
                spec[:, f0:f0 + fw, :] = 0

            if np.random.rand() < 0.3:
                t0 = np.random.randint(0, IMG_WIDTH - 16)
                tw = np.random.randint(1, 16)
                spec[:, :, t0:t0 + tw] = 0

            # Small noise on features
            if np.random.rand() < 0.2:
                feat = feat + 0.02 * torch.randn_like(feat)

        return spec, feat, self.labels[idx]


# === Feature columns used for the MLP branch ===
FEATURE_COLS = [
    "peak_freq_hz", "duration_s", "snr", "bandwidth_hz",
    "spectral_slope", "freq_modulation", "spectral_centroid_hz",
    "peak_power_db", "decay_time_s", "rise_time_s",
    "band_power_0_5_hz", "band_power_5_15_hz", "band_power_15_50_hz",
    "band_power_50_100_hz", "band_power_100_250_hz",
]


# ============================================================
# Model
# ============================================================

class HybridModel(nn.Module):
    """CNN on spectrograms + MLP on features, fused for classification.

    CNN branch extracts visual patterns; feature branch captures the
    handcrafted statistics that Phase 1 labels were based on.
    """

    def __init__(self, num_features, num_classes=NUM_CLASSES):
        super().__init__()

        # CNN branch
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Feature MLP branch
        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Fusion head: 128 (CNN) + 64 (MLP) = 192
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(192, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, spec, feat):
        cnn_out = self.cnn(spec).view(spec.size(0), -1)  # (B, 128)
        mlp_out = self.feature_mlp(feat)                  # (B, 64)
        fused = torch.cat([cnn_out, mlp_out], dim=1)      # (B, 192)
        return self.classifier(fused)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for specs, feats, labels in loader:
        specs = specs.to(device)
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(specs, feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for specs, feats, labels in loader:
        specs = specs.to(device)
        feats = feats.to(device)
        labels = labels.to(device)
        outputs = model(specs, feats)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    n = len(all_labels)
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / n, acc, macro_f1, all_preds, all_labels


def train_model(train_specs, train_feats, train_labels,
                val_specs, val_feats, val_labels):
    """Train the hybrid CNN+MLP. Returns (model, history)."""
    print(f"\n{'='*60}")
    print(f"Training Hybrid CNN+MLP on {DEVICE}")
    print(f"  Train: {len(train_labels):,}, Val: {len(val_labels):,}")
    print(f"  Features: {train_feats.shape[1]}")
    print(f"  Classes: {dict(Counter(train_labels))}")
    print(f"{'='*60}")

    # Datasets
    train_ds = HybridDataset(train_specs, train_feats, train_labels, augment=True)
    val_ds = HybridDataset(val_specs, val_feats, val_labels, augment=False)

    # Weighted sampler for class imbalance
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Model
    model = HybridModel(train_feats.shape[1], NUM_CLASSES).to(DEVICE)
    print(f"  Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, DEVICE
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_F1={val_f1:.3f} | "
            f"lr={lr:.1e} | {elapsed:.1f}s"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (best F1={best_f1:.3f})")
                break

    # Restore best
    model.load_state_dict(best_state)
    model.to(DEVICE)

    return model, history


# ============================================================
# Evaluation & Figures
# ============================================================

def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], label="Accuracy")
    ax2.plot(epochs, history["val_f1"], label="Macro F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics")
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    outpath = FIG_DIR / "cnn_training_curves.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("CNN Test Set Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    outpath = FIG_DIR / "cnn_confusion_matrix.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ============================================================
# Prediction
# ============================================================

class PredictionDataset(Dataset):
    """Dataset for inference — spectrograms + features, no labels."""

    def __init__(self, spectrograms, features):
        self.specs = torch.from_numpy(spectrograms).float()
        self.features = torch.from_numpy(features).float()

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec = self.specs[idx].unsqueeze(0)
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        return spec, self.features[idx]


@torch.no_grad()
def predict_bulk(model, specs, feats):
    """Run inference on bulk events. Returns (predictions, probabilities)."""
    model.eval()
    ds = PredictionDataset(specs, feats)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, num_workers=4, pin_memory=True)

    all_probs = []
    for spec_batch, feat_batch in loader:
        spec_batch = spec_batch.to(DEVICE)
        feat_batch = feat_batch.to(DEVICE)
        logits = model(spec_batch, feat_batch)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    predictions = all_probs.argmax(axis=1)
    confidences = all_probs.max(axis=1)

    return predictions, all_probs, confidences


# ============================================================
# Main Pipeline
# ============================================================

def load_features_for_events(event_ids):
    """Load and normalize handcrafted features for given event IDs.

    Returns (features_array, feature_scaler_params) where features_array
    is z-score normalized.
    """
    all_features = pd.read_parquet(DATA_DIR / "event_features.parquet")
    feat_df = all_features[all_features["event_id"].isin(set(event_ids))]
    feat_df = feat_df.set_index("event_id").loc[event_ids].reset_index()

    # Extract numeric features
    available = [c for c in FEATURE_COLS if c in feat_df.columns]
    X = feat_df[available].values.astype(np.float32)

    # Replace NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, available


def main():
    parser = argparse.ArgumentParser(description="Phase 2 hybrid classifier")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract spectrograms, don't train")
    parser.add_argument("--train-only", action="store_true",
                        help="Train using cached spectrograms")
    parser.add_argument("--predict-only", action="store_true",
                        help="Predict bulk using saved model")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Assign labels ----
    labeled, bulk = assign_labels()

    # ---- Step 2: Extract spectrograms ----
    if not args.predict_only:
        if not args.train_only:
            print("\nExtracting labeled spectrograms...")
            labeled_specs, labeled_eids = extract_spectrograms_batch(
                labeled, CACHE_DIR / "labeled_specs.npz", desc="Labeled"
            )

            print("\nExtracting bulk spectrograms...")
            bulk_specs, bulk_eids = extract_spectrograms_batch(
                bulk, CACHE_DIR / "bulk_specs.npz", desc="Bulk"
            )

            if args.extract_only:
                print("\nExtraction complete.")
                return
        else:
            print("\nLoading cached spectrograms...")
            data = np.load(CACHE_DIR / "labeled_specs.npz", allow_pickle=True)
            labeled_specs = data["spectrograms"]
            labeled_eids = data["event_ids"]

    if args.predict_only:
        print("\nLoading cached data and model...")
        data = np.load(CACHE_DIR / "bulk_specs.npz", allow_pickle=True)
        bulk_specs = data["spectrograms"]
        bulk_eids = data["event_ids"]

        checkpoint = torch.load(DATA_DIR / "cnn_model.pt", map_location=DEVICE,
                                weights_only=False)
        num_features = checkpoint["num_features"]
        model = HybridModel(num_features, NUM_CLASSES).to(DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        feat_mean = checkpoint["feat_mean"]
        feat_std = checkpoint["feat_std"]

        # Load bulk features
        bulk_feats_raw, _ = load_features_for_events(bulk_eids)
        bulk_feats = (bulk_feats_raw - feat_mean) / (feat_std + 1e-8)

    else:
        # ---- Step 3: Load features and build splits ----
        print("\nLoading handcrafted features...")
        labeled_feats_raw, feat_cols = load_features_for_events(labeled_eids)
        print(f"  Feature columns ({len(feat_cols)}): {feat_cols}")

        # Compute normalization from labeled data
        feat_mean = labeled_feats_raw.mean(axis=0)
        feat_std = labeled_feats_raw.std(axis=0)
        labeled_feats = (labeled_feats_raw - feat_mean) / (feat_std + 1e-8)

        # Map event_ids back to labels
        label_map = dict(zip(labeled["event_id"], labeled["label"]))
        labels = np.array([label_map[eid] for eid in labeled_eids])

        # Stratified split
        np.random.seed(RANDOM_SEED)
        indices = np.arange(len(labels))
        np.random.shuffle(indices)

        n_test = int(len(indices) * TEST_FRACTION)
        n_val = int(len(indices) * VAL_FRACTION)

        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        train_specs = labeled_specs[train_idx]
        train_feats = labeled_feats[train_idx]
        train_labels = labels[train_idx]
        val_specs = labeled_specs[val_idx]
        val_feats = labeled_feats[val_idx]
        val_labels = labels[val_idx]
        test_specs = labeled_specs[test_idx]
        test_feats = labeled_feats[test_idx]
        test_labels = labels[test_idx]

        print(f"\nSplit: train={len(train_labels):,}, val={len(val_labels):,}, "
              f"test={len(test_labels):,}")

        # ---- Step 4: Train ----
        model, history = train_model(
            train_specs, train_feats, train_labels,
            val_specs, val_feats, val_labels
        )

        # ---- Step 5: Evaluate on test set ----
        print(f"\n{'='*60}")
        print("Test Set Evaluation")
        print(f"{'='*60}")

        test_ds = HybridDataset(test_specs, test_feats, test_labels, augment=False)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
            num_workers=4, pin_memory=True
        )
        criterion = nn.CrossEntropyLoss()
        _, test_acc, test_f1, test_preds, test_true = evaluate(
            model, test_loader, criterion, DEVICE
        )

        print(f"\n  Test Accuracy: {test_acc:.3f}")
        print(f"  Test Macro F1: {test_f1:.3f}")
        print(f"\n{classification_report(test_true, test_preds, target_names=CLASS_NAMES)}")

        # Figures
        plot_training_curves(history)
        plot_confusion_matrix(test_true, test_preds)

        # Save model + normalization stats
        checkpoint = {
            "model_state": model.state_dict(),
            "history": history,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "class_names": CLASS_NAMES,
            "num_features": len(feat_cols),
            "feat_mean": feat_mean,
            "feat_std": feat_std,
        }
        model_path = DATA_DIR / "cnn_model.pt"
        torch.save(checkpoint, model_path)
        print(f"  Saved model: {model_path}")

        if args.train_only:
            print("\nTraining complete.")
            return

        # Load bulk data
        data = np.load(CACHE_DIR / "bulk_specs.npz", allow_pickle=True)
        bulk_specs = data["spectrograms"]
        bulk_eids = data["event_ids"]

        bulk_feats_raw, _ = load_features_for_events(bulk_eids)
        bulk_feats = (bulk_feats_raw - feat_mean) / (feat_std + 1e-8)

    # ---- Step 6: Predict bulk ----
    print(f"\n{'='*60}")
    print(f"Predicting {len(bulk_eids):,} bulk events...")
    print(f"{'='*60}")

    predictions, probs, confidences = predict_bulk(model, bulk_specs, bulk_feats)

    # Build output DataFrame
    result = pd.DataFrame({
        "event_id": bulk_eids,
        "cnn_label": [CLASS_NAMES[p] for p in predictions],
        "cnn_label_idx": predictions,
        "cnn_confidence": confidences,
        "prob_tphase": probs[:, 0],
        "prob_icequake": probs[:, 1],
        "prob_vessel": probs[:, 2],
    })

    outpath = DATA_DIR / "cnn_predictions.parquet"
    result.to_parquet(outpath, index=False)
    print(f"\n  Saved predictions: {outpath}")

    # Summary
    print(f"\n  Prediction summary:")
    for name in CLASS_NAMES:
        mask = result["cnn_label"] == name
        n = mask.sum()
        med_conf = result.loc[mask, "cnn_confidence"].median()
        high_conf = (result.loc[mask, "cnn_confidence"] >= 0.8).sum()
        print(f"    {name}: {n:,} ({100*n/len(result):.1f}%), "
              f"median conf={med_conf:.2f}, high-conf(>=0.8): {high_conf:,}")

    # Confidence distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(CLASS_NAMES):
        mask = result["cnn_label_idx"] == i
        ax.hist(result.loc[mask, "cnn_confidence"], bins=50, alpha=0.5,
                label=f"{name} ({mask.sum():,})")
    ax.set_xlabel("CNN Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Hybrid Model Prediction Confidence — Bulk Events")
    ax.legend()
    plt.tight_layout()
    outpath_fig = FIG_DIR / "cnn_bulk_confidence.png"
    fig.savefig(outpath_fig, dpi=200)
    plt.close(fig)
    print(f"  Saved: {outpath_fig}")

    print("\nDone.")


if __name__ == "__main__":
    main()
