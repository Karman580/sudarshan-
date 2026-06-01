"""
Feature Quality Diagnostic — Run BEFORE Training
====================================================
Verifies that each feature stream (Frequency, Temporal, Spatial) loads
correctly and carries discriminative signal for Real vs Fake classification.

Usage:
    python diagnostic.py
"""

import os
import sys
import glob
import random
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from model import Config


def collect_csp_video_ids(base_dir):
    """Same collector as run_training.py."""
    video_data = {}
    for fake_real in ["Fake", "Real"]:
        label = 1 if fake_real == "Fake" else 0
        for face_type in ["Single Face", "Multi Face"]:
            category = os.path.join(base_dir, fake_real, face_type)
            if not os.path.exists(category):
                continue
            for video_id in sorted(os.listdir(category)):
                csp_dir = os.path.join(
                    category, video_id, "face_extracted", "csp_tensors_pt"
                )
                if os.path.exists(csp_dir):
                    tensor_files = sorted(glob.glob(os.path.join(csp_dir, "*.pt")))
                    if tensor_files:
                        video_data[video_id] = {
                            "label": label,
                            "csp_dir": csp_dir,
                            "tensor_files": tensor_files,
                        }
    return video_data


def main():
    config = Config()
    print("=" * 70)
    print("Feature Quality Diagnostic")
    print("=" * 70)
    print(f"BASE_DIR:     {config.BASE_DIR}")
    print(f"TEMPORAL_DIR: {config.TEMPORAL_DIR}")
    print(f"SPATIAL_DIR:  {config.SPATIAL_DIR}")
    print()

    # --- Step 1: Collect videos ---
    print("[1/5] Collecting videos...")
    csp_data = collect_csp_video_ids(config.BASE_DIR)

    if len(csp_data) == 0:
        print("✗ ERROR: No CSP videos found! Check BASE_DIR.")
        sys.exit(1)

    labels = {vid: d["label"] for vid, d in csp_data.items()}
    real_ids = [v for v in labels if labels[v] == 0]
    fake_ids = [v for v in labels if labels[v] == 1]
    print(f"  ✓ Total: {len(csp_data)} videos (Real: {len(real_ids)}, Fake: {len(fake_ids)})")

    # --- Step 2: Sample balanced subset ---
    N = 100  # 100 per class
    random.seed(42)
    sample_real = random.sample(real_ids, min(N, len(real_ids)))
    sample_fake = random.sample(fake_ids, min(N, len(fake_ids)))
    sample_ids = sample_real + sample_fake
    print(f"\n[2/5] Sampled {len(sample_real)} Real + {len(sample_fake)} Fake for diagnostics")

    temporal_dir = Path(config.TEMPORAL_DIR)
    spatial_dir = Path(config.SPATIAL_DIR)

    # --- Step 3: Check feature file existence ---
    print("\n[3/5] Checking feature file existence...")
    freq_ok, freq_miss = 0, 0
    temp_ok, temp_miss = 0, 0
    spat_ok, spat_miss = 0, 0

    for vid in sample_ids:
        # Frequency
        if len(csp_data[vid]["tensor_files"]) > 0:
            freq_ok += 1
        else:
            freq_miss += 1

        # Temporal
        if (temporal_dir / f"{vid}_temporal_features.pt").exists():
            temp_ok += 1
        else:
            temp_miss += 1

        # Spatial
        if (spatial_dir / f"{vid}_spatial_features.pt").exists():
            spat_ok += 1
        else:
            spat_miss += 1

    print(f"  Frequency: {freq_ok} OK, {freq_miss} missing")
    print(f"  Temporal:  {temp_ok} OK, {temp_miss} missing")
    print(f"  Spatial:   {spat_ok} OK, {spat_miss} missing")

    if temp_miss > len(sample_ids) * 0.5:
        print("  ⚠ WARNING: >50% temporal features missing! Check TEMPORAL_DIR.")
    if spat_miss > len(sample_ids) * 0.5:
        print("  ⚠ WARNING: >50% spatial features missing! Check SPATIAL_DIR.")

    # --- Step 4: Load features & check shapes/ranges ---
    print("\n[4/5] Loading features & checking shapes...")

    freq_features = []
    temp_features = []
    spat_features = []
    y_labels = []

    for vid in tqdm(sample_ids, desc="Loading"):
        label = labels[vid]

        # Frequency — load first CSP tensor
        try:
            t = torch.load(csp_data[vid]["tensor_files"][0], map_location="cpu")
            phase = t[:, 1, :, :].flatten().numpy()  # Phase channel
            freq_features.append(phase[:1024])  # Take first 1024 values
        except Exception:
            freq_features.append(np.zeros(1024))

        # Temporal
        tf = temporal_dir / f"{vid}_temporal_features.pt"
        if tf.exists():
            t = torch.load(tf, map_location="cpu").flatten().numpy()
            temp_features.append(t[:1024])
        else:
            temp_features.append(np.zeros(1024))

        # Spatial
        sf = spatial_dir / f"{vid}_spatial_features.pt"
        if sf.exists():
            t = torch.load(sf, map_location="cpu").flatten().numpy()
            spat_features.append(t[:1024])
        else:
            spat_features.append(np.zeros(1024))

        y_labels.append(label)

    freq_X = np.array(freq_features)
    temp_X = np.array(temp_features)
    spat_X = np.array(spat_features)
    y = np.array(y_labels)

    # Value range check
    print(f"\n  Frequency — shape: {freq_X.shape}, range: [{freq_X.min():.4f}, {freq_X.max():.4f}], mean: {freq_X.mean():.4f}")
    print(f"  Temporal  — shape: {temp_X.shape}, range: [{temp_X.min():.4f}, {temp_X.max():.4f}], mean: {temp_X.mean():.4f}")
    print(f"  Spatial   — shape: {spat_X.shape}, range: [{spat_X.min():.4f}, {spat_X.max():.4f}], mean: {spat_X.mean():.4f}")

    # --- Step 5: Simple LogReg AUC per modality ---
    print("\n[5/5] Testing discriminative power (LogReg AUC)...")

    for name, X in [("Frequency", freq_X), ("Temporal", temp_X), ("Spatial", spat_X)]:
        # Check if features are all zeros (missing data)
        if np.abs(X).max() < 1e-10:
            print(f"  {name}: ALL ZEROS — features not loaded correctly!")
            continue

        try:
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            clf.fit(X, y)
            probs = clf.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, probs)
            preds = clf.predict(X)
            acc = accuracy_score(y, preds)
            print(f"  {name}: AUC={auc:.4f}, Acc={acc * 100:.1f}%", end="")

            if auc > 0.70:
                print("  ✓ GOOD signal")
            elif auc > 0.55:
                print("  ~ Weak signal")
            else:
                print("  ✗ NO signal (near random)")
        except Exception as e:
            print(f"  {name}: Error — {e}")

    # Combined
    try:
        X_all = np.concatenate([freq_X, temp_X, spat_X], axis=1)
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        clf.fit(X_all, y)
        probs = clf.predict_proba(X_all)[:, 1]
        auc = roc_auc_score(y, probs)
        acc = accuracy_score(y, clf.predict(X_all))
        print(f"\n  Combined:  AUC={auc:.4f}, Acc={acc * 100:.1f}%")
    except Exception as e:
        print(f"\n  Combined: Error — {e}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("If AUC > 0.70 on any stream, training should succeed.")
    print("If AUC ≈ 0.50 on all streams, check feature extraction pipeline.")


if __name__ == "__main__":
    main()
