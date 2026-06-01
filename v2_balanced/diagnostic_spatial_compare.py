"""
Spatial Feature Comparison Diagnostic
====================================================
Compares the discriminative power of the two different spatial feature formats:
1. FR Dataset: 2x256x256 CNN-upsampled features
2. FR_10Mar Dataset: 768-D CLS embeddings + 14x14 activation maps
"""

import os
import glob
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


def collect_videos(base_dir):
    video_data = {}
    for fake_real in ["Fake", "Real"]:
        label = 1 if fake_real == "Fake" else 0
        for face_type in ["Single Face", "Multi Face"]:
            category = os.path.join(base_dir, fake_real, face_type)
            if not os.path.exists(category):
                continue
            for video_id in sorted(os.listdir(category)):
                video_data[video_id] = {"label": label, "path": os.path.join(category, video_id)}
    return video_data


def main():
    print("=" * 70)
    print("Spatial Feature Comparison")
    print("=" * 70)

    # FR Dataset (2x256x256)
    fr_spatial_dir = Path("/workspace/FR/ProcessedDataset/spatial_features_with_masks_10feb_union")
    fr_base_dir = "/workspace/FR/ProcessedDataset"
    
    # FR_10Mar Dataset (768-D CLS + 14x14)
    mar_spatial_dir = Path("/workspace/FR_10Mar/ProcessedDataset/spatial_features_with_masks_updated")
    mar_base_dir = "/workspace/FR_10Mar/ProcessedDataset"

    # --- Test FR Dataset ---
    print("\n[1] Testing FR spatial_features_with_masks_10feb_union (2x256x256 CNN maps)")
    try:
        videos = collect_videos(fr_base_dir)
        real_ids = [v for v in videos if videos[v]["label"] == 0]
        fake_ids = [v for v in videos if videos[v]["label"] == 1]
        
        random.seed(42)
        sample = random.sample(real_ids, min(100, len(real_ids))) + random.sample(fake_ids, min(100, len(fake_ids)))
        
        X, y = [], []
        loaded = 0
        for vid in tqdm(sample):
            sf = fr_spatial_dir / f"{vid}_spatial_features.pt"
            if sf.exists():
                try:
                    # Takes the flattened 2x256x256 tensor (only first 1024 values to avoid memory issues with LogReg)
                    t = torch.load(sf, map_location="cpu").flatten().numpy()
                    X.append(t[:1024]) 
                    y.append(videos[vid]["label"])
                    loaded += 1
                except: pass
                
        if loaded > 0:
            X, y = np.array(X), np.array(y)
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            clf.fit(X, y)
            auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
            print(f"  Result ({loaded} samples): AUC={auc:.4f}, Acc={accuracy_score(y, clf.predict(X))*100:.1f}%")
        else:
            print("  No samples loaded.")
    except Exception as e:
        print(f"  Error: {e}")

    # --- Test FR_10Mar Dataset ---
    print("\n[2] Testing FR_10Mar spatial_features_with_masks_updated (768-D CLS)")
    try:
        videos = collect_videos(mar_base_dir)
        real_ids = [v for v in videos if videos[v]["label"] == 0]
        fake_ids = [v for v in videos if videos[v]["label"] == 1]
        
        random.seed(42)
        sample = random.sample(real_ids, min(100, len(real_ids))) + random.sample(fake_ids, min(100, len(fake_ids)))
        
        X_cls, X_act, y = [], [], []
        loaded = 0
        for vid in tqdm(sample):
            cls_f = mar_spatial_dir / f"{vid}_spatial_embedding.pt"
            act_f = mar_spatial_dir / f"{vid}_activation_map.pt"
            
            if cls_f.exists() and act_f.exists():
                try:
                    cls = torch.load(cls_f, map_location="cpu").flatten().numpy()
                    act = torch.load(act_f, map_location="cpu").flatten().numpy()
                    X_cls.append(cls[:1024])  # usually 768 anyway
                    X_act.append(act[:1024])  # usually 196 (14x14)
                    y.append(videos[vid]["label"])
                    loaded += 1
                except: pass
                
        if loaded > 0:
            y = np.array(y)
            
            # CLS Token
            X = np.array(X_cls)
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            clf.fit(X, y)
            auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
            print(f"  CLS Token ({loaded} samples): AUC={auc:.4f}, Acc={accuracy_score(y, clf.predict(X))*100:.1f}%")
            
            # Activation Map
            X = np.array(X_act)
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            clf.fit(X, y)
            auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
            print(f"  Act Map   ({loaded} samples): AUC={auc:.4f}, Acc={accuracy_score(y, clf.predict(X))*100:.1f}%")
        else:
            print("  No samples loaded.")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    main()
