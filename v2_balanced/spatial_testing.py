import torch
import os
import glob
import numpy as np
from torch.nn.functional import cosine_similarity

# Paths from your config
spatial_dir = "/workspace/FR/ProcessedDataset/temporal_features_18feb"
base_dir = "/workspace/FR/ProcessedDataset"

def get_samples(label_str, num=5):
    # Find video IDs for the specific category
    path = os.path.join(base_dir, label_str, "Single Face")
    vids = os.listdir(path)[:num]
    features = []
    for vid in vids:
        file_path = os.path.join(spatial_dir, f"{vid}_spatial_embedding.pt")
        if os.path.exists(file_path):
            features.append(torch.load(file_path, map_location='cpu').flatten())
    return torch.stack(features) if features else None

print("--- Deepfake Feature Variance Check ---")

real_feats = get_samples("Real")
fake_feats = get_samples("Fake")

if real_feats is not None and fake_feats is not None:
    # Basic Stats
    print(f"Real - Mean: {real_feats.mean():.4f}, Std: {real_feats.std():.4f}, Norm: {torch.norm(real_feats, dim=1).mean():.4f}")
    print(f"Fake - Mean: {fake_feats.mean():.4f}, Std: {fake_feats.std():.4f}, Norm: {torch.norm(fake_feats, dim=1).mean():.4f}")
    
    # Check for "Zero-Collapse" (If features are all zeros or nearly identical)
    cos_sim_within_real = cosine_similarity(real_feats[0].unsqueeze(0), real_feats[1].unsqueeze(0))
    cos_sim_cross_class = cosine_similarity(real_feats[0].unsqueeze(0), fake_feats[0].unsqueeze(0))
    
    print(f"\nCosine Similarity (Real vs Real): {cos_sim_within_real.item():.4f}")
    print(f"Cosine Similarity (Real vs Fake): {cos_sim_cross_class.item():.4f}")
    
    if cos_sim_cross_class > 0.98:
        print("\n[WARNING]: Real and Fake features are >98% similar. The model will struggle to separate them.")
    else:
        print("\n[OK]: Features appear mathematically distinct.")
else:
    print("Error: Could not find feature files. Check your paths!")