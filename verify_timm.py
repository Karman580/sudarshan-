import torch
import cv2
import numpy as np

# Simulate frame input
print("1. Creating dummy frame buffer...")
dummy_frames = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(4)]
print(f"Created {len(dummy_frames)} frames of shape {dummy_frames[0].shape}")

# Extract features
from src.fnet_features import extract_spatial_features, extract_temporal_features
spatial_features, _ = extract_spatial_features(["verify_timm.py"] * 4) # pass fake paths, it will skip cv2.imread and return zeros
print(f"Spatial features: {[s.shape for s in spatial_features]}")

from src.fnet_model import get_fnet_model
model = get_fnet_model()
print("Model loaded.")

features = {
    'frequency': torch.zeros(1, 8, 1, 256, 256),
    'temporal': torch.zeros(1, 2, 256, 256),
    'spatial_cls': torch.zeros(1, 768),
    'spatial_act': torch.zeros(1, 1, 14, 14),
}

with torch.no_grad():
    logits, stream_outputs = model(features)
    print("Logits:", logits)
    print("Stream outputs keys:", stream_outputs.keys())
    print("Success!")
