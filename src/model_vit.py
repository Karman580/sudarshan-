"""
Multi-Modal Deepfake Detection — ViT Spatial Version
=====================================================
Uses FRMar features:
- Frequency: CSP Phase (8 scales)
- Temporal: 2x256x256 CNN features
- Spatial: ViT CLS Token (768-D) + Attention Map (14x14)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class Config:
    BASE_DIR: str = "/workspace/FR/ProcessedDataset"
    TEMPORAL_DIR: str = "/workspace/FR/ProcessedDataset/temporal_features_18feb"
    SPATIAL_DIR: str = "/workspace/FR/ProcessedDataset/spatial_features_with_masks_updated"

    # Shapes
    FREQ_SHAPE = (8, 1, 256, 256)
    TEMPORAL_SHAPE = (2, 256, 256)
    SPATIAL_CLS_SHAPE = (768,)
    SPATIAL_ACT_SHAPE = (1, 14, 14)

    BATCH_SIZE: int = 8
    NUM_EPOCHS: int = 500
    LEARNING_RATE: float = 5e-5
    WEIGHT_DECAY: float = 1e-4
    MAX_GRAD_NORM: float = 1.0
    MAX_FRAMES: int = 32

    DROPOUT_RATE: float = 0.5
    HIDDEN_DIM: int = 512
    FUSION_DIM: int = 256
    DEVICE: str = "cuda"
    CHECKPOINT_DIR: str = "checkpoints"
    SAVE_EVERY: int = 10

    def get_device(self):
        return torch.device(self.DEVICE if torch.cuda.is_available() else "cpu")

# ... (Frequency and Temporal CNNs remain identical) ...

class FrequencyStreamCNN(nn.Module):
    def __init__(self, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.scale_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(128 * 8, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_features = [self.scale_conv(x[:, i, :, :, :]) for i in range(x.size(1))]
        return self.fc(self.scale_fusion(torch.cat(scale_features, dim=1)))

class TemporalStreamCNN(nn.Module):
    def __init__(self, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv_layers(x))


class SpatialStreamViT(nn.Module):
    """
    Processes the ViT CLS token and 14x14 attention map.
    """
    def __init__(self, hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()
        
        # Process 14x14 activation map
        self.act_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7x7
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1) # 1x1
        )
        
        # Combine CLS (768) + Act Map (64) -> 832 -> hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(768 + 64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, cls_token: torch.Tensor, act_map: torch.Tensor) -> torch.Tensor:
        # act_map: (Batch, 1, 14, 14)
        act_feat = self.act_conv(act_map).flatten(1) # (Batch, 64)
        
        # cls_token: (Batch, 768)
        combined = torch.cat([cls_token, act_feat], dim=1) # (Batch, 832)
        
        return self.fc(combined)


class MultiModalFusionClassifierViT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.frequency_stream = FrequencyStreamCNN(config.HIDDEN_DIM, config.DROPOUT_RATE)
        self.temporal_stream = TemporalStreamCNN(config.HIDDEN_DIM, config.DROPOUT_RATE)
        self.spatial_stream = SpatialStreamViT(config.HIDDEN_DIM, config.DROPOUT_RATE)

        # Simple concatenation fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 3, config.FUSION_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE / 2),
        )
        self.classifier_norm = nn.BatchNorm1d(128) # Add this
        self.classifier = nn.Linear(128, 2)

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        freq_out = self.frequency_stream(features["frequency"])
        temp_out = self.temporal_stream(features["temporal"])
        spat_out = self.spatial_stream(features["spatial_cls"], features["spatial_act"])

        combined = torch.cat([freq_out, temp_out, spat_out], dim=1)
        fused = self.fusion(combined)
        logits = self.classifier(self.classifier_norm(fused))

        return logits, {
            "frequency": freq_out, "temporal": temp_out, 
            "spatial": spat_out, "fused": fused,
        }
