# ==============================================================================
# fnet_model.py - F-Net Multi-Modal CNN Model for Deepfake Detection
# ==============================================================================
# Three-stream CNN architecture (Frequency + Temporal + Spatial) with late fusion.
# Adapted from F_NET/F_Net_CNN/deepfake_training_pipeline.py for inference use.
#
# Model: MultiModalFusionClassifier
#   - FrequencyStreamCNN: (B, 8, 1, 256, 256) → 512D
#   - TemporalStreamCNN:  (B, 2, 256, 256)    → 512D
#   - SpatialStreamCNN:   (B, 2, 256, 256)    → 512D
#   - Fusion: 1536D → 256D → 128D → 2 classes
# ==============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ==============================================================================
# INFERENCE CONFIGURATION
# ==============================================================================

class FNetConfig:
    """Configuration for F-Net model (inference only)."""
    HIDDEN_DIM = 512
    FUSION_DIM = 256
    DROPOUT_RATE = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# STREAM ARCHITECTURES
# ==============================================================================

class FrequencyStreamCNN(nn.Module):
    """
    CNN for processing frequency (CSP) features.
    Input: (batch, 8, 1, 256, 256) - 8 scales, phase only
    Output: (batch, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()

        # Process each scale independently then combine
        self.scale_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256 -> 128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
        )

        # Combine scales
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(128 * 8, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        num_scales = x.size(1)

        # Process each scale
        scale_features = []
        for i in range(num_scales):
            scale = x[:, i, :, :, :]  # (batch, 1, 256, 256)
            feat = self.scale_conv(scale)  # (batch, 128, 32, 32)
            scale_features.append(feat)

        # Concatenate all scales
        combined = torch.cat(scale_features, dim=1)  # (batch, 128*8, 32, 32)

        # Fuse scales
        fused = self.scale_fusion(combined)  # (batch, 512, 1, 1)

        # Final FC
        output = self.fc(fused)  # (batch, hidden_dim)
        return output


class TemporalStreamCNN(nn.Module):
    """
    CNN for processing temporal features.
    Input: (batch, 2, 256, 256) - regional dynamics + global stability
    Output: (batch, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256 -> 64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class SpatialStreamCNN(nn.Module):
    """
    CNN for processing spatial features.
    Input: (batch, 2, 256, 256) - ViT embeddings + attention
    Output: (batch, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


# ==============================================================================
# MAIN MULTI-MODAL FUSION MODEL
# ==============================================================================

class MultiModalFusionClassifier(nn.Module):
    """
    F-Net: Three-stream CNN with late fusion.
    Combines frequency, temporal, and spatial streams.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = FNetConfig()

        self.frequency_stream = FrequencyStreamCNN(
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT_RATE
        )

        self.temporal_stream = TemporalStreamCNN(
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT_RATE
        )

        self.spatial_stream = SpatialStreamCNN(
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT_RATE
        )

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 3, config.FUSION_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE / 2)
        )

        # Classification head
        self.classifier = nn.Linear(128, 2)  # Binary: Real vs Fake

    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            features: Dict with 'frequency', 'temporal', 'spatial' tensors

        Returns:
            logits: (batch, 2)
            stream_outputs: Dict with individual stream outputs
        """
        freq_out = self.frequency_stream(features['frequency'])
        temp_out = self.temporal_stream(features['temporal'])
        spat_out = self.spatial_stream(features['spatial'])

        # Concatenate all streams
        combined = torch.cat([freq_out, temp_out, spat_out], dim=1)

        # Fusion
        fused = self.fusion(combined)

        # Classification
        logits = self.classifier(fused)

        stream_outputs = {
            'frequency': freq_out,
            'temporal': temp_out,
            'spatial': spat_out,
            'fused': fused
        }

        return logits, stream_outputs


# ==============================================================================
# MODEL LOADING (SINGLETON)
# ==============================================================================

from src.model_vit import MultiModalFusionClassifierViT, Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
FNET_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_vit_model.pth")
FALLBACK_MODEL_PATH = os.path.join(PROJECT_ROOT, "v2_balanced", "checkpoints", "best_vit_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_fnet_model = None


def get_fnet_model():
    """
    Lazy-load the F-Net model (singleton pattern).
    Model remains FROZEN — no training during inference.
    """
    global _fnet_model

    if _fnet_model is None:
        config = Config()
        _fnet_model = MultiModalFusionClassifierViT(config)

        model_path = FNET_MODEL_PATH if os.path.exists(FNET_MODEL_PATH) else FALLBACK_MODEL_PATH

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            if 'model_state_dict' in checkpoint:
                _fnet_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                _fnet_model.load_state_dict(checkpoint)
            print(f"[INFO] F-Net model loaded from {model_path} on {DEVICE}")
        else:
            print(f"[WARN] F-Net model weights not found at {FNET_MODEL_PATH} or {FALLBACK_MODEL_PATH}. Using uninitialized weights.")
            
        _fnet_model.to(DEVICE)
        _fnet_model.eval()

    return _fnet_model

