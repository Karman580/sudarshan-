"""
Neural Architecture Search — Searchable Supernet Model
=======================================================
Every module is parameterised by an `arch` dict (see nas_search_space.py).
During supernet training a random arch is sampled each iteration;
during evaluation a fixed arch is applied.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


def _make_conv_block(in_c: int, out_c: int, kernel: int = 3,
                     pool: str = "max", use_se: bool = False) -> nn.Sequential:
    """Conv-BN-ReLU-Pool, optionally followed by SE attention."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=kernel // 2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if pool == "max":
        layers.append(nn.MaxPool2d(2))
    else:
        layers.append(nn.AvgPool2d(2))
    if use_se:
        layers.append(SEBlock(out_c))
    return nn.Sequential(*layers)


# ============================================================================
# SEARCHABLE FREQUENCY STREAM
# ============================================================================

class SearchableFrequencyStream(nn.Module):
    def __init__(self, arch: Dict[str, Any]):
        super().__init__()
        channels = list(arch["freq_channels"])  # copy
        num_blocks = arch["freq_num_blocks"]
        pool = arch["freq_pool_type"]
        use_se = arch["freq_use_se"]
        hidden = arch["freq_hidden_dim"]
        dropout = arch["dropout"]

        # Ensure we have enough channel entries; pad with last value
        while len(channels) < num_blocks:
            channels.append(channels[-1])
        channels = channels[:num_blocks]

        # Per-scale conv tower
        blocks = []
        in_c = 1
        for c in channels:
            blocks.append(_make_conv_block(in_c, c, kernel=3, pool=pool, use_se=use_se))
            in_c = c
        self.scale_conv = nn.Sequential(*blocks)

        # Fusion across 8 scales
        fused_c = channels[-1] * 8
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(fused_c, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, 1, 256, 256)
        scale_features = [self.scale_conv(x[:, i]) for i in range(x.size(1))]
        fused = torch.cat(scale_features, dim=1)
        return self.fc(self.scale_fusion(fused))


# ============================================================================
# SEARCHABLE TEMPORAL STREAM
# ============================================================================

class SearchableTemporalStream(nn.Module):
    def __init__(self, arch: Dict[str, Any]):
        super().__init__()
        channels = list(arch["temp_channels"])
        num_blocks = arch["temp_num_blocks"]
        first_k = arch["temp_first_kernel"]
        use_se = arch["temp_use_se"]
        hidden = arch["temp_hidden_dim"]
        dropout = arch["dropout"]

        while len(channels) < num_blocks:
            channels.append(channels[-1])
        channels = channels[:num_blocks]

        blocks: list[nn.Module] = []
        in_c = 2  # temporal has 2 channels
        for i, c in enumerate(channels):
            k = first_k if i == 0 else 3
            stride = 2 if i == 0 else 1
            blocks.append(nn.Sequential(
                nn.Conv2d(in_c, c, kernel_size=k, stride=stride, padding=k // 2),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                *([] if not use_se else [SEBlock(c)]),
            ))
            in_c = c

        blocks.append(nn.AdaptiveAvgPool2d(1))
        self.conv_layers = nn.Sequential(*blocks)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv_layers(x))


# ============================================================================
# SEARCHABLE SPATIAL STREAM (ViT)
# ============================================================================

class SearchableSpatialStream(nn.Module):
    def __init__(self, arch: Dict[str, Any]):
        super().__init__()
        act_channels = list(arch["spatial_act_channels"])
        act_blocks = arch["spatial_act_blocks"]
        cls_proj = arch["spatial_cls_proj_dim"]
        merge = arch["spatial_merge_mode"]
        hidden = arch["spatial_hidden_dim"]
        dropout = arch["dropout"]

        while len(act_channels) < act_blocks:
            act_channels.append(act_channels[-1])
        act_channels = act_channels[:act_blocks]

        # Attention-map conv tower
        blocks: list[nn.Module] = []
        in_c = 1
        for c in act_channels:
            blocks.extend([
                nn.Conv2d(in_c, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_c = c
        blocks.append(nn.AdaptiveAvgPool2d(1))
        self.act_conv = nn.Sequential(*blocks)
        act_out = act_channels[-1]

        # CLS token projection
        self.cls_proj = nn.Sequential(
            nn.Linear(768, cls_proj),
            nn.ReLU(inplace=True),
        )

        # Merge mode
        self.merge_mode = merge
        if merge == "concat":
            fc_in = cls_proj + act_out
        elif merge == "gate":
            # gate acts on cls_proj dimension
            fc_in = cls_proj
            self.gate_proj = nn.Sequential(
                nn.Linear(act_out, cls_proj),
                nn.Sigmoid(),
            )
        elif merge == "bilinear":
            self.bilinear = nn.Bilinear(cls_proj, act_out, cls_proj)
            fc_in = cls_proj

        self.fc = nn.Sequential(
            nn.Linear(fc_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, cls_token: torch.Tensor,
                act_map: torch.Tensor) -> torch.Tensor:
        act_feat = self.act_conv(act_map).flatten(1)
        cls_feat = self.cls_proj(cls_token)

        if self.merge_mode == "concat":
            combined = torch.cat([cls_feat, act_feat], dim=1)
        elif self.merge_mode == "gate":
            gate = self.gate_proj(act_feat)
            combined = cls_feat * gate
        elif self.merge_mode == "bilinear":
            combined = self.bilinear(cls_feat, act_feat)

        return self.fc(combined)


# ============================================================================
# SEARCHABLE FUSION HEAD
# ============================================================================

class SearchableFusionHead(nn.Module):
    def __init__(self, stream_dims: List[int], arch: Dict[str, Any]):
        super().__init__()
        strategy = arch["fusion_strategy"]
        hidden_layers = arch["fusion_hidden_layers"]
        use_bn = arch["fusion_use_bn"]
        dropout = arch["dropout"]
        self.strategy = strategy

        total_in = sum(stream_dims)

        if strategy == "attention":
            self.attn_fc = nn.Linear(total_in, len(stream_dims))
            fc_in = stream_dims[0]  # all streams projected to same dim
            # Project each stream to same dim for weighted sum
            self.stream_projs = nn.ModuleList([
                nn.Linear(d, stream_dims[0]) for d in stream_dims
            ])
        elif strategy == "gated":
            self.gate_fc = nn.Linear(total_in, total_in)
            fc_in = total_in
        else:  # concat
            fc_in = total_in

        # MLP head
        layers: list[nn.Module] = []
        in_d = fc_in
        for h in hidden_layers:
            layers.append(nn.Linear(in_d, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_d = h

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_d, 2)

    def forward(self, stream_outputs: List[torch.Tensor]) -> torch.Tensor:
        cat = torch.cat(stream_outputs, dim=1)

        if self.strategy == "attention":
            weights = F.softmax(self.attn_fc(cat), dim=1)  # (B, num_streams)
            projected = [proj(s) for proj, s in zip(self.stream_projs, stream_outputs)]
            stacked = torch.stack(projected, dim=1)  # (B, num_streams, D)
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        elif self.strategy == "gated":
            gate = torch.sigmoid(self.gate_fc(cat))
            fused = cat * gate
        else:
            fused = cat

        hidden = self.mlp(fused)
        return self.classifier(hidden), hidden


# ============================================================================
# FULL NAS MULTI-MODAL CLASSIFIER
# ============================================================================

class NASMultiModalClassifier(nn.Module):
    """Supernet — architecture defined entirely by `arch` dict."""
    def __init__(self, arch: Dict[str, Any]):
        super().__init__()
        self.arch = arch
        self.frequency_stream = SearchableFrequencyStream(arch)
        self.temporal_stream = SearchableTemporalStream(arch)
        self.spatial_stream = SearchableSpatialStream(arch)

        stream_dims = [
            arch["freq_hidden_dim"],
            arch["temp_hidden_dim"],
            arch["spatial_hidden_dim"],
        ]
        self.fusion_head = SearchableFusionHead(stream_dims, arch)

    def forward(self, features: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        freq_out = self.frequency_stream(features["frequency"])
        temp_out = self.temporal_stream(features["temporal"])
        spat_out = self.spatial_stream(features["spatial_cls"], features["spatial_act"])

        logits, fused = self.fusion_head([freq_out, temp_out, spat_out])

        return logits, {
            "frequency": freq_out,
            "temporal": temp_out,
            "spatial": spat_out,
            "fused": fused,
        }
