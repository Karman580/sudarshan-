import torch
import torch.nn as nn
import timm


class ViTVideoClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0  # remove ImageNet head
        )

        self.classifier = nn.Linear(768, 1)

    def forward(self, x):
        """
        x: (batch, num_faces, 3, 224, 224)
        """

        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        features = self.vit(x)          # (B*T, 768)
        features = features.view(B, T, -1)
        features = features.mean(dim=1) # video-level

        out = self.classifier(features)
        return out
