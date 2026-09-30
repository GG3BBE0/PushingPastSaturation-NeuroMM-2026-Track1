from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


class SpecCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = "resnet18",
        in_chans: int = 26,
        target_size: int = 224,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm required")
        self.num_classes = num_classes
        self.target_size = target_size
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=in_chans, num_classes=0,
            drop_rate=dropout, drop_path_rate=dropout * 2,
        )
        feat_dim = self.backbone.num_features
        self.fc_main = nn.Linear(feat_dim, num_classes)

    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch["spec"]  # (B, C, F, T)
        else:
            x = batch
        # Resize to target_size × target_size
        x = F.interpolate(x, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)
        feat = self.backbone(x)
        logits = self.fc_main(feat)
        if self.num_classes == 1:
            logits = logits.view(-1)
        return {"main": logits}
