from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


class ConcatSpecCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = "maxvit_tiny_tf_384.in1k",
        target_size: int = 384,
        pretrained: bool = True,
        dropout: float = 0.1,
        drop_path: float | None = None,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm required")
        self.num_classes = num_classes
        self.target_size = target_size
        # drop_path defaults to dropout*2 (back-compat); override for big backbones
        dpr = dropout * 2 if drop_path is None else drop_path
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=0,
            drop_rate=dropout, drop_path_rate=dpr,
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, batch):
        x = batch["spec"] if isinstance(batch, dict) else batch  # (B, C, F, T)
        B, C, H, W = x.shape
        # Concat channels vertically: (B, 1, C*H, W)
        x = x.reshape(B, 1, C * H, W)
        # Resize to target_size × target_size for backbone
        x = F.interpolate(x, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)
        feat = self.backbone(x)
        logits = self.head(feat)
        if self.num_classes == 1:
            logits = logits.view(-1)
        return {"main": logits}
