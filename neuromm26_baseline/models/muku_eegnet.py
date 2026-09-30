"""
Muku-style multi-branch EEG model — adapted from HMS muku for NeuroMM-2026.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


class MukuEEGNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = "resnet18",
        in_chans: int = 26,
        n_freqs: int = 8,
        fs: int = 500,
        time_pool: int = 8,
        target_size: int = 224,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("timm is required; pip install timm")

        self.num_classes = num_classes
        self.in_chans = in_chans
        self.n_freqs = n_freqs
        self.fs = fs
        self.time_pool = time_pool
        self.target_size = target_size

        # ---------- 1. 1D temporal conv "DW block" ----------
        # kernel = (1, fs) means "1 second" of temporal conv per channel
        ksize = fs
        pad_l = (ksize - 1) // 2
        pad_r = ksize - 1 - pad_l
        self.pad = nn.ZeroPad2d((pad_l, pad_r, 0, 0))
        self.conv_t = nn.Conv2d(
            in_channels=1, out_channels=n_freqs,
            kernel_size=(1, ksize), padding=0, bias=False,
        )
        self.bn = nn.BatchNorm2d(n_freqs)
        self.act = nn.SiLU()
        self.pool = nn.AvgPool2d((1, time_pool))

        # ---------- 2. Two 2D CNN backbones (separate weights) ----------
        self.backbone_a = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=0,
            drop_rate=dropout, drop_path_rate=dropout * 2,
        )
        self.backbone_b = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=0,
            drop_rate=dropout, drop_path_rate=dropout * 2,
        )
        feat_dim = self.backbone_a.num_features

        # ---------- 3. Heads ----------
        self.fc_main = nn.Linear(feat_dim * 2, num_classes)
        self.fc_a = nn.Linear(feat_dim, num_classes)
        self.fc_b = nn.Linear(feat_dim, num_classes)

        # Pre-compute freq-first reorder indices
        # View A rows (linear): f*C + c  (f = freq idx, c = channel idx)
        # View B rows (target): c*F + f
        # For row i of view B → original = (i % F) * C + (i // F)
        Fn, C = self.n_freqs, self.in_chans
        reorder = torch.tensor(
            [(i % Fn) * C + (i // Fn) for i in range(Fn * C)],
            dtype=torch.long,
        )
        self.register_buffer("freq_reorder", reorder)

    def _make_image(self, raw):
        """raw: (B, C, T) → conv-pool → view_a (B, 1, F*C, T'), view_b (B, 1, F*C, T')."""
        x = raw.unsqueeze(1)                           # (B, 1, C, T)
        x = self.pad(x)
        x = self.conv_t(x)                             # (B, F, C, T)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)                               # (B, F, C, T/pool)
        # View A "channel-first": stack freqs vertically, channels inside each block
        view_a = torch.cat([x[:, f] for f in range(x.shape[1])], dim=1)  # (B, F*C, T')
        view_a = view_a.unsqueeze(1)                                     # (B, 1, F*C, T')
        # View B "freq-first": rows are channel-major (freqs of each channel together)
        view_b = view_a[:, :, self.freq_reorder, :]
        # Resize both to target_size × target_size for backbone
        view_a = F.interpolate(view_a, size=(self.target_size, self.target_size),
                               mode="bilinear", align_corners=False)
        view_b = F.interpolate(view_b, size=(self.target_size, self.target_size),
                               mode="bilinear", align_corners=False)
        return view_a, view_b

    def forward(self, batch):
        """batch: dict with 'eeg' tensor of shape (B, C, T).

        Returns dict with 'main', 'aux_a', 'aux_b' logits.
        At inference time, only 'main' is used.
        """
        if isinstance(batch, dict):
            raw = batch["eeg"]
        else:
            raw = batch
        view_a, view_b = self._make_image(raw)
        feat_a = self.backbone_a(view_a)
        feat_b = self.backbone_b(view_b)
        main = self.fc_main(torch.cat([feat_a, feat_b], dim=1))
        aux_a = self.fc_a(feat_a)
        aux_b = self.fc_b(feat_b)
        if self.num_classes == 1:
            main = main.view(-1)
            aux_a = aux_a.view(-1)
            aux_b = aux_b.view(-1)
        return {"main": main, "aux_a": aux_a, "aux_b": aux_b}


def build_muku_eegnet(
    task: str = "binary",  # "binary" or "task3"
    backbone: str = "resnet18",
    in_chans: int = 26,
    n_freqs: int = 8,
    fs: int = 500,
    pretrained: bool = True,
    dropout: float = 0.1,
) -> MukuEEGNet:
    num_classes = 1 if task == "binary" else 5
    return MukuEEGNet(
        num_classes=num_classes,
        backbone=backbone,
        in_chans=in_chans,
        n_freqs=n_freqs,
        fs=fs,
        pretrained=pretrained,
        dropout=dropout,
    )
