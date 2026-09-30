"""Muku V2 — dual-branch EEG + Spec model for NeuroMM-2026.

Adds a third backbone path on a precomputed time-frequency image (CWT/STFT/superlet),
mirroring HMS muku  `input_type='eeg_spec'` architecture.

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


class MukuEEGNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = "resnet18",
        backbone_spec: str | None = None,
        in_chans: int = 26,
        n_freqs: int = 8,
        fs: int = 500,
        time_pool: int = 8,
        target_size: int = 224,
        target_size_spec: int | None = None,
        spec_use_1ddw: bool = False,
        spec_n_freqs_dw: int = 7,
        spec_dw_kernel: int = 16,
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
        self.target_size_spec = target_size_spec if target_size_spec is not None else target_size
        self.spec_use_1ddw = spec_use_1ddw
        self.spec_n_freqs_dw = spec_n_freqs_dw

        backbone_spec = backbone_spec or backbone

        # ===== 1. 1D temporal conv DW block =====
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

        # ===== 2. Two 2D CNN backbones for raw EEG (channel-first / freq-first) =====
        self.backbone_a = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=0,
            drop_rate=dropout, drop_path_rate=dropout * 2,
        )
        self.backbone_b = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=0,
            drop_rate=dropout, drop_path_rate=dropout * 2,
        )
        feat_dim = self.backbone_a.num_features

        # ===== 3. Spec branch (3rd timm CNN, separate weights) =====
        if spec_use_1ddw:
            # Optional learned 1D filter on the spec time axis (HMS spec_1ddw variant)
            pad_l_s = (spec_dw_kernel - 1) // 2
            pad_r_s = spec_dw_kernel - 1 - pad_l_s
            self.pad_spec = nn.ZeroPad2d((pad_l_s, pad_r_s, 0, 0))
            self.conv_t_spec = nn.Conv2d(
                in_channels=1, out_channels=spec_n_freqs_dw,
                kernel_size=(1, spec_dw_kernel), padding=0, bias=False,
            )
            self.bn_spec = nn.BatchNorm2d(spec_n_freqs_dw)
            self.act_spec = nn.SiLU()
            spec_in_chans = 1  # after DW reshape, we feed as single-channel image
        else:
            spec_in_chans = in_chans  # plain (B, C, F, T)

        self.backbone_spec = timm.create_model(
            backbone_spec, pretrained=pretrained, in_chans=spec_in_chans, num_classes=0,
            drop_rate=dropout, drop_path_rate=dropout * 2,
        )
        feat_dim_spec = self.backbone_spec.num_features

        # ===== 4. Heads =====
        self.fc_main = nn.Linear(feat_dim * 2 + feat_dim_spec, num_classes)
        self.fc_a = nn.Linear(feat_dim, num_classes)
        self.fc_b = nn.Linear(feat_dim, num_classes)
        self.fc_spec = nn.Linear(feat_dim_spec, num_classes)

        # Pre-compute freq-first reorder indices for view B
        Fn, C = self.n_freqs, self.in_chans
        reorder = torch.tensor(
            [(i % Fn) * C + (i // Fn) for i in range(Fn * C)],
            dtype=torch.long,
        )
        self.register_buffer("freq_reorder", reorder)

    def _make_eeg_views(self, raw):
        """raw (B,C,T) → view_a/view_b (B,1,target,target)."""
        x = raw.unsqueeze(1)                              # (B,1,C,T)
        x = self.pad(x)
        x = self.conv_t(x)                                # (B,F,C,T)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)                                  # (B,F,C,T')
        view_a = torch.cat([x[:, f] for f in range(x.shape[1])], dim=1)  # (B,F*C,T')
        view_a = view_a.unsqueeze(1)                                     # (B,1,F*C,T')
        view_b = view_a[:, :, self.freq_reorder, :]
        view_a = F.interpolate(view_a, size=(self.target_size, self.target_size),
                               mode="bilinear", align_corners=False)
        view_b = F.interpolate(view_b, size=(self.target_size, self.target_size),
                               mode="bilinear", align_corners=False)
        return view_a, view_b

    def _make_spec_input(self, spec):
        """spec (B,C,F,T') → (B,Cin,H,W) resized to target_size_spec."""
        if self.spec_use_1ddw:
            # Apply 1D DW per channel along time, then stack DW outputs vertically
            B, C, Fh, Tw = spec.shape
            # Treat (C,F) as rows so DW kernel runs along time axis
            xs = spec.reshape(B, 1, C * Fh, Tw)
            xs = self.pad_spec(xs)
            xs = self.conv_t_spec(xs)                                     # (B, n_dw, C*F, Tw)
            xs = self.bn_spec(xs); xs = self.act_spec(xs)
            # Stack DW filters vertically: (B,1,n_dw*C*F,Tw)
            xs = torch.cat([xs[:, k] for k in range(xs.shape[1])], dim=1).unsqueeze(1)
            xs = F.interpolate(xs, size=(self.target_size_spec, self.target_size_spec),
                               mode="bilinear", align_corners=False)
            return xs
        # Plain path: feed (B,C,F,T) as multi-channel image
        xs = F.interpolate(spec, size=(self.target_size_spec, self.target_size_spec),
                           mode="bilinear", align_corners=False)
        return xs

    def forward(self, batch):
        """batch: dict with keys 'eeg' (B,C,T) and 'spec' (B,C,F,T').

        Returns dict with 'main', 'aux_a', 'aux_b', 'aux_spec' logits.
        """
        raw = batch["eeg"]
        spec = batch["spec"]
        view_a, view_b = self._make_eeg_views(raw)
        feat_a = self.backbone_a(view_a)
        feat_b = self.backbone_b(view_b)

        spec_in = self._make_spec_input(spec)
        feat_spec = self.backbone_spec(spec_in)

        main = self.fc_main(torch.cat([feat_a, feat_b, feat_spec], dim=1))
        aux_a = self.fc_a(feat_a)
        aux_b = self.fc_b(feat_b)
        aux_spec = self.fc_spec(feat_spec)
        if self.num_classes == 1:
            main = main.view(-1)
            aux_a = aux_a.view(-1)
            aux_b = aux_b.view(-1)
            aux_spec = aux_spec.view(-1)
        return {"main": main, "aux_a": aux_a, "aux_b": aux_b, "aux_spec": aux_spec}


def build_muku_eegnet_v2(
    task: str = "binary",
    backbone: str = "resnet18",
    backbone_spec: str | None = None,
    in_chans: int = 26,
    n_freqs: int = 8,
    fs: int = 500,
    spec_use_1ddw: bool = False,
    pretrained: bool = True,
    dropout: float = 0.1,
) -> MukuEEGNetV2:
    num_classes = 1 if task == "binary" else 5
    return MukuEEGNetV2(
        num_classes=num_classes,
        backbone=backbone,
        backbone_spec=backbone_spec,
        in_chans=in_chans,
        n_freqs=n_freqs,
        fs=fs,
        spec_use_1ddw=spec_use_1ddw,
        pretrained=pretrained,
        dropout=dropout,
    )
