"""Muku V3 — quad-branch EEG + Spec + TCN model for NeuroMM-2026.

Adds a TCN (Temporal Convolutional Network) branch to muku V2. This is the
analog of HMS muku `eeg_rnn_spec` input_type, but using a TCN tower
(dilated 1D conv, parallel) instead of a GRU for sequence modeling.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromm26_baseline.models.legacy.tcnet import _EEGFrontend, TemporalBlock, AttnPool1d

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


class MukuEEGNetV3(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = "resnet18",
        backbone_spec: str | None = None,
        in_chans: int = 26,
        # 1D DW (raw branch)
        n_freqs: int = 8,
        fs: int = 500,
        time_pool: int = 8,
        target_size: int = 224,
        # Spec branch
        target_size_spec: int | None = None,
        spec_use_1ddw: bool = False,
        # TCN branch
        tcn_F1: int = 16,
        tcn_D: int = 2,
        tcn_channels: int = 128,
        tcn_layers: int = 8,
        tcn_kernel: int = 5,
        tcn_dilation_base: int = 2,
        tcn_dropout: float = 0.2,
        # Common
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
        backbone_spec = backbone_spec or backbone

        # ===== 1. 1D-DW for raw branch =====
        ksize = fs
        pad_l = (ksize - 1) // 2
        pad_r = ksize - 1 - pad_l
        self.pad = nn.ZeroPad2d((pad_l, pad_r, 0, 0))
        self.conv_t = nn.Conv2d(1, n_freqs, kernel_size=(1, ksize), padding=0, bias=False)
        self.bn = nn.BatchNorm2d(n_freqs)
        self.act = nn.SiLU()
        self.pool = nn.AvgPool2d((1, time_pool))

        # ===== 2. Two timm CNNs for the raw branch (view A / view B) =====
        self.backbone_a = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=0,
            drop_rate=dropout, drop_path_rate=dropout * 2,
        )
        self.backbone_b = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=0,
            drop_rate=dropout, drop_path_rate=dropout * 2,
        )
        # Probe actual feat dim (some timm models like mobilenetv3 have num_features != real output)
        with torch.no_grad():
            probe = torch.zeros(1, 1, target_size, target_size)
            feat_dim = self.backbone_a(probe).shape[-1]

        # ===== 3. Spec branch =====
        if spec_use_1ddw:
            self.pad_spec = nn.ZeroPad2d((7, 8, 0, 0))
            self.conv_t_spec = nn.Conv2d(1, 7, kernel_size=(1, 16), padding=0, bias=False)
            self.bn_spec = nn.BatchNorm2d(7)
            self.act_spec = nn.SiLU()
            spec_in_chans = 1
        else:
            spec_in_chans = in_chans
        self.backbone_spec = timm.create_model(
            backbone_spec, pretrained=pretrained, in_chans=spec_in_chans, num_classes=0,
            drop_rate=dropout, drop_path_rate=dropout * 2,
        )
        with torch.no_grad():
            probe = torch.zeros(1, spec_in_chans, self.target_size_spec, self.target_size_spec)
            feat_dim_spec = self.backbone_spec(probe).shape[-1]

        # ===== 4. TCN branch =====
        # EEGNet-style 1D frontend on raw EEG
        self.tcn_frontend = _EEGFrontend(
            in_chans=in_chans, F1=tcn_F1, D=tcn_D,
            eeg_kernel=32, sep_kernel=16, pool1=4, pool2=2,
            eeg_dropout=tcn_dropout,
        )
        tcn_in = self.tcn_frontend.out_channels   # F2 = F1*D
        # Stack of dilated TCN blocks
        blocks = []
        in_c = tcn_in
        for i in range(tcn_layers):
            dil = tcn_dilation_base ** i
            blocks.append(TemporalBlock(
                in_channels=in_c, out_channels=tcn_channels,
                kernel_size=tcn_kernel, dilation=dil, dropout=tcn_dropout,
            ))
            in_c = tcn_channels
        self.tcn = nn.Sequential(*blocks)
        # Attentive temporal pooling
        self.tcn_pool = AttnPool1d(tcn_channels)
        feat_dim_tcn = tcn_channels

        # ===== 5. Heads =====
        total_feat = feat_dim * 2 + feat_dim_tcn + feat_dim_spec
        self.fc_main = nn.Linear(total_feat, num_classes)
        self.fc_a = nn.Linear(feat_dim, num_classes)
        self.fc_b = nn.Linear(feat_dim, num_classes)
        self.fc_tcn = nn.Linear(feat_dim_tcn, num_classes)
        self.fc_spec = nn.Linear(feat_dim_spec, num_classes)

        # Pre-compute freq-first reorder indices
        Fn, C = self.n_freqs, self.in_chans
        reorder = torch.tensor(
            [(i % Fn) * C + (i // Fn) for i in range(Fn * C)],
            dtype=torch.long,
        )
        self.register_buffer("freq_reorder", reorder)

    def _make_eeg_views(self, raw):
        x = raw.unsqueeze(1)
        x = self.pad(x); x = self.conv_t(x); x = self.bn(x); x = self.act(x); x = self.pool(x)
        view_a = torch.cat([x[:, f] for f in range(x.shape[1])], dim=1).unsqueeze(1)
        view_b = view_a[:, :, self.freq_reorder, :]
        view_a = F.interpolate(view_a, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        view_b = F.interpolate(view_b, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return view_a, view_b

    def _make_spec_input(self, spec):
        if self.spec_use_1ddw:
            B, C, Fh, Tw = spec.shape
            xs = spec.reshape(B, 1, C * Fh, Tw)
            xs = self.pad_spec(xs); xs = self.conv_t_spec(xs); xs = self.bn_spec(xs); xs = self.act_spec(xs)
            xs = torch.cat([xs[:, k] for k in range(xs.shape[1])], dim=1).unsqueeze(1)
            xs = F.interpolate(xs, size=(self.target_size_spec, self.target_size_spec), mode="bilinear", align_corners=False)
            return xs
        return F.interpolate(spec, size=(self.target_size_spec, self.target_size_spec), mode="bilinear", align_corners=False)

    def _make_tcn_feat(self, raw):
        # raw: (B, C, T) → frontend (B, F2, 1, T') → squeeze → (B, F2, T')
        x = raw.unsqueeze(1)               # (B, 1, C, T)
        x = self.tcn_frontend(x)           # (B, F2, 1, T')
        x = x.squeeze(2)                   # (B, F2, T')
        x = self.tcn(x)                    # (B, tcn_channels, T'')
        feat = self.tcn_pool(x)            # (B, tcn_channels)
        return feat

    def forward(self, batch):
        raw = batch["eeg"]
        spec = batch["spec"]
        view_a, view_b = self._make_eeg_views(raw)
        feat_a = self.backbone_a(view_a)
        feat_b = self.backbone_b(view_b)

        feat_tcn = self._make_tcn_feat(raw)

        spec_in = self._make_spec_input(spec)
        feat_spec = self.backbone_spec(spec_in)

        main = self.fc_main(torch.cat([feat_a, feat_b, feat_tcn, feat_spec], dim=1))
        aux_a = self.fc_a(feat_a)
        aux_b = self.fc_b(feat_b)
        aux_tcn = self.fc_tcn(feat_tcn)
        aux_spec = self.fc_spec(feat_spec)
        if self.num_classes == 1:
            main = main.view(-1); aux_a = aux_a.view(-1); aux_b = aux_b.view(-1)
            aux_tcn = aux_tcn.view(-1); aux_spec = aux_spec.view(-1)
        return {"main": main, "aux_a": aux_a, "aux_b": aux_b,
                "aux_tcn": aux_tcn, "aux_spec": aux_spec}


def build_muku_eegnet_v3(
    task: str = "binary",
    backbone: str = "resnet18",
    backbone_spec: str | None = None,
    in_chans: int = 26,
    n_freqs: int = 8,
    fs: int = 500,
    tcn_layers: int = 8,
    pretrained: bool = True,
    dropout: float = 0.1,
) -> MukuEEGNetV3:
    num_classes = 1 if task == "binary" else 5
    return MukuEEGNetV3(
        num_classes=num_classes,
        backbone=backbone,
        backbone_spec=backbone_spec,
        in_chans=in_chans,
        n_freqs=n_freqs,
        fs=fs,
        tcn_layers=tcn_layers,
        pretrained=pretrained,
        dropout=dropout,
    )
