"""Migrated legacy EEG model module: tcnet.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# -----------------------------
# EEG 前端（轻量 EEGNet 风格）
# -----------------------------
class _EEGFrontend(nn.Module):
    """
    输入:  [B, 1, C, T]
    输出:  [B, F2, 1, T']  （F2 = F1 * D）
    """
    def __init__(
        self,
        in_chans: int = 26,
        F1: int = 16,
        D: int = 2,
        eeg_kernel: int = 32,
        sep_kernel: int = 16,
        pool1: int = 4,           # 调小默认池化，保留更多时间步
        pool2: int = 2,
        eeg_dropout: float = 0.2,
        eps: float = 1e-5,
    ):
        super().__init__()
        F2 = F1 * D

        # 1) 时域卷积
        self.conv_t = nn.Conv2d(
            in_channels=1, out_channels=F1,
            kernel_size=(1, eeg_kernel),
            padding=(0, eeg_kernel // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1, eps=eps)

        # 2) 空间深度可分离（通道方向）
        self.conv_s = nn.Conv2d(
            in_channels=F1, out_channels=F2,
            kernel_size=(in_chans, 1),
            groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2, eps=eps)
        self.act = nn.ELU()

        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1))
        self.drop1 = nn.Dropout(eeg_dropout) if eeg_dropout and eeg_dropout > 0 else nn.Identity()

        # 3) 简化的时域可分离（稳定做法：常规 1xk）
        self.conv_sep = nn.Conv2d(
            in_channels=F2, out_channels=F2,
            kernel_size=(1, sep_kernel),
            padding=(0, sep_kernel // 2),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2, eps=eps)

        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2))
        self.drop2 = nn.Dropout(eeg_dropout) if eeg_dropout and eeg_dropout > 0 else nn.Identity()

        # init
        nn.init.kaiming_normal_(self.conv_t.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_s.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_sep.weight, nonlinearity="relu")

        self.out_channels = F2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, C, T]
        x = self.conv_t(x);  x = self.bn1(x)
        x = self.conv_s(x);  x = self.bn2(x); x = self.act(x)
        x = self.pool1(x);   x = self.drop1(x)
        x = self.conv_sep(x); x = self.bn3(x); x = self.act(x)
        x = self.pool2(x);   x = self.drop2(x)
        return x  # [B, F2, 1, T']


# -----------------------------
# 因果卷积（左填充）
# -----------------------------
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=False):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=dilation, bias=bias,
        )
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x):
        # 只在左侧 pad，保证严格因果 & 等长输出
        x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)


# -----------------------------
# TemporalBlock：两层因果膨胀卷积 + 残差（TCN 标配）
# -----------------------------
class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        eps: float = 1e-5,
    ):
        super().__init__()
        # 两层因果卷积（weight norm）
        self.conv1 = weight_norm(CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=False))
        self.bn1   = nn.BatchNorm1d(out_channels, eps=eps)
        self.conv2 = weight_norm(CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation, bias=False))
        self.bn2   = nn.BatchNorm1d(out_channels, eps=eps)

        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # 残差投影
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) \
            if in_channels != out_channels else nn.Identity()

        # init（kaiming 对 ELU 也 OK）
        for m in [self.downsample]:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        y = self.conv1(x); y = self.bn1(y); y = self.act(y); y = self.drop(y)
        y = self.conv2(y); y = self.bn2(y); y = self.act(y); y = self.drop(y)
        res = self.downsample(x)
        return self.act(y + res)


# -----------------------------
# 注意力读出（可选）
# -----------------------------
class AttnPool1d(nn.Module):
    """
    简洁可学习时间注意力池化：
      s_t = u^T h_t；w = softmax(s)；y = Σ w_t h_t
    输入:  [B, C, T] -> 输出: [B, C]
    """
    def __init__(self, channels: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(channels))
        nn.init.normal_(self.u, std=0.02)

    def forward(self, x):
        # x: [B, C, T]
        scores = torch.einsum('bct,c->bt', x, self.u)              # [B, T]
        w = torch.softmax(scores, dim=-1).unsqueeze(1)             # [B, 1, T]
        y = torch.bmm(w, x.transpose(1, 2)).squeeze(1)             # [B, C]
        return y


# -----------------------------
# TCNet 顶层（前端 + TCN 堆叠 + 读出）
# -----------------------------
class TCNet(nn.Module):
    """
    输入: [B, C, T] 或 [B, 1, C, T]
    输出: [B, num_classes] 的 logits
    """
    def __init__(
        self,
        num_classes: int = 1,
        # EEG 前端
        in_chans: int = 26,
        F1: int = 16,
        D: int = 2,
        eeg_kernel: int = 32,
        sep_kernel: int = 16,
        pool1: int = 4,
        pool2: int = 2,
        eeg_dropout: float = 0.2,
        # TCN 堆叠
        tcn_channels: int = 128,
        tcn_layers: int = 8,          # 总“膨胀层数”，每层一个 dilation
        tcn_kernel: int = 5,
        dilation_base: int = 2,
        tcn_dropout: float = 0.2,
        # 统一接口
        dropout: float = 0.3,         # 分类层前的 Dropout
        readout: str = "attn",        # 'mean' | 'last' | 'attn'
        temporal_pool: Optional[int] = 1,  # 进入前端前的时间下采样
        pretrained: bool = False,     # 占位
    ):
        super().__init__()
        self.readout = readout.lower()
        assert self.readout in ["mean", "last", "attn"]

        self.pre_pool = (
            nn.AvgPool1d(kernel_size=temporal_pool, stride=temporal_pool)
            if temporal_pool is not None and temporal_pool > 1 else None
        )

        # EEG 前端
        self.frontend = _EEGFrontend(
            in_chans=in_chans, F1=F1, D=D,
            eeg_kernel=eeg_kernel, sep_kernel=sep_kernel,
            pool1=pool1, pool2=pool2, eeg_dropout=eeg_dropout
        )
        tcn_in = self.frontend.out_channels

        # [B, F2, 1, T'] -> [B, F2, T']
        self.squeeze_hw = nn.Identity()

        # TCN：堆叠 tcn_layers 个 dilation 层（每层两次因果卷积 + 残差）
        blocks = []
        in_c = tcn_in
        for i in range(tcn_layers):
            dil = dilation_base ** i
            blocks.append(
                TemporalBlock(
                    in_channels=in_c,
                    out_channels=tcn_channels,
                    kernel_size=tcn_kernel,
                    dilation=dil,
                    dropout=tcn_dropout,
                )
            )
            in_c = tcn_channels
        self.tcn = nn.Sequential(*blocks)

        # 读出
        if self.readout == "attn":
            self.pool = AttnPool1d(tcn_channels)
        else:
            self.pool = None

        self.pre_logits_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(tcn_channels, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 26, T] 或 [B, 1, 26, T]
        """
        if x.dim() == 3:        # [B, C, T]
            x = x.unsqueeze(1)  # -> [B, 1, C, T]
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Expected x with shape [B, C, T] or [B, 1, C, T], got {tuple(x.shape)}")

        B, _, C, T = x.shape
        # 进入前端前做时间降采样（可选）
        if self.pre_pool is not None:
            x = x.view(B, 1 * C, T)  # -> [B, C, T]
            x = self.pre_pool(x)     # -> [B, C, T']
            T2 = x.shape[-1]
            x = x.view(B, 1, C, T2)

        # EEG 前端
        x = self.frontend(x)          # [B, F2, 1, T']
        x = x.squeeze(2)              # [B, F2, T']

        # TCN 堆叠
        x = self.tcn(x)               # [B, tcn_channels, T'']

        # 读出
        if self.readout == "mean":
            feats = x.mean(dim=-1)            # [B, C]
        elif self.readout == "last":
            feats = x[..., -1]                # [B, C]
        else:  # 'attn'
            feats = self.pool(x)              # [B, C]

        feats = self.pre_logits_drop(feats)
        logits = self.classifier(feats)       # [B, num_classes]
        return logits


# -----------------------------
# 项目统一导出 Net
# -----------------------------
class Net(TCNet):
    def __init__(
        self,
        num_classes: int = 1,
        # EEG
        in_chans: int = 26,
        F1: int = 16,
        D: int = 2,
        eeg_kernel: int = 32,
        sep_kernel: int = 16,
        pool1: int = 4,
        pool2: int = 2,
        eeg_dropout: float = 0.2,
        # TCN
        tcn_channels: int = 128,
        tcn_layers: int = 8,
        tcn_kernel: int = 5,
        dilation_base: int = 2,
        tcn_dropout: float = 0.2,
        # 统一
        dropout: float = 0.3,
        readout: str = "attn",
        temporal_pool: Optional[int] = 1,
        pretrained: bool = False,
    ):
        super().__init__(
            num_classes=num_classes,
            in_chans=in_chans,
            F1=F1, D=D,
            eeg_kernel=eeg_kernel, sep_kernel=sep_kernel,
            pool1=pool1, pool2=pool2, eeg_dropout=eeg_dropout,
            tcn_channels=tcn_channels, tcn_layers=tcn_layers,
            tcn_kernel=tcn_kernel, dilation_base=dilation_base, tcn_dropout=tcn_dropout,
            dropout=dropout, readout=readout,
            temporal_pool=temporal_pool, pretrained=pretrained,
        )
