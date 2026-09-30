"""Migrated legacy EEG model module: actnet.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

# 代码在统计Param和FLOPs时报错，注释，只使用新代码统计参数量和浮点计算
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# 小工具
# -----------------------
def _same_pad_1d(k: int, d: int = 1) -> int:
    return (k - 1) // 2 * d


class SE2d(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hid = max(8, ch // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, ch), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, 1, T]
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        w = self.fc(s).view(b, c, 1, 1)
        return x * w


class SE1d(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hid = max(8, ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(ch, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, ch), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        b, c, _ = x.shape
        w = self.fc(x).view(b, c, 1)
        return x * w


# -----------------------
# EEG 前端：EEGNet 风格 + N 个可分离时域残差块（可大幅放大）
# -----------------------
class SepTemporalResBlock2d(nn.Module):
    """
    Depthwise-Temporal (1xK, groups=C) + PW + BN + GELU + SE + Dropout + 残差
    输入/输出: [B, C, 1, T]
    """
    def __init__(self, ch: int, k: int = 17, p_drop: float = 0.1, se_reduction: int = 8):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size=(1, k), padding=(0, k // 2), groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.GELU()
        self.se = SE2d(ch, se_reduction)
        self.do = nn.Dropout(p_drop)

        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.do(x)
        return x + res


class EEGFront(nn.Module):
    """
    输入: [B, 26, T] 或 [B, 1, 26, T]
    输出: [B, C_feat, T']
    结构:
      temporal(1xK) -> spatial(depthwise 跨通道) -> pointwise 扩展到 C=F1*D*expand
      -> 堆叠 N 个 SepTemporalResBlock2d（中间若干次池化）
    """
    def __init__(
        self,
        in_chans: int = 26,
        F1: int = 16,
        D: int = 2,
        eeg_kernel: int = 64,
        expand: int = 2,
        n_front_blocks: int = 4,
        sep_kernel: int = 17,
        pool_every: int = 2,      # 每多少个 block 做一次时间池化
        pool_size: int = 4,
        dropout: float = 0.2,
        use_se: bool = True,
    ):
        super().__init__()
        F2 = F1 * D
        C = F2 * max(1, int(expand))

        # temporal -> spatial -> pointwise 扩展
        self.conv_t = nn.Conv2d(1, F1, kernel_size=(1, eeg_kernel),
                                padding=(0, eeg_kernel // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv_s = nn.Conv2d(F1, F2, kernel_size=(in_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.act = nn.ELU()
        self.conv_pw = nn.Conv2d(F2, C, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(C)
        self.act_pw = nn.ELU()
        self.se0 = SE2d(C, 8) if use_se else nn.Identity()
        self.do0 = nn.Dropout(dropout)

        # 残差时域块
        blocks = []
        for i in range(n_front_blocks):
            blocks.append(SepTemporalResBlock2d(C, k=sep_kernel, p_drop=dropout, se_reduction=8))
            if (i + 1) % max(1, pool_every) == 0:
                blocks.append(nn.AvgPool2d(kernel_size=(1, pool_size)))
        self.blocks = nn.Sequential(*blocks)

        nn.init.kaiming_normal_(self.conv_t.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_s.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_pw.weight, nonlinearity="relu")

        self.out_ch = C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:   # [B, C, T] -> [B, 1, C, T]
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected [B, 26, T] or [B, 1, 26, T], got {tuple(x.shape)}")

        x = self.conv_t(x)
        x = self.bn1(x)
        x = self.conv_s(x)       # [B, F2, 1, T]
        x = self.bn2(x); x = self.act(x)

        x = self.conv_pw(x)      # [B, C, 1, T]
        x = self.bn_pw(x); x = self.act_pw(x)
        x = self.se0(x)
        x = self.do0(x)

        x = self.blocks(x)       # [B, C, 1, T']
        x = x.squeeze(2)         # [B, C, T']
        return x


# -----------------------
# Conformer 风格窗口内块：MHSA + Conv(GLU, DW-Conv) + SE + 残差（PreNorm）
# -----------------------
class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class ConvModule1d(nn.Module):
    def __init__(self, ch: int, k: int = 7, p_drop: float = 0.2, use_se: bool = True):
        super().__init__()
        self.pw1 = nn.Conv1d(ch, ch * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(ch, ch, kernel_size=k, padding=_same_pad_1d(k), groups=ch)
        self.bn = nn.BatchNorm1d(ch)
        self.swish = nn.SiLU()
        self.pw2 = nn.Conv1d(ch, ch, kernel_size=1)
        self.do = nn.Dropout(p_drop)
        self.se = SE1d(ch, 8) if use_se else nn.Identity()

        nn.init.kaiming_normal_(self.pw1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        y = self.pw1(x)
        y = self.glu(y)
        y = self.dw(y)
        y = self.bn(y)
        y = self.swish(y)
        y = self.pw2(y)
        y = self.se(y)
        y = self.do(y)
        return x + y


class PerWindowConformer(nn.Module):
    """
    输入: [B, C, Tw]；内部使用 batch_first Transformer (tokens=[B,Tw,C])
    """
    def __init__(self, ch: int, heads: int = 6, layers: int = 4, ff_mult: int = 4, conv_k: int = 7, p_drop: float = 0.2):
        super().__init__()
        d_model = ch
        ff_dim = ch * ff_mult
        enc_layers = []
        for _ in range(layers):
            mha = nn.MultiheadAttention(d_model, heads, dropout=p_drop, batch_first=True)
            ff = nn.Sequential(
                nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(p_drop),
                nn.Linear(ff_dim, d_model), nn.Dropout(p_drop),
            )
            attn = PreNorm(d_model, nn.Sequential(nn.Dropout(p_drop), nn.Identity()))
            ffpn = PreNorm(d_model, ff)

            enc_layers.append(nn.ModuleDict({
                "mha": mha,
                "attn_pn": nn.LayerNorm(d_model),
                "conv": ConvModule1d(ch, k=conv_k, p_drop=p_drop, use_se=True),
                "ff": ffpn
            }))
        self.layers = nn.ModuleList(enc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, Tw] -> tokens [B, Tw, C]
        tokens = x.transpose(1, 2)
        for blk in self.layers:
            # MHSA
            z = blk["attn_pn"](tokens)
            z, _ = blk["mha"](z, z, z, need_weights=False)
            tokens = tokens + z
            # Conv 模块（到 1d conv 前转为 [B,C,Tw]）
            y = tokens.transpose(1, 2)
            y = blk["conv"](y)
            tokens = y.transpose(1, 2)
            # FFN
            tokens = tokens + blk["ff"](tokens)
        return tokens.transpose(1, 2)  # [B, C, Tw]


# -----------------------
# 跨窗融合：CLS + 深层 Transformer
# -----------------------
class WindowFusion(nn.Module):
    def __init__(self, ch: int, heads: int = 8, layers: int = 4, ff_mult: int = 4, p_drop: float = 0.2, max_tokens: int = 1024):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, ch))
        self.pos = nn.Parameter(torch.zeros(1, max_tokens + 1, ch))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=ch, nhead=heads,
            dim_feedforward=ch * ff_mult, dropout=p_drop,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(ch)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, Nw, C]
        B, N, C = tokens.shape
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)  # [B, 1+N, C]
        x = x + self.pos[:, : 1 + N, :]
        x = self.enc(x)
        x = self.norm(x[:, 0])              # CLS
        return x


# -----------------------
# 主模型
# -----------------------
class Net(nn.Module):
    """
    - 输入: [B, 26, T] / [B, 1, 26, T]
    - 输出: logits [B, num_classes]
    - 分类器前统一 Dropout
    """
    def __init__(
        self,
        num_classes: int = 1,
        # EEG 前端（加大关键参数）
        F1: int = 24, D: int = 3, eeg_kernel: int = 96,
        expand: int = 4,                # 通道扩展倍数（非常影响参数量）
        n_front_blocks: int = 6,        # 前端可分离残差堆叠层数
        sep_kernel: int = 17,
        pool_every: int = 2, pool_size: int = 4,
        eeg_dropout: float = 0.2, use_se: bool = True,
        # 滑窗
        n_windows: int = 7, win_overlap: float = 0.5,
        # 窗内 Conformer
        t_layers: int = 6, t_heads: int = 8, t_ff_mult: int = 4, t_conv_k: int = 7, t_dropout: float = 0.2,
        # 跨窗融合
        fuse_layers: int = 4, fuse_heads: int = 8, fuse_ff_mult: int = 4, fuse_dropout: float = 0.2,
        # 统一 Dropout
        dropout: float = 0.3,
        # 其他
        in_chans: int = 26,
        temporal_pool: int = 1,
        readout: str = "cls",   # "cls" 或 "mean"
        pretrained: bool = False,
    ):
        super().__init__()
        assert readout in ["cls", "mean"]
        self.readout = readout
        self.n_windows = n_windows
        self.win_overlap = float(win_overlap)

        self.pre_pool = nn.AvgPool1d(temporal_pool, temporal_pool) if temporal_pool and temporal_pool > 1 else None

        # EEG 前端
        self.front = EEGFront(
            in_chans=in_chans, F1=F1, D=D, eeg_kernel=eeg_kernel,
            expand=expand, n_front_blocks=n_front_blocks,
            sep_kernel=sep_kernel, pool_every=pool_every, pool_size=pool_size,
            dropout=eeg_dropout, use_se=use_se
        )
        self.c_feat = self.front.out_ch

        # 窗内 Conformer
        self.per_win = PerWindowConformer(
            ch=self.c_feat, heads=t_heads, layers=t_layers, ff_mult=t_ff_mult, conv_k=t_conv_k, p_drop=t_dropout
        )

        # 跨窗融合
        self.fusion = WindowFusion(ch=self.c_feat, heads=fuse_heads, layers=fuse_layers,
                                   ff_mult=fuse_ff_mult, p_drop=fuse_dropout)

        self.pre_logits_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.c_feat, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    # 切滑窗 + 窗内建模 -> 每窗一个 token
    def _segment_windows(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        Nw = self.n_windows
        if Nw == 1:
            segs = [x]
        else:
            hop = max(1, int(round(T / (Nw - (Nw - 1) * self.win_overlap))))
            wlen = max(hop, int(round(hop / max(1e-6, 1 - self.win_overlap))))
            segs = []
            for i in range(Nw):
                st = i * hop
                ed = min(T, st + wlen)
                if ed - st <= 1:
                    segs.append(x[:, :, -1:].contiguous())
                else:
                    segs.append(x[:, :, st:ed].contiguous())

        tokens = []
        for seg in segs:
            y = self.per_win(seg)     # [B, C, Tw]
            tok = y.mean(dim=-1)      # [B, C]
            tokens.append(tok)
        tokens = torch.stack(tokens, dim=1)  # [B, Nw, C]
        tokens = F.layer_norm(tokens, (C,))
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容 [B,1,26,T]
        if x.dim() == 4:
            x = x.squeeze(1)  # -> [B,26,T]
        if self.pre_pool is not None:
            x = self.pre_pool(x)  # [B,26,T']

        feat = self.front(x)               # [B, C, T']
        tokens = self._segment_windows(feat)  # [B, Nw, C]
        fused = self.fusion(tokens)        # [B, C]
        if self.readout == "mean":
            fused = 0.5 * (fused + tokens.mean(dim=1))
        fused = self.pre_logits_drop(fused)
        logits = self.classifier(fused)    # [B, num_classes]
        return logits
