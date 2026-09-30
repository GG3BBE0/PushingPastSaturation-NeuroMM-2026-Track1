"""Migrated legacy EEG model module: lmda.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

# models/LMDA.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EEGDepthAttention(nn.Module):
    """
    Depth attention over feature channels for maps [B, C, 1, W].
    去除 *C 的放大，避免尺度爆炸。
    """
    def __init__(self, W: int, C: int, k: int = 7):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))          # 保持时间长度
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1),
                              padding=(k // 2, 0), bias=True)       # 沿“通道”轴卷积
        self.softmax = nn.Softmax(dim=-2)                           # 在通道维归一化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, 1, W]
        x_pool = self.adaptive_pool(x)                              # [B, C, 1, W]
        attn_in = x_pool.transpose(-2, -3)                          # [B, 1, C, W]
        a = self.conv(attn_in)                                      # [B, 1, C, W]
        a = self.softmax(a)                                         # 通道注意力
        a = a.transpose(-2, -3)                                     # [B, C, 1, W]
        return x * a                                                # 不再乘 C


class Net(nn.Module):
    """
    LMDAv2（稳定增强版）
    - 输入: [B, 26, T] 或 [B, 1, 26, T]
    - 输出: logits [B, num_classes]
    - 统一: temporal_pool, dropout, pretrained 占位
    """
    def __init__(
        self,
        num_classes: int = 1,
        in_chans: int = 26,
        samples: int = 2000,
        # LMDA 主体超参
        depth: int = 9,
        kernel: int = 51,             # 略缩小默认核，减轻过度平滑（原 75）
        channel_depth1: int = 64,     # 稍加宽以提升表示
        channel_depth2: int = 32,
        avepool: int = 3,             # 温和一些
        da_kernel: int = 7,
        # 统一接口
        dropout: float = 0.3,         # 分类前 dropout
        temporal_pool: int = 1,       # >1 则先时域降采样
        pretrained: bool = False,     # 占位
        mlp_hidden: int = 512,        # 分类头隐藏维
    ):
        super().__init__()
        self.in_chans = in_chans
        self.samples = samples
        self.depth = depth
        self.c1 = channel_depth1
        self.c2 = channel_depth2
        self.kernel = kernel
        self.avepool = avepool
        self.da_kernel = da_kernel
        self.mlp_hidden = mlp_hidden

        # 入口时域降采样（2D 更安全）
        self.pre_pool = (
            nn.AvgPool2d(kernel_size=(1, temporal_pool), stride=(1, temporal_pool))
            if temporal_pool and temporal_pool > 1 else None
        )

        # 可学习的“多深度×导联”权重：先作为logits，前向里做softmax归一化（跨导联）
        self.channel_weight_logit = nn.Parameter(torch.zeros(depth, 1, in_chans))
        # 小扰动有助于破对称
        nn.init.normal_(self.channel_weight_logit, mean=0.0, std=0.01)

        # —— 时间卷积分支（含残差） —— #
        self.time_pw1 = nn.Conv2d(depth, self.c1, kernel_size=(1, 1), bias=False)
        self.time_pw1_bn = nn.BatchNorm2d(self.c1)
        self.time_dw = nn.Conv2d(self.c1, self.c1, kernel_size=(1, self.kernel),
                                 groups=self.c1, bias=False, padding=(0, self.kernel // 2))
        self.time_dw_bn = nn.BatchNorm2d(self.c1)
        self.time_act = nn.GELU()
        # 残差：把原始 depth 通道映射到 c1 并加回
        self.time_res = nn.Conv2d(depth, self.c1, kernel_size=1, bias=False)
        self.time_res_bn = nn.BatchNorm2d(self.c1)

        # 先跑一次到 time_conv 输出，给注意力构造形状
        W_after, C_after = self._infer_temporal_out_shape(samples)
        self.depthAttention = EEGDepthAttention(W=W_after, C=C_after, k=self.da_kernel)

        # —— 通道卷积分支 —— #
        self.chan_pw1 = nn.Conv2d(self.c1, self.c2, kernel_size=1, bias=False)
        self.chan_pw1_bn = nn.BatchNorm2d(self.c2)
        self.chan_spatial = nn.Conv2d(self.c2, self.c2, kernel_size=(in_chans, 1),
                                      groups=self.c2, bias=False)
        self.chan_spatial_bn = nn.BatchNorm2d(self.c2)
        self.chan_act = nn.GELU()

        # 尾部聚合：更温和 + 自适应
        self.tail_pool = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, max(1, self.avepool))),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # 分类头：MLP
        flat_dim = self._infer_flat_dim(samples)
        self.pre_logits_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(flat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(mlp_hidden, num_classes)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _infer_temporal_out_shape(self, T: int):
        dev = self.channel_weight_logit.device
        x = torch.zeros(1, 1, self.in_chans, T, device=dev)
        if self.pre_pool is not None:
            x = self.pre_pool(x)
        # 归一化后的权重（跨导联 softmax）
        w = F.softmax(self.channel_weight_logit, dim=-1)             # [depth,1,C]
        x = torch.einsum('bdcw,hdc->bhcw', x, w)                     # [1, depth, C, T]
        # time branch + 残差
        y_main = self.time_pw1_bn(self.time_pw1(x))
        y_main = self.time_dw_bn(self.time_dw(y_main))
        y_res  = self.time_res_bn(self.time_res(x))
        y = self.time_act(y_main + y_res)                            # [1, c1, C, W]
        _, C_after, _, W_after = y.shape
        return W_after, C_after

    @torch.no_grad()
    def _infer_flat_dim(self, T: int) -> int:
        dev = self.channel_weight_logit.device
        x = torch.zeros(1, 1, self.in_chans, T, device=dev)
        if self.pre_pool is not None:
            x = self.pre_pool(x)
        w = F.softmax(self.channel_weight_logit, dim=-1)
        x = torch.einsum('bdcw,hdc->bhcw', x, w)
        y_main = self.time_pw1_bn(self.time_pw1(x))
        y_main = self.time_dw_bn(self.time_dw(y_main))
        y_res  = self.time_res_bn(self.time_res(x))
        y = self.time_act(y_main + y_res)                            # [1, c1, C, W]
        y = self.depthAttention(y)                                   # [1, c1, C, W]
        y = self.chan_pw1_bn(self.chan_pw1(y))
        y = self.chan_spatial_bn(self.chan_spatial(y))
        y = self.chan_act(y)                                         # [1, c2, 1, W]
        y = self.tail_pool(y)                                        # [1, c2, 1, 1]
        return int(y.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 26, T] 或 [B, 1, 26, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)                                       # [B,1,C,T]
        elif x.dim() != 4:
            raise ValueError(f"Expected [B,26,T] or [B,1,26,T], got {tuple(x.shape)}")

        if self.pre_pool is not None:
            x = self.pre_pool(x)

        # 归一化导联权重
        w = F.softmax(self.channel_weight_logit, dim=-1)             # [depth,1,C]
        x = torch.einsum('bdcw,hdc->bhcw', x, w)                     # [B,depth,C,T']

        # 时间卷积分支（含残差）
        y_main = self.time_pw1_bn(self.time_pw1(x))
        y_main = self.time_dw_bn(self.time_dw(y_main))
        y_res  = self.time_res_bn(self.time_res(x))
        y = self.time_act(y_main + y_res)                            # [B,c1,C,W]

        # 深度注意力
        y = self.depthAttention(y)                                   # [B,c1,C,W]

        # 通道卷积分支
        y = self.chan_pw1_bn(self.chan_pw1(y))
        y = self.chan_spatial_bn(self.chan_spatial(y))
        y = self.chan_act(y)                                         # [B,c2,1,W]

        # 轻量尾部聚合
        y = self.tail_pool(y)                                        # [B,c2,1,1]
        feats = y.flatten(1)                                         # [B, c2]
        feats = self.pre_logits_drop(feats)
        feats = self.head(feats)
        logits = self.classifier(feats)
        return logits
