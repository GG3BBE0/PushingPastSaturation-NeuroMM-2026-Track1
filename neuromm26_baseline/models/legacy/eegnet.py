"""Migrated legacy EEG model module: eegnet.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- Squeeze-Excitation (轻量注意力，可选) ---------
class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(ch, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, 1, T]
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


# --------- 可堆叠的“可分离时域卷积”模块（DW-Conv over time + PW-Conv + BN + Act + Dropout + 可选SE） ---------
class SeparableTemporalBlock(nn.Module):
    def __init__(
        self,
        ch: int,                      # 输入/输出通道（保持恒定，便于堆叠）
        sep_kernel: int = 16,
        dropout: float = 0.0,
        act: str = "elu",
        use_se: bool = False,
        se_reduction: int = 8,
        eps: float = 1e-5,
    ):
        super().__init__()
        if act.lower() == "elu":
            act_layer = nn.ELU()
        elif act.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {act}")

        self.dw = nn.Conv2d(
            in_channels=ch, out_channels=ch,
            kernel_size=(1, sep_kernel),
            padding=(0, sep_kernel // 2),
            groups=ch, bias=False
        )
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(ch, eps=eps)
        self.act = act_layer
        self.do = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.se = SEBlock(ch, se_reduction) if use_se else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw.weight, nonlinearity="relu")

    def forward(self, x):
        # x: [B, C, 1, T]
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.do(x)
        return x


# --------- EEGNet 主干（可放大） ---------
class _EEGNetCore(nn.Module):
    """
    放大版 EEGNet：
      输入:  [B, 26, T] 或 [B, 1, 26, T]
      输出:  [B, C*]  (分类器前的全局聚合特征)
    结构:
      1) Temporal Stem: 1 -> F1, kernel=(1, temporal_kernel)
      2) Spatial Depthwise: F1 -> F1*D, kernel=(C,1), groups=F1  (跨通道空间滤波)
      3) 通道扩展: 1x1 conv 进一步扩张到 ch = F1*D*expand
      4) 堆叠 N 个 SeparableTemporalBlock（DW over time + PW）
         - 每若干块后做一次时间池化（按 pool_seq 指定）
      5) 自适应时间聚合（Avg 或者 Avg+Max 拼接）
    """

    def __init__(
        self,
        in_chans: int = 26,
        F1: int = 16,
        D: int = 2,
        temporal_kernel: int = 64,
        # 放大关键：通道扩展倍数 & 堆叠层数
        expand: int = 2,               # 将 F1*D 扩张到 ch = F1*D*expand
        n_sep_blocks: int = 2,         # 可分离时域卷积块的数量（>=1 直接拉满容量）
        sep_kernel: int = 16,
        block_dropout: float = 0.1,    # 每个 block 内的 dropout
        use_se: bool = False,
        se_reduction: int = 8,
        # 池化策略：每次在时间维做 AvgPool2d((1, p))
        pool_seq: Optional[List[int]] = None,  # e.g. [4, 8]；如果 None，则使用 pool1/pool2
        pool1: int = 4,
        pool2: int = 8,
        # 聚合
        agg_max: bool = False,         # True 则全局聚合时拼接 Avg+Max (通道翻倍)
        act: str = "elu",
        eps: float = 1e-5,
    ):
        super().__init__()
        assert in_chans >= 1 and F1 >= 1 and D >= 1 and n_sep_blocks >= 1
        F2 = F1 * D
        ch = F2 * max(1, int(expand))   # 扩张后的主干通道数（放大参数量最关键的一步）
        self.ch = ch

        # 激活
        if act.lower() == "elu":
            act_layer = nn.ELU()
        elif act.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {act}")

        # 1) Temporal Stem
        self.conv_temporal = nn.Conv2d(
            in_channels=1, out_channels=F1,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1, eps=eps)

        # 2) Spatial Depthwise (跨通道)
        self.conv_spatial = nn.Conv2d(
            in_channels=F1, out_channels=F2,
            kernel_size=(in_chans, 1),
            groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2, eps=eps)
        self.act2 = act_layer

        # 3) 通道扩展（Pointwise 1x1）
        self.expand_pw = nn.Conv2d(F2, ch, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(ch, eps=eps)
        self.act_expand = act_layer

        # 4) N 个 SeparableTemporalBlock + 若干时间池化
        blocks = []
        # 将池化序列准备好：如果没给，就用 [pool1, pool2]（与原始 EEGNet 兼容）
        if pool_seq is None:
            pool_seq = [pool1, pool2]
        # 把池化分配到若干个位置（例如在第 k 块后池化一次）
        # 简单做法：平均分布到后半段；也可全部用在前两次以模仿经典 EEGNet
        pool_positions = []
        if len(pool_seq) > 0:
            # 在第 1/3 和 2/3 位置池化（或尽量靠前）
            idxs = []
            if n_sep_blocks >= 2:
                idxs = [max(1, n_sep_blocks // 3), max(1, (2 * n_sep_blocks) // 3)]
                if idxs[0] == idxs[1] and n_sep_blocks >= 3:
                    idxs[1] = idxs[0] + 1
                idxs = idxs[:len(pool_seq)]
            else:
                idxs = []
            pool_positions = list(zip(idxs[:len(pool_seq)], pool_seq[:len(pool_seq)]))  # [(pos,pool)]
        pool_cursor = 0

        for i in range(1, n_sep_blocks + 1):
            blocks.append(
                SeparableTemporalBlock(
                    ch=ch,
                    sep_kernel=sep_kernel,
                    dropout=block_dropout,
                    act=act,
                    use_se=use_se,
                    se_reduction=se_reduction,
                    eps=eps,
                )
            )
            # 在指定块之后插入池化（时间维）
            if pool_cursor < len(pool_positions) and i == pool_positions[pool_cursor][0]:
                p = pool_positions[pool_cursor][1]
                blocks.append(nn.AvgPool2d(kernel_size=(1, p)))
                pool_cursor += 1

        self.blocks = nn.Sequential(*blocks)

        # 5) 全局聚合
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max = nn.AdaptiveMaxPool2d((1, 1)) if agg_max else None
        self.out_dim = ch * (2 if agg_max else 1)

        # init
        self._init_weights()

    def _init_weights(self):
        for m in [self.conv_temporal, self.conv_spatial, self.expand_pw]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 26, T] 或 [B, 1, 26, T]
        返回: [B, out_dim]
        """
        if x.dim() == 3:  # [B, C, T]
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected x as [B, 26, T] or [B, 1, 26, T], got {tuple(x.shape)}")

        # Stem
        x = self.conv_temporal(x)
        x = self.bn1(x)

        # Spatial
        x = self.conv_spatial(x)  # [B, F2, 1, T]
        x = self.bn2(x)
        x = self.act2(x)

        # Channel expand
        x = self.expand_pw(x)     # [B, ch, 1, T]
        x = self.bn_expand(x)
        x = self.act_expand(x)

        # Stacked Sep-Temporal blocks (+ optional pooling)
        x = self.blocks(x)        # [B, ch, 1, T']

        # Global aggregation
        avg = self.global_avg(x)  # [B, ch, 1, 1]
        if self.global_max is not None:
            mx = self.global_max(x)
            x = torch.cat([avg, mx], dim=1)   # [B, ch*2, 1, 1]
        else:
            x = avg

        x = x.flatten(1)          # [B, out_dim]
        return x


# --------- 封装为项目统一 Net 接口 ---------
class Net(nn.Module):
    """
    - 输入: [B, 26, T] / [B, 1, 26, T]
    - 输出: logits [B, num_classes]
    - 在分类层前统一加 Dropout
    - 通过 expand/n_sep_blocks/head_hidden/head_layers 等参数方便“加大/减小容量”
    """
    def __init__(
        self,
        num_classes: int = 1,
        # 主干放大相关
        F1: int = 32,                 # ↑ 建议放大（原 16 -> 32/48）
        D: int = 4,                   # ↑ 建议放大（原 2 -> 4/6）
        temporal_kernel: int = 64,
        expand: int = 4,              # ↑ 关键增宽倍数（2/4/6/8）
        n_sep_blocks: int = 6,        # ↑ 堆叠层数（2/4/6/8/10）
        sep_kernel: int = 16,
        block_dropout: float = 0.1,
        use_se: bool = True,          # ↑ 建议打开（轻量）
        se_reduction: int = 8,
        pool_seq: Optional[List[int]] = None,  # e.g. [4, 8]；None 表示用 pool1/pool2
        pool1: int = 4,
        pool2: int = 8,
        agg_max: bool = True,         # ↑ Avg+Max 拼接，特征翻倍
        # 分类头
        head_hidden: int = 512,       # ↑ 大一点的 MLP
        head_layers: int = 2,         # ↑ 1~3 层
        dropout: float = 0.3,
        # 统一保留的占位
        pretrained: bool = False,
        temporal_pool: Optional[int] = None,
        in_chans: int = 26,
        act: str = "elu",
    ):
        super().__init__()
        self.backbone = _EEGNetCore(
            in_chans=in_chans,
            F1=F1, D=D,
            temporal_kernel=temporal_kernel,
            expand=expand,
            n_sep_blocks=n_sep_blocks,
            sep_kernel=sep_kernel,
            block_dropout=block_dropout,
            use_se=use_se,
            se_reduction=se_reduction,
            pool_seq=pool_seq,
            pool1=pool1, pool2=pool2,
            agg_max=agg_max,
            act=act,
        )
        in_dim = self.backbone.out_dim

        # 更大的 MLP 头
        mlp = []
        hid = head_hidden
        for i in range(max(0, head_layers)):
            fc = nn.Linear(in_dim if i == 0 else hid, hid)
            nn.init.kaiming_uniform_(fc.weight, a=math.sqrt(5))
            nn.init.zeros_(fc.bias)
            mlp += [fc, nn.ReLU(inplace=True), nn.Dropout(dropout)]
        self.head = nn.Sequential(*mlp) if mlp else nn.Identity()
        head_out = hid if head_layers > 0 else in_dim

        self.pre_logits_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(head_out, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)        # [B, C*]
        feats = self.head(feats)
        feats = self.pre_logits_drop(feats)
        logits = self.classifier(feats) # [B, num_classes]
        return logits
