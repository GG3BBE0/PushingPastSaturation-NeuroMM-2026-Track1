"""Migrated legacy EEG model module: cnn_eeg.

This file was ported from the former old_baseline area and is kept inside
neuromm26_baseline.models.legacy so release builds remain self-contained.
"""

import torch
import torch.nn as nn
from typing import Optional

import torchvision

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

from torchvision.models import (
    convnext_tiny, convnext_base, convnext_large,
    efficientnet_v2_s, mobilenet_v3_large, densenet121
)
import torch.nn.functional as F


def _adapt_densenet_stem_for_small_input(model: nn.Module):
    """
    让 DenseNet 能适配矮高(H=26)输入：
    - conv0 stride: (2,2) -> (1,1)
    - pool0: MaxPool2d -> Identity
    """
    if hasattr(model, "features"):
        if hasattr(model.features, "conv0"):
            # 保持权重不变，仅改步幅
            model.features.conv0.stride = (1, 1)
        if hasattr(model.features, "pool0"):
            model.features.pool0 = nn.Identity()
    return model

# 兼容新旧权重接口
_HAS_ENUM = hasattr(torchvision.models, "ConvNeXt_Tiny_Weights")

def _load_backbone(base: str, pretrained: bool):
    """
    加载 torchvision 模型，并返回 (model, feat_dim, head_locator)
    head_locator: 一个函数，输入 model，返回最后分类全连接层（用于替换为 num_classes）。
    """
    base = base.lower()

    if base == "convnext_tiny":
        if pretrained and _HAS_ENUM:
            m = convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            m = convnext_tiny(weights=None)
        feat_dim = m.classifier[2].in_features
        def _head(model): return model.classifier[2]

    elif base == "convnext_base":
        if pretrained and _HAS_ENUM:
            m = convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        else:
            m = convnext_base(weights=None)
        feat_dim = m.classifier[2].in_features
        def _head(model): return model.classifier[2]

    elif base == "efficientnet_v2_s":
        if pretrained and _HAS_ENUM:
            m = efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        else:
            m = efficientnet_v2_s(weights=None)
        feat_dim = m.classifier[1].in_features
        def _head(model): return model.classifier[1]

    elif base == "mobilenet_v3_large":
        if pretrained and _HAS_ENUM:
            m = mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        else:
            m = mobilenet_v3_large(weights=None)
        # classifier: [0] Linear(960->1280), [1] Hardswish, [2] Dropout, [3] Linear(1280->1000)
        feat_dim = m.classifier[-1].in_features
        def _head(model): return model.classifier[-1]

    elif base == "densenet121":
        if pretrained and _HAS_ENUM:
            m = densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            m = densenet121(weights=None)
        feat_dim = m.classifier.in_features
        def _head(model): return model.classifier

    elif base == "convnext_large":
        m = convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained and _HAS_ENUM else None)
        feat_dim = m.classifier[2].in_features
        def _head(model): return model.classifier[2]
    elif base == "convnext_xlarge":
        # 优先尝试 torchvision（部分版本可能有）
        has_tv_xl = hasattr(torchvision.models, "convnext_xlarge") and hasattr(torchvision.models, "ConvNeXt_XLarge_Weights")
        if has_tv_xl:
            m = torchvision.models.convnext_xlarge(
                weights=torchvision.models.ConvNeXt_XLarge_Weights.IMAGENET1K_V1 if pretrained else None
            )
            feat_dim = m.classifier[2].in_features
            def _head(model): return model.classifier[2]
        else:
            # 回退到 timm（推荐：fb_in22k_ft_in1k；也可尝试 fb_in22k）
            assert _HAS_TIMM, "convnext_xlarge 需要 torchvision>=支持XL 或安装 timm：pip install timm"
            model_name = "convnext_xlarge.fb_in22k_ft_in1k"
            m = timm.create_model(model_name, pretrained=pretrained)
            # timm 的 convnext 通常有 num_features 与 head（Linear）
            feat_dim = m.num_features if hasattr(m, "num_features") else (
                (m.head.in_features if hasattr(m, "head") and hasattr(m.head, "in_features") else None)
            )
            assert feat_dim is not None, "无法从 timm 模型上确定 feat_dim"
            def _head(model):
                # timm 提供 get_classifier()；否则回退到 model.head
                if hasattr(model, "get_classifier"):
                    return model.get_classifier()
                elif hasattr(model, "head"):
                    return model.head
                else:
                    # 最后一招：从模块里找最后一个 Linear（返回它的引用）
                    last_linear = None
                    for mod in model.modules():
                        if isinstance(mod, nn.Linear):
                            last_linear = mod
                    assert last_linear is not None, "无法定位 timm 模型的分类头"
                    return last_linear


    else:
        raise ValueError(f"Unsupported base: {base}")

    return m, feat_dim, _head


def _replace_first_conv_to_1ch(model: nn.Module):
    """
    遍历模型，找到第一个 in_channels=3 的 Conv2d，替换为 in_channels=1，
    并把预训练权重在通道维做均值迁移过去。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            # 构造新卷积
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode if hasattr(module, "padding_mode") else "zeros",
            )
            # 权重迁移
            with torch.no_grad():
                w = module.weight  # [out_c, 3, kH, kW]
                w_mean = w.mean(dim=1, keepdim=True)  # -> [out_c, 1, kH, kW]
                new_conv.weight.copy_(w_mean)
                if module.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(module.bias)

            # 将该卷积层替换回去
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)
            return model  # 只替换第一个就够了

    # 如果未找到 3 通道 conv（极少数情况），不做处理
    return model


class Net(nn.Module):
    """
    通用 EEG-CNN 包装：
    - 输入: [B, 26, 2000] 或 [B, 1, 26, 2000]
    - 可选 temporal_pool 在时间轴做降采样（AvgPool2d kernel=(1, temporal_pool)）
    - 自动把首层改为 1 通道并迁移权重
    - 自动替换最后分类层为 num_classes
    """
    def __init__(
        self,
        num_classes: int = 1,
        base: str = "convnext_tiny",
        pretrained: bool = True,
        temporal_pool: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.base = base.lower()
        self.is_convnext = self.base.startswith("convnext")
        
        m, feat_dim, head_locator = _load_backbone(self.base, pretrained)

        # 仅对 densenet 做 stem 适配（关键一步）
        if self.base == "densenet121":
            m = _adapt_densenet_stem_for_small_input(m)
        
        # # 改首层到 1 通道
        # m = _replace_first_conv_to_1ch(m)
        # 只有非 ConvNeXt 才把第一层改为 1 通道
        if not self.is_convnext:
            m = _replace_first_conv_to_1ch(m)

        # 在进入骨干之前先做时间轴池化（不改变高 H=26）
        if temporal_pool is not None and temporal_pool > 1:
            self.temporal_pool: Optional[nn.Module] = nn.AvgPool2d(kernel_size=(1, temporal_pool), stride=(1, temporal_pool))
        else:
            self.temporal_pool = None

        # 替换分类头维度（不同架构位置不同）
        self.dropout = nn.Dropout(p=dropout) if (dropout and dropout > 0) else nn.Identity()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # 找到并替换 head
        head_ref = head_locator(m)
        # head_ref 可能是 Linear 对象引用，需要替换到原位置
        # 我们通过再次定位父模块完成替换
        # 先扫描，定位该模块路径
        target_path = None
        for name, mod in m.named_modules():
            if mod is head_ref:
                target_path = name
                break
        assert target_path is not None, "Failed to locate classifier head."

        # 构造新的 Linear
        new_head = nn.Linear(self.feat_dim, num_classes)

        # 把 new_head 放回去
        parent = m
        parts = target_path.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_head)

        self.backbone = m

    def forward(self, x):
        # 输入标准化为 [B, 1, 26, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Expected x with shape [B, 26, T] or [B, 1, 26, T], got {x.shape}")

        x = x.float()

        if self.temporal_pool is not None:
            x = self.temporal_pool(x)
            
        if self.is_convnext:
            # 1) 复制为 3 通道，保持 ConvNeXt 预训练分布
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)  # [B,3,H,W]

            # 2) H<32 用“对称反射填充”到 32，避免零带伪影
            H = x.size(2)
            if H < 32:
                need = 32 - H
                pad_top = need // 2
                pad_bot = need - pad_top
                x = F.pad(x, (0, 0, pad_top, pad_bot), mode='reflect')

        # 某些主干（如 ConvNeXt/EfficientNet）内部已包含全局池化与展平
        # 我们仅在进入分类头前加一个 dropout（放在模型内部 head 前已经替换好的）
        # 这里直接 forward
        out = self.backbone(x)
        # 大多数架构 head 在内部，无法轻易在 head 前插入 dropout；
        # 因此改为在输入 backbone 前后都不强插。若想强制，可拆出 features+classifier。
        # 简化：直接返回 logits
        return out
