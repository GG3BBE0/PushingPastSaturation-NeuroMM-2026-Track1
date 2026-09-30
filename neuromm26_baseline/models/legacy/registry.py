"""Registry for migrated EEG-only baseline variants stored in the legacy subpackage."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

import torch.nn as nn


class LegacyEEGModelAdapter(nn.Module):
    """Adapts EEG-only models that consume tensors into the current batch-based API."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch["eeg"])


MODEL_SPECS: dict[str, tuple[str, Mapping[str, object]]] = {
    "resnet18_eeg": ("resnet_eeg", {"num_classes": 1, "base": "resnet18", "pretrained": False, "temporal_pool": 10, "dropout": 0.2}),
    "resnet50_eeg": ("resnet_eeg", {"num_classes": 1, "base": "resnet50", "pretrained": True, "temporal_pool": 10, "dropout": 0.3}),
    "convnext_tiny_eeg": ("cnn_eeg", {"num_classes": 1, "base": "convnext_tiny", "pretrained": True, "temporal_pool": 10, "dropout": 0.3}),
    "convnext_base_eeg": ("cnn_eeg", {"num_classes": 1, "base": "convnext_base", "pretrained": True, "temporal_pool": 10, "dropout": 0.3}),
    "convnext_large_eeg": ("cnn_eeg", {"num_classes": 1, "base": "convnext_large", "pretrained": True, "temporal_pool": 10, "dropout": 0.3}),
    "convnext_xlarge_eeg": ("cnn_eeg", {"num_classes": 1, "base": "convnext_xlarge", "pretrained": True, "temporal_pool": 10, "dropout": 0.3}),
    "efficientnet_v2_s_eeg": ("cnn_eeg", {"num_classes": 1, "base": "efficientnet_v2_s", "pretrained": True, "temporal_pool": 10, "dropout": 0.3}),
    "mobilenet_v3_large_eeg": ("cnn_eeg", {"num_classes": 1, "base": "mobilenet_v3_large", "pretrained": True, "temporal_pool": 10, "dropout": 0.2}),
    "densenet121_eeg": ("cnn_eeg", {"num_classes": 1, "base": "densenet121", "pretrained": True, "temporal_pool": 10, "dropout": 0.3}),
    "lstm_big_eeg": ("rnn_eeg_big", {"num_classes": 1, "base": "lstm_big", "pretrained": False, "temporal_pool": 1, "dropout": 0.3}),
    "lstm_huge_eeg": ("rnn_eeg_big", {"num_classes": 1, "base": "lstm_huge", "pretrained": False, "temporal_pool": 1, "dropout": 0.3}),
    "gru_big_eeg": ("rnn_eeg_big", {"num_classes": 1, "base": "gru_big", "pretrained": False, "temporal_pool": 1, "dropout": 0.3}),
    "eegnet": ("eegnet", {"num_classes": 1, "F1": 32, "D": 4, "expand": 4, "n_sep_blocks": 6, "sep_kernel": 16, "use_se": True, "agg_max": True, "head_hidden": 512, "head_layers": 2, "dropout": 0.35, "pool_seq": [4, 8], "in_chans": 26}),
    "eegnet_xl": ("eegnet", {"num_classes": 1, "F1": 48, "D": 6, "expand": 6, "n_sep_blocks": 8, "sep_kernel": 31, "block_dropout": 0.15, "use_se": True, "se_reduction": 8, "agg_max": True, "head_hidden": 1024, "head_layers": 3, "dropout": 0.4, "pool_seq": [4, 4, 4], "in_chans": 26}),
    "eegnet_m": ("eegnet", {"num_classes": 1, "F1": 24, "D": 3, "expand": 3, "n_sep_blocks": 4, "sep_kernel": 16, "use_se": True, "agg_max": False, "head_hidden": 384, "head_layers": 2, "dropout": 0.3, "pool_seq": [4, 8], "in_chans": 26}),
    "tcnet_eeg": ("tcnet", {"num_classes": 1, "F1": 32, "D": 4, "eeg_kernel": 32, "sep_kernel": 16, "pool1": 4, "pool2": 2, "eeg_dropout": 0.2, "tcn_channels": 128, "tcn_layers": 10, "tcn_kernel": 5, "dilation_base": 2, "tcn_dropout": 0.2, "dropout": 0.3, "readout": "attn", "in_chans": 26, "temporal_pool": 1, "pretrained": False}),
    "tcnet_eeg_l": ("tcnet", {"num_classes": 1, "F1": 48, "D": 4, "eeg_kernel": 32, "sep_kernel": 16, "pool1": 4, "pool2": 2, "eeg_dropout": 0.2, "tcn_channels": 192, "tcn_layers": 12, "tcn_kernel": 5, "dilation_base": 2, "tcn_dropout": 0.25, "dropout": 0.35, "readout": "attn", "in_chans": 26, "temporal_pool": 1, "pretrained": False}),
    "actnet_s": ("actnet", {"num_classes": 1, "in_chans": 26, "F1": 24, "D": 3, "eeg_kernel": 64, "expand": 2, "n_front_blocks": 4, "sep_kernel": 17, "pool_every": 2, "pool_size": 4, "eeg_dropout": 0.15, "use_se": True, "n_windows": 5, "win_overlap": 0.5, "t_layers": 2, "t_heads": 4, "t_ff_mult": 2, "t_conv_k": 7, "t_dropout": 0.15, "fuse_layers": 2, "fuse_heads": 4, "fuse_ff_mult": 2, "fuse_dropout": 0.15, "dropout": 0.3, "temporal_pool": 1, "readout": "cls"}),
    "actnet_m": ("actnet", {"num_classes": 1, "in_chans": 26, "F1": 32, "D": 4, "eeg_kernel": 64, "expand": 2, "n_front_blocks": 6, "sep_kernel": 21, "pool_every": 3, "pool_size": 4, "eeg_dropout": 0.15, "use_se": True, "n_windows": 7, "win_overlap": 0.5, "t_layers": 3, "t_heads": 4, "t_ff_mult": 2, "t_conv_k": 7, "t_dropout": 0.15, "fuse_layers": 3, "fuse_heads": 4, "fuse_ff_mult": 2, "fuse_dropout": 0.15, "dropout": 0.3, "temporal_pool": 1, "readout": "cls"}),
    "lmda_eeg": ("lmda", {"num_classes": 1, "in_chans": 26, "samples": 2000, "depth": 9, "kernel": 51, "channel_depth1": 64, "channel_depth2": 32, "avepool": 3, "da_kernel": 7, "dropout": 0.3, "temporal_pool": 1, "pretrained": False, "mlp_hidden": 512}),
    "vit_base_eeg": ("vit_eeg", {"num_classes": 1, "base": "vit_base_patch16_224", "pretrained": True, "temporal_pool": 1, "dropout": 0.3, "img_size": 224}),
    "vit_small_eeg": ("vit_eeg", {"num_classes": 1, "base": "vit_small_patch16_224", "pretrained": True, "temporal_pool": 1, "dropout": 0.3, "img_size": 224}),
    "vit_large_eeg": ("vit_eeg", {"num_classes": 1, "base": "vit_large_patch16_224", "pretrained": True, "temporal_pool": 1, "dropout": 0.3, "img_size": 224}),
    "vit_base_384_eeg": ("vit_eeg", {"num_classes": 1, "base": "vit_base_patch16_384", "pretrained": True, "temporal_pool": 1, "dropout": 0.3, "img_size": 384}),
    "vit_hybrid_eeg": ("vit_hybrid_eeg", {"num_classes": 1, "base_vit": "vit_base_patch16_224", "base_cnn": "resnet18", "pretrained_vit": True, "pretrained_cnn": True, "img_size": 224, "temporal_pool": 1, "dropout": 0.3, "cnn_out_stage": 2, "fusion": "residual", "freeze_cnn_stem": False}),
    "densenet_vit_eeg": ("densenet_vit_eeg", {"num_classes": 1, "pretrained_dense": True, "vit_name": "vit_base_patch16_224", "pretrained_vit": True, "temporal_pool": 10, "fusion": "residual", "dropout": 0.3}),
    "cbramod": ("cbramod", {"num_classes": 1, "input_channels": 19, "input_samples": 2000, "patch_size": 200, "d_model": 200, "dim_feedforward": 800, "n_layer": 12, "nhead": 8, "dropout": 0.3, "pretrained_path": "models/CBraMod/pretrained_weights/pretrained_weights.pth"}),
    "labram_base": ("labram", {"num_classes": 1, "input_channels": 26, "input_samples": 2000, "patch_size": 200, "embed_dim": 200, "depth": 12, "num_heads": 10, "out_chans": 8, "mlp_ratio": 4.0, "drop_path_rate": 0.1, "dropout": 0.3, "pretrained_path": "models/LaBraM/checkpoints/labram-base.pth"}),
    "labram_large": ("labram", {"num_classes": 1, "input_channels": 26, "input_samples": 2000, "patch_size": 200, "embed_dim": 400, "depth": 24, "num_heads": 16, "out_chans": 16, "mlp_ratio": 4.0, "drop_path_rate": 0.1, "dropout": 0.3, "pretrained_path": ""}),
    "eeg_dino_s": ("eeg_dino", {"num_classes": 1, "model_name": "eeg_dino_s", "dropout": 0.3, "pretrained_path": "models/EEG-DINO/pre-trained-models/model_EEG_DINO_S.pt", "freeze_ratio": 0.0}),
    "eeg_dino_m": ("eeg_dino", {"num_classes": 1, "model_name": "eeg_dino_m", "dropout": 0.3, "pretrained_path": "models/EEG-DINO/pre-trained-models/model_EEG_DINO_M.pt", "freeze_ratio": 0.0}),
    "eeg_dino_l": ("eeg_dino", {"num_classes": 1, "model_name": "eeg_dino_l", "dropout": 0.3, "pretrained_path": "models/EEG-DINO/pre-trained-models/model_EEG_DINO_L.pt", "freeze_ratio": 0.5}),
}

LEGACY_EEG_MODEL_NAMES = tuple(sorted(MODEL_SPECS))


def _build_local_model(module_name: str, kwargs: Mapping[str, object]) -> nn.Module:
    module = importlib.import_module(f"{__package__}.{module_name}")
    model_cls = getattr(module, "Net")
    return LegacyEEGModelAdapter(model_cls(**dict(kwargs)))


def is_legacy_eeg_model_name(model_name: str | None) -> bool:
    return bool(model_name) and model_name in MODEL_SPECS


def build_legacy_eeg_model(model_name: str) -> nn.Module:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported legacy EEG model: {model_name}")
    module_name, kwargs = MODEL_SPECS[model_name]
    return _build_local_model(module_name, kwargs)
