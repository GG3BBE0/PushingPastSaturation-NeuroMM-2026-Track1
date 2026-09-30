"""Migrated legacy EEG model family for NeuroMM 2026.

This subpackage contains self-contained ports of the former experimental EEG-only
architectures that were originally staged in old_baseline. The training runtime
should import only from this package so open-source releases can safely remove
old_baseline without breaking model construction.
"""

from .registry import LEGACY_EEG_MODEL_NAMES, LegacyEEGModelAdapter, build_legacy_eeg_model, is_legacy_eeg_model_name

__all__ = [
    "LEGACY_EEG_MODEL_NAMES",
    "LegacyEEGModelAdapter",
    "build_legacy_eeg_model",
    "is_legacy_eeg_model_name",
]
