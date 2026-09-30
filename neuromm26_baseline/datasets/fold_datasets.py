"""Fold-CV datasets: train when fold != fold_idx, val when fold == fold_idx.

fold_df.csv columns: sample_id, label, label_type, subject_id, label_5, fold
Each sample_id appears exactly once with a single fold value (patient-disjoint
K-fold; same subject_id → same fold).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _load_fold_records(fold_csv: str, fold_idx: int, split: str):
    """split = 'train' (fold != idx) or 'val' (fold == idx)."""
    with Path(fold_csv).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if split == "train":
        rows = [r for r in rows if int(r["fold"]) != fold_idx]
    elif split == "val":
        rows = [r for r in rows if int(r["fold"]) == fold_idx]
    else:
        raise ValueError(split)
    return rows


def _normalize_29_to_26(wave: np.ndarray) -> np.ndarray:
    wave = wave.astype(np.float32, copy=True)
    if wave.shape[0] >= 29:
        wave[:23, ...] = wave[:23, ...] / 1e-3
        wave[23:, ...] = wave[23:, ...] * 1e-2
        heart_wave = wave[23, :] - wave[24, :]
        muscle_wave1 = wave[25, :] - wave[26, :]
        muscle_wave2 = wave[27, :] - wave[28, :]
        heart_muscle = np.stack([heart_wave, muscle_wave1, muscle_wave2], axis=0)
        wave = np.concatenate([wave[:23, ...], heart_muscle], axis=0)
    return wave


def _pad_or_crop(wave, target):
    out = np.zeros(target, dtype=np.float32)
    c, t = wave.shape
    out[: min(c, target[0]), : min(t, target[1])] = wave[: min(c, target[0]), : min(t, target[1])]
    return out


class FoldEEGFeatureDataset(Dataset):
    """EEG-only fold dataset (for muku raw)."""

    def __init__(
        self,
        eeg_feature_root: str,
        fold_csv: str,
        fold_idx: int,
        split: str,
        preload_in_memory: bool = True,
        target_shape: tuple[int, int] = (26, 2000),
    ) -> None:
        self.eeg_feature_root = Path(eeg_feature_root)
        self.records = _load_fold_records(fold_csv, fold_idx, split)
        self.sample_ids = [r["sample_id"] for r in self.records]
        self.labels = [int(r["label"]) for r in self.records]
        self.target_shape = target_shape

        # Preload: dedupe by sample_id (each unique loaded once), then map back
        self._tensor_cache: dict[str, torch.Tensor] | None = None
        if preload_in_memory:
            self._tensor_cache = {}
            unique = list(dict.fromkeys(self.sample_ids))  # preserve order, unique
            for sid in tqdm(unique, desc=f"Preload EEG fold{fold_idx}/{split}"):
                self._tensor_cache[sid] = self._load(sid)

    def _load(self, sid: str) -> torch.Tensor:
        wave = np.load(self.eeg_feature_root / f"{sid}.npy")
        wave = _normalize_29_to_26(wave)
        wave = _pad_or_crop(wave, self.target_shape)
        return torch.from_numpy(wave.copy()).float()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid = self.sample_ids[idx]
        eeg = self._tensor_cache[sid] if self._tensor_cache is not None else self._load(sid)
        return {
            "sample_id": sid,
            "eeg": eeg,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class FoldSpecDataset(Dataset):
    """Pre-computed spec (STFT/CWT) fold dataset."""

    def __init__(
        self,
        spec_root: str,
        fold_csv: str,
        fold_idx: int,
        split: str,
        preload_in_memory: bool = True,
    ) -> None:
        self.spec_root = Path(spec_root)
        self.records = _load_fold_records(fold_csv, fold_idx, split)
        self.sample_ids = [r["sample_id"] for r in self.records]
        self.labels = [int(r["label"]) for r in self.records]

        self._tensor_cache: dict[str, torch.Tensor] | None = None
        if preload_in_memory:
            self._tensor_cache = {}
            unique = list(dict.fromkeys(self.sample_ids))
            for sid in tqdm(unique, desc=f"Preload spec fold{fold_idx}/{split}"):
                self._tensor_cache[sid] = self._load(sid)

    def _load(self, sid: str) -> torch.Tensor:
        arr = np.load(self.spec_root / f"{sid}.npy").astype(np.float32)
        return torch.from_numpy(arr)

    def __len__(self): return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid = self.sample_ids[idx]
        x = self._tensor_cache[sid] if self._tensor_cache is not None else self._load(sid)
        return {
            "sample_id": sid,
            "spec": x,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class FoldEEGSpecDataset(Dataset):
    """Combined fold dataset for muku V2 (returns both raw EEG and a precomputed spec).

    Superlet/CWT cache is kept as float16 in memory to halve RAM (~17 GB train fold);
    cast to float32 in __getitem__.
    """

    def __init__(
        self,
        eeg_feature_root: str,
        spec_root: str,
        fold_csv: str,
        fold_idx: int,
        split: str,
        preload_in_memory: bool = True,
        target_shape: tuple[int, int] = (26, 2000),
    ) -> None:
        self.eeg_feature_root = Path(eeg_feature_root)
        self.spec_root = Path(spec_root)
        self.records = _load_fold_records(fold_csv, fold_idx, split)
        self.sample_ids = [r["sample_id"] for r in self.records]
        self.labels = [int(r["label"]) for r in self.records]
        self.target_shape = target_shape

        self._eeg_cache: dict[str, torch.Tensor] | None = None
        self._spec_cache: dict[str, torch.Tensor] | None = None
        if preload_in_memory:
            self._eeg_cache = {}
            self._spec_cache = {}
            unique = list(dict.fromkeys(self.sample_ids))
            for sid in tqdm(unique, desc=f"Preload eeg+spec fold{fold_idx}/{split}"):
                self._eeg_cache[sid] = self._load_eeg(sid)
                self._spec_cache[sid] = self._load_spec_f16(sid)

    def _load_eeg(self, sid: str) -> torch.Tensor:
        wave = np.load(self.eeg_feature_root / f"{sid}.npy")
        wave = _normalize_29_to_26(wave)
        wave = _pad_or_crop(wave, self.target_shape)
        return torch.from_numpy(wave.copy()).float()

    def _load_spec_f16(self, sid: str) -> torch.Tensor:
        arr = np.load(self.spec_root / f"{sid}.npy")   # already float16 on disk
        return torch.from_numpy(arr)                   # keep float16 in cache

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid = self.sample_ids[idx]
        if self._eeg_cache is not None:
            eeg = self._eeg_cache[sid]
            spec = self._spec_cache[sid].float()        # f16 → f32 on the fly
        else:
            eeg = self._load_eeg(sid)
            spec = self._load_spec_f16(sid).float()
        return {
            "sample_id": sid,
            "eeg": eeg,
            "spec": spec,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
