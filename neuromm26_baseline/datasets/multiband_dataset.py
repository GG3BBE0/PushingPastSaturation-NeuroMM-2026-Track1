"""Dataset that loads pre-computed multi-band EEG features (130 channels).

Drop-in replacement for EEGFeatureDataset.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MultiBandFeatureDataset(Dataset):
    def __init__(
        self,
        multiband_root: str,
        manifest_csv: str,
        split: str,
        preload_in_memory: bool = True,
        target_shape: tuple[int, int] = (130, 2000),
    ) -> None:
        self.multiband_root = Path(multiband_root)
        self.manifest_csv = Path(manifest_csv)
        self.split = split
        self.preload_in_memory = preload_in_memory
        self.target_shape = target_shape

        with self.manifest_csv.open("r", encoding="utf-8-sig", newline="") as f:
            self.records = [r for r in csv.DictReader(f) if r.get("split") == split]
        self.sample_ids = [r["sample_id"] for r in self.records]
        self.labels = [int(r["label"]) if r.get("label", "") != "" else None for r in self.records]

        self._cache: list[torch.Tensor] | None = None
        if preload_in_memory:
            self._cache = []
            for sid in tqdm(self.sample_ids, desc=f"Preloading multiband {split}"):
                self._cache.append(self._load_tensor(sid))

    def _load_tensor(self, sid: str) -> torch.Tensor:
        wave = np.load(self.multiband_root / f"{sid}.npy")  # (130, 2000) float16
        # Ensure shape
        out = np.zeros(self.target_shape, dtype=np.float32)
        c, t = wave.shape
        out[: min(c, self.target_shape[0]), : min(t, self.target_shape[1])] = (
            wave[: min(c, self.target_shape[0]), : min(t, self.target_shape[1])].astype(np.float32)
        )
        return torch.from_numpy(out)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        eeg = self._cache[idx] if self._cache is not None else self._load_tensor(self.sample_ids[idx])
        return {
            "sample_id": self.sample_ids[idx],
            "eeg": eeg,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32) if self.labels[idx] is not None else None,
        }
