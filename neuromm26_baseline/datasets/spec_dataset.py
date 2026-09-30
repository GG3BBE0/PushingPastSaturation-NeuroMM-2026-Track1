"""Dataset that loads pre-computed STFT or CWT spectrograms."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SpecFeatureDataset(Dataset):
    """Generic time-frequency spec dataset (STFT / CWT cache)."""

    def __init__(
        self,
        spec_root: str,
        manifest_csv: str,
        split: str,
        preload_in_memory: bool = True,
    ) -> None:
        self.spec_root = Path(spec_root)
        self.split = split
        with Path(manifest_csv).open("r", encoding="utf-8-sig", newline="") as f:
            self.records = [r for r in csv.DictReader(f) if r.get("split") == split]
        self.sample_ids = [r["sample_id"] for r in self.records]
        self.labels = [int(r["label"]) if r.get("label", "") != "" else None for r in self.records]

        self._cache: list[torch.Tensor] | None = None
        if preload_in_memory:
            self._cache = []
            for sid in tqdm(self.sample_ids, desc=f"Preloading spec {self.spec_root.name} {split}"):
                self._cache.append(self._load(sid))

    def _load(self, sid: str) -> torch.Tensor:
        arr = np.load(self.spec_root / f"{sid}.npy").astype(np.float32)
        return torch.from_numpy(arr)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        x = self._cache[idx] if self._cache is not None else self._load(self.sample_ids[idx])
        return {
            "sample_id": self.sample_ids[idx],
            "spec": x,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32) if self.labels[idx] is not None else None,
        }
