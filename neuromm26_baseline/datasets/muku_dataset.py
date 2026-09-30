"""Muku-style dataset wrappers with EEG augmentation.

Wraps EEGFeatureDataset / Task3EEGFeatureDataset and applies time shift,
random bandpass, channel mask augmentations only during training.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, lfilter
from torch.utils.data import Dataset


class EEGAugmentation:
    """Time shift + random bandpass + channel mask + amplitude scale."""

    def __init__(
        self,
        time_shift_max: int = 100,     # ±100 samples = ±0.2s @ 500Hz
        bandpass_prob: float = 0.1,
        bandpass_low_range: tuple[int, int] = (10, 15),
        bandpass_band: float = 3.0,
        channel_mask_range: tuple[int, int] = (1, 3),
        amp_scale_range: tuple[float, float] = (0.9, 1.1),
        amp_scale_prob: float = 0.5,
        fs: int = 500,
    ):
        self.time_shift_max = time_shift_max
        self.bandpass_prob = bandpass_prob
        self.bandpass_low_range = bandpass_low_range
        self.bandpass_band = bandpass_band
        self.channel_mask_range = channel_mask_range
        self.amp_scale_range = amp_scale_range
        self.amp_scale_prob = amp_scale_prob
        self.fs = fs

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        """eeg: tensor (C, T). Returns augmented tensor (C, T)."""
        x = eeg.numpy().copy()
        x = self._time_shift(x)
        x = self._random_bandpass(x)
        x = self._channel_mask(x)
        x = self._amp_scale(x)
        return torch.from_numpy(x).float()

    def _time_shift(self, x):
        if self.time_shift_max <= 0:
            return x
        shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
        return np.roll(x, shift, axis=-1)

    def _random_bandpass(self, x):
        if np.random.random() > self.bandpass_prob:
            return x
        low = np.random.uniform(*self.bandpass_low_range)
        high = min(low + self.bandpass_band, self.fs / 2 - 1)
        try:
            b, a = butter(2, [low, high], btype="band", fs=self.fs)
            return lfilter(b, a, x, axis=-1).astype(np.float32)
        except ValueError:
            return x

    def _channel_mask(self, x):
        k = np.random.randint(self.channel_mask_range[0], self.channel_mask_range[1] + 1)
        if k <= 0:
            return x
        idx = np.random.choice(x.shape[0], min(k, x.shape[0]), replace=False)
        x[idx] = 0
        return x

    def _amp_scale(self, x):
        if np.random.random() > self.amp_scale_prob:
            return x
        factor = np.random.uniform(*self.amp_scale_range)
        return x * factor


class AugmentedDataset(Dataset):
    """Wrap any dataset that returns dict with 'eeg' tensor; apply augmentation."""

    def __init__(self, base_dataset: Dataset, transform: EEGAugmentation | None = None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        # forward unknown attrs to base dataset (labels, sample_ids etc.)
        return getattr(self.base, name)

    def __getitem__(self, index):
        sample = self.base[index]
        if self.transform is not None and isinstance(sample.get("eeg"), torch.Tensor):
            sample["eeg"] = self.transform(sample["eeg"])
        return sample
