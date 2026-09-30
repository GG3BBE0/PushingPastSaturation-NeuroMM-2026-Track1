"""Pre-compute multi-band EEG features and cache to disk.

Input:  processed/features/eeg/{sample_id}.npy  shape (29, 2000), raw EEG @ 500 Hz
Output: processed/features/multiband/{sample_id}.npy  shape (130, 2000) float16
        = 26 normalized channels + 4 bands × 26 channels each (104) = 130

Bands (Hz):
  delta  0.5–4
  theta  4–8
  alpha  8–13
  beta   13–30

Uses same normalize scheme as EEGFeatureDataset to map (29) → (26).
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm


FS = 500
BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta",  13.0, 30.0),
]

# Pre-compute filter coefficients once
FILTER_COEFFS = []
for name, lo, hi in BANDS:
    b, a = butter(4, [lo, hi], btype="band", fs=FS)
    FILTER_COEFFS.append((name, b, a))


def normalize_29_to_26(wave: np.ndarray) -> np.ndarray:
    """Same logic as EEGFeatureDataset._normalize."""
    wave = wave.astype(np.float32, copy=True)
    if wave.shape[0] >= 29:
        wave[:23, ...] = wave[:23, ...] / 1e-3
        wave[23:, ...] = wave[23:, ...] * 1e-2
        heart_wave = wave[23, :] - wave[24, :]
        muscle_wave1 = wave[25, :] - wave[26, :]
        muscle_wave2 = wave[27, :] - wave[28, :]
        heart_muscle = np.stack([heart_wave, muscle_wave1, muscle_wave2], axis=0)
        wave = np.concatenate([wave[:23, ...], heart_muscle], axis=0)
    return wave  # (26, 2000)


def pad_or_crop(wave, target=(26, 2000)):
    out = np.zeros(target, dtype=np.float32)
    c, t = wave.shape
    out[: min(c, target[0]), : min(t, target[1])] = wave[: min(c, target[0]), : min(t, target[1])]
    return out


def process_one(args):
    in_path, out_path = args
    if out_path.exists():
        return "skip"
    wave = np.load(in_path)
    wave = normalize_29_to_26(wave)
    wave = pad_or_crop(wave)  # (26, 2000)
    # Bandpass each channel for each band
    band_outs = []
    for name, b, a in FILTER_COEFFS:
        # filtfilt zero-phase; pad to handle edges
        filt = filtfilt(b, a, wave, axis=-1, padlen=min(50, wave.shape[-1] - 1))
        band_outs.append(filt.astype(np.float32))
    # Concat: raw + 4 bands = 5 × 26 = 130 channels
    full = np.concatenate([wave] + band_outs, axis=0)  # (130, 2000)
    np.save(out_path, full.astype(np.float16))
    return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", default="neuromm26_datasets/processed/features/eeg")
    parser.add_argument("--out-root", default="neuromm26_datasets/processed/features/multiband")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    eeg_root = Path(args.eeg_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(eeg_root.glob("*.npy"))
    print(f"Found {len(files)} EEG files; output to {out_root}")
    jobs = [(f, out_root / f.name) for f in files]

    with Pool(args.workers) as p:
        results = list(tqdm(p.imap_unordered(process_one, jobs, chunksize=32),
                            total=len(jobs), desc="multiband"))
    n_done = results.count("done")
    n_skip = results.count("skip")
    print(f"Done: {n_done}  Skipped existing: {n_skip}  Total: {len(jobs)}")

    # Disk usage
    total_bytes = sum(f.stat().st_size for f in out_root.glob("*.npy"))
    print(f"Cache size: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
