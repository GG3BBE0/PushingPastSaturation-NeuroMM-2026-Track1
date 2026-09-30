"""Pre-compute STFT spectrograms and cache to disk.

Input:  processed/features/eeg/{sample_id}.npy  shape (29, 2000), raw EEG @ 500 Hz
Output: processed/features/stft/{sample_id}.npy  shape (26, 65, 60) float16

STFT params:
  n_fft = 128       → 65 freq bins (covers 0–250 Hz with 3.9 Hz resolution)
  hop_length = 32   → 60 time bins
  win_length = 128
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


FS = 500
N_FFT = 128
HOP = 32
WIN = 128


def normalize_29_to_26(wave: np.ndarray) -> np.ndarray:
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
    x = torch.from_numpy(wave).float()
    # STFT per channel (batched over channel dim)
    spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                      return_complex=True, center=False)
    # spec shape: (26, n_freq=65, n_time≈60)
    spec = spec.abs()
    spec = torch.log1p(spec)
    np.save(out_path, spec.numpy().astype(np.float16))
    return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", default="neuromm26_datasets/processed/features/eeg")
    parser.add_argument("--out-root", default="neuromm26_datasets/processed/features/stft")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    eeg_root = Path(args.eeg_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(eeg_root.glob("*.npy"))
    print(f"Found {len(files)} EEG files; output to {out_root}")
    print(f"STFT: n_fft={N_FFT} hop={HOP} win={WIN}")
    jobs = [(f, out_root / f.name) for f in files]

    with Pool(args.workers) as p:
        results = list(tqdm(p.imap_unordered(process_one, jobs, chunksize=32),
                            total=len(jobs), desc="stft"))
    n_done = results.count("done")
    n_skip = results.count("skip")
    print(f"Done: {n_done}  Skipped: {n_skip}  Total: {len(jobs)}")

    total_bytes = sum(f.stat().st_size for f in out_root.glob("*.npy"))
    print(f"Cache size: {total_bytes / 1e9:.2f} GB")

    # Sample shape verification
    sample = np.load(jobs[0][1])
    print(f"Sample shape: {sample.shape}  dtype: {sample.dtype}")


if __name__ == "__main__":
    main()
