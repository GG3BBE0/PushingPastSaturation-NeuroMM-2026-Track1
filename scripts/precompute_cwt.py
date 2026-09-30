"""Pre-compute CWT (Continuous Wavelet Transform) and cache to disk.

Input:  processed/features/eeg/{sample_id}.npy  shape (29, 2000), raw EEG @ 500 Hz
Output: processed/features/cwt/{sample_id}.npy  shape (26, 64, 256) float16

Uses Morlet wavelet, 64 freq bins log-spaced 0.5–60 Hz.
Downsamples time axis from 2000 → 256 via cv2.resize to manage disk size.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pywt
from tqdm import tqdm


FS = 500
N_FREQS = 64
FREQ_LOW = 0.5
FREQ_HIGH = 60.0
TARGET_TIME = 256

# Pre-compute scales
_freqs = np.logspace(np.log10(FREQ_LOW), np.log10(FREQ_HIGH), N_FREQS)
SCALES = pywt.central_frequency("morl") * FS / _freqs


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
    # CWT per channel
    cwts = []
    for ch in range(wave.shape[0]):
        coef, _ = pywt.cwt(wave[ch], SCALES, "morl", sampling_period=1.0 / FS)
        # coef shape: (n_freqs=64, n_time=2000), complex
        mag = np.abs(coef).astype(np.float32)  # (64, 2000)
        # Downsample time: (64, 2000) → (64, 256) via cv2.resize
        mag_ds = cv2.resize(mag, (TARGET_TIME, N_FREQS), interpolation=cv2.INTER_LINEAR)
        cwts.append(mag_ds)
    img = np.stack(cwts)  # (26, 64, 256)
    img = np.log1p(img)
    np.save(out_path, img.astype(np.float16))
    return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", default="neuromm26_datasets/processed/features/eeg")
    parser.add_argument("--out-root", default="neuromm26_datasets/processed/features/cwt")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    eeg_root = Path(args.eeg_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(eeg_root.glob("*.npy"))
    print(f"Found {len(files)} EEG files; output to {out_root}")
    print(f"CWT: morlet, {N_FREQS} freqs {FREQ_LOW}–{FREQ_HIGH} Hz, time downsample → {TARGET_TIME}")
    jobs = [(f, out_root / f.name) for f in files]

    with Pool(args.workers) as p:
        results = list(tqdm(p.imap_unordered(process_one, jobs, chunksize=16),
                            total=len(jobs), desc="cwt"))
    n_done = results.count("done")
    n_skip = results.count("skip")
    print(f"Done: {n_done}  Skipped: {n_skip}  Total: {len(jobs)}")

    total_bytes = sum(f.stat().st_size for f in out_root.glob("*.npy"))
    print(f"Cache size: {total_bytes / 1e9:.2f} GB")

    sample = np.load(jobs[0][1])
    print(f"Sample shape: {sample.shape}  dtype: {sample.dtype}")


if __name__ == "__main__":
    main()
