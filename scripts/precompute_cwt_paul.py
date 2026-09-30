"""Pre-compute CWT with Paul wavelet (m=4) — kfuji-style.

Output: processed/features/cwt_paul/{sample_id}.npy  shape (26, 64, 256) float16
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
WAVELET = "cmor1.5-1.0"  # PyWavelets doesn't have Paul directly, use complex morlet

# Actually pywt doesn't have Paul wavelet built-in for cwt. Let me use complex morlet
# with different parameters (B=1.5, C=1.0) as approximation of "different wavelet".
# Or implement Paul manually.

# For Paul wavelet, implement manually:
def paul_cwt(signal: np.ndarray, scales: np.ndarray, m: int = 4) -> np.ndarray:
    """
    Compute CWT with Paul wavelet of order m.

    Paul wavelet (analytic): ψ(t) = (2^m i^m m!) / sqrt(π (2m)!) * (1 - i t)^(-(m+1))

    Returns: (n_scales, n_time) complex
    """
    from scipy.special import factorial
    norm = (2**m * 1j**m * factorial(m)) / np.sqrt(np.pi * factorial(2 * m))
    n = len(signal)
    cwt_result = np.zeros((len(scales), n), dtype=np.complex64)
    for k, s in enumerate(scales):
        # Cap wavelet length to (n - 1) so np.convolve mode='same' returns n samples
        half = min((n - 1) // 2, int(4 * s))
        t = np.arange(-half, half + 1) / s
        wavelet = (norm * (1 - 1j * t) ** (-(m + 1))) / np.sqrt(s)
        wavelet = wavelet.astype(np.complex64)
        out = np.convolve(signal, wavelet, mode="same")
        # Just in case, crop or pad to exact length n
        if len(out) >= n:
            cwt_result[k] = out[:n]
        else:
            cwt_result[k, :len(out)] = out
    return cwt_result


# Pre-compute scales (Paul: freq = (2m+1) / (4π s))
PAUL_M = 4
_freqs = np.logspace(np.log10(FREQ_LOW), np.log10(FREQ_HIGH), N_FREQS)
SCALES_PAUL = (2 * PAUL_M + 1) / (4 * np.pi * _freqs) * FS


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
    cwts = []
    for ch in range(wave.shape[0]):
        coef = paul_cwt(wave[ch], SCALES_PAUL, m=PAUL_M)
        mag = np.abs(coef).astype(np.float32)  # (64, 2000)
        mag_ds = cv2.resize(mag, (TARGET_TIME, N_FREQS), interpolation=cv2.INTER_LINEAR)
        cwts.append(mag_ds)
    img = np.stack(cwts)
    img = np.log1p(img)
    np.save(out_path, img.astype(np.float16))
    return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", default="neuromm26_datasets/processed/features/eeg")
    parser.add_argument("--out-root", default="neuromm26_datasets/processed/features/cwt_paul")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    eeg_root = Path(args.eeg_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(eeg_root.glob("*.npy"))
    print(f"Found {len(files)} EEG files; output to {out_root}")
    print(f"Paul wavelet m={PAUL_M}, {N_FREQS} freqs {FREQ_LOW}–{FREQ_HIGH} Hz, time → {TARGET_TIME}")
    jobs = [(f, out_root / f.name) for f in files]

    with Pool(args.workers) as p:
        results = list(tqdm(p.imap_unordered(process_one, jobs, chunksize=8),
                            total=len(jobs), desc="cwt_paul"))
    n_done = results.count("done")
    n_skip = results.count("skip")
    print(f"Done: {n_done}  Skipped: {n_skip}")

    total_bytes = sum(f.stat().st_size for f in out_root.glob("*.npy"))
    print(f"Cache size: {total_bytes / 1e9:.2f} GB")
    sample = np.load(jobs[0][1])
    print(f"Sample shape: {sample.shape}  dtype: {sample.dtype}")


if __name__ == "__main__":
    main()
