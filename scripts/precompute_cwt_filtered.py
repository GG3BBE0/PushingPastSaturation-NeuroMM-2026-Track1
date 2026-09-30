"""Pre-compute FILTERED CWT cache (bandpass + notch + CAR + robust norm) → CWT.

New preprocessing variant for ensemble diversity (current pipeline does NO
filtering / montage). Pipeline per sample:
  raw (29,2000)
  -> normalize_29_to_26 (scale + ECG/EMG diff pairs) -> (26,2000)
  -> bandpass 0.5-70 Hz (zero-phase filtfilt)
  -> notch 50 Hz (powerline; zero-phase)
  -> CAR (common average reference) on the 23 EEG channels
  -> robust per-channel norm (median / IQR)
  -> Morlet CWT per channel -> log1p -> downsample time -> (26,64,256) f16

Output: processed/features/cwt_filtered/{sample_id}.npy
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch
from tqdm import tqdm

FS = 500
N_FREQS = 64
FREQ_LOW = 0.5
FREQ_HIGH = 60.0
TARGET_TIME = 256
N_EEG = 23  # first 23 channels are EEG; last 3 (of 26) are ECG/EMG diffs

_freqs = np.logspace(np.log10(FREQ_LOW), np.log10(FREQ_HIGH), N_FREQS)
SCALES = pywt.central_frequency("morl") * FS / _freqs

# Precompute filter coefficients (applied along time axis)
_BP_B, _BP_A = butter(4, [0.5, 70.0], btype="band", fs=FS)
_NOTCH_B, _NOTCH_A = iirnotch(50.0, Q=30.0, fs=FS)


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


def filter_signal(wave: np.ndarray) -> np.ndarray:
    """wave (26,2000) -> filtered + CAR + robust norm."""
    x = wave.astype(np.float64)
    # zero-phase bandpass + notch along time
    x = filtfilt(_BP_B, _BP_A, x, axis=-1)
    x = filtfilt(_NOTCH_B, _NOTCH_A, x, axis=-1)
    # CAR on the 23 EEG channels only
    car = x[:N_EEG].mean(axis=0, keepdims=True)
    x[:N_EEG] = x[:N_EEG] - car
    # robust per-channel normalization (median / IQR)
    med = np.median(x, axis=-1, keepdims=True)
    q75 = np.percentile(x, 75, axis=-1, keepdims=True)
    q25 = np.percentile(x, 25, axis=-1, keepdims=True)
    iqr = (q75 - q25)
    x = (x - med) / (iqr + 1e-6)
    return x.astype(np.float32)


def process_one(args):
    in_path, out_path = args
    if out_path.exists():
        return "skip"
    wave = np.load(in_path)
    wave = normalize_29_to_26(wave)
    wave = pad_or_crop(wave)
    wave = filter_signal(wave)
    cwts = []
    for ch in range(wave.shape[0]):
        coef, _ = pywt.cwt(wave[ch], SCALES, "morl", sampling_period=1.0 / FS)
        mag = np.abs(coef).astype(np.float32)
        mag_ds = cv2.resize(mag, (TARGET_TIME, N_FREQS), interpolation=cv2.INTER_LINEAR)
        cwts.append(mag_ds)
    img = np.stack(cwts)
    img = np.log1p(img)
    np.save(out_path, img.astype(np.float16))
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-root", default="neuromm26_datasets/processed/features/eeg")
    ap.add_argument("--out-root", default="neuromm26_datasets/processed/features/cwt_filtered")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    eeg_root = Path(args.eeg_root); out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    files = sorted(eeg_root.glob("*.npy"))
    print(f"Found {len(files)} EEG files -> {out_root}")
    print("Filter: bandpass 0.5-70 + notch 50 + CAR(23eeg) + robust norm, then Morlet CWT")
    jobs = [(f, out_root / f.name) for f in files]
    with Pool(args.workers) as p:
        res = list(tqdm(p.imap_unordered(process_one, jobs, chunksize=16), total=len(jobs), desc="cwt_filt"))
    print(f"done={res.count('done')} skip={res.count('skip')} total={len(jobs)}")
    tot = sum(f.stat().st_size for f in out_root.glob("*.npy"))
    print(f"cache size: {tot/1e9:.2f} GB")
    s = np.load(jobs[0][1]); print(f"sample shape: {s.shape} {s.dtype}")


if __name__ == "__main__":
    main()
