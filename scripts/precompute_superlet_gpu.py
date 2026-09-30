"""GPU-based Superlet Transform precompute.

Implements the (additive) Superlet Transform from Moca et al. 2021:
  S(f, t) = geometric mean over c ∈ {c_min, ..., c_max} of |CWT_Morlet(f, c)|

For each freq f, we average log-magnitudes of multiple Morlet wavelets with
different cycle counts. Higher freq usually wants higher cycles for tightness.

Input:  processed/features/eeg/{sample_id}.npy  shape (29, 2000) float32
Output: processed/features/superlet/{sample_id}.npy  shape (26, 64, 256) float16
        log1p-applied geometric-mean magnitude

Run with --shard-idx + --n-shards to split work across multiple GPUs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


FS = 500
N_FREQS = 64
FREQ_LOW = 0.5
FREQ_HIGH = 60.0
TARGET_TIME = 256
CYCLE_MIN = 3
CYCLE_MAX = 7        # 5 orders: cycles ∈ {3,4,5,6,7}
N_ORDERS = CYCLE_MAX - CYCLE_MIN + 1
T_LEN = 2000


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


def pad_or_crop(wave, target=(26, T_LEN)):
    out = np.zeros(target, dtype=np.float32)
    c, t = wave.shape
    out[: min(c, target[0]), : min(t, target[1])] = wave[: min(c, target[0]), : min(t, target[1])]
    return out


def build_morlet_freq_kernels(freqs: np.ndarray, cycles: list[int], n_time: int, fs: int, device: torch.device):
    """Return precomputed Morlet wavelet FFTs.

    Output shape: (N_ORDERS, N_FREQS, n_time) complex64, on `device`.

    Each kernel is the FFT of a complex Morlet wavelet at given (freq, cycles).
    """
    t = (np.arange(n_time) - n_time // 2) / fs   # centered time axis (s)
    kernels = np.zeros((len(cycles), len(freqs), n_time), dtype=np.complex64)
    for c_i, c in enumerate(cycles):
        for f_i, f in enumerate(freqs):
            sigma = c / (2 * math.pi * f)   # Gaussian width
            wavelet = (1.0 / (sigma * math.sqrt(math.pi))) ** 0.5
            wavelet = wavelet * np.exp(2j * math.pi * f * t) * np.exp(-(t ** 2) / (2 * sigma ** 2))
            kernels[c_i, f_i] = wavelet.astype(np.complex64)
    # FFT (with shift to align)
    kernels_fft = np.fft.fft(np.fft.ifftshift(kernels, axes=-1), axis=-1)
    return torch.from_numpy(kernels_fft).to(device)


@torch.no_grad()
def superlet_batch(eegs: torch.Tensor, morlet_fft: torch.Tensor) -> torch.Tensor:
    """Compute superlet for a batch of EEG signals.

    Args:
        eegs: (B, C, T) float32 — already padded to T_LEN
        morlet_fft: (N_ORDERS, N_FREQS, T) complex64

    Returns:
        (B, C, F, T) float32 — geometric mean of |CWT| over orders.
    """
    B, C, T = eegs.shape
    N_ORDERS_LOCAL, N_F, _ = morlet_fft.shape

    # FFT signal once: (B*C, T)
    x = eegs.reshape(B * C, T).to(torch.complex64)
    X = torch.fft.fft(x, dim=-1)                 # (B*C, T) complex

    # Accumulate log-magnitude in (B*C, N_F, T) float32
    log_acc = torch.zeros(B * C, N_F, T, dtype=torch.float32, device=eegs.device)
    eps = 1e-7

    for o in range(N_ORDERS_LOCAL):
        # (N_F, T) → broadcast → (1, N_F, T)
        K = morlet_fft[o].unsqueeze(0)
        Y = X.unsqueeze(1) * K                   # (B*C, N_F, T) complex
        y = torch.fft.ifft(Y, dim=-1)            # (B*C, N_F, T) complex
        mag = torch.abs(y)                       # (B*C, N_F, T) float
        log_acc = log_acc + torch.log(mag + eps)
        del Y, y, mag

    # Geometric mean: exp(mean log)
    geo_mean = torch.exp(log_acc / N_ORDERS_LOCAL)   # (B*C, N_F, T)
    geo_mean = geo_mean.reshape(B, C, N_F, T)
    return geo_mean


def downsample_time(arr_bcft: np.ndarray, target_t: int) -> np.ndarray:
    """(B,C,F,T) → (B,C,F,target_t) via cv2.resize per (B,C)."""
    B, C, F_, T = arr_bcft.shape
    out = np.empty((B, C, F_, target_t), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            out[b, c] = cv2.resize(arr_bcft[b, c], (target_t, F_), interpolation=cv2.INTER_LINEAR)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-root", default="neuromm26_datasets/processed/features/eeg")
    ap.add_argument("--out-root", default="neuromm26_datasets/processed/features/superlet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[{args.device}] Superlet shard {args.shard_idx}/{args.n_shards}")
    print(f"  freqs: {N_FREQS} log-spaced {FREQ_LOW}-{FREQ_HIGH} Hz")
    print(f"  cycles: {list(range(CYCLE_MIN, CYCLE_MAX+1))} ({N_ORDERS} orders)")

    freqs = np.logspace(np.log10(FREQ_LOW), np.log10(FREQ_HIGH), N_FREQS)
    cycles = list(range(CYCLE_MIN, CYCLE_MAX + 1))
    morlet_fft = build_morlet_freq_kernels(freqs, cycles, T_LEN, FS, device)
    print(f"  morlet_fft: {tuple(morlet_fft.shape)} {morlet_fft.dtype} on {morlet_fft.device}")

    eeg_root = Path(args.eeg_root)
    all_files = sorted(eeg_root.glob("*.npy"))
    files = [f for i, f in enumerate(all_files) if i % args.n_shards == args.shard_idx]
    print(f"  files this shard: {len(files)} / total {len(all_files)}")

    # Filter out already-done
    todo = [f for f in files if not (out_root / f.name).exists()]
    print(f"  todo: {len(todo)} ({len(files)-len(todo)} already cached)")

    # Process in batches
    pbar = tqdm(range(0, len(todo), args.batch_size), desc=f"shard{args.shard_idx}")
    for i_start in pbar:
        batch_files = todo[i_start: i_start + args.batch_size]
        # Load + preprocess on CPU
        eegs = np.zeros((len(batch_files), 26, T_LEN), dtype=np.float32)
        for bi, f in enumerate(batch_files):
            wave = np.load(f)
            wave = normalize_29_to_26(wave)
            wave = pad_or_crop(wave)
            eegs[bi] = wave
        eegs_t = torch.from_numpy(eegs).to(device, non_blocking=True)
        # GPU superlet
        sl = superlet_batch(eegs_t, morlet_fft)            # (B, 26, 64, T_LEN) float32
        sl_np = sl.detach().cpu().numpy()
        # Downsample time + log1p + save
        sl_ds = downsample_time(sl_np, TARGET_TIME)        # (B, 26, 64, 256)
        sl_ds = np.log1p(sl_ds)
        for bi, f in enumerate(batch_files):
            out_path = out_root / f.name
            np.save(out_path, sl_ds[bi].astype(np.float16))

    # Report
    cached = list(out_root.glob("*.npy"))
    total_bytes = sum(f.stat().st_size for f in cached)
    print(f"[{args.device}] DONE shard {args.shard_idx}: total cached = {len(cached)} files, {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
