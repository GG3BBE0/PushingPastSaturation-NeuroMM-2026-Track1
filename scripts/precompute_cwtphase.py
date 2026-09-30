"""Pre-compute CWT PHASE features (orthogonal to the magnitude CWT the pool already uses).

All existing specs keep |coef| (magnitude) and throw away the phase. Phase encodes the temporal
fine-structure / synchrony that a sharp spike imprints. We output cos(angle(coef)) downsampled
in time -> a bounded, CNN-friendly map of local phase structure, in the SAME (26,64,256) format
as cwt/ -> trains on the exact ConcatSpec MaxViT pipeline with zero new integration.

  python scripts/precompute_cwtphase.py            # train (25k)
  python scripts/precompute_cwtphase.py --eeg-root NeuroMM-2026/candidate/candidate/eeg --out-root .../cand
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pywt
from tqdm import tqdm

FS = 500; N_FREQS = 64; FREQ_LOW = 0.5; FREQ_HIGH = 60.0; TARGET_TIME = 256
_freqs = np.logspace(np.log10(FREQ_LOW), np.log10(FREQ_HIGH), N_FREQS)
SCALES = pywt.central_frequency("morl") * FS / _freqs


def normalize_29_to_26(wave):
    wave = wave.astype(np.float32, copy=True)
    if wave.shape[0] >= 29:
        wave[:23] = wave[:23] / 1e-3; wave[23:] = wave[23:] * 1e-2
        h = wave[23] - wave[24]; m1 = wave[25] - wave[26]; m2 = wave[27] - wave[28]
        wave = np.concatenate([wave[:23], np.stack([h, m1, m2], 0)], 0)
    return wave


def pad_or_crop(wave, target=(26, 2000)):
    out = np.zeros(target, dtype=np.float32); c, t = wave.shape
    out[:min(c, target[0]), :min(t, target[1])] = wave[:min(c, target[0]), :min(t, target[1])]
    return out


def process_one(args):
    in_path, out_path = args
    if out_path.exists():
        return "skip"
    wave = pad_or_crop(normalize_29_to_26(np.load(in_path)))
    feats = []
    for ch in range(wave.shape[0]):
        coef, _ = pywt.cwt(wave[ch], SCALES, "morl", sampling_period=1.0 / FS)  # (64,2000) complex
        cphi = np.cos(np.angle(coef)).astype(np.float32)        # phase cosine, bounded [-1,1]
        ds = cv2.resize(cphi, (TARGET_TIME, N_FREQS), interpolation=cv2.INTER_AREA)  # area = local phase mean
        feats.append(ds)
    img = np.stack(feats)  # (26,64,256)
    np.save(out_path, img.astype(np.float16))
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-root", default="neuromm26_datasets/processed/features/eeg")
    ap.add_argument("--out-root", default="neuromm26_datasets/processed/features/cwt_phase")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    eeg_root = Path(args.eeg_root); out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    files = sorted(eeg_root.glob("*.npy"))
    print(f"Found {len(files)} -> {out_root}  (CWT cos-phase, {N_FREQS}x{TARGET_TIME})", flush=True)
    jobs = [(f, out_root / f.name) for f in files]
    with Pool(args.workers) as p:
        res = list(tqdm(p.imap_unordered(process_one, jobs, chunksize=16), total=len(jobs), desc="cwtphase"))
    print(f"done={res.count('done')} skip={res.count('skip')} total={len(jobs)}")
    s = np.load(jobs[0][1]); print(f"sample {s.shape} {s.dtype}")


if __name__ == "__main__":
    main()
