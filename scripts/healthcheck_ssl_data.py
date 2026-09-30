"""Verify SSL data paths + filter consistency (train filt cache == recomputed filt)."""
import os
from pathlib import Path

import numpy as np
import torch

from neuromm26_baseline.datasets.ssl_eeg_dataset import SSLEEGDataset
from neuromm26_baseline.tools.predict_candidate_full_pool import filt_eeg_signal, normalize_29_to_26

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
TRAIN_FILT = REPO / "neuromm26_datasets/processed/features/filt_eeg"
TRAIN_RAW = REPO / "neuromm26_datasets/processed/features/eeg"
CAND = REPO / "NeuroMM-2026/candidate/candidate"
FOLD_CSV = REPO / "fold_df_fixed.csv"

print("== path / count checks ==")
n_filt = len(list(TRAIN_FILT.glob("*.npy")))
n_raw = len(list(TRAIN_RAW.glob("*.npy")))
cand_ids = [l.strip() for l in (CAND / "candidate_ids.txt").read_text().splitlines() if l.strip()]
n_cand_eeg = len(list((CAND / "eeg").glob("*.npy")))
print(f"  train filt_eeg cache : {n_filt}")
print(f"  train raw eeg cache  : {n_raw}")
print(f"  candidate ids        : {len(cand_ids)}")
print(f"  candidate eeg files  : {n_cand_eeg}")

print("== dataset (preload=False) ==")
ds = SSLEEGDataset(str(FOLD_CSV), str(TRAIN_FILT), str(CAND), cand_filt_root=None,
                   use_train=True, use_candidate=True, preload=False)
print(f"  len = {len(ds)} (expect ~45426)")
# find a train and a candidate entry
i_tr = next(i for i, (k, _) in enumerate(ds.entries) if k == "train")
i_ca = next(i for i, (k, _) in enumerate(ds.entries) if k == "cand")
s_tr = ds[i_tr]["eeg"]; s_ca = ds[i_ca]["eeg"]
print(f"  train sample : {tuple(s_tr.shape)} dtype={s_tr.dtype} finite={torch.isfinite(s_tr).all().item()}")
print(f"  cand  sample : {tuple(s_ca.shape)} dtype={s_ca.dtype} finite={torch.isfinite(s_ca).all().item()}")
assert s_tr.shape == (26, 2000) and s_ca.shape == (26, 2000)
assert torch.isfinite(s_tr).all() and torch.isfinite(s_ca).all()

print("== filter consistency: train filt cache == recomputed filt_eeg_signal(normalize(raw)) ==")
tr_sid = ds.entries[i_tr][1]
cache = np.load(TRAIN_FILT / f"{tr_sid}.npy").astype(np.float32)        # (26,2000) f16->f32
raw = np.load(TRAIN_RAW / f"{tr_sid}.npy")                              # (29,2000)
recomp = filt_eeg_signal(normalize_29_to_26(raw)).astype(np.float32)   # (26,2000)
maxdiff = float(np.abs(cache - recomp).max())
print(f"  sid={tr_sid} cache{cache.shape} recomp{recomp.shape} max|diff|={maxdiff:.5f} (f16 cache ~<0.01)")
assert maxdiff < 0.02, "filter mismatch — pretrain/finetune inputs would diverge"

print("== candidate on-the-fly == recomputed (sanity) ==")
ca_sid = ds.entries[i_ca][1]
raw_c = np.load(CAND / "eeg" / f"{ca_sid}.npy")
recomp_c = filt_eeg_signal(normalize_29_to_26(raw_c)).astype(np.float32)
maxdiff_c = float(np.abs(s_ca.numpy() - recomp_c).max())
print(f"  sid={ca_sid} max|diff|={maxdiff_c:.6f} (expect ~f16 rounding <0.02, dataset stores f16)")
assert maxdiff_c < 0.02

print("\nALL SSL DATA HEALTH CHECKS PASSED")
