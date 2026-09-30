"""Compute the muku_specnchc candidate logit (kind=mukuv1, eeg) and save to a temp npy."""
import sys
import os
from pathlib import Path
import numpy as np, torch
from neuromm26_baseline.tools.predict_candidate_full_pool import infer_arch, RES, normalize_29_to_26
REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
CAND = REPO / "NeuroMM-2026/candidate/candidate"
gpu, out = sys.argv[1], sys.argv[2]
ids = [l.strip() for l in (CAND / "candidate_ids.txt").read_text().splitlines() if l.strip()]
print(f"loading {len(ids)} eeg candidates ...", flush=True)
eeg = {sid: torch.from_numpy(normalize_29_to_26(np.load(CAND / "eeg" / f"{sid}.npy"))) for sid in ids}
empty = {}
caches = (eeg, empty, empty, empty, empty, empty, empty, empty)  # (eeg,cwt,sl,cwtfilt,stft,paul,filteeg,multiband)
e = {"name": "muku specnchc convnext", "prefix": "muku_specnchc_fold__convnext_pico_d1_in1k",
     "root": RES, "kind": "mukuv1", "backbone": "convnext_pico.d1_in1k", "feature": "eeg", "tsize": 224}
logit = infer_arch(e, ids, caches, torch.device(f"cuda:{gpu}"), 128)
np.save(out, logit)
print(f"saved {out}: mean {logit.mean():.3f} std {logit.std():.3f}", flush=True)
