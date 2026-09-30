"""Worker: compute ONE arch's candidate logits on a given GPU, save to a temp .npy (no npz write,
so workers can run in parallel on GPU 1,2 without racing the shared npz). Merge happens separately.
Usage: python scripts/nchc_one_logit.py <prefix> <backbone> <feature> <tsize> <gpu> <out_npy>
"""
import sys
import os
from pathlib import Path
import numpy as np, torch
from neuromm26_baseline.tools.predict_candidate_full_pool import (
    infer_arch, RES, filt_eeg_signal, normalize_29_to_26,
)

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
CANDDIR = REPO / "NeuroMM-2026/candidate/candidate"
prefix, backbone, feature, tsize, gpu, out_npy = sys.argv[1:7]
tsize = int(tsize)
e = {"name": prefix, "prefix": prefix, "root": RES, "kind": "concat",
     "backbone": backbone, "feature": feature, "tsize": tsize}

ids = [l.strip() for l in (CANDDIR / "candidate_ids.txt").read_text().splitlines() if l.strip()]
caches_d = {k: {} for k in ("eeg", "cwt", "sl", "cwtfilt", "stft", "paul", "filteeg", "multiband")}
sub = {"cwt": ("cwt", "cwt"), "cwt_paul": ("paul", "cwt_paul"), "cwt_filtered": ("cwtfilt", "cwt_filtered"),
       "stft": ("stft", "stft"), "superlet": ("sl", "superlet")}
if feature in sub:
    key, dname = sub[feature]
    for sid in ids:
        caches_d[key][sid] = torch.from_numpy(np.load(CANDDIR / dname / f"{sid}.npy"))
elif feature == "filt_eeg":
    for sid in ids:
        caches_d["filteeg"][sid] = torch.from_numpy(filt_eeg_signal(normalize_29_to_26(np.load(CANDDIR / "eeg" / f"{sid}.npy"))))
else:
    raise SystemExit(f"feature '{feature}' unsupported")

caches = (caches_d["eeg"], caches_d["cwt"], caches_d["sl"], caches_d["cwtfilt"],
          caches_d["stft"], caches_d["paul"], caches_d["filteeg"], caches_d["multiband"])
device = torch.device(f"cuda:{gpu}")
print(f"[{prefix}] inferring on cuda:{gpu} ...", flush=True)
logit = infer_arch(e, ids, caches, device, 96)
np.save(out_npy, logit)
print(f"[{prefix}] saved {out_npy}: shape {logit.shape} mean {logit.mean():.3f} std {logit.std():.3f}", flush=True)
