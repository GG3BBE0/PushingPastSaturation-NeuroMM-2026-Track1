"""Append ONE arch's candidate logits to candidate_arch_logits.npz WITHOUT re-running
the whole ~2.7h predict. Reuses predict's own infer_arch (identical inference path);
only the candidate feature-cache load is replicated (verbatim from predict). Use when a
single new arch (e.g. an NCHC return) is added on top of an already-computed pool.

Usage: python scripts/append_arch_to_npz.py <name> <prefix> <backbone> <feature> <tsize> [kind]
       (kind defaults to "concat"; pass e.g. "neuromae" for non-concat archs)
"""
import sys
import os
from pathlib import Path
import numpy as np, torch
from neuromm26_baseline.tools.predict_candidate_full_pool import (
    infer_arch, RES, filt_eeg_signal, normalize_29_to_26,
)

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
CANDDIR = REPO / "NeuroMM-2026/candidate/candidate"

argv = sys.argv[1:]
name, prefix, backbone, feature, tsize = argv[:5]
kind = argv[5] if len(argv) > 5 else "concat"
tsize = int(tsize)
e = {"name": name, "prefix": prefix, "root": RES, "kind": kind,
     "backbone": backbone, "feature": feature, "tsize": tsize}

ids = [l.strip() for l in (CANDDIR / "candidate_ids.txt").read_text().splitlines() if l.strip()]
print(f"candidate ids: {len(ids)}; loading '{feature}' candidate cache ...", flush=True)

empty = {}
caches_d = {k: {} for k in ("eeg", "cwt", "sl", "cwtfilt", "stft", "paul", "filteeg", "multiband")}
# verbatim from predict_candidate_full_pool.main() candidate loaders
if feature == "cwt":
    for sid in ids: caches_d["cwt"][sid] = torch.from_numpy(np.load(CANDDIR / "cwt" / f"{sid}.npy"))
elif feature == "cwt_paul":
    for sid in ids: caches_d["paul"][sid] = torch.from_numpy(np.load(CANDDIR / "cwt_paul" / f"{sid}.npy"))
elif feature == "cwt_filtered":
    for sid in ids: caches_d["cwtfilt"][sid] = torch.from_numpy(np.load(CANDDIR / "cwt_filtered" / f"{sid}.npy"))
elif feature == "stft":
    for sid in ids: caches_d["stft"][sid] = torch.from_numpy(np.load(CANDDIR / "stft" / f"{sid}.npy"))
elif feature == "superlet":
    for sid in ids: caches_d["sl"][sid] = torch.from_numpy(np.load(CANDDIR / "superlet" / f"{sid}.npy"))
elif feature == "filt_eeg":
    # filtered raw EEG, computed on-the-fly (verbatim from predict main()) -> batch["eeg"]
    for sid in ids:
        wave = normalize_29_to_26(np.load(CANDDIR / "eeg" / f"{sid}.npy"))
        caches_d["filteeg"][sid] = torch.from_numpy(filt_eeg_signal(wave))
else:
    raise SystemExit(f"feature '{feature}' not supported by this fast-append helper")

caches = (caches_d["eeg"], caches_d["cwt"], caches_d["sl"], caches_d["cwtfilt"],
          caches_d["stft"], caches_d["paul"], caches_d["filteeg"], caches_d["multiband"])
device = torch.device("cuda:1")
print("inferring (5-fold checkpoint average) ...", flush=True)
logit = infer_arch(e, ids, caches, device, 96)
print(f"new logit: shape {logit.shape} mean {logit.mean():.3f} std {logit.std():.3f}", flush=True)

d = np.load(NPZ, allow_pickle=True)
names = [str(n) for n in d["names"]]
X = d["logits"]
npz_ids = [str(x) for x in d["ids"]]
assert npz_ids == ids, "npz id order mismatch vs candidate_ids.txt"
if name in names:
    raise SystemExit(f"'{name}' already in npz ({len(names)} archs) — aborting to avoid dup")

# sanity: same-backbone sibling correlation (heavyaug384 vs plain maxvit384)
sib = "ConcatCWT maxvit384"
if sib in names:
    j = names.index(sib)
    r = float(np.corrcoef(logit, X[:, j])[0, 1])
    print(f"sanity: corr(new, '{sib}') = {r:.4f}  (expect high ~0.9 for same backbone)", flush=True)

names.append(name)
X2 = np.concatenate([X, logit[:, None]], axis=1)
np.savez(NPZ, names=np.array(names), logits=X2, ids=np.array(npz_ids))
print(f"appended '{name}': npz now {X2.shape[1]} archs", flush=True)
