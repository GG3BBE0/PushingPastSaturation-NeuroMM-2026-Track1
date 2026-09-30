"""Noise-robust NM weighting (zero-risk, no retraining, no relabeling, no candidate touch).

Hypothesis: the 187 confident-learning-flagged OOF samples are label noise; deriving NM
ensemble weights to fit them slightly mis-tunes the weights. Test: nested honest OOF where
each fold's weight DERIVATION excludes the flagged samples, but every fold is PREDICTED and
SCORED on its full ORIGINAL labels. If honest OOF rises vs the baseline (derive-on-all), the
noise was hurting weight derivation and the robust weights generalize better.

Run capped: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/nm_noise_robust.py
"""
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps

sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import POOL_LOOKUP, load_oof, derive_nm, TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
FOLD = REPO / "fold_df_fixed.csv"
NOISE = REPO / "neuromm26_results/fold4_noise_sids.txt"

flagged = {l.strip() for l in NOISE.read_text().splitlines() if l.strip()}
# exclude the not-yet-trained clean/base experiment archs from this pool analysis
EXCLUDE_NAMES = {"ConcatCWT clean256", "ConcatCWT cleanbase256", "NeuroMAE v1",
                 "GNN orthopara", "Spec3D superlet"}
names = [str(n) for n in np.load(NPZ, allow_pickle=True)["names"] if str(n) not in EXCLUDE_NAMES]
print(f"pool: {len(names)} archs; flagged-noise sids: {len(flagged)}", flush=True)

fdf = pd.read_csv(FOLD); fdf["sample_id"] = fdf["sample_id"].astype(str)
sid_fold = dict(zip(fdf["sample_id"], fdf["fold"].astype(int)))

ref_s = ref_y = None; cols = []; ap = []
for nm in names:
    pfx, root = POOL_LOOKUP[nm]
    s, l, y = load_oof(pfx, root); ss = s.astype(str)
    if ref_s is None:
        ref_s, ref_y = ss, y.astype(int); cols.append(l)
    else:
        im = {x: i for i, x in enumerate(ss)}
        cols.append(l[np.array([im[x] for x in ref_s])])
    ap.append(aps(y, 1 / (1 + np.exp(-l))))
X = np.stack(cols, 1); ap = np.array(ap)
rf = np.array([sid_fold[s] for s in ref_s])
is_noise = np.array([s in flagged for s in ref_s])
print(f"flagged in OOF: {is_noise.sum()}", flush=True)

# trim (same two-dim rule)
wb, _ = derive_nm(X, ref_y, n_restarts=3)
keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
Xt = X[:, keep]
print(f"trim -> {keep.sum()} archs", flush=True)


def nested(exclude_noise_in_derivation):
    pred = np.zeros(len(ref_y))
    for k in range(5):
        tr, te = rf != k, rf == k
        if exclude_noise_in_derivation:
            tr = tr & ~is_noise
        wk, _ = derive_nm(Xt[tr], ref_y[tr], n_restarts=3)
        pred[te] = Xt[te] @ wk
    return aps(ref_y, pred)  # scored on FULL original labels


base = nested(False)
robust = nested(True)
print(f"\nhonest nested OOF (original labels):")
print(f"  baseline (derive on all)        : {base:.4f}")
print(f"  noise-robust (derive ex-flagged): {robust:.4f}")
print(f"  Δ = {robust - base:+.4f}   "
      f"{'ROBUST helps' if robust - base > 0.001 else ('within noise floor' if abs(robust-base) <= 0.001 else 'ROBUST hurts')}", flush=True)
