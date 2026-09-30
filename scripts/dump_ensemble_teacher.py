"""Dump the nchc_n2filt ENSEMBLE honest-OOF prediction per TRAIN sample (sid -> prob) = the REAL
teacher the candidate-specialist used. The single-arch (cwt) sim teacher was pessimistic (more wrong
on hard samples). This faithful teacher tests whether a more-accurate teacher shrinks the fold4 harm.
Reproduces the nchc_n2filt pool + trim+bagged NM on OOF, then writes the per-sample OOF prediction.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score as aps

sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import (POOL_LOOKUP, load_oof, derive_nm, derive_bagged_nm, apply_cap,
                                       TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW, REPO)

NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
SWAP = {"ConcatCWT maxvit256": "ConcatCWT pseudo_nchc 256", "ConcatPaul maxvit256": "ConcatPaul pseudo_nchc 256",
        "ConcatFilt maxvit256": "ConcatFilt pseudo_nchc 256"}
PSE = ("pseudo", "relabel", "relbase", "rank256", "NeuroMAE", "GNN orthopara", "Spec3D", "r3nchc", " spec ", "_big_", "_soft_", "_ns_")
npz_names = [str(n) for n in np.load(NPZ, allow_pickle=True)["names"]]
canon = [n for n in npz_names if not any(k in n for k in PSE)]
pool = [SWAP.get(n, n) for n in canon]

ref = None; Xo = []; ap = []
for nm in pool:
    pfx, root = POOL_LOOKUP[nm]; s, l, y = load_oof(pfx, root); ss = s.astype(str); l = np.clip(l, -30, 30)
    if ref is None:
        ref = ss; yy = y.astype(int); Xo.append(l)
    else:
        im = {x: i for i, x in enumerate(ss)}; Xo.append(l[np.array([im[x] for x in ref])])
    ap.append(aps(y, 1 / (1 + np.exp(-l))))
Xo = np.stack(Xo, 1); ap = np.array(ap)
wb, _ = derive_nm(Xo, yy, n_restarts=4)
keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
wD = apply_cap(derive_bagged_nm(Xo[:, keep], yy, n_iter=40), 0.20)
prob = 1 / (1 + np.exp(-(Xo[:, keep] @ wD)))
print(f"ensemble teacher OOF AUPRC = {aps(yy, prob):.4f} over {len(ref)} train samples", flush=True)
(REPO / "ensemble_teacher_oof.txt").write_text("\n".join(f"{s} {p:.6f}" for s, p in zip(ref, prob)) + "\n")
print(f"wrote ensemble_teacher_oof.txt ({len(ref)})", flush=True)
