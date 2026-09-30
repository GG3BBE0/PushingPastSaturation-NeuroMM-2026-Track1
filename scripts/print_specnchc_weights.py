"""Reproduce the EXACT NM weights of the 0.9846 specnchc submission (mirrors build_specnchc_submission.py
through the trim + bagged Nelder-Mead, then prints per-arch weight sorted desc). OOF-only, no build.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score as aps
sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import (POOL_LOOKUP, load_oof, derive_nm, derive_bagged_nm, apply_cap,
                                       TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW, REPO)

NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
SWAP_ALL = {
    "ConcatCWT maxvit256": "ConcatCWT specnchc 256",
    "ConcatPaul maxvit256": "ConcatPaul specnchc 256",
    "ConcatFilt maxvit256": "ConcatFilt specnchc 256",
    "ConcatSuperlet maxvit384": "ConcatSuperlet specnchc 384",
    "muku raw convnext_pico": "muku specnchc convnext",
}
PSE = ("pseudo", "relabel", "relbase", "rank256", "NeuroMAE", "GNN orthopara", "Spec3D", "r3nchc",
       " spec ", "_big_", "_soft_", "_ns_", "specnchc", "soft")
d = np.load(NPZ, allow_pickle=True)
npz_names = [str(n) for n in d["names"]]
canon = [n for n in npz_names if not any(k in n for k in PSE)]
pool = [SWAP_ALL.get(n, n) for n in canon]

ref_s = ref_y = None; Xo = []; ap = []
for nm in pool:
    pfx, root = POOL_LOOKUP[nm]; s, l, y = load_oof(pfx, root); ss = s.astype(str); l = np.clip(l, -30, 30)
    if ref_s is None:
        ref_s, ref_y = ss, y.astype(int); Xo.append(l)
    else:
        im = {x: i for i, x in enumerate(ss)}; Xo.append(l[np.array([im[x] for x in ref_s])])
    ap.append(aps(y, 1 / (1 + np.exp(-l))))
Xo = np.stack(Xo, 1); ap = np.array(ap)

wb, _ = derive_nm(Xo, ref_y, n_restarts=4)
keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
poolk = [p for p, k in zip(pool, keep) if k]
apk = ap[keep]
Xot = Xo[:, keep]
wD = apply_cap(derive_bagged_nm(Xot, ref_y, n_iter=40), 0.20)
oof = aps(ref_y, Xot @ wD)

order = np.argsort(-wD)
print(f"\n=== specnchc (public 0.9846) ensemble NM weights ===")
print(f"pool: {len(pool)} archs -> trim -> {len(poolk)} kept; ensemble OOF (in-sample) = {oof:.4f}\n")
print(f"{'rank':>4} {'weight':>8} {'single-OOF':>10}  arch")
for r, i in enumerate(order, 1):
    print(f"{r:>4} {wD[i]*100:>7.2f}% {apk[i]:>10.4f}  {poolk[i]}")
print(f"\nsum weights = {wD.sum():.4f}; specnchc-member weight = "
      f"{sum(w for p, w in zip(poolk, wD) if 'specnchc' in p)*100:.1f}%; "
      f"trimmed {len(pool)-len(poolk)} near-zero archs")
