"""Fast reusable OOF-only gate: does ADDING one new arch's OOF to the canonical pool raise the
honest nested OOF? Multi-seed paired-Δ, force-keep (exempt from trim). No candidate inference
needed -> seconds-to-minutes per arch. Use for every campaign experiment before committing.

  python scripts/honest_add_check.py <oof_prefix> [root=RES]
  e.g. python scripts/honest_add_check.py riemann_logeuc_multiband_fold__filteeg
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score as aps

sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import POOL_LOOKUP, load_oof, RES, REPO, TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW

PREFIX = sys.argv[1]
NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
FOLD = REPO / "fold_df_fixed.csv"
SEEDS = [11, 23, 37]
EXCL = {"ConcatCWT relabel256", "ConcatCWT relbase256", "ConcatPaul relabel256", "ConcatSTFT relabel256",
        "ConcatCWT rank256", "ConcatPaul rank256", "NeuroMAE v1", "GNN orthopara", "Spec3D superlet"}

base_names = [str(n) for n in np.load(NPZ, allow_pickle=True)["names"] if str(n) not in EXCL]
fdf = pd.read_csv(FOLD); fdf["sample_id"] = fdf["sample_id"].astype(str)
sid_fold = dict(zip(fdf["sample_id"], fdf["fold"].astype(int)))

ref_s = ref_y = None; cols = []; ap = []
for nm in base_names:
    pfx, root = POOL_LOOKUP[nm]; s, l, y = load_oof(pfx, root); ss = s.astype(str)
    if ref_s is None:
        ref_s, ref_y = ss, y.astype(int); cols.append(l)
    else:
        im = {x: i for i, x in enumerate(ss)}; cols.append(l[np.array([im[x] for x in ref_s])])
    ap.append(aps(y, 1 / (1 + np.exp(-l))))
# the new arch
s, l, y = load_oof(PREFIX, RES); ss = s.astype(str); im = {x: i for i, x in enumerate(ss)}
new_col = l[np.array([im[x] for x in ref_s])]
new_ap = aps(ref_y, 1 / (1 + np.exp(-new_col)))
X = np.stack(cols, 1); ap = np.array(ap)
rf = np.array([sid_fold[s] for s in ref_s])
# corr vs strongest pool arch
best_j = int(np.argmax(ap))
corr = float(np.corrcoef(new_col, X[:, best_j])[0, 1])
print(f"new arch '{PREFIX}': 5-fold OOF AUPRC={new_ap:.4f}  corr-vs-strongest={corr:.3f}", flush=True)


def nm(Xa, ya, seed):
    M = Xa.shape[1]
    def neg(w):
        w = np.maximum(w, 0); sm = w.sum(); w = w / sm if sm > 0 else np.ones_like(w) / M
        return -aps(ya, Xa @ w)
    rng = np.random.default_rng(seed); bw, ba = None, -1
    for _ in range(2):
        w0 = np.ones(M) / M + 0.05 * rng.standard_normal(M); w0 = np.maximum(w0, 1e-3); w0 /= w0.sum()
        r = minimize(neg, w0, method="Nelder-Mead", options={"maxiter": 8000, "xatol": 1e-7, "fatol": 1e-7})
        w = np.maximum(r.x, 0); w /= w.sum(); a = aps(ya, Xa @ w)
        if a > ba: ba, bw = a, w
    return bw


def honest(withnew, seed):
    Xb = np.concatenate([X, new_col[:, None]], 1) if withnew else X
    apv = np.concatenate([ap, [new_ap]]) if withnew else ap
    wb = nm(Xb, ref_y, seed)
    keep = ~((apv < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
    if withnew: keep[-1] = True
    Xt = Xb[:, keep]
    pred = np.zeros(len(ref_y))
    for k in range(5):
        tr, te = rf != k, rf == k
        pred[te] = Xt[te] @ nm(Xt[tr], ref_y[tr], seed + k)
    return aps(ref_y, pred)


ds = []
for sd in SEEDS:
    b = honest(False, sd); w = honest(True, sd); ds.append(w - b)
    print(f"  seed{sd}: base={b:.4f} +new={w:.4f}  Δ={w-b:+.4f}", flush=True)
md = float(np.mean(ds))
verdict = "ROBUST+ (add it)" if min(ds) > 0.0005 else ("mixed" if max(ds) > 0 else "NO")
print(f"meanΔ={md:+.4f}  {verdict}", flush=True)
