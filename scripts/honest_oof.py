"""Honest (nested) OOF for the ensemble.

regnm_real.py derives NM weights by maximizing AUPRC on the SAME 25426 OOF samples it
then reports -> optimistic / in-sample -> OOF rises without public gain (v6/v7).
This evaluator is out-of-sample for the weights: for each held-out fold k, derive NM
weights on the OTHER 4 folds' OOF and predict fold k; concatenate the 5 held-out
predictions -> an honest ensemble OOF that tracks public far better. Use it to judge
whether a new arch ACTUALLY helps before spending a submission.

Usage: python scripts/honest_oof.py [comma,separated,arch,names,to,EXCLUDE]
"""
import sys
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as aps
sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import POOL_LOOKUP, load_oof, derive_nm, TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
FOLD = REPO / "fold_df_fixed.csv"
exclude = set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else set()

names = [str(n) for n in np.load(NPZ, allow_pickle=True)["names"] if str(n) not in exclude]
print(f"pool: {len(names)} archs (excluded: {sorted(exclude) if exclude else 'none'})", flush=True)

fold_df = pd.read_csv(FOLD); fold_df["sample_id"] = fold_df["sample_id"].astype(str)
sid_fold = dict(zip(fold_df["sample_id"], fold_df["fold"]))

ref_s = ref_y = None; cols = []; ap = []
for nm in names:
    pfx, root = POOL_LOOKUP[nm]
    s, l, y = load_oof(pfx, root); ss = s.astype(str)
    if ref_s is None:
        ref_s, ref_y = ss, y; cols.append(l)
    else:
        im = {x: i for i, x in enumerate(ss)}
        cols.append(l[np.array([im[x] for x in ref_s])])
    ap.append(aps(y, 1 / (1 + np.exp(-l))))
X = np.stack(cols, 1); ap = np.array(ap)
rf = np.array([sid_fold[s] for s in ref_s])

# trim (same two-dim rule as regnm), using full-pool baseline NM weight
wb, _ = derive_nm(X, ref_y, n_restarts=3)
keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
Xt = X[:, keep]
print(f"trim -> {keep.sum()} archs", flush=True)

# in-sample (optimistic) reference
w_in, oof_in = derive_nm(Xt, ref_y, n_restarts=4)
print(f"in-sample (optimistic) OOF: {oof_in:.4f}", flush=True)

# honest nested: weights from the other 4 folds, predict the held-out fold
pred = np.zeros(len(ref_y))
for k in range(5):
    tr, te = rf != k, rf == k
    wk, _ = derive_nm(Xt[tr], ref_y[tr], n_restarts=3)
    pred[te] = Xt[te] @ wk
    print(f"  fold{k} honest OOF: {aps(ref_y[te], pred[te]):.4f}", flush=True)
honest = aps(ref_y, pred)
print(f"\nHONEST nested OOF: {honest:.4f}   (in-sample {oof_in:.4f}, optimism gap +{oof_in - honest:.4f})", flush=True)
