"""Gate ALL NCHC returns (4-job + spec3d) — OOF-only, fast NM.

Gates: 1 validate, 2 Spearman corr (<0.85), 3 honest nested-CV OOF (THE gate).
Fast local Nelder-Mead (maxiter 8000, identical settings across all configs -> fair relative Δ).
Bar = v5-best honest baseline (current 48 minus the 2 known-bad correlated archs).
Integration rule: arch worth adding ONLY if it raises honest OOF above the bar.
"""
import sys, time, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score as aps
sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import POOL_LOOKUP, load_oof, RES, REPO, \
    TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
FOLD = REPO / "fold_df_fixed.csv"
NPZ = RES / "candidate_arch_logits.npz"

NEW = {
    "GNN orthopara":        ("eeg_gnn_fold__gcn", RES),
    "Spec3D scratch":       ("spec3d_superlet_scratch_fold__r3d_18", RES),
    "ConcatPaul hvy384":    ("concat_cwt_paul_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatFilt hvy384":    ("concat_cwt_filtered_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
    "Superlet maxvit_base hvy384": ("concat_superlet_heavyaug_fold__maxvit_base_tf_384_in1k", RES),
}

def fast_nm(X, y, n_restarts=2, maxiter=8000, seed=42):
    M = X.shape[1]
    def neg(w):
        w = np.maximum(w, 0); sm = w.sum(); w = w/sm if sm > 0 else np.ones_like(w)/M
        return -aps(y, X @ w)
    rng = np.random.default_rng(seed); bw, ba = None, -1
    for _ in range(n_restarts):
        w0 = np.ones(M)/M + 0.05*rng.standard_normal(M); w0 = np.maximum(w0, 1e-3); w0 /= w0.sum()
        r = minimize(neg, w0, method="Nelder-Mead",
                     options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6})
        w = np.maximum(r.x, 0); w /= w.sum(); a = aps(y, X @ w)
        if a > ba: ba, bw = a, w
    return bw, ba

fold_df = pd.read_csv(FOLD); fold_df["sample_id"] = fold_df["sample_id"].astype(str)
sid_fold = dict(zip(fold_df["sample_id"], fold_df["fold"]))

# ---- 1. validate ----
log("=== GATE 1: validate per-fold AUPRC ===")
for name, (pfx, root) in NEW.items():
    s, l, y = load_oof(pfx, root); ss = s.astype(str)
    folds = np.array([sid_fold[x] for x in ss])
    per = [aps(y[folds == k], 1/(1+np.exp(-l[folds == k]))) for k in range(5)]
    log(f"  {name:30s} {'/'.join(f'{p:.3f}' for p in per)} mean={np.mean(per):.4f} pooled={aps(y,1/(1+np.exp(-l))):.4f}")

# ---- build 48-arch pool ----
log("=== load pool ===")
pool_names = [str(n) for n in np.load(NPZ, allow_pickle=True)["names"]]
ref_s = None; pool_cols = {}; pool_ap = {}
for nm in pool_names:
    pfx, root = POOL_LOOKUP[nm]; s, l, y = load_oof(pfx, root); ss = s.astype(str)
    if ref_s is None:
        ref_s, ref_y = ss, y.astype(np.int32); im = {x: i for i, x in enumerate(ref_s)}
    col = np.empty(len(ref_s)); col[[im[x] for x in ss]] = l
    pool_cols[nm] = col.astype(np.float64); pool_ap[nm] = aps(ref_y, 1/(1+np.exp(-l)))
rf = np.array([sid_fold[s] for s in ref_s])
new_aligned = {}
for name, (pfx, root) in NEW.items():
    s, l, y = load_oof(pfx, root); ss = s.astype(str); im2 = {x: i for i, x in enumerate(ss)}
    new_aligned[name] = np.array([l[im2[x]] for x in ref_s]).astype(np.float64)
log(f"aligned: {len(ref_s)} samples, pool={len(pool_names)}")

# ---- 2. correlation gate ----
log("=== GATE 2: Spearman(OOF) vs pool — <0.85 ===")
for name, col in new_aligned.items():
    cors = sorted(((spearmanr(col, pool_cols[pn]).correlation, pn) for pn in pool_names), key=lambda t: -t[0])
    mx = cors[0][0]
    log(f"  {name:30s} max={mx:.3f} vs '{cors[0][1]}'  [{'PASS' if mx<0.85 else 'FAIL-redundant'}]  next: {cors[1][0]:.3f},{cors[2][0]:.3f}")

# ---- 3. honest gate ----
def honest(names_subset, extra=None):
    cols = [pool_cols[n] for n in names_subset]; ap = [pool_ap[n] for n in names_subset]
    if extra:
        for n, c in extra.items(): cols.append(c); ap.append(aps(ref_y, 1/(1+np.exp(-c))))
    X = np.stack(cols, 1); ap = np.array(ap)
    wb, _ = fast_nm(X, ref_y, n_restarts=2)
    keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW)); Xt = X[:, keep]
    _, oof_in = fast_nm(Xt, ref_y, n_restarts=3)
    pred = np.zeros(len(ref_y))
    for k in range(5):
        tr, te = rf != k, rf == k
        wk, _ = fast_nm(Xt[tr], ref_y[tr], n_restarts=2); pred[te] = Xt[te] @ wk
    return aps(ref_y, pred), oof_in, int(keep.sum())

log(f"=== GATE 3: honest nested OOF (THE gate). OMP={os.environ.get('OMP_NUM_THREADS')} ===")
b48 = honest(pool_names)
log(f"  ref  48 (current/v7)        honest={b48[0]:.4f} in={b48[1]:.4f} keep={b48[2]}")
drop_known = {"ConcatCWT heavyaug384", "ConcatCWT focalheavy256"}
base = [n for n in pool_names if n not in drop_known]
bb = honest(base)
BAR = bb[0]
log(f"  BAR  46 (v5,-2 corr archs)  honest={bb[0]:.4f} in={bb[1]:.4f} keep={bb[2]}   <== BAR to beat")
log("  --- each new arch onto v5-best (46) ---")
res = {}
for name, col in new_aligned.items():
    h, i, k = honest(base, extra={name: col}); res[name] = h; d = h - BAR
    v = "HELPS" if d > 0.0005 else ("neutral" if d > -0.0005 else "HURTS")
    log(f"  +{name:30s} honest={h:.4f} (Δ{d:+.4f}) in={i:.4f} keep={k}  [{v}]")
# orthogonal-paradigm combo + everything
log("  --- combos ---")
for combo_name, keys in [("GNN+Spec3D", ["GNN orthopara", "Spec3D scratch"]),
                          ("all 5 new", list(new_aligned.keys()))]:
    extra = {k_: new_aligned[k_] for k_ in keys}
    h, i, k = honest(base, extra=extra)
    log(f"  +{combo_name:30s} honest={h:.4f} (Δ{h-BAR:+.4f}) keep={k}")
log(f"DONE {time.time()-t0:.0f}s")
