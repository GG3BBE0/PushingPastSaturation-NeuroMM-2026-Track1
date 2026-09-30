"""Gate the NCHC 4-job return — OOF-only (no 2.7h candidate inference needed).

Three gates, in order of cheapness:
  1. validate  : per-fold + pooled AUPRC of each new arch vs README
  2. corr gate : Spearman(OOF logits) vs all pool archs -> max-corr < 0.85 (proven metric;
                 same-family @384 heavyaug expected redundant)
  3. honest gate (THE gate): nested-CV honest OOF. derive NM weights on 4 folds, predict the
                 held-out fold, rotate. An arch is worth integrating ONLY if it raises honest
                 OOF above the v5-best baseline (0.8813). In-sample regnm OOF is UNSAFE
                 (it green-lit v7 which lost public).

eva02 (folds 0,1 only) and muku_focal (fold0 only) are PILOTS -> incomplete -> report-only.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score as aps
sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import POOL_LOOKUP, load_oof, derive_nm, RES, REPO, \
    TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

FOLD = REPO / "fold_df_fixed.csv"
NPZ = RES / "candidate_arch_logits.npz"

# ---- the 4 integration-testable new archs (full 5-fold) ----
NEW = {
    "GNN orthopara":        ("eeg_gnn_fold__gcn", RES),
    "ConcatPaul hvy384":    ("concat_cwt_paul_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatFilt hvy384":    ("concat_cwt_filtered_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatSuperlet maxvit_base hvy384": ("concat_superlet_heavyaug_fold__maxvit_base_tf_384_in1k", RES),
}

fold_df = pd.read_csv(FOLD); fold_df["sample_id"] = fold_df["sample_id"].astype(str)
sid_fold = dict(zip(fold_df["sample_id"], fold_df["fold"]))

# ============ 1. VALIDATE new archs ============
log("=== GATE 1: validate per-fold AUPRC ===")
new_oof = {}   # name -> (sids, logits, y) aligned
for name, (pfx, root) in NEW.items():
    try:
        s, l, y = load_oof(pfx, root)
    except FileNotFoundError as e:
        log(f"  {name}: MISSING fold -> {e}"); continue
    ss = s.astype(str)
    folds = np.array([sid_fold[x] for x in ss])
    per = [aps(y[folds == k], 1/(1+np.exp(-l[folds == k]))) for k in range(5)]
    pooled = aps(y, 1/(1+np.exp(-l)))
    new_oof[name] = (ss, l.astype(np.float64), y.astype(np.int32))
    log(f"  {name:38s} folds={'/'.join(f'{p:.3f}' for p in per)} mean={np.mean(per):.4f} pooled={pooled:.4f} n={len(y)}")

# ============ build pool OOF matrix (the 48 npz archs) ============
log("=== loading 48-arch pool OOF ===")
pool_names = [str(n) for n in np.load(NPZ, allow_pickle=True)["names"]]
ref_s = None; pool_cols = {}; pool_ap = {}
for nm in pool_names:
    pfx, root = POOL_LOOKUP[nm]
    s, l, y = load_oof(pfx, root); ss = s.astype(str)
    if ref_s is None:
        ref_s, ref_y = ss, y.astype(np.int32)
        im = {x: i for i, x in enumerate(ref_s)}
    col = np.empty(len(ref_s)); col[[im[x] for x in ss]] = l
    pool_cols[nm] = col.astype(np.float64)
    pool_ap[nm] = aps(ref_y, 1/(1+np.exp(-l)))
rf = np.array([sid_fold[s] for s in ref_s])
log(f"pool ref aligned: {len(ref_s)} samples, {len(pool_names)} archs")

# align new archs onto ref_s order
new_aligned = {}
for name, (ss, l, y) in new_oof.items():
    im2 = {x: i for i, x in enumerate(ss)}
    new_aligned[name] = np.array([l[im2[x]] for x in ref_s])

# ============ 2. CORRELATION GATE ============
log("=== GATE 2: Spearman(OOF logits) vs pool — max-corr < 0.85 ===")
for name, col in new_aligned.items():
    cors = sorted(((spearmanr(col, pool_cols[pn]).correlation, pn) for pn in pool_names),
                  key=lambda t: -t[0])
    mx = cors[0][0]
    flag = "PASS<0.85" if mx < 0.85 else "FAIL>=0.85(redundant)"
    log(f"  {name:38s} max-corr={mx:.3f} vs '{cors[0][1]}'  [{flag}]")
    log(f"      next: " + "; ".join(f"{c:.3f} {n}" for c, n in cors[1:4]))

# ============ 3. HONEST NESTED-CV OOF GATE ============
def honest(names_subset, extra=None):
    """names_subset: list of pool arch names; extra: dict name->aligned col to append.
    Returns (honest_oof, in_sample_oof, n_after_trim)."""
    cols = [pool_cols[n] for n in names_subset]
    ap = [pool_ap[n] for n in names_subset]
    nm_all = list(names_subset)
    if extra:
        for n, c in extra.items():
            cols.append(c); ap.append(aps(ref_y, 1/(1+np.exp(-c)))); nm_all.append(n)
    X = np.stack(cols, 1); ap = np.array(ap)
    wb, _ = derive_nm(X, ref_y, n_restarts=2)
    keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
    Xt = X[:, keep]
    w_in, oof_in = derive_nm(Xt, ref_y, n_restarts=3)
    pred = np.zeros(len(ref_y))
    for k in range(5):
        tr, te = rf != k, rf == k
        wk, _ = derive_nm(Xt[tr], ref_y[tr], n_restarts=2)
        pred[te] = Xt[te] @ wk
    return aps(ref_y, pred), oof_in, int(keep.sum())

log("=== GATE 3: honest nested OOF (THE gate). bar to beat = best baseline ===")
import os
log(f"(threads OMP={os.environ.get('OMP_NUM_THREADS')} )")

base48 = honest(pool_names)
log(f"  baseline 48 (current npz/v7)         honest={base48[0]:.4f} in={base48[1]:.4f} keep={base48[2]}")

# v5-best: drop the two known-bad correlated archs
drop_known = {"ConcatCWT heavyaug384", "ConcatCWT focalheavy256"}
base46_names = [n for n in pool_names if n not in drop_known]
base46 = honest(base46_names)
log(f"  baseline 46 (v5, -heavy384/focal256) honest={base46[0]:.4f} in={base46[1]:.4f} keep={base46[2]}  <== BAR")
BAR = base46[0]

# each new arch added onto the v5-best baseline (the cleanest pool)
for name, col in new_aligned.items():
    h, i, k = honest(base46_names, extra={name: col})
    d = h - BAR
    verdict = "HELPS" if d > 0.0005 else ("neutral" if d > -0.0005 else "HURTS")
    log(f"  +{name:36s} honest={h:.4f} (Δ{d:+.4f} vs bar) in={i:.4f} keep={k}  [{verdict}]")

# also add onto the full-48 (does it help even the saturated pool?)
log("  --- added onto full-48 (saturated) ---")
for name, col in new_aligned.items():
    h, i, k = honest(pool_names, extra={name: col})
    d = h - base48[0]
    log(f"  48+{name:34s} honest={h:.4f} (Δ{d:+.4f} vs 48) keep={k}")

# best-case: add ALL passing new archs at once onto v5-best
log("=== combo: v5-best + all 4 new ===")
hc, ic, kc = honest(base46_names, extra=new_aligned)
log(f"  v5-best + all4  honest={hc:.4f} (Δ{hc-BAR:+.4f} vs bar) keep={kc}")

log(f"DONE in {time.time()-t0:.0f}s")
