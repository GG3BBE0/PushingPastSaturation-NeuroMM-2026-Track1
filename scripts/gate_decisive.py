"""Decisive multi-seed honest test. Run1 said +GNN +0.0028; run2 (better-converged) said
+0.0000 (GNN trimmed). To settle it: FORCE-KEEP the new arch (exempt from the trim rule —
GNN is weak OOF 0.547 but NOT redundant, corr 0.54, so trim's 'weak AND useless' intent
shouldn't drop it) and measure honest(BAR+arch) - honest(BAR) across 3 NM seeds at high
maxiter. If the per-seed Δ is reliably > +0.001, the arch is a marginal-but-real add; if it
straddles 0, it's NM noise and we DO NOT submit. BAR is recomputed per seed (paired Δ)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score as aps
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import POOL_LOOKUP, load_oof, RES, REPO, TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW

MAXITER = 16000
SEEDS = [11, 23, 37]
FOLD = REPO / "fold_df_fixed.csv"; NPZ = RES / "candidate_arch_logits.npz"
NEW = {"GNN": ("eeg_gnn_fold__gcn", RES),
       "Spec3D": ("spec3d_superlet_scratch_fold__r3d_18", RES),
       "NeuroMAE": ("neuromae_fold__v1", RES)}
fold_df = pd.read_csv(FOLD); fold_df["sample_id"] = fold_df["sample_id"].astype(str)
sid_fold = dict(zip(fold_df["sample_id"], fold_df["fold"]))
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
    try:
        s, l, y = load_oof(pfx, root)
    except FileNotFoundError:
        print(f"  (skip {name}: no OOF yet)", flush=True); continue
    ss = s.astype(str); im2 = {x: i for i, x in enumerate(ss)}
    new_aligned[name] = np.array([l[im2[x]] for x in ref_s]).astype(np.float64)

def nm(X, y, n_restarts, seed):
    M = X.shape[1]
    def neg(w):
        w = np.maximum(w, 0); sm = w.sum(); w = w/sm if sm > 0 else np.ones_like(w)/M
        return -aps(y, X @ w)
    rng = np.random.default_rng(seed); bw, ba = None, -1
    for _ in range(n_restarts):
        w0 = np.ones(M)/M + 0.05*rng.standard_normal(M); w0 = np.maximum(w0, 1e-3); w0/=w0.sum()
        r = minimize(neg, w0, method="Nelder-Mead", options={"maxiter": MAXITER, "xatol":1e-7, "fatol":1e-7})
        w = np.maximum(r.x, 0); w/=w.sum(); a = aps(y, X@w)
        if a > ba: ba, bw = a, w
    return bw, ba

def honest(base_names, extra_key, seed, force_keep):
    cols = [pool_cols[n] for n in base_names]; ap = [pool_ap[n] for n in base_names]
    nfix = len(base_names)
    if extra_key:
        cols.append(new_aligned[extra_key]); ap.append(aps(ref_y, 1/(1+np.exp(-new_aligned[extra_key]))))
    X = np.stack(cols, 1); ap = np.array(ap)
    wb, _ = nm(X, ref_y, 2, seed)
    keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
    if extra_key and force_keep: keep[nfix] = True  # exempt the new arch from trim
    Xt = X[:, keep]
    pred = np.zeros(len(ref_y))
    for k in range(5):
        tr, te = rf != k, rf == k; wk, _ = nm(Xt[tr], ref_y[tr], 2, seed+k); pred[te] = Xt[te] @ wk
    return aps(ref_y, pred), int(keep.sum())

def worker(cfg):
    label, extra_key, seed, force = cfg
    base = [n for n in pool_names if n not in {"ConcatCWT heavyaug384", "ConcatCWT focalheavy256"}]
    h, keep = honest(base, extra_key, seed, force)
    return label, seed, h, keep

if __name__ == "__main__":
    t0 = time.time()
    cfgs = []
    avail_new = [k for k in NEW if k in new_aligned]  # only test archs whose OOF exists
    for seed in SEEDS:
        cfgs.append(("BAR", None, seed, False))
        for k in avail_new:
            cfgs.append((f"+{k}(forcekeep)", k, seed, True))
    print(f"decisive: maxiter={MAXITER}, seeds={SEEDS}, testing {avail_new}, "
          f"{len(cfgs)} parallel workers", flush=True)
    rows = {}
    with ProcessPoolExecutor(max_workers=len(cfgs)) as ex:
        for label, seed, h, keep in ex.map(worker, cfgs):
            rows.setdefault(label, {})[seed] = (h, keep); print(f"  done {label} seed{seed}: {h:.4f} keep{keep}", flush=True)
    print("\n=== per-seed honest OOF ===", flush=True)
    bar = {s: rows["BAR"][s][0] for s in SEEDS}
    for label in ["BAR"] + [f"+{k}(forcekeep)" for k in avail_new]:
        hs = [rows[label][s][0] for s in SEEDS]
        if label == "BAR":
            print(f"{label:20s} " + " ".join(f"s{s}={rows[label][s][0]:.4f}" for s in SEEDS) + f"  mean={np.mean(hs):.4f}", flush=True)
        else:
            ds = [rows[label][s][0] - bar[s] for s in SEEDS]
            print(f"{label:20s} " + " ".join(f"s{s}={rows[label][s][0]:.4f}(Δ{rows[label][s][0]-bar[s]:+.4f})" for s in SEEDS)
                  + f"  meanΔ={np.mean(ds):+.4f}  {'ROBUST+' if min(ds)>0.0005 else ('mixed' if max(ds)>0 else 'NO')}", flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)
