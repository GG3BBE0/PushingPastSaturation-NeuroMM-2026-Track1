"""Honest-OOF gate, parallel across configs (each config -> 1 forked single-thread worker).
9 configs run concurrently; wall ~= one config. maxiter 8000 keeps NM converged.
Run with OMP_NUM_THREADS=1 so each worker stays on its own core."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score as aps
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import POOL_LOOKUP, load_oof, RES, REPO, \
    TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW

FOLD = REPO / "fold_df_fixed.csv"
NPZ = RES / "candidate_arch_logits.npz"
NEW = {
    "GNN":        ("eeg_gnn_fold__gcn", RES),
    "Spec3D":     ("spec3d_superlet_scratch_fold__r3d_18", RES),
    "Paul384":    ("concat_cwt_paul_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
    "Filt384":    ("concat_cwt_filtered_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
    "MaxBase384": ("concat_superlet_heavyaug_fold__maxvit_base_tf_384_in1k", RES),
}

# ---- module-level load (inherited by forked workers) ----
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
    s, l, y = load_oof(pfx, root); ss = s.astype(str); im2 = {x: i for i, x in enumerate(ss)}
    new_aligned[name] = np.array([l[im2[x]] for x in ref_s]).astype(np.float64)

def fast_nm(X, y, n_restarts, maxiter=8000, seed=42):
    M = X.shape[1]
    def neg(w):
        w = np.maximum(w, 0); sm = w.sum(); w = w/sm if sm > 0 else np.ones_like(w)/M
        return -aps(y, X @ w)
    rng = np.random.default_rng(seed); bw, ba = None, -1
    for _ in range(n_restarts):
        w0 = np.ones(M)/M + 0.05*rng.standard_normal(M); w0 = np.maximum(w0, 1e-3); w0 /= w0.sum()
        r = minimize(neg, w0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6})
        w = np.maximum(r.x, 0); w /= w.sum(); a = aps(y, X @ w)
        if a > ba: ba, bw = a, w
    return bw, ba

def honest_worker(cfg):
    label, base_names, extra_keys = cfg
    cols = [pool_cols[n] for n in base_names]; ap = [pool_ap[n] for n in base_names]
    for k_ in extra_keys:
        c = new_aligned[k_]; cols.append(c); ap.append(aps(ref_y, 1/(1+np.exp(-c))))
    X = np.stack(cols, 1); ap = np.array(ap)
    wb, _ = fast_nm(X, ref_y, 2); keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW)); Xt = X[:, keep]
    _, oof_in = fast_nm(Xt, ref_y, 3)
    pred = np.zeros(len(ref_y))
    perf = []
    for k in range(5):
        tr, te = rf != k, rf == k
        wk, _ = fast_nm(Xt[tr], ref_y[tr], 2); pred[te] = Xt[te] @ wk
        perf.append(aps(ref_y[te], pred[te]))
    return label, aps(ref_y, pred), oof_in, int(keep.sum()), perf

if __name__ == "__main__":
    t0 = time.time()
    drop_known = {"ConcatCWT heavyaug384", "ConcatCWT focalheavy256"}
    base46 = [n for n in pool_names if n not in drop_known]
    configs = [
        ("ref48 (current/v7)", pool_names, []),
        ("BAR  v5-best(46)",   base46, []),
        ("+Spec3D",            base46, ["Spec3D"]),
        ("+GNN",               base46, ["GNN"]),
        ("+Paul384",           base46, ["Paul384"]),
        ("+Filt384",           base46, ["Filt384"]),
        ("+MaxBase384",        base46, ["MaxBase384"]),
        ("+GNN+Spec3D",        base46, ["GNN", "Spec3D"]),
        ("+all5",              base46, list(NEW.keys())),
    ]
    print(f"launching {len(configs)} parallel honest configs, n={len(ref_y)}, pos={int(ref_y.sum())}", flush=True)
    out = {}
    with ProcessPoolExecutor(max_workers=len(configs)) as ex:
        for label, h, oin, keep, perf in ex.map(honest_worker, configs):
            out[label] = (h, oin, keep, perf)
            print(f"  done {label}", flush=True)
    BAR = out["BAR  v5-best(46)"][0]
    print("\n=== HONEST NESTED OOF RESULTS ===", flush=True)
    print(f"{'config':22s} {'honest':>8s} {'Δvsbar':>8s} {'insmp':>7s} {'keep':>4s}  per-fold(0..4)", flush=True)
    for label, _, _ in configs:
        h, oin, keep, perf = out[label]
        d = h - BAR
        tag = "" if label.startswith(("ref", "BAR")) else (" HELPS" if d > 0.0005 else (" neutral" if d > -0.0005 else " HURTS"))
        print(f"{label:22s} {h:8.4f} {d:+8.4f} {oin:7.4f} {keep:4d}  {'/'.join(f'{p:.3f}' for p in perf)}{tag}", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
