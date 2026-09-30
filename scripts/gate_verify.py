"""Robustness re-run of the honest gate for the KEY configs, with a DIFFERENT NM seed
and HIGHER maxiter, to confirm +GNN's honest gain is not Nelder-Mead local-optimum noise.
If +GNN stays ~+0.002..0.003 over BAR across (seed42,mi8000) and (seed7,mi14000), it's real."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score as aps
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import POOL_LOOKUP, load_oof, RES, REPO, TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW

SEED = 7; MAXITER = 14000
FOLD = REPO / "fold_df_fixed.csv"; NPZ = RES / "candidate_arch_logits.npz"
NEW = {
    "GNN":     ("eeg_gnn_fold__gcn", RES),
    "Spec3D":  ("spec3d_superlet_scratch_fold__r3d_18", RES),
    "Paul384": ("concat_cwt_paul_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
}
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

def nm(X, y, n_restarts, seed):
    M = X.shape[1]
    def neg(w):
        w = np.maximum(w, 0); sm = w.sum(); w = w/sm if sm>0 else np.ones_like(w)/M
        return -aps(y, X @ w)
    rng = np.random.default_rng(seed); bw, ba = None, -1
    for _ in range(n_restarts):
        w0 = np.ones(M)/M + 0.05*rng.standard_normal(M); w0 = np.maximum(w0,1e-3); w0/=w0.sum()
        r = minimize(neg, w0, method="Nelder-Mead", options={"maxiter": MAXITER, "xatol":1e-7,"fatol":1e-7})
        w = np.maximum(r.x,0); w/=w.sum(); a = aps(y, X@w)
        if a>ba: ba,bw=a,w
    return bw,ba

def worker(cfg):
    label, base_names, keys = cfg
    cols=[pool_cols[n] for n in base_names]; ap=[pool_ap[n] for n in base_names]
    for k_ in keys:
        c=new_aligned[k_]; cols.append(c); ap.append(aps(ref_y,1/(1+np.exp(-c))))
    X=np.stack(cols,1); ap=np.array(ap)
    wb,_=nm(X,ref_y,2,SEED); keep=~((ap<TRIM_CUTOFF_OOF)&(wb<TRIM_CUTOFF_NMW)); Xt=X[:,keep]
    _,oin=nm(Xt,ref_y,3,SEED)
    pred=np.zeros(len(ref_y))
    for k in range(5):
        tr,te=rf!=k,rf==k; wk,_=nm(Xt[tr],ref_y[tr],2,SEED+k); pred[te]=Xt[te]@wk
    return label, aps(ref_y,pred), oin, int(keep.sum())

if __name__=="__main__":
    t0=time.time()
    base=[n for n in pool_names if n not in {"ConcatCWT heavyaug384","ConcatCWT focalheavy256"}]
    cfgs=[("BAR v5-best(46)",base,[]),("+Spec3D",base,["Spec3D"]),("+GNN",base,["GNN"]),
          ("+Paul384",base,["Paul384"]),("+GNN+Paul384",base,["GNN","Paul384"])]
    print(f"VERIFY seed={SEED} maxiter={MAXITER}; {len(cfgs)} configs parallel", flush=True)
    out={}
    with ProcessPoolExecutor(max_workers=len(cfgs)) as ex:
        for label,h,oin,keep in ex.map(worker,cfgs):
            out[label]=(h,oin,keep); print(f"  done {label}", flush=True)
    BAR=out["BAR v5-best(46)"][0]
    print(f"\n{'config':18s} {'honest':>8s} {'Δvsbar':>8s} {'insmp':>7s} keep", flush=True)
    for label,_,_ in cfgs:
        h,oin,keep=out[label]; print(f"{label:18s} {h:8.4f} {h-BAR:+8.4f} {oin:7.4f} {keep}", flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)
