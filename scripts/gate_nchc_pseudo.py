"""Gate the NCHC 6-arch pseudo retrain vs the local 2-arch (public 0.9769). NCHC bigger-batch
training lifted single-arch OOF (paul256 +0.0177, superlet384 +0.0218), so a stronger superlet384
might no longer dilute. Honest nested-OOF meanΔ vs canonical across 5 NM seeds (tighter than 3).
A config beats the bar (L2) only if meanΔ > L2 + 0.005 (the real noise floor). OOF-only, no GPU.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score as aps

sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import POOL_LOOKUP, load_oof, TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW, REPO

NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
FOLD = REPO / "fold_df_fixed.csv"
SEEDS = [11, 23, 37, 53, 71]
EXPERIMENT = {"ConcatCWT relabel256", "ConcatCWT relbase256", "ConcatPaul relabel256", "ConcatSTFT relabel256",
              "ConcatCWT rank256", "ConcatPaul rank256", "NeuroMAE v1", "GNN orthopara", "Spec3D superlet",
              "ConcatCWT pseudo256", "ConcatPaul pseudo256", "ConcatFilt pseudo256", "ConcatSTFT pseudo256",
              "ConcatSuperlet pseudo384", "ConcatCWT pseudo384",
              "muku raw pseudo convnext_pico", "muku raw pseudo resnet18", "muku Filt pseudo convnext_pico",
              "ConcatCWT pseudo_r2 256", "ConcatPaul pseudo_r2 256",
              "ConcatCWT pseudo_nchc 256", "ConcatPaul pseudo_nchc 256", "ConcatFilt pseudo_nchc 256",
              "ConcatSTFT pseudo_nchc 256", "ConcatSuperlet pseudo_nchc 384", "ConcatCWT pseudo_nchc 384"}
base = [str(n) for n in np.load(NPZ, allow_pickle=True)["names"] if str(n) not in EXPERIMENT]

L = {"ConcatCWT maxvit256": "ConcatCWT pseudo256", "ConcatPaul maxvit256": "ConcatPaul pseudo256"}  # local 2-arch
N = {"ConcatCWT maxvit256": "ConcatCWT pseudo_nchc 256", "ConcatPaul maxvit256": "ConcatPaul pseudo_nchc 256"}
CONFIGS = {
    "L2 local2arch(bar) ": L,
    "N2 nchc2arch       ": N,
    "N2+sup384(nchc)    ": {**N, "ConcatSuperlet maxvit384": "ConcatSuperlet pseudo_nchc 384"},
    "N2+filt256(nchc)   ": {**N, "ConcatFilt maxvit256": "ConcatFilt pseudo_nchc 256"},
    "N4(2+sup+filt nchc)": {**N, "ConcatSuperlet maxvit384": "ConcatSuperlet pseudo_nchc 384",
                             "ConcatFilt maxvit256": "ConcatFilt pseudo_nchc 256"},
    "mix loc-cwt+nchc-paul": {"ConcatCWT maxvit256": "ConcatCWT pseudo256", "ConcatPaul maxvit256": "ConcatPaul pseudo_nchc 256"},
}

fdf = pd.read_csv(FOLD); fdf["sample_id"] = fdf["sample_id"].astype(str)
sid_fold = dict(zip(fdf["sample_id"], fdf["fold"].astype(int)))


def build(names):
    ref = None; cols = []; ap = []
    for nm in names:
        pfx, root = POOL_LOOKUP[nm]; s, l, y = load_oof(pfx, root); ss = s.astype(str)
        if ref is None:
            ref = ss; yy = y.astype(int); cols.append(l)
        else:
            im = {x: i for i, x in enumerate(ss)}; cols.append(l[np.array([im[x] for x in ref])])
        ap.append(aps(y, 1 / (1 + np.exp(-l))))
    return ref, yy, np.stack(cols, 1), np.array(ap)


def nm(X, yy, seed):
    M = X.shape[1]
    def neg(w):
        w = np.maximum(w, 0); sm = w.sum(); w = w / sm if sm > 0 else np.ones_like(w) / M
        return -aps(yy, X @ w)
    rng = np.random.default_rng(seed); bw, ba = None, -1
    for _ in range(2):
        w0 = np.ones(M) / M + 0.05 * rng.standard_normal(M); w0 = np.maximum(w0, 1e-3); w0 /= w0.sum()
        r = minimize(neg, w0, method="Nelder-Mead", options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-7})
        w = np.maximum(r.x, 0); w /= w.sum(); a = aps(yy, X @ w)
        if a > ba: ba, bw = a, w
    return bw


def honest_eval(X, yy, ap, rf_local, seed):
    wb = nm(X, yy, seed); keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW)); Xt = X[:, keep]
    pred = np.zeros(len(yy))
    for k in range(5):
        tr, te = rf_local != k, rf_local == k; pred[te] = Xt[te] @ nm(Xt[tr], yy[tr], seed + k)
    return aps(yy, pred)


# base built once; base honest cached per seed (was recomputed per config = 6x waste)
refB, yB, XB, apB = build(base)
RFB = np.array([sid_fold[s] for s in refB])
base_h = {sd: honest_eval(XB, yB, apB, RFB, sd) for sd in SEEDS}

print(f"NCHC pseudo ensemble gate ({len(SEEDS)} seeds, honest paired-Δ vs canonical)\n", flush=True)
res = {}
for tag, swap in CONFIGS.items():
    repl = [swap.get(n, n) for n in base]
    refR, yR, XR, apR = build(repl)
    RFR = np.array([sid_fold[s] for s in refR])
    ds = [honest_eval(XR, yR, apR, RFR, sd) - base_h[sd] for sd in SEEDS]
    res[tag.strip()] = float(np.mean(ds))
    print(f"{tag}  Δ/seed={['%+.4f'%x for x in ds]}  meanΔ={np.mean(ds):+.4f}", flush=True)

bar = res["L2 local2arch(bar)"]
print(f"\nbar = L2 local 2-arch meanΔ {bar:+.4f} (public 0.9769). Beats bar iff meanΔ > {bar+0.005:+.4f}:", flush=True)
for tag, md in res.items():
    if "bar" not in tag:
        print(f"  {tag:22s} {md:+.4f}  -> {'BEATS bar — PROBE public' if md > bar+0.005 else 'within noise / no gain'}", flush=True)
