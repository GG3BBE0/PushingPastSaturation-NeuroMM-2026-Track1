"""Gate NCHC pseudo2 (5 seeds, cached base honest). Bar = nchc N2+filt (current best, public 0.9778).
  r3nchc N2+filt   : aggressive r3 labels @ NCHC -> does aggression beat conservative? (expect no)
  +muku_nchc       : add the ORTHOGONAL time-domain muku (NCHC bigger-batch) on top of N2+filt
  muku_nchc only   : swap just muku raw -> muku_nchc (does the orthogonal family alone help?)
A config beats the bar only if meanΔ > bar + 0.005. OOF only, no GPU.
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
SEEDS = [11, 23, 37]
PSE = {"pseudo256", "pseudo_r2", "pseudo_nchc", "pseudo384", "pseudo_r3", "r3nchc", "relabel", "relbase",
       "rank256", "NeuroMAE", "GNN orthopara", "Spec3D"}
allnames = [str(n) for n in np.load(NPZ, allow_pickle=True)["names"]]
base = [n for n in allnames if not any(k in n for k in PSE)]

NCHC = {"ConcatCWT maxvit256": "ConcatCWT pseudo_nchc 256", "ConcatPaul maxvit256": "ConcatPaul pseudo_nchc 256",
        "ConcatFilt maxvit256": "ConcatFilt pseudo_nchc 256"}
MUKU = {"muku raw convnext_pico": "muku raw pseudo_nchc convnext_pico", "muku raw resnet18": "muku raw pseudo_nchc resnet18"}
CONFIGS = {
    "nchc N2+filt (bar) ": NCHC,
    "r3nchc N2+filt     ": {"ConcatCWT maxvit256": "ConcatCWT r3nchc 256", "ConcatPaul maxvit256": "ConcatPaul r3nchc 256",
                             "ConcatFilt maxvit256": "ConcatFilt r3nchc 256"},
    "N2+filt + muku_nchc": {**NCHC, **MUKU},
    "muku_nchc only     ": MUKU,
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
        r = minimize(neg, w0, method="Nelder-Mead", options={"maxiter": 2500, "xatol": 1e-6, "fatol": 1e-6})
        w = np.maximum(r.x, 0); w /= w.sum(); a = aps(yy, X @ w)
        if a > ba: ba, bw = a, w
    return bw


def honest_eval(X, yy, ap, rf, seed):
    wb = nm(X, yy, seed); keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW)); Xt = X[:, keep]
    pred = np.zeros(len(yy))
    for k in range(5):
        tr, te = rf != k, rf == k; pred[te] = Xt[te] @ nm(Xt[tr], yy[tr], seed + k)
    return aps(yy, pred)


refB, yB, XB, apB = build(base)
RFB = np.array([sid_fold[s] for s in refB])
base_h = {sd: honest_eval(XB, yB, apB, RFB, sd) for sd in SEEDS}

print(f"NCHC pseudo2 gate ({len(SEEDS)} seeds, honest paired-Δ vs canonical)\n", flush=True)
res = {}
for tag, swap in CONFIGS.items():
    repl = [swap.get(n, n) for n in base]
    refR, yR, XR, apR = build(repl)
    RFR = np.array([sid_fold[s] for s in refR])
    ds = [honest_eval(XR, yR, apR, RFR, sd) - base_h[sd] for sd in SEEDS]
    res[tag.strip()] = float(np.mean(ds))
    print(f"{tag}  Δ/seed={['%+.4f'%x for x in ds]}  meanΔ={np.mean(ds):+.4f}", flush=True)

bar = res["nchc N2+filt (bar)"]
print(f"\nbar = nchc N2+filt {bar:+.4f} (public 0.9778). Beats bar iff meanΔ > {bar+0.005:+.4f}:", flush=True)
for tag, md in res.items():
    if "bar" not in tag:
        print(f"  {tag:20s} {md:+.4f}  (vs bar {md-bar:+.4f})  {'BEATS' if md>bar+0.005 else 'within noise'}", flush=True)
