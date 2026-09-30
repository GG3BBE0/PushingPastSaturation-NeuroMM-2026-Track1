"""Strategy-B (public-gated) lever that directly fights ensemble blend-dampening: take the
nchc_n2filt (0.9778) blend and UP-WEIGHT the 3 pseudo arches (cwt/paul/filt nchc) beyond their
real-OOF-optimal NM weight -> forces the prediction further toward the candidate distribution.
No GPU. Builds several factors so the user can probe the public sweet spot. Reports OOF (real, will
DROP as factor rises) + Spearman vs 0.9778 (drops = bigger candidate shift) so the aggression of each
variant is explicit.
"""
import csv, io, shutil, sys, zipfile
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score as aps

sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import (POOL_LOOKUP, load_oof, derive_nm, derive_bagged_nm, apply_cap,
                                       TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW, REPO)

NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
SWAP = {"ConcatCWT maxvit256": "ConcatCWT pseudo_nchc 256",
        "ConcatPaul maxvit256": "ConcatPaul pseudo_nchc 256",
        "ConcatFilt maxvit256": "ConcatFilt pseudo_nchc 256"}
PSEUDO = set(SWAP.values())
EXPERIMENT = {"ConcatCWT relabel256", "ConcatCWT relbase256", "ConcatPaul relabel256", "ConcatSTFT relabel256",
              "ConcatCWT rank256", "ConcatPaul rank256", "NeuroMAE v1", "GNN orthopara", "Spec3D superlet",
              "ConcatCWT pseudo256", "ConcatPaul pseudo256", "ConcatFilt pseudo256", "ConcatSTFT pseudo256",
              "ConcatSuperlet pseudo384", "ConcatCWT pseudo384",
              "muku raw pseudo convnext_pico", "muku raw pseudo resnet18", "muku Filt pseudo convnext_pico",
              "ConcatCWT pseudo_r2 256", "ConcatPaul pseudo_r2 256",
              "ConcatCWT pseudo_nchc 256", "ConcatPaul pseudo_nchc 256", "ConcatFilt pseudo_nchc 256",
              "ConcatSTFT pseudo_nchc 256", "ConcatSuperlet pseudo_nchc 384", "ConcatCWT pseudo_nchc 384",
              "ConcatCWT pseudo_r3 256", "ConcatPaul pseudo_r3 256", "ConcatFilt pseudo_r3 256"}

d = np.load(NPZ, allow_pickle=True)
npz_names = [str(n) for n in d["names"]]; Xc_all = d["logits"]; ids = [str(x) for x in d["ids"]]
col = {n: i for i, n in enumerate(npz_names)}
canon = [n for n in npz_names if n not in EXPERIMENT]
pool = [SWAP.get(n, n) for n in canon]

ref_y = None; Xo = []; Xc = []; ap = []
for nm in pool:
    pfx, root = POOL_LOOKUP[nm]; s, l, y = load_oof(pfx, root); ss = s.astype(str)
    if ref_y is None:
        ref_s = ss; ref_y = y.astype(int); Xo.append(l)
    else:
        im = {x: i for i, x in enumerate(ss)}; Xo.append(l[np.array([im[x] for x in ref_s])])
    ap.append(aps(y, 1 / (1 + np.exp(-l)))); Xc.append(Xc_all[:, col[nm]])
Xo = np.stack(Xo, 1); Xc = np.stack(Xc, 1); ap = np.array(ap)

wb, _ = derive_nm(Xo, ref_y, n_restarts=2)
keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
poolk = [p for p, k in zip(pool, keep) if k]
Xot, Xct = Xo[:, keep], Xc[:, keep]
w0 = apply_cap(derive_bagged_nm(Xot, ref_y, n_iter=12), 0.20)
is_pseudo = np.array([p in PSEUDO for p in poolk])
base_oof = aps(ref_y, Xot @ w0)
print(f"base nchc_n2filt: OOF {base_oof:.4f}, pseudo-arch total weight {w0[is_pseudo].sum():.3f} "
      f"({[f'{p}={w:.3f}' for p,w in zip(poolk,w0) if p in PSEUDO]})", flush=True)

z = zipfile.ZipFile(REPO / "submissions/submission_test1_nchc_n2filt.zip")
b = {r["sample_id"]: float(r["prediction"]) for r in csv.DictReader(io.TextIOWrapper(z.open("submission.csv")))}
bvec = np.array([b[s] for s in ids])

print("\nfactor  pseudo_wt  OOF(real)  rho_vs_0.9778   file", flush=True)
for fac in [1.5, 2.0, 3.0, 5.0]:
    w = w0.copy(); w[is_pseudo] *= fac; w = w / w.sum()
    oof = aps(ref_y, Xot @ w)
    prob = 1.0 / (1.0 + np.exp(-(Xct @ w)))
    rho = spearmanr(prob, bvec).statistic
    tag = f"upw{str(fac).replace('.0','').replace('.','p')}"
    out = REPO / f"submission_test1_{tag}.csv"
    with out.open("w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["sample_id", "prediction"])
        for sid, p in zip(ids, prob):
            wr.writerow([sid, f"{p:.6f}"])
    staged = REPO / "submissions/submission.csv"; shutil.copy(out, staged)
    zname = REPO / f"submissions/submission_test1_{tag}.zip"
    with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(staged, arcname="submission.csv")
    staged.unlink(); out.unlink()
    print(f"  {fac:.1f}x   {w[is_pseudo].sum():.3f}     {oof:.4f}    {rho:.4f}       {zname.name}", flush=True)
