"""Build the pseudo-label submission: the canonical 48-arch ensemble with ConcatCWT + ConcatPaul
maxvit256 REPLACED by their pseudo-retrained versions. Trim+bagged Nelder-Mead on OOF (real
labels), applied to the candidate logits (pseudo columns from append_arch_to_npz). Public LB is
the real test of pseudo-labeling.
"""
import csv
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score as aps

sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import (POOL_LOOKUP, load_oof, derive_nm, derive_bagged_nm, apply_cap,
                                       TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW, REPO)

NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
PRED = REPO / "neuromm26_results/predictions"
# HIGH-WEIGHT arches only (low-weight ones DILUTE — verified: 2-arch +0.0088 > 4-arch +0.0082 >
# 7-arch +0.0051; 3-arch w/ superlet384#1 = ROBUST+). Auto-included only if the pseudo arch is ready.
SWAP_ALL = {
    "ConcatCWT maxvit256": ("ConcatCWT pseudo256", "concat_cwt_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    "ConcatPaul maxvit256": ("ConcatPaul pseudo256", "concat_cwt_paul_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    "ConcatSuperlet maxvit384": ("ConcatSuperlet pseudo384", "concat_superlet_pseudo_fold__maxvit_tiny_tf_384_in1k"),
    "ConcatCWT maxvit384": ("ConcatCWT pseudo384", "concat_cwt_pseudo_fold__maxvit_tiny_tf_384_in1k"),
}
EXPERIMENT = {"ConcatCWT relabel256", "ConcatCWT relbase256", "ConcatPaul relabel256", "ConcatSTFT relabel256",
              "ConcatCWT rank256", "ConcatPaul rank256", "NeuroMAE v1", "GNN orthopara", "Spec3D superlet",
              # ALL pseudo arches excluded from the canonical scan; only SWAP_ALL ones get swapped IN
              "ConcatCWT pseudo256", "ConcatPaul pseudo256", "ConcatFilt pseudo256", "ConcatSTFT pseudo256",
              "ConcatSuperlet pseudo384", "ConcatCWT pseudo384",
              "muku raw pseudo convnext_pico", "muku raw pseudo resnet18", "muku Filt pseudo convnext_pico"}

d = np.load(NPZ, allow_pickle=True)
npz_names = [str(n) for n in d["names"]]; Xc_all = d["logits"]; ids = [str(x) for x in d["ids"]]
col = {n: i for i, n in enumerate(npz_names)}

def ready(pseudo_name, pfx):  # OOF (5 folds) + candidate logits both present
    oof = all((PRED / f"{pfx}__fold{f}__seed0_oof.npz").exists() for f in range(5))
    return oof and pseudo_name in col

SWAP = {o: ps for o, (ps, pfx) in SWAP_ALL.items() if ready(ps, pfx)}
EXPERIMENT |= {ps for _, (ps, _) in SWAP_ALL.items()}  # keep pseudo cols out of the canonical scan
canon = [n for n in npz_names if n not in EXPERIMENT]
pool = [SWAP.get(n, n) for n in canon]
print(f"pool {len(pool)} archs; {len(SWAP)} pseudo swaps: {SWAP}", flush=True)

# assemble OOF + candidate matrices aligned
ref_s = ref_y = None; Xo = []; Xc = []; ap = []
for nm in pool:
    pfx, root = POOL_LOOKUP[nm]; s, l, y = load_oof(pfx, root); ss = s.astype(str)
    if ref_s is None:
        ref_s, ref_y = ss, y.astype(int); Xo.append(l)
    else:
        im = {x: i for i, x in enumerate(ss)}; Xo.append(l[np.array([im[x] for x in ref_s])])
    ap.append(aps(y, 1 / (1 + np.exp(-l))))
    Xc.append(Xc_all[:, col[nm]])
Xo = np.stack(Xo, 1); Xc = np.stack(Xc, 1); ap = np.array(ap)

# trim + bagged + cap (config D)
wb, _ = derive_nm(Xo, ref_y, n_restarts=4)
keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
Xot, Xct = Xo[:, keep], Xc[:, keep]
wC = derive_bagged_nm(Xot, ref_y, n_iter=40)
wD = apply_cap(wC, 0.20)
oof = aps(ref_y, Xot @ wD)
print(f"trim {keep.sum()} archs; pseudo-ensemble OOF (real labels) = {oof:.4f}  (v3 baseline 0.8974)", flush=True)

prob = 1.0 / (1.0 + np.exp(-(Xct @ wD)))
TAG = f"pseudo{len(SWAP)}"  # name by #pseudo arches so versions don't overwrite
out = REPO / f"submission_test1_{TAG}.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["sample_id", "prediction"])
    for sid, p in zip(ids, prob):
        w.writerow([sid, f"{p:.6f}"])

# Spearman vs v3
z = zipfile.ZipFile(REPO / "submissions/submission_test1_regnm_v3.zip")
import io
v3 = list(csv.DictReader(io.TextIOWrapper(z.open("submission.csv"))))
v3p = {r["sample_id"]: float(r["prediction"]) for r in v3}
rho = spearmanr(prob, [v3p[s] for s in ids]).statistic
print(f"Spearman rho(pseudo, v3) = {rho:.4f}  (healthy 0.93-0.995)", flush=True)

subdir = REPO / "submissions"; staged = subdir / "submission.csv"; shutil.copy(out, staged)
zname = subdir / f"submission_test1_{TAG}.zip"
with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(staged, arcname="submission.csv")
staged.unlink()
print(f"WROTE {zname}  (range {prob.min():.4f}-{prob.max():.4f} mean {prob.mean():.4f})", flush=True)
