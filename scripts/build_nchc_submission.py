"""Build the NCHC N2+filt submission: canonical pool with ConcatCWT + ConcatPaul + ConcatFilt
maxvit256 REPLACED by their NCHC (bigger-batch) pseudo retrains. This config won the 5-seed honest
gate: meanΔ +0.0136 vs local-2arch bar +0.0032 (increment +0.0104, all 5 seeds positive).
Output: submission_test1_nchc_n2filt.zip.
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
SWAP_ALL = {
    "ConcatCWT maxvit256": ("ConcatCWT pseudo_nchc 256", "concat_cwt_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    "ConcatPaul maxvit256": ("ConcatPaul pseudo_nchc 256", "concat_cwt_paul_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    "ConcatFilt maxvit256": ("ConcatFilt pseudo_nchc 256", "concat_cwt_filtered_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
}
# every pseudo arch (local/r2/nchc) excluded from the canonical scan; only SWAP_ALL swapped IN
EXPERIMENT = {"ConcatCWT relabel256", "ConcatCWT relbase256", "ConcatPaul relabel256", "ConcatSTFT relabel256",
              "ConcatCWT rank256", "ConcatPaul rank256", "NeuroMAE v1", "GNN orthopara", "Spec3D superlet",
              "ConcatCWT pseudo256", "ConcatPaul pseudo256", "ConcatFilt pseudo256", "ConcatSTFT pseudo256",
              "ConcatSuperlet pseudo384", "ConcatCWT pseudo384",
              "muku raw pseudo convnext_pico", "muku raw pseudo resnet18", "muku Filt pseudo convnext_pico",
              "ConcatCWT pseudo_r2 256", "ConcatPaul pseudo_r2 256",
              "ConcatCWT pseudo_nchc 256", "ConcatPaul pseudo_nchc 256", "ConcatFilt pseudo_nchc 256",
              "ConcatSTFT pseudo_nchc 256", "ConcatSuperlet pseudo_nchc 384", "ConcatCWT pseudo_nchc 384"}

d = np.load(NPZ, allow_pickle=True)
npz_names = [str(n) for n in d["names"]]; Xc_all = d["logits"]; ids = [str(x) for x in d["ids"]]
col = {n: i for i, n in enumerate(npz_names)}


def ready(pseudo_name, pfx):
    oof = all((REPO / "neuromm26_results/predictions" / f"{pfx}__fold{f}__seed0_oof.npz").exists() for f in range(5))
    return oof and pseudo_name in col


SWAP = {o: ps for o, (ps, pfx) in SWAP_ALL.items() if ready(ps, pfx)}
assert len(SWAP) == 3, f"NCHC arches not all ready (got {SWAP}); need cwt+paul+filt candidate logits in npz"
canon = [n for n in npz_names if n not in EXPERIMENT]
pool = [SWAP.get(n, n) for n in canon]
print(f"pool {len(pool)} archs; {len(SWAP)} NCHC swaps: {SWAP}", flush=True)

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

wb, _ = derive_nm(Xo, ref_y, n_restarts=4)
keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
Xot, Xct = Xo[:, keep], Xc[:, keep]
wC = derive_bagged_nm(Xot, ref_y, n_iter=40)
wD = apply_cap(wC, 0.20)
oof = aps(ref_y, Xot @ wD)
print(f"trim {keep.sum()} archs; NCHC N2+filt ensemble OOF (real labels) = {oof:.4f}  (v3 baseline 0.8974)", flush=True)

prob = 1.0 / (1.0 + np.exp(-(Xct @ wD)))
out = REPO / "submission_test1_nchc_n2filt.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["sample_id", "prediction"])
    for sid, p in zip(ids, prob):
        w.writerow([sid, f"{p:.6f}"])

z = zipfile.ZipFile((REPO/"submissions/submission_test1_pseudo.zip") if (REPO/"submissions/submission_test1_pseudo.zip").exists() else (REPO/"submissions/submission_test1_regnm_v3.zip"))  # round-1 2-arch (0.9769)
r1 = list(csv.DictReader(io.TextIOWrapper(z.open("submission.csv"))))
r1p = {r["sample_id"]: float(r["prediction"]) for r in r1}
rho = spearmanr(prob, [r1p[s] for s in ids]).statistic
print(f"Spearman rho(nchc_n2filt, 2-arch 0.9769) = {rho:.4f}  (healthy 0.93-0.995)", flush=True)

subdir = REPO / "submissions"; staged = subdir / "submission.csv"; shutil.copy(out, staged)
zname = subdir / "submission_test1_nchc_n2filt.zip"
with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(staged, arcname="submission.csv")
staged.unlink()
print(f"WROTE {zname}  (range {prob.min():.4f}-{prob.max():.4f} mean {prob.mean():.4f})", flush=True)
