"""Build the candidate-SPECIALIST submission: canonical pool with ConcatCWT+Paul+Filt maxvit256
REPLACED by their candidate-specialist versions (11336 candidates @ weight 1.0; real-OOF +0.024~+0.044
per arch). The NM (derived on REAL OOF) naturally up-weights them since they're stronger -> a
specialist-dominant blend. Output: submission_test1_spec.zip. Public-gated probe vs 0.9778, but the
real-OOF brake is GREEN (single arches stronger), so this is the lowest-risk of the aggressive bets.
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
    "ConcatCWT maxvit256": ("ConcatCWT spec 256", "concat_cwt_spec_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    "ConcatPaul maxvit256": ("ConcatPaul spec 256", "concat_cwt_paul_spec_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    "ConcatFilt maxvit256": ("ConcatFilt spec 256", "concat_cwt_filtered_spec_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
}
PSE = ("pseudo", "relabel", "relbase", "rank256", "NeuroMAE", "GNN orthopara", "Spec3D", "r3nchc", " spec ")
d = np.load(NPZ, allow_pickle=True)
npz_names = [str(n) for n in d["names"]]; Xc_all = d["logits"]; ids = [str(x) for x in d["ids"]]
col = {n: i for i, n in enumerate(npz_names)}


def ready(ps, pfx):
    return all((REPO / "neuromm26_results/predictions" / f"{pfx}__fold{f}__seed0_oof.npz").exists() for f in range(5)) and ps in col


SWAP = {o: ps for o, (ps, pfx) in SWAP_ALL.items() if ready(ps, pfx)}
assert len(SWAP) == 3, f"spec arches not all in npz yet ({SWAP}) — run the candidate-logit append first"
canon = [n for n in npz_names if not any(k in n for k in PSE)]
pool = [SWAP.get(n, n) for n in canon]
print(f"pool {len(pool)} archs; {len(SWAP)} specialist swaps: {SWAP}", flush=True)

ref_s = ref_y = None; Xo = []; Xc = []; ap = []
for nm in pool:
    pfx, root = POOL_LOOKUP[nm]; s, l, y = load_oof(pfx, root); ss = s.astype(str)
    if ref_s is None:
        ref_s, ref_y = ss, y.astype(int); Xo.append(l)
    else:
        im = {x: i for i, x in enumerate(ss)}; Xo.append(l[np.array([im[x] for x in ref_s])])
    ap.append(aps(y, 1 / (1 + np.exp(-np.clip(l, -30, 30))))); Xc.append(Xc_all[:, col[nm]])
Xo = np.stack(Xo, 1); Xc = np.stack(Xc, 1); ap = np.array(ap)

wb, _ = derive_nm(Xo, ref_y, n_restarts=4)
keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
poolk = [p for p, k in zip(pool, keep) if k]
Xot, Xct = Xo[:, keep], Xc[:, keep]
wD = apply_cap(derive_bagged_nm(Xot, ref_y, n_iter=40), 0.20)
oof = aps(ref_y, Xot @ wD)
spec_wt = sum(w for p, w in zip(poolk, wD) if "spec" in p)
print(f"trim {keep.sum()} archs; specialist ensemble OOF (real labels) = {oof:.4f} (v3 0.8974, nchc 0.9048); "
      f"specialist total NM weight = {spec_wt:.3f}", flush=True)

prob = 1.0 / (1.0 + np.exp(-(Xct @ wD)))
out = REPO / "submission_test1_spec.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["sample_id", "prediction"])
    for sid, p in zip(ids, prob):
        w.writerow([sid, f"{p:.6f}"])

z = zipfile.ZipFile(REPO / "submissions/submission_test1_nchc_n2filt.zip")
b = {r["sample_id"]: float(r["prediction"]) for r in csv.DictReader(io.TextIOWrapper(z.open("submission.csv")))}
rho = spearmanr(prob, [b[s] for s in ids]).statistic
print(f"Spearman rho(spec, 0.9778) = {rho:.4f}  (lower = bigger candidate shift)", flush=True)

subdir = REPO / "submissions"; staged = subdir / "submission.csv"; shutil.copy(out, staged)
zname = subdir / "submission_test1_spec.zip"
with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(staged, arcname="submission.csv")
staged.unlink()
print(f"WROTE {zname}  (range {prob.min():.4f}-{prob.max():.4f} mean {prob.mean():.4f})", flush=True)
