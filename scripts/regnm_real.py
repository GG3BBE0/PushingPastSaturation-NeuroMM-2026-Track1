"""regnm_real: REAL Nelder-Mead ensemble weighting (NOT SLSQP).

SLSQP (scripts/regnm_fast.py) stays near-uniform on the non-smooth AUPRC objective
-> simple-mean -> ~ -0.02 public. This script uses the proven Nelder-Mead funcs from
regularized_nm_submission.py: trim (OOF<0.68 AND baseline NMw<0.005) + bagged NM.
NO cos block (cos is split-sensitive noise; the ~2h slowness of the full script was
half from per-config cos). Reads candidate_arch_logits.npz (current pool) + per-arch
OOF; writes submission_test1_regnm_<TAG>.{csv,zip} + Spearman vs v3 + upload gate.

Usage: python scripts/regnm_real.py <TAG> [bag_n=25]
"""
import csv, shutil, zipfile, sys
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import (POOL_LOOKUP, load_oof, derive_nm, derive_bagged_nm,
                                       TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW)

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
RES = REPO / "neuromm26_results"
FOLD = REPO / "fold_df_fixed.csv"
NPZ = RES / "candidate_arch_logits.npz"
CAND = REPO / "NeuroMM-2026/candidate/candidate/candidate_ids.txt"
TAG = sys.argv[1] if len(sys.argv) > 1 else "v6"
BAG_N = int(sys.argv[2]) if len(sys.argv) > 2 else 25
V3_OOF = 0.8974  # current best (regnm_v3 -> public 0.9711)


def load_v3_probs():
    """v3 reference probs: loose csv at repo root, else from the v3 submission zip."""
    csvp = REPO / "submission_test1_regnm_v3.csv"
    if csvp.exists():
        r = list(csv.DictReader(csvp.open()))
        return {x["sample_id"]: float(x["prediction"]) for x in r}
    zp = REPO / "submissions" / "submission_test1_regnm_v3.zip"
    if zp.exists():
        with zipfile.ZipFile(zp) as z, z.open("submission.csv") as fh:
            r = list(csv.DictReader(line.decode() for line in fh))
            return {x["sample_id"]: float(x["prediction"]) for x in r}
    return None


d = np.load(NPZ)
names = [str(n) for n in d["names"]]
Xc = d["logits"]
ids_npz = [str(x) for x in d["ids"]]
print(f"npz archs: {len(names)}  candidate logits: {Xc.shape}")

fold_df = pd.read_csv(FOLD)
fold_df["sample_id"] = fold_df["sample_id"].astype(str)
sid_fold = dict(zip(fold_df["sample_id"], fold_df["fold"]))

# ---- assemble OOF matrix aligned to a common sample order ----
ref_s = ref_y = None
cols, ap = [], []
for nm in names:
    pfx, root = POOL_LOOKUP[nm]
    s, l, y = load_oof(pfx, root)
    ss = s.astype(str)
    if ref_s is None:
        ref_s, ref_y = ss, y
        cols.append(l)
    else:
        im = {x: i for i, x in enumerate(ss)}
        cols.append(l[np.array([im[x] for x in ref_s])])
    ap.append(average_precision_score(y, 1 / (1 + np.exp(-l))))
X = np.stack(cols, 1)
ap = np.array(ap)
rf = np.array([sid_fold[s] for s in ref_s])

# ---- two-dim trim rule (needs full-pool baseline NM weights) ----
wb, ab = derive_nm(X, ref_y)
keep = ~((ap < TRIM_CUTOFF_OOF) & (wb < TRIM_CUTOFF_NMW))
Xt, Xct = X[:, keep], Xc[:, keep]
namesT = [n for i, n in enumerate(names) if keep[i]]
apT = ap[keep]
dropped = [n for i, n in enumerate(names) if not keep[i]]
print(f"baseline full-pool NM OOF {ab:.4f} | pool {len(names)} -> trim {keep.sum()} "
      f"(dropped {len(dropped)}: {dropped})")

# ---- trim + bagged Nelder-Mead ----
wB, oB = derive_nm(Xt, ref_y)
wC = derive_bagged_nm(Xt, ref_y, n_iter=BAG_N)
oC = average_precision_score(ref_y, Xt @ wC)
f2 = average_precision_score(ref_y[rf == 2], Xt[rf == 2] @ wC)
f4 = average_precision_score(ref_y[rf == 4], Xt[rf == 4] @ wC)
print(f"trim NM OOF {oB:.4f} | trim+bagged OOF {oC:.4f} | fold2 {f2:.4f} fold4 {f4:.4f} "
      f"| maxw {wC.max():.3f}")

# ---- candidate predictions + write submission ----
ids = [l.strip() for l in CAND.read_text().splitlines() if l.strip()]
assert ids_npz == ids, "candidate id order mismatch between npz and candidate_ids.txt"
prob = 1 / (1 + np.exp(-(Xct @ wC)))
out = REPO / f"submission_test1_regnm_{TAG}.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_id", "prediction"])
    for sid, p in zip(ids, prob):
        w.writerow([sid, f"{p:.6f}"])
sd = REPO / "submissions"
sd.mkdir(exist_ok=True)
st = sd / "submission.csv"
shutil.copy(out, st)
zf = sd / f"submission_test1_regnm_{TAG}.zip"
with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as zz:
    zz.write(st, arcname="submission.csv")
st.unlink()
print(f"zipped {zf}")

# ---- Spearman vs v3 ----
rho = None
v3 = load_v3_probs()
if v3 is not None:
    rho = float(spearmanr(prob, np.array([v3[i] for i in ids])).statistic)
    print(f"Spearman vs v3 (public 0.9711): {rho:.4f}")
else:
    print("(v3 reference not found — skip Spearman)")

print("\ntop weights:")
for i in np.argsort(-wC)[:14]:
    print(f"  {namesT[i]:34s} {wC[i] * 100:6.2f}%  (single-arch OOF {apT[i]:.3f})")

# ---- upload gate ----
ok_oof = oC > V3_OOF
ok_rho = (rho is not None) and (0.93 <= rho <= 0.995)
print("\n=== UPLOAD GATE ===")
print(f"  OOF {oC:.4f} > v3 {V3_OOF}        : {'PASS' if ok_oof else 'FAIL'}")
print(f"  Spearman {rho if rho is not None else 'NA'} in [0.93, 0.995]: {'PASS' if ok_rho else 'FAIL'}")
print(f"  VERDICT: {'UPLOAD CANDIDATE' if (ok_oof and ok_rho) else 'DO NOT UPLOAD — v3 0.9711 stays best'}")
