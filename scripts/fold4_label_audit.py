"""fold4 diagnosis: is the low fold4 AUPRC LABEL NOISE (fixable by selective cleaning) or
genuine HARDNESS (fixable by oversampling/aug)? Free — uses existing 48-arch OOF, no GPU.

Confident-learning style: ensemble OOF prob is an out-of-fold predictor. A label=1 sample the
ensemble confidently calls negative is a likely false-positive label; label=0 called strongly
positive is a likely missed spike. We quantify these per fold and for DA00103D, and estimate a
'noise-corrected' fold4 AUPRC (remove the most-confident label-disagreements from fold4 val).
"""
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps

sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import POOL_LOOKUP, load_oof

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
FOLD = REPO / "fold_df_fixed.csv"
NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"

fdf = pd.read_csv(FOLD); fdf["sample_id"] = fdf["sample_id"].astype(str)
sid_lab = dict(zip(fdf["sample_id"], fdf["label"].astype(int)))
sid_subj = dict(zip(fdf["sample_id"], fdf["subject_id"].astype(str)))
sid_fold = dict(zip(fdf["sample_id"], fdf["fold"].astype(int)))

# ensemble OOF prob = mean of strong archs (OOF>0.80), simple + robust (no NM needed for audit)
pool_names = [str(n) for n in np.load(NPZ, allow_pickle=True)["names"]]
ref_s = None; probs = []; used = []
for nm in pool_names:
    pfx, root = POOL_LOOKUP[nm]
    s, l, y = load_oof(pfx, root); ss = s.astype(str)
    if aps(y, 1 / (1 + np.exp(-l))) < 0.80:
        continue
    if ref_s is None:
        ref_s, ref_y = ss, y.astype(int)
        im = {x: i for i, x in enumerate(ref_s)}
    p = 1 / (1 + np.exp(-l))
    probs.append(p[[ {x: i for i, x in enumerate(ss)}[x] for x in ref_s ]])
    used.append(nm)
P = np.mean(probs, axis=0)  # ensemble OOF prob aligned to ref_s
y = ref_y
fold = np.array([sid_fold[s] for s in ref_s])
subj = np.array([sid_subj[s] for s in ref_s])
print(f"ensemble OOF prob = mean of {len(used)} strong archs; N={len(ref_s)} overall AUPRC={aps(y,P):.4f}\n")

# confident-learning thresholds (self-confidence per class)
t1 = P[y == 1].mean()          # avg P(spike) on label-1
t0 = (1 - P[y == 0]).mean()    # avg P(non-spike) on label-0
print(f"CL thresholds: t1(meanP|y=1)={t1:.3f}  t0(mean(1-P)|y=0)={t0:.3f}")

# label errors: y=1 confidently negative (P<=1-t0) ; y=0 confidently positive (P>=t1)
err_fp = (y == 1) & (P <= 1 - t0)   # labeled spike, ensemble sure non-spike -> likely mislabeled+
err_fn = (y == 0) & (P >= t1)       # labeled non-spike, ensemble sure spike -> likely missed spike
err = err_fp | err_fn
print(f"estimated label errors: {err.sum()} / {len(y)} = {100*err.mean():.2f}%  "
      f"(false-pos labels {err_fp.sum()}, missed-spike labels {err_fn.sum()})\n")

print(f"{'fold':<6}{'N':<7}{'pos%':<7}{'err%':<7}{'err_fp':<8}{'err_fn':<8}{'AUPRC':<8}")
for f in range(5):
    m = fold == f
    print(f"{f:<6}{m.sum():<7}{100*y[m].mean():<7.2f}{100*err[m].mean():<7.2f}"
          f"{(err_fp & m).sum():<8}{(err_fn & m).sum():<8}{aps(y[m],P[m]):<8.4f}")

# DA00103D + worst fold4 subjects
print("\nfold4 subjects (label-noise suspects):")
f4 = fold == 4
for s in pd.unique(subj[f4]):
    ms = f4 & (subj == s)
    if ms.sum() < 30:
        continue
    print(f"  {s:<12} n={ms.sum():<4} pos={int(y[ms].sum()):<4}({100*y[ms].mean():4.1f}%) "
          f"err={int(err[ms].sum()):<3}({100*err[ms].mean():4.1f}%) "
          f"meanP={P[ms].mean():.3f} AUPRC={aps(y[ms],P[ms]) if 0<y[ms].sum()<ms.sum() else float('nan'):.3f}")

# noise-corrected fold4 AUPRC: drop the flagged label-error samples from fold4 val
keep4 = f4 & ~err
print(f"\nfold4 AUPRC  original={aps(y[f4],P[f4]):.4f}  "
      f"after removing {int((f4&err).sum())} flagged-noise samples={aps(y[keep4],P[keep4]):.4f}")
print("  (big jump => fold4 is largely LABEL NOISE; small jump => genuine HARDNESS)")

# top-15 most confident label-error suspects in fold4
print("\ntop fold4 label-error suspects (|P - y| largest):")
idx4 = np.where(f4 & err)[0]
order = idx4[np.argsort(-np.abs(P[idx4] - y[idx4]))][:15]
for i in order:
    kind = "spike-labeled-but-NEG" if y[i] == 1 else "nonspike-labeled-but-POS"
    print(f"  {ref_s[i]:<26} subj={subj[i]:<10} y={y[i]} P={P[i]:.3f}  {kind}")

# ---- dump flagged sids, split by treatment (consensus-justified, see missed_spike_consensus.py) ----
# err_fn (y=0, ensemble-confident-positive) = high-consensus missed spikes -> RELABEL y0->1
# err_fp (y=1, ensemble-confident-negative, incl DA00103D) = AMBIGUOUS -> DOWN-WEIGHT only
(REPO / "neuromm26_results/fold4_noise_sids.txt").write_text(
    "\n".join(ref_s[i] for i in np.where(err)[0]) + "\n")
relabel = [ref_s[i] for i in np.where(err_fn)[0]]
downwt = [ref_s[i] for i in np.where(err_fp)[0]]
(REPO / "neuromm26_results/fold4_relabel_sids.txt").write_text("\n".join(relabel) + "\n")
(REPO / "neuromm26_results/fold4_downweight_sids.txt").write_text("\n".join(downwt) + "\n")
print(f"\nwrote {int(err.sum())} flagged: {len(relabel)} RELABEL (y0->1) + {len(downwt)} DOWN-WEIGHT")
