"""Build a pseudo-labeled augmented training set.

High-confidence-ONLY to limit self-amplification: the v3 ensemble's candidate probs are
UNCALIBRATED (negatives cluster ~0.32-0.35, not 0), so we select by RELATIVE confidence:
  pseudo-pos = prob >= HI   (clearly-separated high tail)
  pseudo-neg = prob <= LO   (tight low cluster)
The uncertain middle is discarded. Selected candidates get hard labels, are symlinked into the
train feature dirs, and added to fold_df_pseudo.csv with fold=-1 (ALWAYS train, NEVER val) so the
5-fold OOF still validates on REAL train labels only (honest).
"""
import csv
import io
import zipfile
import os
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
V3 = REPO / "submissions/submission_test1_regnm_v3.zip"
FOLD = REPO / "fold_df_fixed.csv"
OUT_CSV = REPO / "fold_df_pseudo.csv"
CAND = REPO / "NeuroMM-2026/candidate/candidate"
FEAT = REPO / "neuromm26_datasets/processed/features"
# feature dirs to symlink (arches we'll retrain on the augmented set)
FEATS = ["cwt", "cwt_paul", "superlet", "cwt_filtered", "stft", "eeg"]
# CAREFUL design: positive-DOMINANT (confident positives are safe+valuable for rare-positive
# AUPRC); negatives are RISKY (the low cluster ~0.32-0.35 may hide missed spikes) so take only
# the tightest few. ALL pseudo are down-weighted at train time (see --noise-sids pseudo_sids.txt).
HI, LO = 0.90, 0.330

# 1. candidate probs
z = zipfile.ZipFile(V3)
rows = list(csv.DictReader(io.TextIOWrapper(z.open("submission.csv"))))
ids = [r["sample_id"] for r in rows]
p = np.array([float(r["prediction"]) for r in rows])

pos = [(ids[i], 1) for i in range(len(ids)) if p[i] >= HI]
neg = [(ids[i], 0) for i in range(len(ids)) if p[i] <= LO]
pseudo = pos + neg
print(f"pseudo-pos={len(pos)}  pseudo-neg={len(neg)}  total={len(pseudo)} "
      f"pos_rate={len(pos)/max(len(pseudo),1):.3f}  (discarded middle={len(ids)-len(pseudo)})", flush=True)

# 2. symlink selected candidate features into the train feature dirs
linked = {f: 0 for f in FEATS}
for sid, _ in pseudo:
    for f in FEATS:
        src = CAND / f / f"{sid}.npy"
        dst = FEAT / f / f"{sid}.npy"
        if src.exists() and not dst.exists():
            dst.symlink_to(src); linked[f] += 1
print(f"symlinked features: {linked}", flush=True)

# 3. build fold_df_pseudo.csv = real fold_df + pseudo rows (fold=-1, unique PSEUDO subject)
with FOLD.open(encoding="utf-8-sig") as f:
    real = list(csv.DictReader(f))
cols = real[0].keys()
with OUT_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in real:
        w.writerow(r)
    for sid, lab in pseudo:
        w.writerow({"sample_id": sid, "label": lab, "label_type": lab,
                    "subject_id": f"PSEUDO_{sid}", "label_5": lab, "fold": -1})
print(f"wrote {OUT_CSV} : {len(real)} real + {len(pseudo)} pseudo = {len(real)+len(pseudo)} rows", flush=True)
print("pseudo rows are fold=-1 -> always TRAIN, never VAL -> OOF stays honest on real labels", flush=True)

# pseudo sids for TRAIN-TIME DOWN-WEIGHTING (--noise-sids pseudo_sids.txt --noise-weight 0.4)
(REPO / "pseudo_sids.txt").write_text("\n".join(sid for sid, _ in pseudo) + "\n")
print(f"wrote pseudo_sids.txt ({len(pseudo)}) for down-weighting", flush=True)
