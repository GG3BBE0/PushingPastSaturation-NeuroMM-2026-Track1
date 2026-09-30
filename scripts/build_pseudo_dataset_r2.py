"""Round-2 pseudo dataset: re-select from the BETTER source (pseudo3, public 0.9769+ vs v3 0.9711).
Iterative pseudo-labeling — guarded by the honest-OOF/ROBUST+ gate (real labels only), so any
self-amplification shows up as gate failure, not a silent private collapse.

pseudo3 is better-calibrated (negatives separated lower: p50 0.241 vs v3 0.352), so round-2 can
take slightly more positives (HI 0.85; tail compressed -> 0.85_p3 ~ 0.90_v3) and the tightest
~3% negatives (now well-separated) while staying positive-dominant. Separate outputs
(fold_df_pseudo_r2.csv / pseudo_sids_r2.txt) so round-1 stays intact & reproducible.
"""
import csv, io, zipfile
import os
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
SRC = REPO / "submissions/submission_test1_pseudo3.zip"   # round-2 source = best available
FOLD = REPO / "fold_df_fixed.csv"
OUT_CSV = REPO / "fold_df_pseudo_r2.csv"
CAND = REPO / "NeuroMM-2026/candidate/candidate"
FEAT = REPO / "neuromm26_datasets/processed/features"
FEATS = ["cwt", "cwt_paul", "superlet", "cwt_filtered", "stft", "eeg"]
HI = 0.85          # ~2425 confident positives (tail compressed in pseudo3)
NEG_PCT = 3.0      # tightest 3% (~600) negatives — now well-separated, still positive-dominant

z = zipfile.ZipFile(SRC)
rows = list(csv.DictReader(io.TextIOWrapper(z.open("submission.csv"))))
ids = [r["sample_id"] for r in rows]
p = np.array([float(r["prediction"]) for r in rows])
LO = float(np.percentile(p, NEG_PCT))

pos = [(ids[i], 1) for i in range(len(ids)) if p[i] >= HI]
neg = [(ids[i], 0) for i in range(len(ids)) if p[i] <= LO]
pseudo = pos + neg
print(f"round-2 source={SRC.name}  HI={HI} LO={LO:.3f}(p{NEG_PCT})", flush=True)
print(f"pseudo-pos={len(pos)}  pseudo-neg={len(neg)}  total={len(pseudo)}  "
      f"pos_rate={len(pos)/max(len(pseudo),1):.3f}  (discarded middle={len(ids)-len(pseudo)})", flush=True)

linked = {f: 0 for f in FEATS}
for sid, _ in pseudo:
    for f in FEATS:
        src = CAND / f / f"{sid}.npy"; dst = FEAT / f / f"{sid}.npy"
        if src.exists() and not dst.exists():
            dst.symlink_to(src); linked[f] += 1
print(f"symlinked new features: {linked}", flush=True)

with FOLD.open(encoding="utf-8-sig") as f:
    real = list(csv.DictReader(f))
cols = real[0].keys()
with OUT_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in real:
        w.writerow(r)
    for sid, lab in pseudo:
        w.writerow({"sample_id": sid, "label": lab, "label_type": lab,
                    "subject_id": f"PSEUDO2_{sid}", "label_5": lab, "fold": -1})
print(f"wrote {OUT_CSV}: {len(real)} real + {len(pseudo)} pseudo = {len(real)+len(pseudo)} rows (fold=-1)", flush=True)

(REPO / "pseudo_sids_r2.txt").write_text("\n".join(sid for sid, _ in pseudo) + "\n")
print(f"wrote pseudo_sids_r2.txt ({len(pseudo)}) for down-weighting", flush=True)
