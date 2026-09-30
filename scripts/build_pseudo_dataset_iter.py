"""Parameterized iterative pseudo dataset builder (round-N self-training engine).
Each round re-selects pseudo labels from the BEST current ensemble submission (a cleaner source =
better labels), gated downstream on real-label honest OOF (so self-amplification is caught, not
silently submitted). Generalizes build_pseudo_dataset_r2.py.

Usage: python scripts/build_pseudo_dataset_iter.py <source_zip> <tag> [HI] [NEG_PCT]
  e.g. python scripts/build_pseudo_dataset_iter.py submissions/submission_test1_nchc_n2filt.zip r3 0.85 3
Outputs fold_df_pseudo_<tag>.csv + pseudo_sids_<tag>.txt (round-1/2 untouched).
"""
import csv, io, sys, zipfile
import os
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
FOLD = REPO / "fold_df_fixed.csv"
CAND = REPO / "NeuroMM-2026/candidate/candidate"
FEAT = REPO / "neuromm26_datasets/processed/features"
FEATS = ["cwt", "cwt_paul", "superlet", "cwt_filtered", "stft", "eeg"]

argv = sys.argv[1:]
src = REPO / argv[0]
tag = argv[1]
HI = float(argv[2]) if len(argv) > 2 else 0.85
NEG_PCT = float(argv[3]) if len(argv) > 3 else 3.0
OUT_CSV = REPO / f"fold_df_pseudo_{tag}.csv"

z = zipfile.ZipFile(src)
rows = list(csv.DictReader(io.TextIOWrapper(z.open("submission.csv"))))
ids = [r["sample_id"] for r in rows]
p = np.array([float(r["prediction"]) for r in rows])
LO = float(np.percentile(p, NEG_PCT))

pos = [(ids[i], 1) for i in range(len(ids)) if p[i] >= HI]
neg = [(ids[i], 0) for i in range(len(ids)) if p[i] <= LO]
pseudo = pos + neg
print(f"iter source={src.name}  tag={tag}  HI={HI} LO={LO:.3f}(p{NEG_PCT})", flush=True)
print(f"pseudo-pos={len(pos)} pseudo-neg={len(neg)} total={len(pseudo)} "
      f"pos_rate={len(pos)/max(len(pseudo),1):.3f} (discarded middle={len(ids)-len(pseudo)})", flush=True)

linked = {f: 0 for f in FEATS}
for sid, _ in pseudo:
    for f in FEATS:
        s = CAND / f / f"{sid}.npy"; d = FEAT / f / f"{sid}.npy"
        if s.exists() and not d.exists():
            d.symlink_to(s); linked[f] += 1
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
                    "subject_id": f"PSEUDO_{tag}_{sid}", "label_5": lab, "fold": -1})
print(f"wrote {OUT_CSV}: {len(real)} real + {len(pseudo)} pseudo (fold=-1)", flush=True)
(REPO / f"pseudo_sids_{tag}.txt").write_text("\n".join(sid for sid, _ in pseudo) + "\n")
print(f"wrote pseudo_sids_{tag}.txt ({len(pseudo)})", flush=True)
