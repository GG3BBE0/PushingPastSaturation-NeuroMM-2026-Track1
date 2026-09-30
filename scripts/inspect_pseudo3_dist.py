"""Inspect the pseudo3 candidate prob distribution to choose round-2 pseudo thresholds.
Compare vs v3 (round-1 source) at the same nominal HI/LO and by percentile, so round-2 selection
is grounded in the new (cleaner) ensemble's actual separation, not blindly reused 0.90/0.330.
"""
import csv, io, zipfile
import os
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))


def probs(zp):
    z = zipfile.ZipFile(zp)
    rows = list(csv.DictReader(io.TextIOWrapper(z.open("submission.csv"))))
    return {r["sample_id"]: float(r["prediction"]) for r in rows}


v3 = probs(REPO / "submissions/submission_test1_regnm_v3.zip")
p3 = probs(REPO / "submissions/submission_test1_pseudo3.zip")
ids = list(p3)
a3 = np.array([p3[i] for i in ids]); av = np.array([v3[i] for i in ids])

for name, a in [("v3", av), ("pseudo3", a3)]:
    qs = np.percentile(a, [50, 75, 90, 95, 97, 98, 99])
    print(f"{name}: min={a.min():.3f} mean={a.mean():.3f} max={a.max():.3f} | "
          f"p50={qs[0]:.3f} p75={qs[1]:.3f} p90={qs[2]:.3f} p95={qs[3]:.3f} p97={qs[4]:.3f} p98={qs[5]:.3f} p99={qs[6]:.3f}")

print("\n-- pseudo3 selection counts at candidate thresholds --")
for HI in [0.92, 0.90, 0.88, 0.85, 0.82]:
    print(f"  HI={HI}: pos={(a3>=HI).sum()}")
for LO in [0.30, 0.31, 0.32, 0.33, 0.34]:
    print(f"  LO={LO}: neg={(a3<=LO).sum()}")

# round-1 set (from v3) for overlap reference
r1pos = {ids[i] for i in range(len(ids)) if av[i] >= 0.90}
r1neg = {ids[i] for i in range(len(ids)) if av[i] <= 0.330}
print(f"\nround-1 (v3): pos={len(r1pos)} neg={len(r1neg)}")
# how stable are round-1 positives under pseudo3?
if r1pos:
    kept = np.array([p3[i] for i in r1pos])
    print(f"round-1 positives under pseudo3: mean={kept.mean():.3f} min={kept.min():.3f} "
          f">=0.85: {(kept>=0.85).sum()}/{len(r1pos)} (stability of the pos set)")
