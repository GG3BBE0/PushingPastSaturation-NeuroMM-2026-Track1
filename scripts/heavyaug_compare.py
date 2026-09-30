"""Compare heavy-aug vs baseline ConcatSuperlet maxvit384 per-fold OOF AUPRC."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
RES = REPO / "neuromm26_results"
BASELINE_PREFIX = "concat_superlet_fold__maxvit_tiny_tf_384_in1k"
HEAVY_PREFIX = "concat_superlet_heavyaug_fold__maxvit_tiny_tf_384_in1k"
REPORT = REPO / "HEAVYAUG_REPORT.md"


def per_fold_oof(prefix):
    folds = {}
    for f in range(5):
        path = RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz"
        if not path.exists():
            folds[f] = None
            continue
        d = np.load(path)
        p = 1 / (1 + np.exp(-d["logits"].astype(np.float64)))
        folds[f] = average_precision_score(d["labels"], p)
    return folds


b = per_fold_oof(BASELINE_PREFIX)
h = per_fold_oof(HEAVY_PREFIX)

lines = ["# Heavy Aug Ablation — ConcatSuperlet maxvit384", "",
         "| fold | baseline | heavy aug | Δ |", "|---|---|---|---|"]
b_vals, h_vals = [], []
for f in range(5):
    bv = b.get(f); hv = h.get(f)
    bs = f"{bv:.4f}" if bv is not None else "—"
    hs = f"{hv:.4f}" if hv is not None else "—"
    d = (hv - bv) if (bv is not None and hv is not None) else None
    ds = f"{d:+.4f}" if d is not None else "—"
    lines.append(f"| {f} | {bs} | {hs} | {ds} |")
    if bv is not None: b_vals.append(bv)
    if hv is not None: h_vals.append(hv)

if b_vals and h_vals and len(b_vals) == 5 and len(h_vals) == 5:
    b_avg = sum(b_vals) / 5
    h_avg = sum(h_vals) / 5
    lines.append(f"| **avg** | **{b_avg:.4f}** | **{h_avg:.4f}** | **{h_avg - b_avg:+.4f}** |")
    lines.append("")
    lines.append("## Decision")
    if h_avg - b_avg >= 0.003:
        lines.append(f"✅ +{h_avg - b_avg:.4f} ≥ +0.003 → **HEAVY AUG WORKS**. "
                     "Consider rolling out to ConcatFilt / ConcatPaul / ConcatCWT maxvit family.")
    elif h_avg - b_avg >= 0.000:
        lines.append(f"~ +{h_avg - b_avg:.4f} marginal → keep but don't expand.")
    else:
        lines.append(f"❌ {h_avg - b_avg:+.4f} → heavy aug HURT or no help. Stop aug path.")

print("\n".join(lines))
REPORT.write_text("\n".join(lines))
print(f"\nWrote {REPORT}")
