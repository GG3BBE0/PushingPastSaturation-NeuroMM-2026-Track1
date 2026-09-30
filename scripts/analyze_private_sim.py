"""Analyze the private-safety simulation. For each fold k:
  baseline  = canonical cwt OOF AUPRC on fold-k   (model trained on other folds, did NOT pseudo-train on fold-k)
  specialist= cwt_simfold{k} OOF AUPRC on fold-k   (SAME but ALSO trained on fold-k's pseudo-copies @ weight 1.0)
Δ = specialist - baseline = the EFFECT of the candidate-specialist trick on the pseudo-trained samples'
TRUE-label score = a DIRECT proxy for the private-LB effect (which our honest OOF gate cannot see).
Δ > 0 across folds  -> pseudo-training the held-out samples HELPS their true score -> private SAFE.
"""
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import average_precision_score as aps

PRED = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent)) / "neuromm26_results/predictions"
BASE = "concat_cwt_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"


def auprc(pfx, k):
    p = PRED / f"{pfx}__fold{k}__seed0_oof.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return aps(d["labels"], 1 / (1 + np.exp(-np.clip(d["logits"], -30, 30))))


print("fold   baseline(no pseudo)   specialist(pseudo-trained)   Δ(private effect)")
ds = []
for k in range(5):
    b = auprc(BASE, k); s = auprc(f"cwt_simfold{k}", k)
    if b is None or s is None:
        print(f"  {k}: {'baseline' if b is None else 'specialist'} not ready"); continue
    ds.append(s - b)
    print(f"  {k}     {b:.4f}                {s:.4f}                {s-b:+.4f}")
if ds:
    md = float(np.mean(ds))
    print(f"\nmeanΔ = {md:+.4f}  over {len(ds)} folds")
    if md > 0.002:
        print("VERDICT: pseudo-training the held-out 'fake-private' samples IMPROVES their TRUE score "
              "-> the candidate-specialist generalizes, does NOT memorize-and-hurt -> PRIVATE SAFE.")
    elif md > -0.002:
        print("VERDICT: ~neutral on the pseudo-trained samples' true score -> private roughly unaffected "
              "(gain is generalization on the UNtrained candidates; no private harm).")
    else:
        print("VERDICT: pseudo-training HURTS the held-out samples' true score -> private-LB RISK is real "
              "-> prefer the lower-memorization soft/noisy-student version + keep 0.9778 as the safe final.")
