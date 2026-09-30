"""Analyze the new-arch private-safety sim. Per fold k and per arch (superlet384, muku), compare the
SPECIALIST sim (trained WITH fold-k ensemble-teacher pseudo, eval on fold-k TRUE labels) vs the CANONICAL
baseline (trained WITHOUT fold-k). Δ = specialist − canonical on the held-out (private-proxy) true score.
Δ > 0 = specializing the held-out samples HELPS their true score => the swapped-in specnchc member is
private-SAFE-to-positive (the blind-spot the honest-OOF gate cannot see).
"""
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score as aps

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
PRED = REPO / "neuromm26_results/predictions"
REAL = REPO / "neuromm26_real_5fold_result/predictions"
FOLDS = [0, 1, 4]

ARCHS = {
    "superlet384": {
        "sim": lambda k: PRED / f"superlet_simensfold{k}__fold{k}__seed0_oof.npz",
        "canon": lambda k: PRED / f"concat_superlet_fold__maxvit_tiny_tf_384_in1k__fold{k}__seed0_oof.npz",
    },
    "muku_raw   ": {
        "sim": lambda k: PRED / f"muku_simensfold{k}__fold{k}__seed0_oof.npz",
        "canon": lambda k: REAL / f"muku_fold__convnext_pico_d1_in1k__fold{k}__seed0_oof.npz",
    },
}


def ap_of(p):
    if not p.exists():
        return None, None
    d = np.load(p, allow_pickle=True)
    sid = np.array([str(s) for s in d["sample_ids"]])
    prob = 1 / (1 + np.exp(-np.clip(d["logits"].astype(np.float64), -30, 30)))
    y = d["labels"].astype(int)
    return {s: (pr, yy) for s, pr, yy in zip(sid, prob, y)}, y


print("=== new-arch private-safety sim (Δ = specialist − canonical on held-out TRUE labels) ===\n")
summary = {}
for arch, fns in ARCHS.items():
    deltas = []
    for k in FOLDS:
        sm, _ = ap_of(fns["sim"](k))
        cm, _ = ap_of(fns["canon"](k))
        if sm is None or cm is None:
            print(f"{arch} fold{k}: {'sim PENDING' if sm is None else 'canon MISSING'}")
            continue
        ids = [s for s in sm if s in cm]  # common val ids (= real fold-k)
        y = np.array([sm[s][1] for s in ids])
        a_sim = aps(y, np.array([sm[s][0] for s in ids]))
        a_can = aps(y, np.array([cm[s][0] for s in ids]))
        d = a_sim - a_can
        deltas.append(d)
        print(f"{arch} fold{k}: specialist={a_sim:.4f}  canonical={a_can:.4f}  Δ={d:+.4f}  "
              f"{'HELP' if d > 0 else 'HURT'}  (n={len(ids)}, pos={int(y.sum())})")
    if deltas:
        md = float(np.mean(deltas))
        summary[arch] = md
        print(f"{arch} >>> meanΔ over {len(deltas)} folds = {md:+.4f}  "
              f"({'NET-POSITIVE → private safe' if md > 0 else 'NET-NEGATIVE → flag'})\n")

print("--- comparison vs cwt ensemble-teacher sim (memory: cwt meanΔ +0.0318, 1/5 neg) ---")
for arch, md in summary.items():
    print(f"  {arch}: meanΔ {md:+.4f}")
print("\nspecnchc submission = {cwt,paul,filt}(cwt-sim +0.0318) + superlet384 + muku above. "
      "All net-positive => private should GAIN like public, not collapse.")
