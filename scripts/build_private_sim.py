"""PRIVATE-SAFETY SIMULATION builder (answers: does training on pseudo-labeled samples HELP or HURT
those samples' TRUE-label score? = the candidate-specialist's private-LB blind spot, measured directly).

For test-fold k: treat fold-k as 'fake private candidates'. Teacher = the canonical cwt model that
NEVER saw fold-k (its honest OOF on fold-k = a clean model-trained-on-other-folds predicting fold-k,
exactly like the ensemble predicting the real unseen candidates). Pseudo-label fold-k with the teacher
(HI=0.60 pos / bottom-35% neg, same as the real spec selection), add those as DISTINCT-id copies
(`{sid}__PS`, features symlinked) at fold=-1 (train), keep the REAL fold-k as the val set.

Then train_spec_concat_fold --fold-idx k --fold-csv fold_df_sim_k.csv evaluates on fold-k REAL labels:
  specialist (trained WITH fold-k pseudo) vs baseline (canonical cwt, trained WITHOUT fold-k).
specialist > baseline  -> pseudo-training the held-out samples HELPS their true score -> private SAFE.
specialist < baseline  -> it HURTS -> private risk is real.

Usage: python scripts/build_private_sim.py <k>
"""
import csv, sys
import os
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
FOLD = REPO / "fold_df_fixed.csv"
PRED = REPO / "neuromm26_results/predictions"
CWT = REPO / "neuromm26_datasets/processed/features/cwt"
TEACHER = "concat_cwt_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"   # clean canonical cwt (honest OOF = teacher)
HI, NEG_PCT = 0.60, 35.0
k = int(sys.argv[1])
teacher_file = sys.argv[2] if len(sys.argv) > 2 else None  # faithful ENSEMBLE teacher (sid prob lines)
TAG = "ens" if teacher_file else ""

d = np.load(PRED / f"{TEACHER}__fold{k}__seed0_oof.npz", allow_pickle=True)
sids = [str(s) for s in d["sample_ids"]]; prob = 1 / (1 + np.exp(-np.clip(d["logits"], -30, 30)))
if teacher_file:  # override single-arch probs with the ensemble teacher's probs for these sids
    tm = {}
    for ln in Path(teacher_file).read_text().splitlines():
        if ln.strip():
            s, p = ln.split()[:2]; tm[s] = float(p)
    prob = np.array([tm.get(s, 0.5) for s in sids])
lo = float(np.percentile(prob, NEG_PCT))
sel = []  # (sid, pseudo_label)
for sid, p in zip(sids, prob):
    if p >= HI: sel.append((sid, 1))
    elif p <= lo: sel.append((sid, 0))
y = d["labels"].astype(int); sidset = {s: int(yy) for s, yy in zip(sids, y)}
# teacher pseudo-label accuracy on fold-k (how good the labels we train on are)
acc = np.mean([pl == sidset[s] for s, pl in sel]) if sel else float("nan")
print(f"fold{k}: teacher pseudo {len(sel)} of {len(sids)} (pos {sum(pl for _,pl in sel)}); "
      f"pseudo-label ACC vs real = {acc:.3f}; LO={lo:.3f}", flush=True)

# symlink {sid}__PS.npy -> {sid}.npy so the pseudo-copies load the same features
n_link = 0
for sid, _ in sel:
    src = CWT / f"{sid}.npy"; dst = CWT / f"{sid}__PS{TAG}.npy"
    if src.exists() and not dst.exists():
        dst.symlink_to(src); n_link += 1
print(f"symlinked {n_link} __PS{TAG} feature copies", flush=True)

with FOLD.open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
cols = rows[0].keys()
out = REPO / f"fold_df_sim{TAG}_{k}.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows:
        w.writerow(r)
    for sid, pl in sel:  # pseudo-copies: distinct id, fold=-1 (train), pseudo label
        w.writerow({"sample_id": f"{sid}__PS{TAG}", "label": pl, "label_type": pl,
                    "subject_id": f"SIM{TAG}_{sid}", "label_5": pl, "fold": -1})
print(f"wrote {out}: {len(rows)} real + {len(sel)} pseudo-copies (fold=-1); val = real fold{k}", flush=True)
