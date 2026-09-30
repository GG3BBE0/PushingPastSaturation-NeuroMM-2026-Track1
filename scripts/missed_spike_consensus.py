"""Are the flagged samples TRUE label noise (relabel-worthy) or model disagreement (only
down-weight)? Decide by INDEPENDENT-ARCH CONSENSUS: if many of the ~19 strong archs each
INDEPENDENTLY call a y=0 sample positive, it is very unlikely they are all wrong the same way
-> it is a real missed spike -> relabeling (y=0->1) is justified (a stronger fix than
down-weighting). Low consensus -> keep to down-weight only.

Run capped: OMP_NUM_THREADS=1 python scripts/missed_spike_consensus.py
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
NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
FOLD = REPO / "fold_df_fixed.csv"
NOISE = REPO / "neuromm26_results/fold4_noise_sids.txt"

flagged = {l.strip() for l in NOISE.read_text().splitlines() if l.strip()}
fdf = pd.read_csv(FOLD); fdf["sample_id"] = fdf["sample_id"].astype(str)
sid_lab = dict(zip(fdf["sample_id"], fdf["label"].astype(int)))
sid_subj = dict(zip(fdf["sample_id"], fdf["subject_id"].astype(str)))

EXCLUDE = {"ConcatCWT clean256", "ConcatCWT cleanbase256", "NeuroMAE v1", "GNN orthopara", "Spec3D superlet"}
names = [str(n) for n in np.load(NPZ, allow_pickle=True)["names"] if str(n) not in EXCLUDE]

# per-arch prob, keep only strong archs (OOF>0.80), aligned
ref_s = None; probs = []; used = []
for nm in names:
    pfx, root = POOL_LOOKUP[nm]; s, l, y = load_oof(pfx, root); ss = s.astype(str)
    if aps(y, 1 / (1 + np.exp(-l))) < 0.80:
        continue
    if ref_s is None:
        ref_s = ss
    im = {x: i for i, x in enumerate(ss)}
    probs.append((1 / (1 + np.exp(-l)))[[im[x] for x in ref_s]]); used.append(nm)
Pa = np.stack(probs, 1)  # (N, n_arch)
y = np.array([sid_lab[s] for s in ref_s])
n_arch = Pa.shape[1]
print(f"{n_arch} strong archs; consensus = fraction of archs each independently predicting positive (P>0.5)\n", flush=True)

# fraction of archs calling each sample positive
frac_pos = (Pa > 0.5).mean(1)
flag = np.array([s in flagged for s in ref_s])

# missed-spike direction: y=0 flagged. high frac_pos = strong consensus it's really a spike.
miss = flag & (y == 0)
print(f"=== 'missed spike' suspects (y=0, flagged): {miss.sum()} ===")
fp = frac_pos[miss]
for thr in [0.9, 0.8, 0.7, 0.5]:
    print(f"  {(fp >= thr).sum():3d} have >= {int(thr*100)}% of archs independently calling them positive")
print(f"  -> high-consensus (>=80%) missed-spikes are strong RELABEL candidates")

# DA00103D direction: y=1 flagged. low frac_pos = consensus they're really negative.
spk = flag & (y == 1)
print(f"\n=== 'spike-labeled-but-NEG' suspects (y=1, flagged): {spk.sum()} ===")
fn = 1 - frac_pos[spk]
for thr in [0.9, 0.8, 0.7, 0.5]:
    print(f"  {(fn >= thr).sum():3d} have >= {int(thr*100)}% of archs independently calling them NEGATIVE")
da = spk & np.array([sid_subj[s] == "DA00103D" for s in ref_s])
print(f"  of which DA00103D: {da.sum()} (mean arch-neg-consensus {(1-frac_pos[da]).mean():.2f})")
print("  -> these are AMBIGUOUS (noise vs model blind-spot); down-weight, do NOT flip")

print(f"\nVERDICT: relabel the {(frac_pos[miss] >= 0.8).sum()} high-consensus missed-spikes (y0->1) is "
      f"{'JUSTIFIED' if (frac_pos[miss] >= 0.8).sum() >= 30 else 'marginal'}; keep DA00103D direction as down-weight only.")
