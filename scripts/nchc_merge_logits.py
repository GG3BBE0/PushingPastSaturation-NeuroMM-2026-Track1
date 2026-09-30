"""Merge the 3 worker temp-logit npys into candidate_arch_logits.npz in ONE write (no race).
Usage: python scripts/nchc_merge_logits.py <name1> <npy1> <name2> <npy2> ..."""
import sys
import os
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))

d = np.load(NPZ, allow_pickle=True)
names = [str(n) for n in d["names"]]; X = d["logits"]; ids = d["ids"]
add_names, add_cols = [], []
for name, npy in pairs:
    if name in names or name in add_names:
        print(f"  skip '{name}' (already present)", flush=True); continue
    logit = np.load(npy)
    assert logit.shape[0] == X.shape[0], f"{name}: {logit.shape} vs {X.shape}"
    add_names.append(name); add_cols.append(logit[:, None])
    print(f"  + '{name}': mean {logit.mean():.3f} std {logit.std():.3f}", flush=True)
if add_cols:
    X2 = np.concatenate([X] + add_cols, axis=1)
    np.savez(NPZ, names=np.array(names + add_names), logits=X2, ids=ids)
    print(f"merged {len(add_cols)} -> npz now {X2.shape[1]} archs", flush=True)
else:
    print("nothing to merge", flush=True)
