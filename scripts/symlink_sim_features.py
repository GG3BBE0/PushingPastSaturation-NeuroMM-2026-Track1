"""Mirror the __PSens pseudo-copy ids of fold_df_simens_{k}.csv into ANOTHER feature dir, so a
non-cwt arch (superlet @384 / muku eeg) can private-sim with the SAME ensemble-teacher pseudo
selection the cwt sim used. The selection is teacher-driven (arch-independent) — only the feature
symlinks differ per arch. {base}__PSens.npy -> {base}.npy in features/<dir>/.

Usage: python scripts/symlink_sim_features.py <feature_dir_name> <k>
  e.g. python scripts/symlink_sim_features.py superlet 4
       python scripts/symlink_sim_features.py eeg 4
"""
import csv, sys
import os
from pathlib import Path

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
FEAT = REPO / "neuromm26_datasets/processed/features"
fdir, k = sys.argv[1], int(sys.argv[2])
D = FEAT / fdir
assert D.is_dir(), f"no feature dir {D}"

csv_path = REPO / f"fold_df_simens_{k}.csv"
with csv_path.open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
ps = [r["sample_id"] for r in rows if r["sample_id"].endswith("__PSens")]

n_link = n_skip = n_missing = 0
for sid in ps:
    base = sid[:-len("__PSens")]
    src = D / f"{base}.npy"
    dst = D / f"{sid}.npy"
    if not src.exists():
        n_missing += 1
        continue
    if dst.exists() or dst.is_symlink():
        n_skip += 1
        continue
    dst.symlink_to(src)
    n_link += 1
print(f"[{fdir} fold{k}] {len(ps)} __PSens ids: linked {n_link}, existed {n_skip}, missing-src {n_missing}",
      flush=True)
