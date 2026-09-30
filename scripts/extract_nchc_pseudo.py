"""Safely extract the NCHC 6-arch pseudo OOF, RENAMING prefix '_pseudo_fold__' -> '_pseudo_nchc_fold__'
so it does NOT clobber the local pseudo OOF (the local cwt256+paul256 = the public-0.9769 set).
Validates each (keys + per-fold AUPRC) and prints NCHC-vs-local single-arch OOF so we can see if the
bigger-batch NCHC training actually improved each arch before any ensemble gate. OOF only (no GPU).
"""
import io, zipfile
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score as aps

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
ZIP = REPO / "concat_pseudo_6arch_5fold_228587.zip"
PRED = REPO / "neuromm26_results/predictions"

z = zipfile.ZipFile(ZIP)
oof = [n for n in z.namelist() if n.endswith("_oof.npz")]
written = []
for n in oof:
    base = n.split("predictions/")[-1]
    assert "_pseudo_fold__" in base, base
    new = base.replace("_pseudo_fold__", "_pseudo_nchc_fold__")
    data = z.read(n)
    d = np.load(io.BytesIO(data), allow_pickle=True)
    assert set(d.files) >= {"sample_ids", "logits", "labels"}, (new, d.files)
    (PRED / new).write_bytes(data)
    written.append(new)
print(f"extracted+renamed {len(written)} NCHC OOF -> predictions/ (with _nchc suffix)", flush=True)

# NCHC-vs-local single-arch OOF (pooled across folds), to see if bigger-batch training helped
def pooled_oof(prefix):
    L = []; Y = []
    for f in range(5):
        p = PRED / f"{prefix}__fold{f}__seed0_oof.npz"
        if not p.exists():
            return None
        d = np.load(p)
        L.append(d["logits"]); Y.append(d["labels"])
    L = np.concatenate(L); Y = np.concatenate(Y)
    return aps(Y, 1 / (1 + np.exp(-L)))

PAIRS = [
    ("ConcatCWT 256", "concat_cwt_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", "concat_cwt_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("ConcatPaul 256", "concat_cwt_paul_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", "concat_cwt_paul_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("ConcatFilt 256", "concat_cwt_filtered_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", "concat_cwt_filtered_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("ConcatSTFT 256", "concat_stft_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", "concat_stft_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("ConcatSuperlet 384", "concat_superlet_pseudo_fold__maxvit_tiny_tf_384_in1k", "concat_superlet_pseudo_nchc_fold__maxvit_tiny_tf_384_in1k"),
    ("ConcatCWT 384", "concat_cwt_pseudo_fold__maxvit_tiny_tf_384_in1k", "concat_cwt_pseudo_nchc_fold__maxvit_tiny_tf_384_in1k"),
]
print("\narch                 local-OOF  NCHC-OOF   Δ(nchc-local)", flush=True)
for name, locp, ncp in PAIRS:
    lo = pooled_oof(locp); nc = pooled_oof(ncp)
    los = f"{lo:.4f}" if lo is not None else "  n/a "
    print(f"{name:20s} {los}     {nc:.4f}    {('%+.4f'%(nc-lo)) if lo is not None else 'n/a'}", flush=True)
