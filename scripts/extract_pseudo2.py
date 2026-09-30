"""Extract NCHC pseudo2 OOF (25 files, NEW prefixes _r3nchc / _pseudo_nchc -> no clobber). Validate
and report single-arch OOF vs references:
  aggressive r3nchc cwt/paul/filt  vs  the nchc (round-1) versions  -> did r3 labels + bigger batch change strength?
  muku_nchc convnext/resnet18      vs  local muku_pseudo (round-1)  -> did NCHC bigger-batch strengthen the orthogonal muku?
OOF only, no GPU.
"""
import io, zipfile
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score as aps

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
ZIP = REPO / "pseudo2_5arch_5fold_230695.zip"
PRED = REPO / "neuromm26_results/predictions"

z = zipfile.ZipFile(ZIP)
n = 0
for entry in z.namelist():
    if entry.endswith("_oof.npz"):
        base = entry.split("predictions/")[-1]
        data = z.read(entry)
        d = np.load(io.BytesIO(data), allow_pickle=True)
        assert set(d.files) >= {"sample_ids", "logits", "labels"}, (base, d.files)
        (PRED / base).write_bytes(data); n += 1
print(f"extracted {n} pseudo2 OOF (expect 25)\n", flush=True)


def pooled(prefix):
    L, Y = [], []
    for f in range(5):
        p = PRED / f"{prefix}__fold{f}__seed0_oof.npz"
        if not p.exists():
            return None
        d = np.load(p); L.append(d["logits"]); Y.append(d["labels"])
    return aps(np.concatenate(Y), 1 / (1 + np.exp(-np.concatenate(L))))


PAIRS = [
    ("CWT  256 (r1nchc->r3nchc)", "concat_cwt_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k",
     "concat_cwt_pseudo_r3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("Paul 256 (r1nchc->r3nchc)", "concat_cwt_paul_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k",
     "concat_cwt_paul_pseudo_r3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("Filt 256 (r1nchc->r3nchc)", "concat_cwt_filtered_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k",
     "concat_cwt_filtered_pseudo_r3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("muku convnext (local->nchc)", "muku_pseudo_fold__convnext_pico_d1_in1k",
     "muku_pseudo_nchc_fold__convnext_pico_d1_in1k"),
    ("muku resnet18 (local->nchc)", "muku_pseudo_fold__resnet18",
     "muku_pseudo_nchc_fold__resnet18"),
]
print("arch                          ref-OOF   new-OOF    Δ", flush=True)
for name, refp, newp in PAIRS:
    r = pooled(refp); nw = pooled(newp)
    rs = f"{r:.4f}" if r is not None else "  n/a "
    print(f"{name:28s}  {rs}    {nw:.4f}   {('%+.4f'%(nw-r)) if r is not None else 'n/a'}", flush=True)
