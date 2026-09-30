"""Extract ONLY the N2+filt arches' ckpts (cwt256, paul256, filt256) from the NCHC 6-arch zip,
renaming prefix '_pseudo_fold__' -> '_pseudo_nchc_fold__' so they don't clobber local pseudo ckpts.
15 ckpts (3 arch x 5 fold). The @384 + stft ckpts are skipped (N2+filt doesn't need them)."""
import zipfile
import os
from pathlib import Path

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
ZIP = REPO / "concat_pseudo_6arch_5fold_228587.zip"
WANT = [  # @256 only (exclude the @384 cwt which shares 'concat_cwt_pseudo' substring)
    "concat_cwt_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k",
    "concat_cwt_paul_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k",
    "concat_cwt_filtered_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k",
]
z = zipfile.ZipFile(ZIP)
n = 0
for entry in z.namelist():
    if not entry.endswith(".pt"):
        continue
    if not any(f"checkpoints/{w}__fold" in entry for w in WANT):
        continue
    new = entry.replace("_pseudo_fold__", "_pseudo_nchc_fold__")
    dst = REPO / new
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(z.read(entry))
    n += 1
    print(f"  {new}", flush=True)
print(f"extracted {n} NCHC ckpts (expect 15)", flush=True)
