"""Extract NCHC 4-job return. Predictions + logs now (small, for gating);
checkpoints deferred (large, extract only for archs that pass the gate)."""
import zipfile, pathlib

ZIP = "nchc_4jobs_20260602.zip"
REPO = pathlib.Path(".")
z = zipfile.ZipFile(ZIP)
names = z.namelist()

# 1) all OOF predictions + logs + README (cheap)
small = [n for n in names if (
    n.startswith("neuromm26_results/predictions/") or
    n.startswith("_logs_/") or n == "README.txt")]
for n in small:
    z.extract(n, REPO)
print(f"extracted {len(small)} small files (predictions + logs + README)")

ck = [n for n in names if n.startswith("neuromm26_results/checkpoints/")]
print(f"DEFERRED {len(ck)} checkpoint files (extract post-gate)")
# list new OOF prefixes
pref = sorted({pathlib.Path(n).name.split('__fold')[0]
               for n in small if n.endswith('_oof.npz')})
print("new OOF arch prefixes:")
for p in pref: print("  ", p)
