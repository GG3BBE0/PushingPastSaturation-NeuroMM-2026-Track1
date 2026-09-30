"""Build the NCHC-specialist (specnchc full) submission: canonical pool with cwt/paul/filt/superlet/muku
REPLACED by their NCHC specialist versions. Gate winner: honest +0.0386 vs spec-local +0.0254 (=public
0.9818) -> +0.0131 robust (5/5 seeds). Output: submission_test1_specnchc.zip.
"""
import csv, io, shutil, sys, zipfile
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score as aps
sys.path.insert(0, str(Path(__file__).parent))
from regularized_nm_submission import (POOL_LOOKUP, load_oof, derive_nm, derive_bagged_nm, apply_cap,
                                       TRIM_CUTOFF_OOF, TRIM_CUTOFF_NMW, REPO)
NPZ = REPO / "neuromm26_results/candidate_arch_logits.npz"
SWAP_ALL = {
    "ConcatCWT maxvit256": ("ConcatCWT specnchc 256", "concat_cwt_specnchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    "ConcatPaul maxvit256": ("ConcatPaul specnchc 256", "concat_cwt_paul_specnchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    "ConcatFilt maxvit256": ("ConcatFilt specnchc 256", "concat_cwt_filtered_specnchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    "ConcatSuperlet maxvit384": ("ConcatSuperlet specnchc 384", "concat_superlet_specnchc_fold__maxvit_tiny_tf_384_in1k"),
    "muku raw convnext_pico": ("muku specnchc convnext", "muku_specnchc_fold__convnext_pico_d1_in1k"),
}
PSE = ("pseudo", "relabel", "relbase", "rank256", "NeuroMAE", "GNN orthopara", "Spec3D", "r3nchc",
       " spec ", "_big_", "_soft_", "_ns_", "specnchc")
d = np.load(NPZ, allow_pickle=True)
npz_names = [str(n) for n in d["names"]]; Xc_all = d["logits"]; ids = [str(x) for x in d["ids"]]
col = {n: i for i, n in enumerate(npz_names)}
def ready(ps, pfx):
    return all((REPO/"neuromm26_results/predictions"/f"{pfx}__fold{f}__seed0_oof.npz").exists() for f in range(5)) and ps in col
SWAP = {o: ps for o,(ps,pfx) in SWAP_ALL.items() if ready(ps,pfx)}
print(f"{len(SWAP)}/5 specnchc swaps ready: {list(SWAP.values())}", flush=True)
assert len(SWAP) == 5, "not all specnchc arches ready (need muku candidate logit merged)"
canon = [n for n in npz_names if not any(k in n for k in PSE)]
pool = [SWAP.get(n, n) for n in canon]
ref_s=ref_y=None; Xo=[]; Xc=[]; ap=[]
for nm in pool:
    pfx,root=POOL_LOOKUP[nm]; s,l,y=load_oof(pfx,root); ss=s.astype(str); l=np.clip(l,-30,30)
    if ref_s is None: ref_s,ref_y=ss,y.astype(int); Xo.append(l)
    else: im={x:i for i,x in enumerate(ss)}; Xo.append(l[np.array([im[x] for x in ref_s])])
    ap.append(aps(y,1/(1+np.exp(-l)))); Xc.append(Xc_all[:,col[nm]])
Xo=np.stack(Xo,1); Xc=np.stack(Xc,1); ap=np.array(ap)
wb,_=derive_nm(Xo,ref_y,n_restarts=4)
keep=~((ap<TRIM_CUTOFF_OOF)&(wb<TRIM_CUTOFF_NMW)); poolk=[p for p,k in zip(pool,keep) if k]
Xot,Xct=Xo[:,keep],Xc[:,keep]
wD=apply_cap(derive_bagged_nm(Xot,ref_y,n_iter=40),0.20)
oof=aps(ref_y,Xot@wD); snw=sum(w for p,w in zip(poolk,wD) if "specnchc" in p)
print(f"trim {keep.sum()}; specnchc-full ensemble OOF = {oof:.4f} (spec-local 0.9167, nchc 0.9048); specnchc NM wt {snw:.3f}", flush=True)
prob=1/(1+np.exp(-(Xct@wD)))
out=REPO/"submission_test1_specnchc.csv"
with out.open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["sample_id","prediction"])
    for sid,p in zip(ids,prob): w.writerow([sid,f"{p:.6f}"])
_ref=next((REPO/"submissions"/z for z in ("submission_test1_spec.zip","submission_test1_regnm_v3.zip","submission_test1_nchc_n2filt.zip") if (REPO/"submissions"/z).exists()), None)
if _ref is not None:  # optional Spearman sanity vs an available reference (none shipped by default)
    z=zipfile.ZipFile(_ref)
    b={r["sample_id"]:float(r["prediction"]) for r in csv.DictReader(io.TextIOWrapper(z.open("submission.csv")))}
    rho=spearmanr(prob,[b[s] for s in ids]).statistic
    print(f"Spearman rho(specnchc, {_ref.name}) = {rho:.4f}", flush=True)
sub=REPO/"submissions"; st=sub/"submission.csv"; shutil.copy(out,st)
zn=sub/"submission_test1_specnchc.zip"
with zipfile.ZipFile(zn,"w",zipfile.ZIP_DEFLATED) as zf: zf.write(st,arcname="submission.csv")
st.unlink()
print(f"WROTE {zn} (range {prob.min():.4f}-{prob.max():.4f} mean {prob.mean():.4f})", flush=True)
