import os
"""Spec-view diversity (STFT + Paul wavelet) on GPU 1,2. Fast-first ordering."""
import os, subprocess, time
from datetime import datetime
from pathlib import Path
REPO=os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent)); PY="python"
LOGDIR=Path(REPO)/f"neuromm26_results/logs/disp_specviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True,exist_ok=True); MASTER=LOGDIR/"MASTER.log"; GPUS=[1,2]; FOLD_CSV="fold_df_fixed.csv"
def now(): return datetime.now().strftime("%H:%M:%S")
def log(m):
    line=f"[{now()}] {m}"; print(line,flush=True); open(MASTER,"a").write(line+"\n")
def speccnn(spec,bb,fold,bs,lr1,lr2):
    return {"id":f"spec_{spec}_{bb.split('.')[0]}__fold{fold}",
        "cmd":[PY,"-m","neuromm26_baseline.tools.train_spec_cnn_fold","--spec-type",spec,"--backbone",bb,
               "--fold-csv",FOLD_CSV,"--fold-idx",str(fold),"--seed","0","--stage1-epochs","10","--stage2-epochs","30",
               "--stage1-lr",str(lr1),"--stage2-lr",str(lr2),"--batch-size",str(bs),"--eval-batch-size","128"]}
def concat(spec,bb,ts,fold,bs,lr1,lr2):
    return {"id":f"concat_{spec}_{bb.split('.')[0]}__fold{fold}",
        "cmd":[PY,"-m","neuromm26_baseline.tools.train_spec_concat_fold","--spec-type",spec,"--backbone",bb,
               "--target-size",str(ts),"--fold-csv",FOLD_CSV,"--fold-idx",str(fold),"--seed","0",
               "--stage1-epochs","10","--stage2-epochs","30","--stage1-lr",str(lr1),"--stage2-lr",str(lr2),
               "--batch-size",str(bs),"--eval-batch-size","96","--mixup-alpha","0.4","--mixup-prob","0.5"]}
# fast-first: convnext SpecCNN (both specs) then maxvit Concat (both specs)
q=[]
for f in range(5): q.append(speccnn("stft","convnext_pico.d1_in1k",f,64,7e-4,7e-5))
for f in range(5): q.append(speccnn("cwt_paul","convnext_pico.d1_in1k",f,64,7e-4,7e-5))
for f in range(5): q.append(concat("stft","maxvit_rmlp_tiny_rw_256.sw_in1k",256,f,48,7e-4,7e-5))
for f in range(5): q.append(concat("cwt_paul","maxvit_rmlp_tiny_rw_256.sw_in1k",256,f,48,7e-4,7e-5))
log(f"Total jobs: {len(q)} on GPUs {GPUS} (fast-first, no resort)")
gp={g:None for g in GPUS}; gj={g:None for g in GPUS}
def launch(g,jb):
    lp=LOGDIR/f"gpu{g}__{jb['id']}.log"; env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    log(f"GPU {g} launching {jb['id']}")
    gp[g]=subprocess.Popen(jb["cmd"],cwd=REPO,env=env,stdout=open(lp,"w"),stderr=subprocess.STDOUT); gj[g]=jb["id"]
def chk(g):
    p=gp[g]
    if p is None: return True
    if p.poll() is not None:
        log(f"GPU {g} freed ('{gj[g]}' exit={p.returncode})"); gp[g]=None; gj[g]=None; return True
    return False
while q or any(p is not None for p in gp.values()):
    for g in GPUS:
        if gp[g] is not None: chk(g)
    for g in GPUS:
        if gp[g] is None and q: launch(g,q.pop(0))
    time.sleep(30)
log("ALL DONE"); (LOGDIR/"ALL_DONE").touch()
