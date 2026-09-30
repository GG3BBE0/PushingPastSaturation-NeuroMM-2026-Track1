import os
"""Wave 3: re-run the failed concat_superlet convnext (5 folds) on GPU 1-4."""
import os, subprocess, time
from datetime import datetime
from pathlib import Path
REPO=os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY="python"
LOGDIR=Path(REPO)/f"neuromm26_results/logs/disp_diverse3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER=LOGDIR/"MASTER.log"
GPUS=[1,2,3,4]; FOLD_CSV="fold_df_fixed.csv"
def now(): return datetime.now().strftime("%H:%M:%S")
def log(m):
    line=f"[{now()}] {m}"; print(line,flush=True)
    open(MASTER,"a").write(line+"\n")
def job(fold):
    return {"id":f"concat_superlet_convnext__fold{fold}","eta_min":90,
        "cmd":[PY,"-m","neuromm26_baseline.tools.train_spec_concat_fold",
               "--spec-type","superlet","--backbone","convnext_tiny.fb_in22k_ft_in1k_384",
               "--target-size","384","--fold-csv",FOLD_CSV,"--fold-idx",str(fold),"--seed","0",
               "--stage1-epochs","10","--stage2-epochs","30","--stage1-lr","9e-4","--stage2-lr","9e-5",
               "--batch-size","80","--eval-batch-size","96","--mixup-alpha","0.4","--mixup-prob","0.5"]}
jobs=[job(f) for f in range(5)]
log(f"Total jobs: {len(jobs)} on GPUs {GPUS}")
gp={g:None for g in GPUS}; gj={g:None for g in GPUS}; q=list(jobs)
def launch(g,jb):
    lp=LOGDIR/f"gpu{g}__{jb['id']}.log"; env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(g)
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
