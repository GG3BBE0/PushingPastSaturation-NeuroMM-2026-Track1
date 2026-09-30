import os
"""Wave 4: 3 more strong ConcatSpecCNN variants x5 fold on GPU 1-4.
  1. ConcatSuperlet maxvit384 (maxvit_tiny_tf_384, superlet)
  2. ConcatCWT coatnet (coatnet_0_rw_224, cwt @224)
  3. ConcatCWT convnext_small (convnext_small @384, cwt)
"""
import os, subprocess, time
from datetime import datetime
from pathlib import Path
REPO=os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY="python"
LOGDIR=Path(REPO)/f"neuromm26_results/logs/disp_diverse4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True,exist_ok=True); MASTER=LOGDIR/"MASTER.log"
GPUS=[1,2,3,4]; FOLD_CSV="fold_df_fixed.csv"
def now(): return datetime.now().strftime("%H:%M:%S")
def log(m):
    line=f"[{now()}] {m}"; print(line,flush=True); open(MASTER,"a").write(line+"\n")
def cc(bb,ts,spec,fold,bs,lr1,lr2,eta):
    return {"id":f"concat_{spec}_{bb.split('.')[0]}__fold{fold}","eta_min":eta,
        "cmd":[PY,"-m","neuromm26_baseline.tools.train_spec_concat_fold",
               "--spec-type",spec,"--backbone",bb,"--target-size",str(ts),
               "--fold-csv",FOLD_CSV,"--fold-idx",str(fold),"--seed","0",
               "--stage1-epochs","10","--stage2-epochs","30","--stage1-lr",str(lr1),"--stage2-lr",str(lr2),
               "--batch-size",str(bs),"--eval-batch-size","96","--mixup-alpha","0.4","--mixup-prob","0.5"]}
jobs=[]
for f in range(5):
    jobs.append(cc("maxvit_tiny_tf_384.in1k",384,"superlet",f,48,8e-4,8e-5,95))
    jobs.append(cc("convnext_small.fb_in22k_ft_in1k_384",384,"cwt",f,48,7e-4,7e-5,90))
    jobs.append(cc("coatnet_0_rw_224.sw_in1k",224,"cwt",f,80,9e-4,9e-5,55))
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
        if gp[g] is None and q: q.sort(key=lambda j:-j["eta_min"]); launch(g,q.pop(0))
    time.sleep(30)
log("ALL DONE"); (LOGDIR/"ALL_DONE").touch()
