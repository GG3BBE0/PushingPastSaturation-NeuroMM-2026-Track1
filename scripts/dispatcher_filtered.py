import os
"""Filtered-CWT models on GPU 1,2: ConcatCWT maxvit256 + CWT SpecCNN convnext_pico, x5 fold."""
import os, subprocess, time
from datetime import datetime
from pathlib import Path
REPO=os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY="python"
LOGDIR=Path(REPO)/f"neuromm26_results/logs/disp_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True,exist_ok=True); MASTER=LOGDIR/"MASTER.log"
GPUS=[1,2]; FOLD_CSV="fold_df_fixed.csv"; SPEC="cwt_filtered"
def now(): return datetime.now().strftime("%H:%M:%S")
def log(m):
    line=f"[{now()}] {m}"; print(line,flush=True); open(MASTER,"a").write(line+"\n")
def concat(bb,ts,fold,bs,lr1,lr2,eta):
    return {"id":f"concat_{SPEC}_{bb.split('.')[0]}__fold{fold}","eta_min":eta,
        "cmd":[PY,"-m","neuromm26_baseline.tools.train_spec_concat_fold","--spec-type",SPEC,
               "--backbone",bb,"--target-size",str(ts),"--fold-csv",FOLD_CSV,"--fold-idx",str(fold),"--seed","0",
               "--stage1-epochs","10","--stage2-epochs","30","--stage1-lr",str(lr1),"--stage2-lr",str(lr2),
               "--batch-size",str(bs),"--eval-batch-size","96","--mixup-alpha","0.4","--mixup-prob","0.5"]}
def speccnn(bb,fold,bs,lr1,lr2,eta):
    return {"id":f"spec_{SPEC}_{bb.split('.')[0]}__fold{fold}","eta_min":eta,
        "cmd":[PY,"-m","neuromm26_baseline.tools.train_spec_cnn_fold","--spec-type",SPEC,
               "--backbone",bb,"--fold-csv",FOLD_CSV,"--fold-idx",str(fold),"--seed","0",
               "--stage1-epochs","10","--stage2-epochs","30","--stage1-lr",str(lr1),"--stage2-lr",str(lr2),
               "--batch-size",str(bs),"--eval-batch-size","128"]}
jobs=[]
for f in range(5):
    jobs.append(concat("maxvit_rmlp_tiny_rw_256.sw_in1k",256,f,48,7e-4,7e-5,70))
    jobs.append(speccnn("convnext_pico.d1_in1k",f,64,7e-4,7e-5,25))
log(f"Total jobs: {len(jobs)} on GPUs {GPUS}")
gp={g:None for g in GPUS}; gj={g:None for g in GPUS}; q=list(jobs)
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
q.sort(key=lambda j:-j["eta_min"])
while q or any(p is not None for p in gp.values()):
    for g in GPUS:
        if gp[g] is not None: chk(g)
    for g in GPUS:
        if gp[g] is None and q: launch(g,q.pop(0))
    time.sleep(30)
log("ALL DONE"); (LOGDIR/"ALL_DONE").touch()
