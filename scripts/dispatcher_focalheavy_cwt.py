import os
"""focal + heavy-aug rollout #2: ConcatCWT maxvit256, ALL 5 folds, GPU 1+2 both immediate.
Validated recipe (Paul focal+heavy = +0.0227 single-arch OOF, superadditive). Next carrier = CWT.
Compare vs concat_cwt_heavyaug256 (heavy-aug-alone +0.0036) to isolate focal's marginal effect.
"""
import os, subprocess, time
from datetime import datetime
from pathlib import Path
REPO=os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY="python"
GPUS=[1,2]
LOGDIR=Path(REPO)/f"neuromm26_results/logs/disp_focalheavy_cwt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True,exist_ok=True); MASTER=LOGDIR/"MASTER.log"
PREFIX="concat_cwt_focalheavy_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"
def log(m):
    line=f"[{datetime.now().strftime('%H:%M:%S')}] {m}"; print(line,flush=True); open(MASTER,"a").write(line+"\n")
def cmd(fold):
    return [PY,"-m","neuromm26_baseline.tools.train_spec_concat_fold","--spec-type","cwt",
            "--backbone","maxvit_rmlp_tiny_rw_256.sw_in1k","--target-size","256",
            "--fold-csv","fold_df_fixed.csv","--fold-idx",str(fold),"--seed","0",
            "--stage1-epochs","10","--stage2-epochs","30","--stage1-lr","7e-4","--stage2-lr","7e-5",
            "--batch-size","48","--eval-batch-size","96","--loss","focal","--focal-gamma","2.0",
            "--mixup-alpha","0.4","--mixup-prob","0.7","--time-mask-max","24","--n-time-masks","2",
            "--freq-mask-max","16","--n-freq-masks","2","--channel-drop-max","4","--n-channel-masks","2",
            "--spec-aug-prob","0.7","--exp-prefix",PREFIX]
def done(f):
    return (Path(REPO)/"neuromm26_results/checkpoints"/f"{PREFIX}__fold{f}__seed0"/"best.pt").exists() and \
           (Path(REPO)/"neuromm26_results/predictions"/f"{PREFIX}__fold{f}__seed0_oof.npz").exists()
q=[f for f in range(5) if not done(f)]
log(f"focal+heavy CWT maxvit256, folds {q} on GPU {GPUS} (both immediate)")
proc={g:None for g in GPUS}; pjob={g:None for g in GPUS}
def launch(g,fold):
    lp=LOGDIR/f"gpu{g}__{PREFIX}__fold{fold}.log"; env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(g); env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    log(f"GPU {g} launching fold{fold}")
    proc[g]=subprocess.Popen(cmd(fold),cwd=REPO,env=env,stdout=lp.open("w"),stderr=subprocess.STDOUT); pjob[g]=fold
def chk(g):
    p=proc[g]
    if p and p.poll() is not None:
        log(f"GPU {g} freed (fold{pjob[g]} exit={p.returncode})"); proc[g]=None; pjob[g]=None
while q or any(p is not None for p in proc.values()):
    for g in GPUS:
        if proc[g] is not None: chk(g)
    for g in GPUS:
        if proc[g] is None and q: launch(g,q.pop(0))
    time.sleep(30)
log("ALL DONE"); (LOGDIR/"ALL_DONE").touch()
