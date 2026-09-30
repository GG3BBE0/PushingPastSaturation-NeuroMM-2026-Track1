"""B plan: ConcatCWT maxvit_tiny_tf_384 @384 on {stft, cwt_paul, cwt_filtered} x 5 fold = 15 jobs.
Edit GPUS, REPO, PY for your machine.
"""
import os, subprocess, time
from datetime import datetime
import os
from pathlib import Path
# === EDIT FOR REMOTE MACHINE ===
REPO=os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))   # remote repo path
PY="python"  # remote python
GPUS=[1,2]   # remote available GPUs
# ================================
LOGDIR=Path(REPO)/f"neuromm26_results/logs/disp_B_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True,exist_ok=True); MASTER=LOGDIR/"MASTER.log"
FOLD_CSV="fold_df_fixed.csv"; BB="maxvit_tiny_tf_384.in1k"; TS=384
def now(): return datetime.now().strftime("%H:%M:%S")
def log(m):
    line=f"[{now()}] {m}"; print(line,flush=True); open(MASTER,"a").write(line+"\n")
def cc(spec,fold,bs=128,lr1=1e-3,lr2=1e-4):
    return {"id":f"concat_{spec}_maxvit384__fold{fold}",
        "cmd":[PY,"-m","neuromm26_baseline.tools.train_spec_concat_fold","--spec-type",spec,"--backbone",BB,
               "--target-size",str(TS),"--fold-csv",FOLD_CSV,"--fold-idx",str(fold),"--seed","0",
               "--stage1-epochs","10","--stage2-epochs","30","--stage1-lr",str(lr1),"--stage2-lr",str(lr2),
               "--batch-size",str(bs),"--eval-batch-size","128","--mixup-alpha","0.4","--mixup-prob","0.5"]}

def done(spec, fold):
    """Skip jobs whose ckpt + OOF are both already on disk (idempotent resume)."""
    prefix = f"concat_{spec}_fold__maxvit_tiny_tf_384_in1k"
    ck = Path(REPO) / "neuromm26_results" / "checkpoints" / f"{prefix}__fold{fold}__seed0" / "best.pt"
    oof = Path(REPO) / "neuromm26_results" / "predictions" / f"{prefix}__fold{fold}__seed0_oof.npz"
    return ck.exists() and oof.exists()

q = []
skipped = []
for spec in ["stft","cwt_paul","cwt_filtered"]:
    for f in range(5):
        if done(spec, f):
            skipped.append(f"concat_{spec}_maxvit384__fold{f}")
        else:
            q.append(cc(spec, f))
log(f"Total jobs: {len(q)} on GPUs {GPUS} (maxvit_tiny_tf_384 @384, bs=128, lr 1e-3/1e-4, expandable_segments)")
if skipped:
    log(f"Skipping {len(skipped)} already-complete jobs: {skipped}")
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
