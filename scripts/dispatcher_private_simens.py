"""Faithful private sim with the ENSEMBLE teacher (vs the pessimistic single-arch teacher). Re-run the
3 folds that were NEGATIVE in the single-arch sim (0, 1, 4 — esp. fold4 -0.0438) using the real
ensemble teacher's pseudo-labels. If the harm shrinks, the real candidate-specialist is safer than the
pessimistic sim showed. GPU 1,2. cwt_simensfold{k} -> compare vs canonical baseline.
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
import os
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
GPUS = [1, 2]
RES = Path(REPO) / "neuromm26_results"
FOLDS = [4, 1, 0]  # the 3 negatives, fold4 (worst) first
TEACHER = "ensemble_teacher_oof.txt"

logdir = RES / f"logs/disp_private_simens_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logdir.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    (logdir / "MASTER.log").open("a").write(f"{m}\n")


for k in FOLDS:  # build each ens-teacher sim dataset
    r = subprocess.run([PY, "scripts/build_private_sim.py", str(k), TEACHER], cwd=REPO, capture_output=True, text=True)
    log(f"build ens fold{k}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-200:]}")

jobs = [{"k": k} for k in FOLDS
        if not (RES / "predictions" / f"cwt_simensfold{k}__fold{k}__seed0_oof.npz").exists()]
gp = {g: None for g in GPUS}; gj = {g: None for g in GPUS}


def launch(g, jb):
    k = jb["k"]
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
           "--spec-type", "cwt", "--backbone", "maxvit_rmlp_tiny_rw_256.sw_in1k",
           "--target-size", "256", "--fold-idx", str(k), "--seed", "0",
           "--fold-csv", f"fold_df_simens_{k}.csv", "--loss", "focal", "--focal-gamma", "2.0",
           "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
           "--batch-size", "64", "--eval-batch-size", "96", "--exp-prefix", f"cwt_simensfold{k}"]
    log(f"GPU {g} launching ens fold{k}")
    gp[g] = subprocess.Popen(cmd, cwd=REPO, env=env,
                             stdout=open(logdir / f"gpu{g}__ensfold{k}.log", "w"), stderr=subprocess.STDOUT)
    gj[g] = k


def chk(g):
    p = gp[g]
    if p is not None and p.poll() is not None:
        log(f"GPU {g} freed (ens fold{gj[g]} exit={p.returncode})"); gp[g] = None; gj[g] = None


log(f"faithful ens-teacher sim: {len(jobs)} jobs on GPU {GPUS}")
q = list(jobs)
while q or any(p is not None for p in gp.values()):
    for g in GPUS:
        chk(g)
    for g in GPUS:
        if gp[g] is None and q:
            launch(g, q.pop(0))
    time.sleep(20)
log("ALL DONE")
(logdir / "ALL_DONE").touch()
