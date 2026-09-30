"""Run the private-safety simulation for all 5 folds: build each fold's sim dataset, then train a cwt
specialist (trained WITH that fold's pseudo-copies) on GPU 3,4,5. Compare each fold's specialist
real-OOF AUPRC vs the canonical baseline -> does pseudo-training the held-out samples help/hurt their
TRUE score (the private-LB question). GPU 3,4,5 (user-authorized this run).
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
import os
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
GPUS = [3, 4, 5]
RES = Path(REPO) / "neuromm26_results"

logdir = RES / f"logs/disp_private_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logdir.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    (logdir / "MASTER.log").open("a").write(f"{m}\n")


# 1. build all 5 sim datasets (fast: file ops + symlinks)
for k in range(5):
    r = subprocess.run([PY, "scripts/build_private_sim.py", str(k)], cwd=REPO, capture_output=True, text=True)
    log(f"build fold{k}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-200:]}")

# 2. train cwt specialist per fold (eval on real fold-k)
def done(k):
    return (RES / "predictions" / f"cwt_simfold{k}__fold{k}__seed0_oof.npz").exists()


jobs = [{"id": f"cwt_simfold{k}", "k": k} for k in range(5) if not done(k)]
gp = {g: None for g in GPUS}; gj = {g: None for g in GPUS}


def launch(g, jb):
    k = jb["k"]
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
           "--spec-type", "cwt", "--backbone", "maxvit_rmlp_tiny_rw_256.sw_in1k",
           "--target-size", "256", "--fold-idx", str(k), "--seed", "0",
           "--fold-csv", f"fold_df_sim_{k}.csv", "--loss", "focal", "--focal-gamma", "2.0",
           "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
           "--batch-size", "64", "--eval-batch-size", "96", "--exp-prefix", f"cwt_simfold{k}"]
    log(f"GPU {g} launching {jb['id']} (train WITH fold{k} pseudo, eval on real fold{k})")
    gp[g] = subprocess.Popen(cmd, cwd=REPO, env=env,
                             stdout=open(logdir / f"gpu{g}__{jb['id']}.log", "w"), stderr=subprocess.STDOUT)
    gj[g] = jb["id"]


def chk(g):
    p = gp[g]
    if p is not None and p.poll() is not None:
        log(f"GPU {g} freed ('{gj[g]}' exit={p.returncode})"); gp[g] = None; gj[g] = None


log(f"private-safety sim: {len(jobs)} jobs on GPU {GPUS}")
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
