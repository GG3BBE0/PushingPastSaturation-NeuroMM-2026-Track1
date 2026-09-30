"""Heavy-aug ablation: ConcatSuperlet maxvit_tiny_tf_384 × 5 folds.

Baseline (existing) per-fold OOF for this arch:
  fold0=0.9539  fold1=0.9145  fold2=0.7786  fold3=0.9215  fold4=0.6397  avg=0.8487

Heavy aug additions vs baseline SpecAug(t=(0,12), f=(0,8), prob=0.5):
  - time mask max 12 -> 24, n=1 -> 2
  - freq mask max 8 -> 16, n=1 -> 2
  - channel drop max 0 -> 4 (new), n=2
  - prob 0.5 -> 0.7
  - mixup prob 0.5 -> 0.7
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
import os
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
FOLD_CSV = "fold_df_fixed.csv"
GPUS = [1, 2]
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_heavyaug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


def job(fold):
    prefix = "concat_superlet_heavyaug_fold__maxvit_tiny_tf_384_in1k"
    return {
        "id": f"{prefix}__fold{fold}",
        "eta_min": 90,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                "--spec-type", "superlet",
                "--backbone", "maxvit_tiny_tf_384.in1k",
                "--target-size", "384",
                "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
                "--batch-size", "24", "--eval-batch-size", "64",
                "--mixup-alpha", "0.4", "--mixup-prob", "0.7",
                "--time-mask-max", "24", "--n-time-masks", "2",
                "--freq-mask-max", "16", "--n-freq-masks", "2",
                "--channel-drop-max", "4", "--n-channel-masks", "2",
                "--spec-aug-prob", "0.7",
                "--exp-prefix", prefix],
    }


jobs = [job(f) for f in range(5)]


def launch(g, jb):
    lp = LOGDIR / f"gpu{g}__{jb['id']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(f"GPU {g} launching {jb['id']} (eta={jb['eta_min']}m)")
    gp[g] = subprocess.Popen(jb["cmd"], cwd=REPO, env=env,
                             stdout=lp.open("w"), stderr=subprocess.STDOUT)
    gj[g] = jb["id"]


def chk(g):
    p = gp[g]
    if p is None:
        return True
    if p.poll() is not None:
        log(f"GPU {g} freed ('{gj[g]}' exit={p.returncode})")
        gp[g] = None
        gj[g] = None
        return True
    return False


q = sorted(jobs, key=lambda j: -j["eta_min"])
gp = {g: None for g in GPUS}
gj = {g: None for g in GPUS}

log(f"Total jobs: {len(q)} on GPUs {GPUS}")
while q or any(p is not None for p in gp.values()):
    for g in GPUS:
        if gp[g] is not None:
            chk(g)
    for g in GPUS:
        if gp[g] is None and q:
            launch(g, q.pop(0))
    time.sleep(30)

log("ALL DONE")
(LOGDIR / "ALL_DONE").touch()
