"""GPU 1,2,3,4 dispatcher: 3 new diverse architectures × 5-fold = 15 runs.

Adds attention/transformer models to the conv-heavy pool:
  1. ConcatCWT MaxViT  (maxvit_rmlp_tiny_rw_256 @256, cwt)
  2. muku V3 SwinV2    (swinv2_cr_tiny_ns_224 @224, superlet)
  3. ConcatCWT convnext_tiny on SUPERLET (@384)

5-fold on fold_df_fixed.csv → OOF saved to neuromm26_results/predictions/.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_diverse_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"
GPUS = [1, 2, 3, 4]
FOLD_CSV = "fold_df_fixed.csv"


def now():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


def concat_maxvit(fold):
    return {
        "id": f"concat_cwt_maxvit__fold{fold}", "eta_min": 70,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                "--spec-type", "cwt", "--backbone", "maxvit_rmlp_tiny_rw_256.sw_in1k",
                "--target-size", "256", "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "1e-3", "--stage2-lr", "1e-4",
                "--batch-size", "96", "--eval-batch-size", "128",
                "--mixup-alpha", "0.4", "--mixup-prob", "0.5"],
    }


def muku_v3_swinv2(fold):
    return {
        "id": f"muku_v3_swinv2__fold{fold}", "eta_min": 120,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_muku_v3_fold",
                "--backbone", "swinv2_cr_tiny_ns_224.sw_in1k", "--spec-type", "superlet",
                "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "8e-4", "--stage2-lr", "8e-5",
                "--batch-size", "48", "--eval-batch-size", "96"],
    }


def concat_convnext_superlet(fold):
    return {
        "id": f"concat_superlet_convnext__fold{fold}", "eta_min": 90,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                "--spec-type", "superlet", "--backbone", "convnext_tiny.fb_in22k_ft_in1k_384",
                "--target-size", "384", "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "9e-4", "--stage2-lr", "9e-5",
                "--batch-size", "80", "--eval-batch-size", "96",
                "--mixup-alpha", "0.4", "--mixup-prob", "0.5"],
    }


jobs = []
for f in range(5):
    jobs.append(muku_v3_swinv2(f))           # heaviest first
    jobs.append(concat_convnext_superlet(f))
    jobs.append(concat_maxvit(f))

log(f"Total jobs: {len(jobs)}  on GPUs {GPUS}")

gpu_proc = {g: None for g in GPUS}
gpu_job_id = {g: None for g in GPUS}
job_queue = list(jobs)


def launch(gpu_id, job):
    log_path = LOGDIR / f"gpu{gpu_id}__{job['id']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log(f"GPU {gpu_id} launching {job['id']} (eta {job['eta_min']}min)")
    proc = subprocess.Popen(job["cmd"], cwd=REPO, env=env,
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    gpu_proc[gpu_id] = proc
    gpu_job_id[gpu_id] = job["id"]


def check(gpu_id):
    p = gpu_proc[gpu_id]
    if p is None:
        return True
    if p.poll() is not None:
        log(f"GPU {gpu_id} freed ('{gpu_job_id[gpu_id]}' exit={p.returncode})")
        gpu_proc[gpu_id] = None
        gpu_job_id[gpu_id] = None
        return True
    return False


while job_queue or any(p is not None for p in gpu_proc.values()):
    for g in GPUS:
        if gpu_proc[g] is not None:
            check(g)
    for g in GPUS:
        if gpu_proc[g] is None and job_queue:
            job_queue.sort(key=lambda j: -j["eta_min"])
            launch(g, job_queue.pop(0))
    time.sleep(30)

log("ALL DONE")
(LOGDIR / "ALL_DONE").touch()
