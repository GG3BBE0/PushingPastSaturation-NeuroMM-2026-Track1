"""GPU 1, 2 dispatcher for muku V3 5-fold sweep.

15 jobs = 3 backbones (test-set strong: effv2s_b0, convnext_pico, mobilenetv3_large) × 5 folds.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_muku_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"
GPUS = [1, 2]
FOLD_CSV = "fold_df_fixed.csv"
SPEC_TYPE = "superlet"


def now():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


def make_job(backbone, fold, bs, lr1, lr2):
    name = backbone.replace("/", "-").replace(".", "_")
    return {
        "id": f"v3__{name}__fold{fold}",
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_muku_v3_fold",
                "--backbone", backbone, "--spec-type", SPEC_TYPE,
                "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", str(lr1), "--stage2-lr", str(lr2),
                "--batch-size", str(bs), "--eval-batch-size", str(min(128, bs * 2))],
        "eta_min": 90 if "convnext_pico" in backbone else 70,
    }


backbones = [
    ("convnext_pico.d1_in1k", 48, 7e-4, 7e-5),
    ("tf_efficientnet_b0.ns_jft_in1k", 48, 7e-4, 7e-5),
    ("mobilenetv3_large_100.ra_in1k", 64, 8e-4, 8e-5),
]

jobs = []
for bb, bs, lr1, lr2 in backbones:
    for f in range(5):
        jobs.append(make_job(bb, f, bs, lr1, lr2))

log(f"Total jobs: {len(jobs)}  on GPUs {GPUS}")
gpu_proc = {g: None for g in GPUS}
gpu_job_id = {g: None for g in GPUS}
job_queue = list(jobs)


def launch_job(gpu_id, job):
    log_path = LOGDIR / f"gpu{gpu_id}__{job['id']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log(f"GPU {gpu_id} launching {job['id']} (eta {job['eta_min']} min)")
    proc = subprocess.Popen(job["cmd"], cwd=REPO, env=env,
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    gpu_proc[gpu_id] = proc
    gpu_job_id[gpu_id] = job["id"]


def check_finished(gpu_id):
    p = gpu_proc[gpu_id]
    if p is None:
        return True
    ret = p.poll()
    if ret is not None:
        log(f"GPU {gpu_id} freed ('{gpu_job_id[gpu_id]}' exit={ret})")
        gpu_proc[gpu_id] = None
        gpu_job_id[gpu_id] = None
        return True
    return False


while job_queue or any(p is not None for p in gpu_proc.values()):
    for g in GPUS:
        if gpu_proc[g] is not None:
            check_finished(g)
    for g in GPUS:
        if gpu_proc[g] is None and job_queue:
            job_queue.sort(key=lambda j: -j["eta_min"])
            job = job_queue.pop(0)
            launch_job(g, job)
    time.sleep(30)

log("ALL DONE")
(LOGDIR / "ALL_DONE").touch()
