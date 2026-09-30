"""Dispatcher for legacy/tcnet_eeg + mobilenet_v3_large_eeg × 5 fold on GPU 1, 2.

Waits for V3 sweep to finish (looks for ALL_DONE marker in latest
disp_muku_v3_* directory), then dispatches 5 + 5 = 10 legacy jobs.
"""

from __future__ import annotations

import glob
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"
GPUS = [1, 2]
FOLD_CSV = "fold_df_fixed.csv"
MODEL_NAMES = ["tcnet_eeg", "mobilenet_v3_large_eeg"]


def now():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


def wait_for_v3():
    """Wait for V3 sweep ALL_DONE marker."""
    v3_dirs = sorted(glob.glob(f"{REPO}/neuromm26_results/logs/disp_muku_v3_*"))
    if not v3_dirs:
        log("No V3 dispatcher dir found — running standalone")
        return
    v3_latest = v3_dirs[-1]
    marker = Path(v3_latest) / "ALL_DONE"
    if marker.exists():
        log(f"V3 already complete: {marker}")
        return
    log(f"Waiting for V3 to finish: {marker}")
    while not marker.exists():
        time.sleep(60)
    log("V3 done — starting tcnet sweep")


def make_job(model_name, fold):
    return {
        "id": f"legacy_{model_name}__fold{fold}",
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_legacy_fold",
                "--model-name", model_name,
                "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "8e-4", "--stage2-lr", "8e-5",
                "--batch-size", "96", "--eval-batch-size", "128"],
        "eta_min": 25,
    }


wait_for_v3()

jobs = []
for model_name in MODEL_NAMES:
    for f in range(5):
        jobs.append(make_job(model_name, f))
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
            job = job_queue.pop(0)
            launch_job(g, job)
    time.sleep(30)

log("ALL DONE")
(LOGDIR / "ALL_DONE").touch()
