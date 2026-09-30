"""Phase 3 dispatcher: only GPU 1, 2.

Manages remaining 5 CWT resnet18 fold jobs. Waits for orphaned GPU 1/2
processes to finish first, then dispatches new work.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_p3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


# Orphaned PIDs (current running jobs)
# GPU 1: PID 2521232 (cwt convnext_pico fold 4)
# GPU 2: PID 2519913 (cwt convnext_pico fold 3)
gpu_busy = {1: True, 2: True}
gpu_proc = {1: 2521232, 2: 2519913}
gpu_job_id = {1: "cwt__convnext_pico__fold4 (orphan)", 2: "cwt__convnext_pico__fold3 (orphan)"}


def make_cwt_cmd(fold):
    return {
        "id": f"cwt__resnet18__fold{fold}",
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_cnn_fold",
                "--spec-type", "cwt", "--backbone", "resnet18",
                "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "8e-4", "--stage2-lr", "8e-5",
                "--batch-size", "96", "--eval-batch-size", "128"],
    }


# Remaining queue: 5 cwt resnet18 folds
job_queue = [make_cwt_cmd(f) for f in range(5)]
log(f"Total remaining jobs: {len(job_queue)}")
log(f"Waiting for orphaned PIDs: GPU 1 = {gpu_proc[1]}, GPU 2 = {gpu_proc[2]}")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def launch_job(gpu_id, job):
    log_path = LOGDIR / f"gpu{gpu_id}__{job['id']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log(f"GPU {gpu_id} launching {job['id']}")
    proc = subprocess.Popen(job["cmd"], cwd=REPO, env=env,
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    gpu_busy[gpu_id] = True
    gpu_proc[gpu_id] = proc
    gpu_job_id[gpu_id] = job["id"]


def check_finished(gpu_id):
    p = gpu_proc[gpu_id]
    if p is None:
        return False
    if isinstance(p, int):
        if not pid_alive(p):
            log(f"GPU {gpu_id} freed (external PID {p} done: {gpu_job_id[gpu_id]})")
            gpu_busy[gpu_id] = False
            gpu_proc[gpu_id] = None
            gpu_job_id[gpu_id] = None
            return True
        return False
    else:
        ret = p.poll()
        if ret is not None:
            log(f"GPU {gpu_id} freed (job '{gpu_job_id[gpu_id]}' exit={ret})")
            gpu_busy[gpu_id] = False
            gpu_proc[gpu_id] = None
            gpu_job_id[gpu_id] = None
            return True
        return False


while job_queue or any(gpu_busy.values()):
    for g in [1, 2]:
        if gpu_busy[g]:
            check_finished(g)
    for g in [1, 2]:
        if not gpu_busy[g] and job_queue:
            job = job_queue.pop(0)
            launch_job(g, job)
    time.sleep(20)

log("ALL DONE")
(LOGDIR / "ALL_DONE").touch()
