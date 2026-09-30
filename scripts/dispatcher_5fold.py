"""3-GPU job dispatcher for remaining 5-fold sweep work.

- GPU 1 already has effv2s_b0 fold 4 (PID 2203775) running
- Wait for that to finish before dispatching to GPU 1
- GPU 0, 2 take jobs from queue immediately
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"


def now():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


def make_muku_cmd(backbone, fold, bs, lr1, lr2):
    name = backbone.replace("/", "-").replace(".", "_")
    return {
        "id": f"muku__{name}__fold{fold}",
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
                "--backbone", backbone, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", str(lr1), "--stage2-lr", str(lr2),
                "--batch-size", str(bs), "--eval-batch-size", "128"],
        "eta_min": 70 if "resnet18" in backbone else 130,
    }


def make_cwt_cmd(backbone, fold, bs, lr1, lr2):
    name = backbone.replace("/", "-").replace(".", "_")
    return {
        "id": f"cwt__{name}__fold{fold}",
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_cnn_fold",
                "--spec-type", "cwt", "--backbone", backbone, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", str(lr1), "--stage2-lr", str(lr2),
                "--batch-size", str(bs), "--eval-batch-size", "128"],
        "eta_min": 15 if "resnet18" in backbone else 25,
    }


def make_v2_cmd(fold):
    return {
        "id": f"v2__fold{fold}",
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                "--spec-type", "cwt", "--backbone", "convnext_tiny.fb_in22k_ft_in1k_384",
                "--target-size", "384", "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
                "--batch-size", "48", "--eval-batch-size", "64",
                "--mixup-alpha", "0.4", "--mixup-prob", "0.5"],
        "eta_min": 50,
    }


# Build remaining job list (longest jobs first to balance)
jobs = []

# muku convnext_pico × 5 folds (~130 min each)
for f in range(5):
    jobs.append(make_muku_cmd("convnext_pico.d1_in1k", f, 64, 7e-4, 7e-5))

# V2 × 5 folds (~50 min each, slowest after big muku)
for f in range(5):
    jobs.append(make_v2_cmd(f))

# CWT-Morlet × 3 backbones × 5 folds (~15-25 min each, smallest)
for f in range(5):
    jobs.append(make_cwt_cmd("tf_efficientnet_b0.ns_jft_in1k", f, 64, 7e-4, 7e-5))
for f in range(5):
    jobs.append(make_cwt_cmd("convnext_pico.d1_in1k", f, 64, 7e-4, 7e-5))
for f in range(5):
    jobs.append(make_cwt_cmd("resnet18", f, 96, 8e-4, 8e-5))

log(f"Total jobs: {len(jobs)}")
log(f"Initial GPU state: GPU 1 busy with effv2s_b0 fold 4 (PID 2203775)")

# GPU state: True = busy, False = free
gpu_busy = {0: False, 1: True, 2: False}  # GPU 1 starts busy
gpu_proc = {0: None, 1: 2203775, 2: None}  # GPU 1 has external PID
gpu_job_id = {0: None, 1: "muku__tf_efficientnet_b0_ns_jft_in1k__fold4", 2: None}

job_queue = list(jobs)


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
    log(f"GPU {gpu_id} launching {job['id']} (eta {job['eta_min']} min)")
    proc = subprocess.Popen(job["cmd"], cwd=REPO, env=env,
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    gpu_busy[gpu_id] = True
    gpu_proc[gpu_id] = proc
    gpu_job_id[gpu_id] = job["id"]


def check_finished(gpu_id):
    """Returns True if the GPU finished and was freed."""
    p = gpu_proc[gpu_id]
    if p is None:
        return False
    if isinstance(p, int):
        # external PID
        if not pid_alive(p):
            log(f"GPU {gpu_id} freed (external PID {p} finished: {gpu_job_id[gpu_id]})")
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


# Main scheduling loop
while job_queue or any(gpu_busy.values()):
    # Check who's free
    for g in [0, 1, 2]:
        if gpu_busy[g]:
            check_finished(g)

    # Dispatch jobs to free GPUs (longest job first)
    for g in [0, 1, 2]:
        if not gpu_busy[g] and job_queue:
            job_queue.sort(key=lambda j: -j["eta_min"])
            job = job_queue.pop(0)
            launch_job(g, job)

    time.sleep(20)

log("ALL DONE")
# Write marker
(LOGDIR / "ALL_DONE").touch()
