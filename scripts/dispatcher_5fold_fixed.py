"""5-fold REAL CV dispatcher using fold_df_fixed.csv (patient-disjoint).

Uses only GPU 1, 2. Manages 35 jobs.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
FOLD_CSV = "fold_df_fixed.csv"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


def make_muku_cmd(backbone, fold, bs, lr1, lr2):
    name = backbone.replace("/", "-").replace(".", "_")
    return {
        "id": f"muku__{name}__fold{fold}",
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
                "--backbone", backbone, "--fold-csv", FOLD_CSV,
                "--fold-idx", str(fold), "--seed", "0",
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
                "--spec-type", "cwt", "--backbone", backbone,
                "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", str(lr1), "--stage2-lr", str(lr2),
                "--batch-size", str(bs), "--eval-batch-size", "128"],
        "eta_min": 65 if "resnet18" in backbone else 100,
    }


def make_v2_cmd(fold):
    return {
        "id": f"v2__fold{fold}",
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                "--spec-type", "cwt", "--backbone", "convnext_tiny.fb_in22k_ft_in1k_384",
                "--target-size", "384", "--fold-csv", FOLD_CSV,
                "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
                "--batch-size", "48", "--eval-batch-size", "64",
                "--mixup-alpha", "0.4", "--mixup-prob", "0.5"],
        "eta_min": 210,
    }


# Build full 35-run queue (longest first to balance load)
jobs = []
# V2 (slowest, eta 210)
for f in range(5):
    jobs.append(make_v2_cmd(f))
# muku effv2s_b0 + convnext_pico (medium-slow, eta 130)
for f in range(5):
    jobs.append(make_muku_cmd("tf_efficientnet_b0.ns_jft_in1k", f, 64, 7e-4, 7e-5))
for f in range(5):
    jobs.append(make_muku_cmd("convnext_pico.d1_in1k", f, 64, 7e-4, 7e-5))
# CWT effv2s_b0 + convnext_pico (eta 100)
for f in range(5):
    jobs.append(make_cwt_cmd("tf_efficientnet_b0.ns_jft_in1k", f, 64, 7e-4, 7e-5))
for f in range(5):
    jobs.append(make_cwt_cmd("convnext_pico.d1_in1k", f, 64, 7e-4, 7e-5))
# muku resnet18 (eta 70)
for f in range(5):
    jobs.append(make_muku_cmd("resnet18", f, 128, 1e-3, 1e-4))
# CWT resnet18 (fastest, eta 65)
for f in range(5):
    jobs.append(make_cwt_cmd("resnet18", f, 96, 8e-4, 8e-5))

log(f"Total jobs: {len(jobs)}")
log(f"Using GPU 1, 2 only")
log(f"Fold CSV: {FOLD_CSV}")

gpu_busy = {1: False, 2: False}
gpu_proc = {1: None, 2: None}
gpu_job_id = {1: None, 2: None}
job_queue = list(jobs)


def launch_job(gpu_id, job):
    log_path = LOGDIR / f"gpu{gpu_id}__{job['id']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log(f"GPU {gpu_id} launching {job['id']} (eta {job['eta_min']} min)  [{len(job_queue)} jobs in queue]")
    proc = subprocess.Popen(job["cmd"], cwd=REPO, env=env,
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    gpu_busy[gpu_id] = True
    gpu_proc[gpu_id] = proc
    gpu_job_id[gpu_id] = job["id"]


def check_finished(gpu_id):
    p = gpu_proc[gpu_id]
    if p is None:
        return False
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
            job_queue.sort(key=lambda j: -j["eta_min"])
            job = job_queue.pop(0)
            launch_job(g, job)
    time.sleep(20)

log("ALL DONE")
(LOGDIR / "ALL_DONE").touch()
