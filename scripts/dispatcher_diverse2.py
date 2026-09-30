"""GPU 1,2,3,4 dispatcher WAVE 2: 3 more diverse architectures × 5-fold = 15 runs.

  1. ConcatCWT MaxViT @384 (maxvit_tiny_tf_384, cwt)   — suguuuuu's resolution
  2. CWT-Morlet SwinV2     (swinv2_cr_tiny_ns_224, cwt) — attention on CWT (SpecCNN @224)
  3. CWT-Morlet caformer   (caformer_s18, cwt)          — MetaFormer on CWT

Launched after wave-1 completes. OOF → neuromm26_results/predictions/.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_diverse2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def concat_maxvit384(fold):
    return {
        "id": f"concat_cwt_maxvit384__fold{fold}", "eta_min": 95,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                "--spec-type", "cwt", "--backbone", "maxvit_tiny_tf_384.in1k",
                "--target-size", "384", "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "8e-4", "--stage2-lr", "8e-5",
                "--batch-size", "48", "--eval-batch-size", "64",
                "--mixup-alpha", "0.4", "--mixup-prob", "0.5"],
    }


def cwt_swinv2(fold):
    return {
        "id": f"spec_cwt_swinv2__fold{fold}", "eta_min": 30,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_cnn_fold",
                "--spec-type", "cwt", "--backbone", "swinv2_cr_tiny_ns_224.sw_in1k",
                "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "8e-4", "--stage2-lr", "8e-5",
                "--batch-size", "64", "--eval-batch-size", "96"],
    }


def cwt_caformer(fold):
    return {
        "id": f"spec_cwt_caformer__fold{fold}", "eta_min": 30,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_cnn_fold",
                "--spec-type", "cwt", "--backbone", "caformer_s18.sail_in22k_ft_in1k",
                "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "8e-4", "--stage2-lr", "8e-5",
                "--batch-size", "64", "--eval-batch-size", "96"],
    }


jobs = []
for f in range(5):
    jobs.append(concat_maxvit384(f))   # slowest first
    jobs.append(cwt_swinv2(f))
    jobs.append(cwt_caformer(f))

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
