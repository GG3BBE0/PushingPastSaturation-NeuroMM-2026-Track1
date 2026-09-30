"""GPU 1,2 dispatcher: full-data (train+val ~90%/10% monitor) retrain of the
28-ckpt core architectures, multi-seed.

7 architectures × 3 seeds = 21 runs. Each seed uses fold_full_seed{s}.csv
(fold 0 = 10% patient-disjoint monitor, fold 1 = 90% train), --fold-idx 0.
Output → neuromm26_fulldata_result/.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_fulldata_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"
GPUS = [1, 2]
OUT_ROOT = "neuromm26_fulldata_result"
SEEDS = [0, 1, 2]


def now():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


def muku_cmd(bb, seed, bs, lr1, lr2, eta):
    return {
        "id": f"muku__{bb.replace('/', '-').replace('.', '_')}__seed{seed}",
        "eta_min": eta,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
                "--backbone", bb, "--fold-csv", f"fold_full_seed{seed}.csv",
                "--fold-idx", "0", "--seed", str(seed),
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", str(lr1), "--stage2-lr", str(lr2),
                "--batch-size", str(bs), "--eval-batch-size", "128",
                "--output-root", OUT_ROOT],
    }


def cwt_cmd(bb, seed, bs, lr1, lr2, eta):
    return {
        "id": f"spec_cwt__{bb.replace('/', '-').replace('.', '_')}__seed{seed}",
        "eta_min": eta,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_cnn_fold",
                "--spec-type", "cwt", "--backbone", bb, "--fold-csv", f"fold_full_seed{seed}.csv",
                "--fold-idx", "0", "--seed", str(seed),
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", str(lr1), "--stage2-lr", str(lr2),
                "--batch-size", str(bs), "--eval-batch-size", "128",
                "--output-root", OUT_ROOT],
    }


def v2_cmd(seed, eta):
    return {
        "id": f"concat_cwt__convnext_tiny__seed{seed}",
        "eta_min": eta,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                "--spec-type", "cwt", "--backbone", "convnext_tiny.fb_in22k_ft_in1k_384",
                "--target-size", "384", "--fold-csv", f"fold_full_seed{seed}.csv",
                "--fold-idx", "0", "--seed", str(seed),
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
                "--batch-size", "48", "--eval-batch-size", "64",
                "--mixup-alpha", "0.4", "--mixup-prob", "0.5",
                "--output-root", OUT_ROOT],
    }


jobs = []
for s in SEEDS:
    jobs.append(muku_cmd("convnext_pico.d1_in1k", s, 64, 7e-4, 7e-5, 75))
    jobs.append(muku_cmd("tf_efficientnet_b0.ns_jft_in1k", s, 64, 7e-4, 7e-5, 65))
    jobs.append(muku_cmd("resnet18", s, 128, 1e-3, 1e-4, 45))
    jobs.append(cwt_cmd("convnext_pico.d1_in1k", s, 64, 7e-4, 7e-5, 25))
    jobs.append(cwt_cmd("tf_efficientnet_b0.ns_jft_in1k", s, 64, 7e-4, 7e-5, 25))
    jobs.append(cwt_cmd("resnet18", s, 96, 8e-4, 8e-5, 15))
    jobs.append(v2_cmd(s, 55))

log(f"Total jobs: {len(jobs)}  on GPUs {GPUS}  out={OUT_ROOT}")

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
