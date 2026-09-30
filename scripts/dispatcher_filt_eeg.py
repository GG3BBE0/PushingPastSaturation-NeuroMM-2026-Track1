"""Filtered raw-EEG models on GPU 1,2.

4 archs x 5 folds = 20 jobs. All use filt_eeg cache as raw-EEG input.

Archs:
  1. muku V1  convnext_pico   on filt_eeg                  (exp: muku_filteeg_fold__convnext_pico_d1_in1k)
  2. muku V1  resnet18        on filt_eeg                  (exp: muku_filteeg_fold__resnet18)
  3. legacy   tcnet_eeg       on filt_eeg                  (exp: legacy_tcnet_filteeg_fold)
  4. muku V2  convnext_pico   on filt_eeg + raw superlet   (exp: muku_v2_filteeg_sl_fold__convnext_pico_d1_in1k)
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
FILT_ROOT = "neuromm26_datasets/processed/features/filt_eeg"

GPUS = [1, 2]
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_filt_eeg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"


def now():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


def muku_v1(bb, fold, bs, lr1, lr2, eta):
    bb_safe = bb.replace("/", "-").replace(".", "_")
    prefix = f"muku_filteeg_fold__{bb_safe}"
    return {
        "id": f"{prefix}__fold{fold}",
        "eta_min": eta,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
                "--backbone", bb, "--fold-csv", FOLD_CSV,
                "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", str(lr1), "--stage2-lr", str(lr2),
                "--batch-size", str(bs), "--eval-batch-size", "128",
                "--eeg-feature-root", FILT_ROOT,
                "--exp-prefix", prefix],
    }


def legacy_tcnet(fold, bs, lr1, lr2, eta):
    prefix = "legacy_tcnet_filteeg_fold"
    return {
        "id": f"{prefix}__fold{fold}",
        "eta_min": eta,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_legacy_fold",
                "--model-name", "tcnet_eeg",
                "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", str(lr1), "--stage2-lr", str(lr2),
                "--batch-size", str(bs), "--eval-batch-size", "128",
                "--eeg-feature-root", FILT_ROOT,
                "--exp-prefix", prefix],
    }


def muku_v2_sl(bb, fold, bs, lr1, lr2, eta):
    bb_safe = bb.replace("/", "-").replace(".", "_")
    prefix = f"muku_v2_filteeg_sl_fold__{bb_safe}"
    return {
        "id": f"{prefix}__fold{fold}",
        "eta_min": eta,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_muku_v2_fold",
                "--backbone", bb, "--spec-type", "superlet",
                "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", str(lr1), "--stage2-lr", str(lr2),
                "--batch-size", str(bs), "--eval-batch-size", str(min(128, bs * 2)),
                "--eeg-feature-root", FILT_ROOT,
                "--exp-prefix", prefix],
    }


jobs = []
for f in range(5):
    # 🥇 muku V1 convnext_pico filt_eeg  -- predicted highest weight
    jobs.append(muku_v1("convnext_pico.d1_in1k", f, bs=128, lr1=1e-3, lr2=1e-4, eta=40))
    # 🥈 legacy tcnet_eeg filt_eeg  -- pure time-domain, biggest theoretical filtering gain
    jobs.append(legacy_tcnet(f, bs=64, lr1=5e-4, lr2=5e-5, eta=35))
    # 🥉 muku V2 sl convnext_pico filt_eeg  -- eeg branch filtered, superlet branch raw
    jobs.append(muku_v2_sl("convnext_pico.d1_in1k", f, bs=128, lr1=1e-3, lr2=1e-4, eta=55))
    # 🔧 muku V1 resnet18 filt_eeg  -- cheap diversity
    jobs.append(muku_v1("resnet18", f, bs=128, lr1=1e-3, lr2=1e-4, eta=30))


def launch(g, jb):
    lp = LOGDIR / f"gpu{g}__{jb['id']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(f"GPU {g} launching {jb['id']} (eta={jb['eta_min']}m)")
    gp[g] = subprocess.Popen(jb["cmd"], cwd=REPO, env=env,
                             stdout=open(lp, "w"), stderr=subprocess.STDOUT)
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


# longest-eta-first so the slowest jobs start while plenty of time remains
q = sorted(jobs, key=lambda j: -j["eta_min"])
gp = {g: None for g in GPUS}
gj = {g: None for g in GPUS}

log(f"Total jobs: {len(q)} on GPUs {GPUS}")
log(f"Filt EEG cache: {FILT_ROOT}")

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
