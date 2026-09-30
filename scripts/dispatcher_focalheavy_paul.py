"""Focal + heavy-aug pilot (FULL 5-fold): ConcatPaul maxvit256.

Tests whether focal loss is superadditive with heavy SpecAug — the stacked
combination is not covered by either NCHC batch (batch1=focal-only,
batch2-B=heavyaug-only).

Compare vs concat_cwt_paul_heavyaug256 (heavy-aug-alone) → isolates focal's
marginal effect. Full 5 folds so P1 guard (>=4/5 fold win) can be evaluated.

GPU scheduling:
- GPU 2 is free now → starts immediately.
- GPU 1 is busy with predict/regnm v4 → joins once the marker file
  `neuromm26_results/logs/V4_GPU1_FREE` appears (touched manually or by the
  predict+regnm chain). Until then only GPU 2 runs.
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
import os
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
FOLD_CSV = "fold_df_fixed.csv"
GPU1_FREE_MARKER = Path(REPO) / "neuromm26_results/logs/V4_GPU1_FREE"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_focalheavy_paul_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"

PREFIX = "concat_cwt_paul_focalheavy_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


def cmd(fold):
    return [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
            "--spec-type", "cwt_paul",
            "--backbone", "maxvit_rmlp_tiny_rw_256.sw_in1k", "--target-size", "256",
            "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
            "--stage1-epochs", "10", "--stage2-epochs", "30",
            "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
            "--batch-size", "48", "--eval-batch-size", "96",
            "--loss", "focal", "--focal-gamma", "2.0",
            "--mixup-alpha", "0.4", "--mixup-prob", "0.7",
            "--time-mask-max", "24", "--n-time-masks", "2",
            "--freq-mask-max", "16", "--n-freq-masks", "2",
            "--channel-drop-max", "4", "--n-channel-masks", "2",
            "--spec-aug-prob", "0.7",
            "--exp-prefix", PREFIX]


def done(fold):
    ck = Path(REPO) / "neuromm26_results/checkpoints" / f"{PREFIX}__fold{fold}__seed0" / "best.pt"
    oof = Path(REPO) / "neuromm26_results/predictions" / f"{PREFIX}__fold{fold}__seed0_oof.npz"
    return ck.exists() and oof.exists()


q = [f for f in range(5) if not done(f)]
skipped = [f for f in range(5) if done(f)]
log(f"FULL 5-fold focal+heavyaug ConcatPaul maxvit256. Queue folds: {q}, skip(done): {skipped}")
log(f"GPU 2 active immediately; GPU 1 joins when {GPU1_FREE_MARKER.name} appears.")

# gp/gj keyed by GPU id; GPU 1 only usable after marker
proc = {1: None, 2: None}
pjob = {1: None, 2: None}


def gpu1_available():
    return GPU1_FREE_MARKER.exists()


def launch(g, fold):
    lp = LOGDIR / f"gpu{g}__{PREFIX}__fold{fold}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(f"GPU {g} launching fold{fold}")
    proc[g] = subprocess.Popen(cmd(fold), cwd=REPO, env=env, stdout=lp.open("w"), stderr=subprocess.STDOUT)
    pjob[g] = fold


def chk(g):
    p = proc[g]
    if p is None:
        return
    if p.poll() is not None:
        log(f"GPU {g} freed (fold{pjob[g]} exit={p.returncode})")
        proc[g] = None
        pjob[g] = None


while q or any(p is not None for p in proc.values()):
    for g in (1, 2):
        chk(g)
    # GPU 2 always available; GPU 1 only after marker
    usable = [2] + ([1] if gpu1_available() else [])
    for g in usable:
        if proc[g] is None and q:
            launch(g, q.pop(0))
    time.sleep(30)

log("ALL DONE")
(LOGDIR / "ALL_DONE").touch()
