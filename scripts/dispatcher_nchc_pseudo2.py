"""NCHC pseudo round-2 batch (BOTH strategy lines), 2xH100 -> bump batch + scale LR.
  AGGRESSIVE (public-gated): cwt/paul/filt @256 on fold_df_pseudo_r3 (source=nchc_n2filt 0.9778,
    HI=0.80, weight 0.5). The clean aggression test at NCHC quality -> probe public.
  HONEST (CV-gated): muku raw convnext_pico + resnet18 on fold_df_pseudo (round-1, weight 0.4).
    Time-domain diversity the NCHC concat batch never touched -> honest-gate.
New exp-prefixes (_r3nchc / _pseudo_nchc) so nothing clobbers existing OOF/ckpts.
GPU 0,1 on NCHC. Returns OOF (real-label val) + ckpts to scp back.
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("REPO_DIR", "/work/<your-account>/NeuroMM-2026_Baseline")
PY = os.environ.get("PY", "python")
GPUS = [0, 1]  # NCHC
RES = Path(REPO) / "neuromm26_results"
EEG = "neuromm26_datasets/processed/features/eeg"

jobs = []

# --- AGGRESSIVE line: concat cwt/paul/filt @256 on round-3 labels (weight 0.5) ---
AGG = [("cwt",          "concat_cwt_pseudo_r3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
       ("cwt_paul",     "concat_cwt_paul_pseudo_r3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
       ("cwt_filtered", "concat_cwt_filtered_pseudo_r3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k")]
for spec, prefix in AGG:
    for f in range(5):
        if (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists():
            continue
        jobs.append({"id": f"{prefix}__fold{f}",
                     "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                             "--spec-type", spec, "--backbone", "maxvit_rmlp_tiny_rw_256.sw_in1k",
                             "--target-size", "256", "--fold-idx", str(f), "--seed", "0",
                             "--fold-csv", "fold_df_pseudo_r3.csv", "--loss", "focal", "--focal-gamma", "2.0",
                             "--stage1-epochs", "10", "--stage2-epochs", "30",
                             "--stage1-lr", "1.0e-3", "--stage2-lr", "1.0e-4",
                             "--batch-size", "128", "--eval-batch-size", "128",
                             "--noise-sids", "pseudo_sids_r3.txt", "--noise-weight", "0.5",
                             "--exp-prefix", prefix]})

# --- HONEST line: muku raw convnext_pico + resnet18 on round-1 labels (weight 0.4) ---
HON = [("convnext_pico.d1_in1k", "muku_pseudo_nchc_fold__convnext_pico_d1_in1k"),
       ("resnet18",              "muku_pseudo_nchc_fold__resnet18")]
for bb, prefix in HON:
    for f in range(5):
        if (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists():
            continue
        jobs.append({"id": f"{prefix}__fold{f}",
                     "cmd": [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
                             "--backbone", bb, "--eeg-feature-root", EEG,
                             "--fold-idx", str(f), "--seed", "0", "--fold-csv", "fold_df_pseudo.csv",
                             "--loss", "focal", "--focal-gamma", "2.0",
                             "--stage1-epochs", "10", "--stage2-epochs", "30",
                             "--stage1-lr", "2.0e-3", "--stage2-lr", "2.0e-4",
                             "--batch-size", "256", "--eval-batch-size", "256",
                             "--noise-sids", "pseudo_sids.txt", "--noise-weight", "0.4",
                             "--exp-prefix", prefix]})

logdir = RES / f"logs/disp_nchc_pseudo2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logdir.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    (logdir / "MASTER.log").open("a").write(f"{m}\n")


gp = {g: None for g in GPUS}; gj = {g: None for g in GPUS}


def launch(g, jb):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(f"GPU {g} launching {jb['id']}")
    gp[g] = subprocess.Popen(jb["cmd"], cwd=REPO, env=env,
                             stdout=open(logdir / f"gpu{g}__{jb['id']}.log", "w"), stderr=subprocess.STDOUT)
    gj[g] = jb["id"]


def chk(g):
    p = gp[g]
    if p is not None and p.poll() is not None:
        log(f"GPU {g} freed ('{gj[g]}' exit={p.returncode})"); gp[g] = None; gj[g] = None


log(f"NCHC pseudo2: {len(jobs)} jobs on GPU {GPUS}  (aggressive cwt/paul/filt @r3 + honest muku @r1)")
q = list(jobs)
while q or any(p is not None for p in gp.values()):
    for g in GPUS:
        chk(g)
    for g in GPUS:
        if gp[g] is None and q:
            launch(g, q.pop(0))
    time.sleep(30)
log("ALL DONE")
(logdir / "ALL_DONE").touch()
