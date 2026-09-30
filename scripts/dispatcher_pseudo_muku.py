"""Pseudo wave-4: the muku family on fold_df_pseudo (pooled, GPU 1,2 only). muku is NOT in the
NCHC concat batch, so this adds ~30% NM weight of pseudo coverage (raw + filt time-domain).
Fast (~40min/fold). Down-weighted pseudo (0.4), val on real labels (honest OOF).
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
GPUS = [1, 2]
RES = Path(REPO) / "neuromm26_results"
EEG = "neuromm26_datasets/processed/features/eeg"
FILT = "neuromm26_datasets/processed/features/filt_eeg"
# (backbone, eeg_root, exp_prefix)
ARCHS = [("convnext_pico.d1_in1k", EEG,  "muku_pseudo_fold__convnext_pico_d1_in1k"),
         ("resnet18",              EEG,  "muku_pseudo_fold__resnet18"),
         ("convnext_pico.d1_in1k", FILT, "muku_filteeg_pseudo_fold__convnext_pico_d1_in1k")]

logdir = RES / f"logs/disp_pseudo_muku_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logdir.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    (logdir / "MASTER.log").open("a").write(f"{m}\n")


def done(prefix, f):
    return (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists()


jobs = [{"id": f"{prefix}__fold{f}", "bb": bb, "root": root, "prefix": prefix, "fold": f}
        for bb, root, prefix in ARCHS for f in range(5) if not done(prefix, f)]
gp = {g: None for g in GPUS}; gj = {g: None for g in GPUS}


def launch(g, jb):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
           "--backbone", jb["bb"], "--eeg-feature-root", jb["root"],
           "--fold-idx", str(jb["fold"]), "--seed", "0", "--fold-csv", "fold_df_pseudo.csv",
           "--loss", "focal", "--focal-gamma", "2.0",
           "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "1e-3", "--stage2-lr", "1e-4",
           "--batch-size", "128", "--eval-batch-size", "128",
           "--noise-sids", "pseudo_sids.txt", "--noise-weight", "0.4", "--exp-prefix", jb["prefix"]]
    log(f"GPU {g} launching {jb['id']}")
    gp[g] = subprocess.Popen(cmd, cwd=REPO, env=env,
                             stdout=open(logdir / f"gpu{g}__{jb['id']}.log", "w"), stderr=subprocess.STDOUT)
    gj[g] = jb["id"]


def chk(g):
    p = gp[g]
    if p is not None and p.poll() is not None:
        log(f"GPU {g} freed ('{gj[g]}' exit={p.returncode})"); gp[g] = None; gj[g] = None


log(f"pseudo wave-4 muku: {len(jobs)} jobs on GPU {GPUS}")
q = list(jobs)
while q or any(p is not None for p in gp.values()):
    for g in GPUS:
        chk(g)
    for g in GPUS:
        if gp[g] is None and q:
            launch(g, q.pop(0))
    time.sleep(20)
log("ALL DONE")
(logdir / "ALL_DONE").touch()
