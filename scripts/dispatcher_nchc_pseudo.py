"""NCHC big pseudo-label retrain: the top contributors retrained on fold_df_pseudo (real train +
high-confidence pseudo candidates, down-weighted). 2xH100 -> bump batch + scale LR. The more
high-weight arches we retrain on pseudo, the stronger the public-LB shift toward the candidate
distribution. GPU 0,1 on NCHC. Returns OOF (real-label val) + checkpoints to scp back.
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
FOLD_CSV = "fold_df_pseudo.csv"
BS = "128"      # H100 80GB
LR1, LR2 = "1.0e-3", "1.0e-4"   # scaled ~1.4x for the larger batch

# (spec, target_size, backbone, exp_prefix)
ARCHS = [
    ("cwt",          256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "concat_cwt_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("cwt_paul",     256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "concat_cwt_paul_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("cwt_filtered", 256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "concat_cwt_filtered_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("stft",         256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "concat_stft_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("superlet",     384, "maxvit_tiny_tf_384.in1k",         "concat_superlet_pseudo_fold__maxvit_tiny_tf_384_in1k"),
    ("cwt",          384, "maxvit_tiny_tf_384.in1k",         "concat_cwt_pseudo_fold__maxvit_tiny_tf_384_in1k"),
]

logdir = RES / f"logs/disp_nchc_pseudo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logdir.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    (logdir / "MASTER.log").open("a").write(f"{m}\n")


def done(prefix, f):
    return (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists()


jobs = []
for spec, tsize, bb, prefix in ARCHS:
    bs = "96" if tsize == 384 else BS
    for f in range(5):
        if done(prefix, f):
            continue
        jobs.append({"id": f"{prefix}__fold{f}",
                     "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                             "--spec-type", spec, "--backbone", bb, "--target-size", str(tsize),
                             "--fold-idx", str(f), "--seed", "0", "--fold-csv", FOLD_CSV,
                             "--loss", "focal", "--focal-gamma", "2.0",
                             "--stage1-epochs", "10", "--stage2-epochs", "30",
                             "--stage1-lr", LR1, "--stage2-lr", LR2,
                             "--batch-size", bs, "--eval-batch-size", "128",
                             "--noise-sids", "pseudo_sids.txt", "--noise-weight", "0.4",
                             "--exp-prefix", prefix]})

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


log(f"NCHC pseudo: {len(jobs)} jobs on GPU {GPUS}  (fold_df_pseudo, pseudo down-weight 0.4)")
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
