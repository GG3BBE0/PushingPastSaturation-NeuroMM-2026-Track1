"""Local pseudo wave-3: the highest-weight @384 arches on fold_df_pseudo (pooled, GPU 1,2 only).
ConcatSuperlet maxvit384 (NM #1, ~19%) + ConcatCWT maxvit384. Same exp-prefixes as the NCHC batch
-> coordinated skip-done. @384 is slow locally (~6h/fold) but these carry the most ensemble weight,
so retraining them on pseudo is the biggest lever toward 0.985.
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
# (spec, backbone, exp_prefix) — superlet first (NM #1)
ARCHS = [("superlet", "maxvit_tiny_tf_384.in1k", "concat_superlet_pseudo_fold__maxvit_tiny_tf_384_in1k"),
         ("cwt", "maxvit_tiny_tf_384.in1k", "concat_cwt_pseudo_fold__maxvit_tiny_tf_384_in1k")]

logdir = RES / f"logs/disp_pseudo_384_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logdir.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    (logdir / "MASTER.log").open("a").write(f"{m}\n")


def done(prefix, f):
    return (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists()


jobs = [{"id": f"{prefix}__fold{f}", "spec": spec, "bb": bb, "prefix": prefix, "fold": f}
        for spec, bb, prefix in ARCHS for f in range(5) if not done(prefix, f)]
gp = {g: None for g in GPUS}; gj = {g: None for g in GPUS}


def launch(g, jb):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
           "--spec-type", jb["spec"], "--backbone", jb["bb"], "--target-size", "384",
           "--fold-idx", str(jb["fold"]), "--seed", "0", "--fold-csv", "fold_df_pseudo.csv",
           "--loss", "focal", "--focal-gamma", "2.0",
           "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
           "--batch-size", "24", "--eval-batch-size", "48",
           "--noise-sids", "pseudo_sids.txt", "--noise-weight", "0.4", "--exp-prefix", jb["prefix"]]
    log(f"GPU {g} launching {jb['id']}")
    gp[g] = subprocess.Popen(cmd, cwd=REPO, env=env,
                             stdout=open(logdir / f"gpu{g}__{jb['id']}.log", "w"), stderr=subprocess.STDOUT)
    gj[g] = jb["id"]


def chk(g):
    p = gp[g]
    if p is not None and p.poll() is not None:
        log(f"GPU {g} freed ('{gj[g]}' exit={p.returncode})"); gp[g] = None; gj[g] = None


log(f"pseudo wave-3 @384: {len(jobs)} jobs on GPU {GPUS} (Superlet#1 + CWT)")
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
