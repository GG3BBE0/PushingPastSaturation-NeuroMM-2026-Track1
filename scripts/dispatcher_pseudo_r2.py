"""Round-2 pseudo retrain: ConcatCWT + ConcatPaul maxvit256 on fold_df_pseudo_r2 (re-selected from
pseudo3, the better source). noise-weight 0.4 kept == round-1 to ISOLATE the source-quality effect
(does a cleaner pseudo source alone beat round-1?). GPU 1,2 ONLY. Pooled so neither GPU idles.
Gate after: verify_replace round-2 arches vs canonical; compare meanΔ to round-1's +0.0058.
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
ARCHS = [("cwt", "concat_cwt_pseudo_r2_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
         ("cwt_paul", "concat_cwt_paul_pseudo_r2_fold__maxvit_rmlp_tiny_rw_256_sw_in1k")]

logdir = RES / f"logs/disp_pseudo_r2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logdir.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    (logdir / "MASTER.log").open("a").write(f"{m}\n")


def done(prefix, f):
    return (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists()


jobs = []
for spec, prefix in ARCHS:
    for f in range(5):
        if not done(prefix, f):
            jobs.append({"id": f"{prefix}__fold{f}", "spec": spec, "prefix": prefix, "fold": f})

gp = {g: None for g in GPUS}; gj = {g: None for g in GPUS}


def launch(g, jb):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
           "--spec-type", jb["spec"], "--backbone", "maxvit_rmlp_tiny_rw_256.sw_in1k",
           "--target-size", "256", "--fold-idx", str(jb["fold"]), "--seed", "0",
           "--fold-csv", "fold_df_pseudo_r2.csv", "--loss", "focal", "--focal-gamma", "2.0",
           "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
           "--batch-size", "64", "--eval-batch-size", "96",
           "--noise-sids", "pseudo_sids_r2.txt", "--noise-weight", "0.4", "--exp-prefix", jb["prefix"]]
    log(f"GPU {g} launching {jb['id']}")
    gp[g] = subprocess.Popen(cmd, cwd=REPO, env=env,
                             stdout=open(logdir / f"gpu{g}__{jb['id']}.log", "w"), stderr=subprocess.STDOUT)
    gj[g] = jb["id"]


def chk(g):
    p = gp[g]
    if p is not None and p.poll() is not None:
        log(f"GPU {g} freed ('{gj[g]}' exit={p.returncode})"); gp[g] = None; gj[g] = None


log(f"pooled pseudo round-2: {len(jobs)} jobs on GPU {GPUS}")
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
