"""Consensus-justified label-CORRECTION experiment: ConcatCWT maxvit256, A/B on GPU 1,2.

  Arm A (corrected): RELABEL the 121 high-consensus missed-spikes (y=0->1, 100% of 19 archs
                     independently call them positive) + down-weight the 66 ambiguous
                     DA00103D-direction flags (0.15).
  Arm B (baseline):  identical recipe + same per-sample-loss code path, but no relabel and
                     down-weight 1.0 (no-op) -> original labels.

A-vs-B honest-OOF (on ORIGINAL val labels) isolates the correction effect. val labels are NEVER
relabeled, so the metric stays honest/conservative.
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
RELABEL = "neuromm26_results/fold4_relabel_sids.txt"      # 121 missed-spikes (y0->1)
DOWNWT = "neuromm26_results/fold4_downweight_sids.txt"    # 66 ambiguous (down-weight)
# (prefix, relabel?, noise_weight)
ARMS = [("concat_cwt_relabel_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", True, "0.15"),
        ("concat_cwt_relbase_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", False, "1.0")]
GPUS = [1, 2]
RES = Path(REPO) / "neuromm26_results"

logdir = RES / f"logs/disp_relabel_cwt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logdir.mkdir(parents=True, exist_ok=True)
master = logdir / "MASTER.log"


def log(m):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with master.open("a") as f:
        f.write(line + "\n")


def done(prefix, f):
    return (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists()


jobs = []
for prefix, relabel, nw in ARMS:
    for f in range(5):
        if done(prefix, f):
            continue
        cmd = [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
               "--spec-type", "cwt", "--backbone", "maxvit_rmlp_tiny_rw_256.sw_in1k",
               "--target-size", "256", "--fold-idx", str(f), "--seed", "0",
               "--loss", "focal", "--focal-gamma", "2.0",
               "--stage1-epochs", "10", "--stage2-epochs", "30",
               "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
               "--batch-size", "64", "--eval-batch-size", "96",
               "--noise-sids", DOWNWT, "--noise-weight", nw,
               "--exp-prefix", prefix]
        if relabel:
            cmd += ["--relabel-sids", RELABEL]
        jobs.append({"id": f"{prefix}__fold{f}", "cmd": cmd})

gp = {g: None for g in GPUS}; gj = {g: None for g in GPUS}


def launch(g, jb):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(f"GPU {g} launching {jb['id']}")
    gp[g] = subprocess.Popen(jb["cmd"], cwd=REPO, env=env,
                             stdout=open(logdir / f"gpu{g}__{jb['id']}.log", "w"),
                             stderr=subprocess.STDOUT)
    gj[g] = jb["id"]


def chk(g):
    p = gp[g]
    if p is not None and p.poll() is not None:
        log(f"GPU {g} freed ('{gj[g]}' exit={p.returncode})")
        gp[g] = None; gj[g] = None


log(f"relabel-CWT A/B: {len(jobs)} jobs on GPU {GPUS}; relabel={RELABEL} downwt={DOWNWT}")
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
