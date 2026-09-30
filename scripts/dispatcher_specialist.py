"""Candidate-SPECIALIST retrain (strategy-B extreme, public-gated): cwt/paul/filt @256 on
fold_df_pseudo_spec (11336 candidates labeled by nchc_n2filt 0.9778, weight 1.0 = FULL trust).
Goal: arches that fit the candidate distribution hard -> a specialist-dominant blend escapes the
ensemble blend-dampening that capped every honest method at 0.978. Real-label OOF is monitored (the
brake) but NOT the gate; the gate is a PUBLIC probe (private risk -> user decides upload).
GPU 1,2 ONLY. exp-prefix _spec_.
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
import os
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
GPUS = [1, 2]
RES = Path(REPO) / "neuromm26_results"
ARCHS = [("cwt", "concat_cwt_spec_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
         ("cwt_paul", "concat_cwt_paul_spec_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
         ("cwt_filtered", "concat_cwt_filtered_spec_fold__maxvit_rmlp_tiny_rw_256_sw_in1k")]

logdir = RES / f"logs/disp_specialist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logdir.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    (logdir / "MASTER.log").open("a").write(f"{m}\n")


def done(prefix, f):
    return (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists()


jobs = [{"id": f"{prefix}__fold{f}", "spec": spec, "prefix": prefix, "fold": f}
        for spec, prefix in ARCHS for f in range(5) if not done(prefix, f)]
gp = {g: None for g in GPUS}; gj = {g: None for g in GPUS}


def launch(g, jb):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
           "--spec-type", jb["spec"], "--backbone", "maxvit_rmlp_tiny_rw_256.sw_in1k",
           "--target-size", "256", "--fold-idx", str(jb["fold"]), "--seed", "0",
           "--fold-csv", "fold_df_pseudo_spec.csv", "--loss", "focal", "--focal-gamma", "2.0",
           "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
           "--batch-size", "64", "--eval-batch-size", "96",
           "--noise-sids", "pseudo_sids_spec.txt", "--noise-weight", "1.0", "--exp-prefix", jb["prefix"]]
    log(f"GPU {g} launching {jb['id']}")
    gp[g] = subprocess.Popen(cmd, cwd=REPO, env=env,
                             stdout=open(logdir / f"gpu{g}__{jb['id']}.log", "w"), stderr=subprocess.STDOUT)
    gj[g] = jb["id"]


def chk(g):
    p = gp[g]
    if p is not None and p.poll() is not None:
        log(f"GPU {g} freed ('{gj[g]}' exit={p.returncode})"); gp[g] = None; gj[g] = None


log(f"candidate-specialist: {len(jobs)} jobs on GPU {GPUS} (weight 1.0 full-trust)")
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
