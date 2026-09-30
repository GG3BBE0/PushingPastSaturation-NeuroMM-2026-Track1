"""NCHC candidate-SPECIALIST round-3 (teacher = specnchc public 0.9846, the current best). Same proven
recipe that took 0.9778->0.9818->0.9846: retrain the 5 specnchc arches on fold_df_pseudo_spec3 (13132
candidates labeled by the 0.9846 ensemble @ weight 1.0 FULL trust). Each teacher iteration so far jumped
the score; this is the next round on a cleaner (0.9846) teacher. New exp-prefixes (_spec3nchc) so nothing
clobbers specnchc. GPU 0,1 on NCHC. Real-label OOF = the BRAKE; gate (honest nested OOF) decides upload.
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("REPO_DIR", "/work/<your-account>/NeuroMM-2026_Baseline")
PY = os.environ.get("PY", "python")
GPUS = [0, 1]  # NCHC
RES = Path(REPO) / "neuromm26_results"
EEG = "neuromm26_datasets/processed/features/eeg"
FOLD_CSV = "fold_df_pseudo_spec3.csv"
SIDS = "pseudo_sids_spec3.txt"

jobs = []
# concat specialists: (spec, tsize, backbone, bs, lr1, lr2, prefix)
CONCAT = [
    ("cwt",          256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "128", "1.0e-3", "1.0e-4", "concat_cwt_spec3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("cwt_paul",     256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "128", "1.0e-3", "1.0e-4", "concat_cwt_paul_spec3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("cwt_filtered", 256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "128", "1.0e-3", "1.0e-4", "concat_cwt_filtered_spec3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("superlet",     384, "maxvit_tiny_tf_384.in1k",          "96", "9.0e-4", "9.0e-5", "concat_superlet_spec3nchc_fold__maxvit_tiny_tf_384_in1k"),
]
for spec, tsize, bb, bs, lr1, lr2, prefix in CONCAT:
    for f in range(5):
        if (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists():
            continue
        jobs.append({"id": f"{prefix}__fold{f}",
                     "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                             "--spec-type", spec, "--backbone", bb, "--target-size", str(tsize),
                             "--fold-idx", str(f), "--seed", "0", "--fold-csv", FOLD_CSV,
                             "--loss", "focal", "--focal-gamma", "2.0",
                             "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", lr1, "--stage2-lr", lr2,
                             "--batch-size", bs, "--eval-batch-size", bs,
                             "--noise-sids", SIDS, "--noise-weight", "1.0", "--exp-prefix", prefix]})
# muku specialist (orthogonal time-domain)
MUKU = [("convnext_pico.d1_in1k", "muku_spec3nchc_fold__convnext_pico_d1_in1k")]
for bb, prefix in MUKU:
    for f in range(5):
        if (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists():
            continue
        jobs.append({"id": f"{prefix}__fold{f}",
                     "cmd": [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
                             "--backbone", bb, "--eeg-feature-root", EEG,
                             "--fold-idx", str(f), "--seed", "0", "--fold-csv", FOLD_CSV,
                             "--loss", "focal", "--focal-gamma", "2.0",
                             "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "2.0e-3", "--stage2-lr", "2.0e-4",
                             "--batch-size", "256", "--eval-batch-size", "256",
                             "--noise-sids", SIDS, "--noise-weight", "1.0", "--exp-prefix", prefix]})

logdir = RES / f"logs/disp_nchc_specialist3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


log(f"NCHC specialist3 (teacher=0.9846): {len(jobs)} jobs on GPU {GPUS} (spec3 labels, weight 1.0)")
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
