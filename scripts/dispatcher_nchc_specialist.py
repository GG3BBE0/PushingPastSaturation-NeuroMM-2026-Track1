"""NCHC candidate-SPECIALIST (strategy-B, the 0.985 attempt at full strength). Retrain the top arches
on fold_df_pseudo_spec (11336 candidates labeled by nchc_n2filt 0.9778, weight 1.0 = FULL trust) with
2xH100 bigger batch. A specialist-dominant blend of these escapes the ensemble blend-dampening that
capped every honest method at 0.978. Includes superlet384 (#1 weight, slow locally) + muku convnext
(orthogonal). New exp-prefixes (_specnchc) so nothing clobbers. GPU 0,1 on NCHC.
Real-label OOF is the BRAKE (monitor collapse), the gate is a PUBLIC probe (user decides upload).
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

jobs = []
# concat specialists: (spec, tsize, backbone, bs, lr1, lr2, prefix)
CONCAT = [
    ("cwt",          256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "128", "1.0e-3", "1.0e-4", "concat_cwt_specnchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("cwt_paul",     256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "128", "1.0e-3", "1.0e-4", "concat_cwt_paul_specnchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("cwt_filtered", 256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "128", "1.0e-3", "1.0e-4", "concat_cwt_filtered_specnchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("superlet",     384, "maxvit_tiny_tf_384.in1k",          "96", "9.0e-4", "9.0e-5", "concat_superlet_specnchc_fold__maxvit_tiny_tf_384_in1k"),
]
for spec, tsize, bb, bs, lr1, lr2, prefix in CONCAT:
    for f in range(5):
        if (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists():
            continue
        jobs.append({"id": f"{prefix}__fold{f}",
                     "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                             "--spec-type", spec, "--backbone", bb, "--target-size", str(tsize),
                             "--fold-idx", str(f), "--seed", "0", "--fold-csv", "fold_df_pseudo_spec.csv",
                             "--loss", "focal", "--focal-gamma", "2.0",
                             "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", lr1, "--stage2-lr", lr2,
                             "--batch-size", bs, "--eval-batch-size", bs,
                             "--noise-sids", "pseudo_sids_spec.txt", "--noise-weight", "1.0", "--exp-prefix", prefix]})
# muku specialist (orthogonal time-domain)
MUKU = [("convnext_pico.d1_in1k", "muku_specnchc_fold__convnext_pico_d1_in1k")]
for bb, prefix in MUKU:
    for f in range(5):
        if (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists():
            continue
        jobs.append({"id": f"{prefix}__fold{f}",
                     "cmd": [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
                             "--backbone", bb, "--eeg-feature-root", EEG,
                             "--fold-idx", str(f), "--seed", "0", "--fold-csv", "fold_df_pseudo_spec.csv",
                             "--loss", "focal", "--focal-gamma", "2.0",
                             "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "2.0e-3", "--stage2-lr", "2.0e-4",
                             "--batch-size", "256", "--eval-batch-size", "256",
                             "--noise-sids", "pseudo_sids_spec.txt", "--noise-weight", "1.0", "--exp-prefix", prefix]})

logdir = RES / f"logs/disp_nchc_specialist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


log(f"NCHC candidate-specialist: {len(jobs)} jobs on GPU {GPUS} (spec labels, weight 1.0)")
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
