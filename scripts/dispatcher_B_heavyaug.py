"""Heavy-aug rollout (NCHC): ConcatPaul + ConcatCWT maxvit256 × 5 folds = 10 jobs.

Replicates the validated heavy-aug recipe that lifted ConcatSuperlet maxvit384
OOF 0.8487 -> 0.8654 (+0.0167) and is being applied to ConcatFilt locally.
Targets the two high-NM-weight carriers that have NOT had heavy-aug yet:
  - ConcatPaul maxvit256  (NM 16%)  spec=cwt_paul
  - ConcatCWT  maxvit256  (NM 12%)  spec=cwt

Parallel over 2 GPUs. REPO / PY / GPUS are patched in-place by B_heavyaug.sb
(SLURM gives CUDA_VISIBLE_DEVICES=0,1 inside the job).
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
import os
from pathlib import Path

# === PATCHED BY B_heavyaug.sb (keep these exact no-space assignments) ===
REPO=os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY="python"
GPUS=[1,2]
# ========================================================================

FOLD_CSV = "fold_df_fixed.csv"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_B_heavyaug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


# (spec_type, prefix) — naming matches existing heavy-aug convention for POOL discovery
ARCHS = [
    ("cwt_paul", "concat_cwt_paul_heavyaug_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("cwt",      "concat_cwt_heavyaug_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
]


def job(spec, prefix, fold):
    return {
        "id": f"{prefix}__fold{fold}",
        "eta_min": 120,
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                "--spec-type", spec,
                "--backbone", "maxvit_rmlp_tiny_rw_256.sw_in1k",
                "--target-size", "256",
                "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
                "--batch-size", "48", "--eval-batch-size", "96",
                "--mixup-alpha", "0.4", "--mixup-prob", "0.7",
                "--time-mask-max", "24", "--n-time-masks", "2",
                "--freq-mask-max", "16", "--n-freq-masks", "2",
                "--channel-drop-max", "4", "--n-channel-masks", "2",
                "--spec-aug-prob", "0.7",
                "--exp-prefix", prefix],
    }


jobs = [job(spec, prefix, f) for (spec, prefix) in ARCHS for f in range(5)]


def launch(g, jb):
    lp = LOGDIR / f"gpu{g}__{jb['id']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(f"GPU {g} launching {jb['id']} (eta={jb['eta_min']}m)")
    gp[g] = subprocess.Popen(jb["cmd"], cwd=REPO, env=env,
                             stdout=lp.open("w"), stderr=subprocess.STDOUT)
    gj[g] = jb["id"]


def chk(g):
    p = gp[g]
    if p is None:
        return True
    if p.poll() is not None:
        log(f"GPU {g} freed ('{gj[g]}' exit={p.returncode})")
        gp[g] = None
        gj[g] = None
        return True
    return False


q = sorted(jobs, key=lambda j: -j["eta_min"])
gp = {g: None for g in GPUS}
gj = {g: None for g in GPUS}

log(f"Total jobs: {len(q)} on GPUs {GPUS} (ConcatPaul + ConcatCWT maxvit256 heavy-aug)")
while q or any(p is not None for p in gp.values()):
    for g in GPUS:
        if gp[g] is not None:
            chk(g)
    for g in GPUS:
        if gp[g] is None and q:
            launch(g, q.pop(0))
    time.sleep(30)

log("ALL DONE")
(LOGDIR / "ALL_DONE").touch()
