"""LOCAL round-3 specialist hedge (teacher = specnchc 0.9846). Trains the 3 dominant @256 concats
(cwt/paul/filt) + muku spec3 locally as a fallback in case NCHC round-3 doesn't return before deadline.
superlet@384 is DEFERRED to NCHC (too slow locally). GPU 2,3 (2-wide) to stay <=3 concurrent CPU-aug
alongside the private-sim's sup_f4 (avoids the num_workers=0 CPU-dataloader starvation). Each completed
spec3 arch can be swapped into the specnchc pool for an incremental probe (like specnchc itself).

Run: python scripts/dispatcher_specialist3_local.py   (after muku private-sim frees the GPUs)
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
import os
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
GPUS = [3, 4]  # 2-wide (GPU2 has muk_f0); +sup_f4 on GPU1 = 3 concurrent CPU-aug. bump after sup_f4 done
RES = Path(REPO) / "neuromm26_results"
EEG = "neuromm26_datasets/processed/features/eeg"
FOLD_CSV = "fold_df_pseudo_spec3.csv"
SIDS = "pseudo_sids_spec3.txt"

jobs = []
CONCAT = [
    ("cwt",          256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "64", "7e-4", "7e-5", "concat_cwt_spec3_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("cwt_paul",     256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "64", "7e-4", "7e-5", "concat_cwt_paul_spec3_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
    ("cwt_filtered", 256, "maxvit_rmlp_tiny_rw_256.sw_in1k", "64", "7e-4", "7e-5", "concat_cwt_filtered_spec3_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
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
MUKU = [("convnext_pico.d1_in1k", "muku_spec3_fold__convnext_pico_d1_in1k")]
for bb, prefix in MUKU:
    for f in range(5):
        if (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists():
            continue
        jobs.append({"id": f"{prefix}__fold{f}",
                     "cmd": [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
                             "--backbone", bb, "--eeg-feature-root", EEG,
                             "--fold-idx", str(f), "--seed", "0", "--fold-csv", FOLD_CSV,
                             "--loss", "focal", "--focal-gamma", "2.0",
                             "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
                             "--batch-size", "128", "--eval-batch-size", "128",
                             "--noise-sids", SIDS, "--noise-weight", "1.0", "--exp-prefix", prefix]})

logdir = RES / f"logs/disp_specialist3_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


log(f"LOCAL specialist3 (teacher=0.9846): {len(jobs)} jobs on GPU {GPUS} (spec3, weight 1.0)")
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
