"""NCHC batch B — heavy-aug @384 rollout (carrier-quality axis).

Apply the validated heavy-aug recipe (superlet @384: OOF 0.8487→0.8654 +0.0167)
to the 3 strong specs that have NOT had heavy-aug @384 yet: cwt, cwt_paul, cwt_filtered.
Same backbone maxvit_tiny_tf_384 @384. NOTE: +0.0167 is NOT linearly extrapolated —
each spec may gain less; net ensemble expected +0.002~0.005.

Trainer unchanged (heavy-aug knobs already exist). 3 spec × 5 fold = 15 jobs.
REPO / PY / GPUS patched in-place by B_nchc_heavyaug384.sb.
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
from pathlib import Path

# === PATCHED BY .sb (keep no-space assignments) ===
REPO=os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY="python"
GPUS=[1,2]
# ===================================================

FOLD_CSV = "fold_df_fixed.csv"
BB = "maxvit_tiny_tf_384.in1k"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_nchc_heavyaug384_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


SPECS = ["cwt", "cwt_paul", "cwt_filtered"]   # superlet already done


def make_job(spec, fold):
    prefix = f"concat_{spec}_heavyaug_fold__maxvit_tiny_tf_384_in1k"
    cmd = [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
           "--spec-type", spec, "--backbone", BB, "--target-size", "384",
           "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
           "--stage1-epochs", "10", "--stage2-epochs", "30",
           "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
           "--batch-size", "24", "--eval-batch-size", "64",
           "--mixup-alpha", "0.4", "--mixup-prob", "0.7",
           "--time-mask-max", "24", "--n-time-masks", "2",
           "--freq-mask-max", "16", "--n-freq-masks", "2",
           "--channel-drop-max", "4", "--n-channel-masks", "2",
           "--spec-aug-prob", "0.7",
           "--exp-prefix", prefix]
    ck = Path(REPO) / "neuromm26_results" / "checkpoints" / f"{prefix}__fold{fold}__seed0" / "best.pt"
    oof = Path(REPO) / "neuromm26_results" / "predictions" / f"{prefix}__fold{fold}__seed0_oof.npz"
    return {"id": f"{prefix}__fold{fold}", "eta_min": 200, "cmd": cmd,
            "done": ck.exists() and oof.exists()}


jobs = [make_job(s, f) for s in SPECS for f in range(5)]
skipped = [j["id"] for j in jobs if j["done"]]
q = sorted([j for j in jobs if not j["done"]], key=lambda j: -j["eta_min"])
if skipped:
    log(f"Skipping {len(skipped)} already-complete")
log(f"Total to run: {len(q)} on GPUs {GPUS} (cwt/paul/filt heavy-aug @384)")


def launch(g, jb):
    lp = LOGDIR / f"gpu{g}__{jb['id']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(f"GPU {g} launching {jb['id']}")
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


gp = {g: None for g in GPUS}
gj = {g: None for g in GPUS}
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
