"""NCHC B-plan v2: Heavy-aug rollout on maxvit_tiny @384 (Paul / Filt / CWT) x 5 fold = 15 jobs.

Edit GPUS, REPO, PY for the NCHC node (the .sb script will patch these automatically).

Heavy aug params copied from the successful ConcatSuperlet heavyaug ablation:
  mixup-prob 0.7, time-mask-max 24 x2, freq-mask-max 16 x2, channel-drop-max 4 x2, spec-aug-prob 0.7

Output prefix: concat_{spec}_heavyaug_fold__maxvit_tiny_tf_384_in1k (kept separate from baseline)
"""
import os, subprocess, time
from datetime import datetime
from pathlib import Path
# === EDIT FOR REMOTE MACHINE ===
REPO = "/work/<your-account>/NeuroMM-2026_Baseline"           # remote NCHC path
PY = "/work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/envs/neuromm26-baseline/bin/python"
GPUS = [0, 1]   # NCHC SLURM exposes 2 GPUs as 0,1
# ================================
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_B_heavyaug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"

BB = "maxvit_tiny_tf_384.in1k"
TS = 384
FOLD_CSV = "fold_df_fixed.csv"


def log(m):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(MASTER, "a").write(line + "\n")


def cc(spec, fold, bs=24, lr1=7e-4, lr2=7e-5):
    """Heavy-aug command for ConcatSpec maxvit_tiny @384 on given spec basis."""
    prefix = f"concat_{spec}_heavyaug_fold__maxvit_tiny_tf_384_in1k"
    return {
        "id": f"{prefix}__fold{fold}",
        "prefix": prefix,
        "eta_min": 240,  # ~4h per fold based on ConcatSuperlet heavyaug timing
        "cmd": [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                "--spec-type", spec,
                "--backbone", BB, "--target-size", str(TS),
                "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
                "--stage1-epochs", "10", "--stage2-epochs", "30",
                "--stage1-lr", str(lr1), "--stage2-lr", str(lr2),
                "--batch-size", str(bs), "--eval-batch-size", "64",
                "--mixup-alpha", "0.4", "--mixup-prob", "0.7",
                "--time-mask-max", "24", "--n-time-masks", "2",
                "--freq-mask-max", "16", "--n-freq-masks", "2",
                "--channel-drop-max", "4", "--n-channel-masks", "2",
                "--spec-aug-prob", "0.7",
                "--exp-prefix", prefix],
    }


def done(spec, fold):
    """Idempotent skip: both best.pt and OOF .npz already present."""
    prefix = f"concat_{spec}_heavyaug_fold__maxvit_tiny_tf_384_in1k"
    ck = Path(REPO) / "neuromm26_results" / "checkpoints" / f"{prefix}__fold{fold}__seed0" / "best.pt"
    oof = Path(REPO) / "neuromm26_results" / "predictions" / f"{prefix}__fold{fold}__seed0_oof.npz"
    return ck.exists() and oof.exists()


# Build queue: 3 specs × 5 folds = 15 jobs
SPECS = ["cwt_paul", "cwt_filtered", "cwt"]
q, skipped = [], []
for spec in SPECS:
    for f in range(5):
        if done(spec, f):
            skipped.append(f"concat_{spec}_heavyaug__fold{f}")
        else:
            q.append(cc(spec, f))

# Longest-eta-first (all same eta here, but consistent w/ pattern)
q.sort(key=lambda j: -j["eta_min"])

log(f"Total jobs: {len(q)} on GPUs {GPUS} (heavy aug rollout, maxvit_tiny @384)")
if skipped:
    log(f"Skipping {len(skipped)} already-complete jobs: {skipped}")

gp = {g: None for g in GPUS}
gj = {g: None for g in GPUS}


def launch(g, jb):
    lp = LOGDIR / f"gpu{g}__{jb['id']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(f"GPU {g} launching {jb['id']} (eta={jb['eta_min']}m)")
    gp[g] = subprocess.Popen(jb["cmd"], cwd=REPO, env=env,
                             stdout=open(lp, "w"), stderr=subprocess.STDOUT)
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
