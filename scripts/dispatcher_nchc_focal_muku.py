"""NCHC batch A — focal loss on muku carriers (T1 binary, loss axis).

muku raw convnext_pico + muku Filt convnext_pico → main-head focal (aux stays BCE).
These ~21% NM-weight carriers were not covered by the ConcatSpec focal batch.

PILOT first: muku raw convnext_pico fold0, focal ±pos_weight (2 jobs), to decide
whether focal+pos_weight double-counts imbalance. Then full 2-arch × 5-fold.

REPO / PY / GPUS patched in-place by B_nchc_focal_muku.sb.
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

PILOT=1          # 1 = pilot (muku raw convnext_pico fold0, ±pos_weight); 0 = full 2×5
POS_WEIGHT=1     # full only: 1 keep pos_weight, 0 --no-pos-weight (set per pilot)
FOCAL_GAMMA="2.0"

FOLD_CSV = "fold_df_fixed.csv"
EEG_ROOT = "neuromm26_datasets/processed/features/eeg"
FILT_ROOT = "neuromm26_datasets/processed/features/filt_eeg"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_nchc_focal_muku_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


# (tag, eeg_root) — muku V1 convnext_pico on raw eeg and on filt_eeg
CARRIERS = [
    ("muku_focal",        EEG_ROOT),
    ("muku_filteeg_focal", FILT_ROOT),
]
BB = "convnext_pico.d1_in1k"
BB_SAFE = BB.replace("/", "-").replace(".", "_")


def make_job(tag, eeg_root, fold, pos_weight):
    suffix = "" if pos_weight else "_nopw"
    prefix = f"{tag}{suffix}_fold__{BB_SAFE}"
    cmd = [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
           "--backbone", BB, "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
           "--stage1-epochs", "10", "--stage2-epochs", "30",
           "--stage1-lr", "1e-3", "--stage2-lr", "1e-4",
           "--batch-size", "128", "--eval-batch-size", "128",
           "--eeg-feature-root", eeg_root,
           "--loss", "focal", "--focal-gamma", FOCAL_GAMMA,
           "--exp-prefix", prefix]
    if not pos_weight:
        cmd.append("--no-pos-weight")
    ck = Path(REPO) / "neuromm26_results" / "checkpoints" / f"{prefix}__fold{fold}__seed0" / "best.pt"
    oof = Path(REPO) / "neuromm26_results" / "predictions" / f"{prefix}__fold{fold}__seed0_oof.npz"
    return {"id": f"{prefix}__fold{fold}", "eta_min": 45, "cmd": cmd,
            "done": ck.exists() and oof.exists()}


jobs = []
if PILOT:
    tag, root = CARRIERS[0]  # muku raw convnext_pico
    jobs.append(make_job(tag, root, 0, pos_weight=True))
    jobs.append(make_job(tag, root, 0, pos_weight=False))
    log("PILOT: muku raw convnext_pico fold0, focal gamma 2.0, ±pos_weight (2 jobs)")
else:
    for (tag, root) in CARRIERS:
        for f in range(5):
            jobs.append(make_job(tag, root, f, pos_weight=bool(POS_WEIGHT)))
    log(f"FULL: 2 carriers × 5 fold, focal gamma 2.0, pos_weight={'on' if POS_WEIGHT else 'off'}")

skipped = [j["id"] for j in jobs if j["done"]]
q = sorted([j for j in jobs if not j["done"]], key=lambda j: -j["eta_min"])
if skipped:
    log(f"Skipping {len(skipped)} already-complete: {skipped}")
log(f"Total to run: {len(q)} on GPUs {GPUS}")


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
