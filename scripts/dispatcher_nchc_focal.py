"""NCHC focal-loss rollout — retrain top ConcatSpec carriers with focal loss (T1 binary).

Non-backbone lever (axis 4): BCE optimizes calibration; focal down-weights easy
samples → focuses on the pos/neg-inseparable hard samples the per-subject
diagnosis flagged. Same recipe as each carrier's BASELINE (NOT heavy-aug) so the
focal effect is isolated vs the existing baseline carrier.

PILOT first: 1 carrier (ConcatSuperlet maxvit384) fold0 in BOTH ±pos_weight, to
decide whether focal+pos_weight double-counts imbalance. Then full 4×5 rollout.

REPO / PY / GPUS patched in-place by B_nchc_focal.sb (SLURM → CUDA_VISIBLE_DEVICES=0,1).
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
from pathlib import Path

# === PATCHED BY B_nchc_focal.sb (keep no-space assignments) ===
REPO=os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY="python"
GPUS=[1,2]
# ==============================================================

# Switches (edit in .sb or here)
PILOT=1          # 1 = run pilot (1 carrier fold0, ±pos_weight); 0 = full 4×5 rollout
POS_WEIGHT=1     # full-rollout only: 1 = keep pos_weight, 0 = --no-pos-weight (set per pilot result)
FOCAL_GAMMA="2.0"

FOLD_CSV = "fold_df_fixed.csv"
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_nchc_focal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


# (spec, backbone, target_size, batch) — the top ConcatSpec carriers, baseline recipe
CARRIERS = [
    ("superlet",     "maxvit_tiny_tf_384.in1k",          384, 24),   # #1 NM weight
    ("cwt_paul",     "maxvit_rmlp_tiny_rw_256.sw_in1k",   256, 48),   # 16%
    ("cwt",          "maxvit_rmlp_tiny_rw_256.sw_in1k",   256, 48),   # 12%
    ("cwt_filtered", "maxvit_rmlp_tiny_rw_256.sw_in1k",   256, 48),   # 8%
]


def make_job(spec, bb, ts, bs, fold, pos_weight, eta):
    bb_safe = bb.replace("/", "-").replace(".", "_")
    suffix = "" if pos_weight else "_nopw"
    prefix = f"concat_{spec}_focal{suffix}_fold__{bb_safe}"
    cmd = [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
           "--spec-type", spec, "--backbone", bb, "--target-size", str(ts),
           "--fold-csv", FOLD_CSV, "--fold-idx", str(fold), "--seed", "0",
           "--stage1-epochs", "10", "--stage2-epochs", "30",
           "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
           "--batch-size", str(bs), "--eval-batch-size", "96",
           "--mixup-alpha", "0.4", "--mixup-prob", "0.5",
           "--loss", "focal", "--focal-gamma", FOCAL_GAMMA,
           "--exp-prefix", prefix]
    if not pos_weight:
        cmd.append("--no-pos-weight")
    ck = Path(REPO) / "neuromm26_results" / "checkpoints" / f"{prefix}__fold{fold}__seed0" / "best.pt"
    oof = Path(REPO) / "neuromm26_results" / "predictions" / f"{prefix}__fold{fold}__seed0_oof.npz"
    return {"id": f"{prefix}__fold{fold}", "eta_min": eta, "cmd": cmd,
            "done": ck.exists() and oof.exists()}


jobs = []
if PILOT:
    # 1 carrier (ConcatSuperlet maxvit384) fold0, both ±pos_weight = 2 jobs
    spec, bb, ts, bs = CARRIERS[0]
    jobs.append(make_job(spec, bb, ts, bs, 0, pos_weight=True,  eta=200))
    jobs.append(make_job(spec, bb, ts, bs, 0, pos_weight=False, eta=200))
    log("PILOT mode: ConcatSuperlet maxvit384 fold0, focal gamma 2.0, ±pos_weight (2 jobs)")
else:
    for (spec, bb, ts, bs) in CARRIERS:
        eta = 200 if ts == 384 else 110
        for f in range(5):
            jobs.append(make_job(spec, bb, ts, bs, f, pos_weight=bool(POS_WEIGHT), eta=eta))
    log(f"FULL rollout: {len(CARRIERS)} carriers × 5 folds, focal gamma 2.0, "
        f"pos_weight={'on' if POS_WEIGHT else 'off'}")

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
