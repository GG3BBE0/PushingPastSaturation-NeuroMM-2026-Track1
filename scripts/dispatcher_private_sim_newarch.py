"""PRIVATE-SAFETY SIM for the TWO NEW specnchc members (superlet384 + muku), the only specnchc arches
NOT yet covered by the cwt ensemble-teacher sim (+0.0318). Same mechanism: ensemble teacher pseudo-labels
held-out fold-k (reuse fold_df_simens_{k}.csv), train the arch as a specialist (fold=-1 always-train
pseudo-copies @ full trust), eval on fold-k TRUE labels. Δ = specialist − canonical = does swapping the
specnchc member for its canonical counterpart HELP or HURT the held-out (private-proxy) true score.

Recipe = the specnchc member's loss (focal) on the canonical backbone/lr (so Δ = the real swap effect on
the blind spot). Baseline = existing canonical OOF (superlet384 / muku raw). GPU 1-5, folds [4,1,0]
(4=hard/label-noise worst case, 1=was-negative-single-arch, 0=clean). superlet@384 ~6.7h/fold, muku ~35min.

Run: python scripts/dispatcher_private_sim_newarch.py
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
import os
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
GPUS = [1, 2, 3, 4, 5]
RES = Path(REPO) / "neuromm26_results"
EEG = "neuromm26_datasets/processed/features/eeg"
FOLDS = [4, 1, 0]

logdir = RES / f"logs/disp_private_sim_newarch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logdir.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    (logdir / "MASTER.log").open("a").write(f"{m}\n")


# 1) symlink __PSens pseudo-copies into superlet + eeg feature dirs for each fold
for fdir in ("superlet", "eeg"):
    for k in FOLDS:
        r = subprocess.run([PY, "scripts/symlink_sim_features.py", fdir, str(k)],
                           cwd=REPO, capture_output=True, text=True)
        log(f"symlink {fdir} f{k}: {(r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr).strip() else 'ok'}")


def sup_cmd(k):
    return [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
            "--spec-type", "superlet", "--backbone", "maxvit_tiny_tf_384.in1k", "--target-size", "384",
            "--fold-idx", str(k), "--seed", "0", "--fold-csv", f"fold_df_simens_{k}.csv",
            "--loss", "focal", "--focal-gamma", "2.0",
            "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "8e-4", "--stage2-lr", "8e-5",
            "--batch-size", "24", "--eval-batch-size", "32", "--exp-prefix", f"superlet_simensfold{k}"]


def muk_cmd(k):
    return [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
            "--backbone", "convnext_pico.d1_in1k", "--eeg-feature-root", EEG,
            "--fold-idx", str(k), "--seed", "0", "--fold-csv", f"fold_df_simens_{k}.csv",
            "--loss", "focal", "--focal-gamma", "2.0",
            "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
            "--batch-size", "64", "--eval-batch-size", "64", "--exp-prefix", f"muku_simensfold{k}"]


# 2) build job queue (longest eta first: superlet @384, then muku), skip done
jobs = []
for k in FOLDS:
    if not (RES / "predictions" / f"superlet_simensfold{k}__fold{k}__seed0_oof.npz").exists():
        jobs.append({"tag": f"sup_f{k}", "cmd": sup_cmd(k)})
for k in FOLDS:
    if not (RES / "predictions" / f"muku_simensfold{k}__fold{k}__seed0_oof.npz").exists():
        jobs.append({"tag": f"muk_f{k}", "cmd": muk_cmd(k)})

gp = {g: None for g in GPUS}; gtag = {g: None for g in GPUS}


def launch(g, jb):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(f"GPU {g} launching {jb['tag']}")
    gp[g] = subprocess.Popen(jb["cmd"], cwd=REPO, env=env,
                             stdout=open(logdir / f"gpu{g}__{jb['tag']}.log", "w"), stderr=subprocess.STDOUT)
    gtag[g] = jb["tag"]


def chk(g):
    p = gp[g]
    if p is not None and p.poll() is not None:
        log(f"GPU {g} freed ({gtag[g]} exit={p.returncode})"); gp[g] = None; gtag[g] = None


log(f"private-sim newarch: {len(jobs)} jobs ({[j['tag'] for j in jobs]}) on GPU {GPUS}")
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
