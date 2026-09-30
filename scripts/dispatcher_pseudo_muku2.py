"""muku-pseudo ORTHOGONAL expansion: add diverse backbones to the time-domain muku family on
round-1 pseudo (the proven conservative recipe). muku-pseudo is surprisingly strong (~0.85) AND
orthogonal to every spectrogram arch -> a few diverse strong muku members push the ensemble in a
NEW direction (the one lever that beats blend-dampening). New backbones: effv2s_b0 (raw+filt) +
mobilenetv3 (raw). GPU 1,2 ONLY. round-1 labels, weight 0.4.
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
EEG = "neuromm26_datasets/processed/features/eeg"
FILT = "neuromm26_datasets/processed/features/filt_eeg"
# (backbone, eeg_root, exp_prefix)
ARCHS = [("tf_efficientnet_b0.ns_jft_in1k",            EEG,  "muku_pseudo_fold__tf_efficientnet_b0_ns_jft_in1k"),
         ("mobilenetv3_large_100.ra4_e3600_r224_in1k", EEG,  "muku_pseudo_fold__mobilenetv3_large_100_ra4_in1k"),
         ("tf_efficientnet_b0.ns_jft_in1k",            FILT, "muku_filteeg_pseudo_fold__tf_efficientnet_b0_ns_jft_in1k")]

logdir = RES / f"logs/disp_pseudo_muku2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logdir.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    (logdir / "MASTER.log").open("a").write(f"{m}\n")


def done(prefix, f):
    return (RES / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists()


jobs = [{"id": f"{prefix}__fold{f}", "bb": bb, "root": root, "prefix": prefix, "fold": f}
        for bb, root, prefix in ARCHS for f in range(5) if not done(prefix, f)]
gp = {g: None for g in GPUS}; gj = {g: None for g in GPUS}


def launch(g, jb):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [PY, "-m", "neuromm26_baseline.tools.train_muku_fold",
           "--backbone", jb["bb"], "--eeg-feature-root", jb["root"],
           "--fold-idx", str(jb["fold"]), "--seed", "0", "--fold-csv", "fold_df_pseudo.csv",
           "--loss", "focal", "--focal-gamma", "2.0",
           "--stage1-epochs", "10", "--stage2-epochs", "30", "--stage1-lr", "1e-3", "--stage2-lr", "1e-4",
           "--batch-size", "128", "--eval-batch-size", "128",
           "--noise-sids", "pseudo_sids.txt", "--noise-weight", "0.4", "--exp-prefix", jb["prefix"]]
    log(f"GPU {g} launching {jb['id']}")
    gp[g] = subprocess.Popen(cmd, cwd=REPO, env=env,
                             stdout=open(logdir / f"gpu{g}__{jb['id']}.log", "w"), stderr=subprocess.STDOUT)
    gj[g] = jb["id"]


def chk(g):
    p = gp[g]
    if p is not None and p.poll() is not None:
        log(f"GPU {g} freed ('{gj[g]}' exit={p.returncode})"); gp[g] = None; gj[g] = None


log(f"muku-pseudo orthogonal expansion: {len(jobs)} jobs on GPU {GPUS}")
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
