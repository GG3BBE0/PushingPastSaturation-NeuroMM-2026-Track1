"""Heavy-aug rollout #1: ConcatFilt maxvit256 × 5 folds sequential on GPU 2.

Baseline ConcatFilt maxvit256 single-arch OOF: 0.8298 (NM weight 11.33% rank #2).
Heavy aug params copy from successful ConcatSuperlet heavyaug ablation.

Sequential (1 GPU): 5 folds × ~2.5h each ≈ 12-15h.
"""
from __future__ import annotations
import os, subprocess, time
from datetime import datetime
import os
from pathlib import Path

REPO = os.environ.get("NEUROMM_REPO", str(Path(__file__).resolve().parent.parent))
PY = "python"
FOLD_CSV = "fold_df_fixed.csv"
GPU = 2
LOGDIR = Path(REPO) / f"neuromm26_results/logs/disp_heavyaug_filt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER = LOGDIR / "MASTER.log"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MASTER.open("a") as f:
        f.write(line + "\n")


PREFIX = "concat_cwt_filtered_heavyaug_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"


def cmd(fold):
    return [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
            "--spec-type", "cwt_filtered",
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
            "--exp-prefix", PREFIX]


log(f"Heavy aug rollout #1 starting: {PREFIX} on GPU {GPU} (sequential 5 folds)")
for f in range(5):
    log(f"--- fold {f} starting ---")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    lp = LOGDIR / f"gpu{GPU}__{PREFIX}__fold{f}.log"
    t0 = time.time()
    p = subprocess.Popen(cmd(f), cwd=REPO, env=env, stdout=lp.open("w"), stderr=subprocess.STDOUT)
    p.wait()
    dt = (time.time() - t0) / 60
    log(f"fold {f} done (exit={p.returncode}, took {dt:.1f}min)")

log("ALL DONE")
(LOGDIR / "ALL_DONE").touch()
