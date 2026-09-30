"""Pseudo-label campaign. Retrains the top @256
concat arches on fold_df_pseudo (real train + 7794 high-confidence pseudo-labeled candidates,
fold=-1 => always train, never val => OOF still validates on REAL labels). GPU 1,2 ONLY.

After training: honest_add_check on REAL OOF (must not degrade real-label generalization), then
mark done so the submission (pseudo arches replacing originals + regnm) can be built. Public-LB is
the real test of pseudo-labeling; the OOF gate just guards against real-data regression.
"""
import os
import subprocess
import time
from pathlib import Path

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
PY = "python"
RES = REPO / "neuromm26_results"; PRED = RES / "predictions"
REPORT = REPO / "WAKEUP_REPORT_pseudo.md"
GPUS = [1, 2]
FOLD_CSV = "fold_df_pseudo.csv"
# (spec, prefix) — retrain the top @256 contributors on the pseudo-augmented set
ARCHS = [("cwt", "concat_cwt_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k"),
         ("cwt_paul", "concat_cwt_paul_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k")]


def log(m):
    print(f"[{time.strftime('%m-%d %H:%M')}] {m}", flush=True)


def train_5fold(spec, prefix):
    todo = [f for f in range(5) if not (PRED / f"{prefix}__fold{f}__seed0_oof.npz").exists()]
    procs = {g: None for g in GPUS}; q = list(todo)
    while q or any(procs.values()):
        for g in GPUS:
            if procs[g] is not None and procs[g].poll() is not None:
                procs[g] = None
        for g in GPUS:
            if procs[g] is None and q:
                f = q.pop(0)
                cmd = [PY, "-m", "neuromm26_baseline.tools.train_spec_concat_fold",
                       "--spec-type", spec, "--backbone", "maxvit_rmlp_tiny_rw_256.sw_in1k",
                       "--target-size", "256", "--fold-idx", str(f), "--seed", "0",
                       "--fold-csv", FOLD_CSV, "--loss", "focal", "--focal-gamma", "2.0",
                       "--stage1-epochs", "10", "--stage2-epochs", "30",
                       "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
                       "--batch-size", "64", "--eval-batch-size", "96",
                       "--noise-sids", "pseudo_sids.txt", "--noise-weight", "0.4",  # down-weight pseudo
                       "--exp-prefix", prefix]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                lp = RES / "logs" / f"pseudo_{prefix}_fold{f}.log"; lp.parent.mkdir(parents=True, exist_ok=True)
                procs[g] = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=open(lp, "w"), stderr=subprocess.STDOUT)
                log(f"GPU{g} pseudo-train {prefix} fold{f}")
        time.sleep(30)


def add_check(prefix):
    r = subprocess.run([PY, "scripts/honest_add_check.py", prefix], cwd=str(REPO),
                       env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                       capture_output=True, text=True, timeout=3600)
    return (r.stdout or "") + (r.stderr or "")


lines = [f"# Pseudo-label campaign (start {time.strftime('%Y-%m-%d %H:%M')})\n",
         "Retrained ConcatCWT + ConcatPaul maxvit256 on real train + 7794 high-confidence pseudo "
         "candidates (2763 pos / 5031 neg). val = REAL labels only (honest OOF).\n"]
REPORT.write_text("\n".join(lines))

log("waiting for rank-loss to free GPU 1,2 (_logs_/GATE_RANK_DONE) ...")
while not (REPO / "_logs_/GATE_RANK_DONE").exists():
    time.sleep(120)

for spec, prefix in ARCHS:
    log(f"=== pseudo retrain {prefix} ===")
    train_5fold(spec, prefix)
    out = add_check(prefix)
    log(out.strip().splitlines()[-1] if out.strip() else "(no output)")
    lines.append(f"## {prefix}\n```\n{out.strip()}\n```\n")
    REPORT.write_text("\n".join(lines))

lines.append("\n**PSEUDO TRAINING DONE.** Next (auto on re-invoke): append candidate logits for the "
             "pseudo arches, regnm with them REPLACING originals -> pseudo submission -> user submits to "
             "test public LB. add-check above is the REAL-label guard (pseudo must not wreck real OOF).\n")
REPORT.write_text("\n".join(lines))
(REPO / "_logs_/PSEUDO_TRAIN_DONE").touch()
log("PSEUDO TRAINING COMPLETE")
