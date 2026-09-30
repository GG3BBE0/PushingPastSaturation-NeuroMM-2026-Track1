"""Self-driving overnight campaign on GPU 1,2 ONLY (policy). Waits for the running rank-loss to
free the GPUs, then trains a QUEUE of new-paradigm experiments 5-fold, honest-gates each, and
writes a cumulative WAKEUP_REPORT_overnight.md. A ROBUST+ add-check is flagged as SUBMIT-WORTHY.

Each experiment reuses the proven ConcatSpec MaxViT pipeline but changes the INPUT (orthogonal
phase) or the OBJECTIVE (AUPRC ranking) -> genuinely different from the BCE-spectrogram pool.
"""
import os
import subprocess
import time
from pathlib import Path

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
PY = "python"
RES = REPO / "neuromm26_results"; PRED = RES / "predictions"
REPORT = REPO / "WAKEUP_REPORT_overnight.md"
GPUS = [1, 2]  # POLICY: ONLY 1,2
PHASE_CACHE = REPO / "neuromm26_datasets/processed/features/cwt_phase"

# (label, spec, loss, prefix, extra_args)
EXPERIMENTS = [
    ("phase-CWT MaxViT (orthogonal: phase not magnitude)", "cwt_phase", "focal",
     "concat_phase_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", []),
    ("rank-loss STFT (AUPRC surrogate)", "stft", "rank",
     "concat_stft_rank_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", []),
    ("rank-loss Filt (AUPRC surrogate)", "cwt_filtered", "rank",
     "concat_filt_rank_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", []),
]


def log(m):
    print(f"[{time.strftime('%m-%d %H:%M')}] {m}", flush=True)


def train_5fold(spec, loss, prefix, extra):
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
                       "--target-size", "256", "--fold-idx", str(f), "--seed", "0", "--loss", loss,
                       "--stage1-epochs", "10", "--stage2-epochs", "30",
                       "--stage1-lr", "7e-4", "--stage2-lr", "7e-5",
                       "--batch-size", "64", "--eval-batch-size", "96", "--exp-prefix", prefix] + extra
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                lp = RES / "logs" / f"overnight_{prefix}_fold{f}.log"; lp.parent.mkdir(parents=True, exist_ok=True)
                procs[g] = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                            stdout=open(lp, "w"), stderr=subprocess.STDOUT)
                log(f"GPU{g} train {prefix} fold{f}")
        time.sleep(30)


def add_check(prefix):
    r = subprocess.run([PY, "scripts/honest_add_check.py", prefix], cwd=str(REPO),
                       env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                       capture_output=True, text=True, timeout=3600)
    return (r.stdout or "") + (r.stderr or "")


def write(lines):
    REPORT.write_text("\n".join(lines))


lines = [f"# Overnight campaign — target public 0.985 (started {time.strftime('%Y-%m-%d %H:%M')})\n",
         "GPU 1,2 only. Each arch reuses the MaxViT pipeline with a NEW input (phase) or objective (rank).\n"]
write(lines)

log("waiting for rank-loss to free GPU 1,2 (_logs_/GATE_RANK_DONE) ...")
while not (REPO / "_logs_/GATE_RANK_DONE").exists():
    time.sleep(120)
rg = REPO / "_logs_/gate_rank.log"
lines.append("## rank-loss (AUPRC surrogate, ConcatCWT+Paul) result\n```\n"
             + (rg.read_text()[-1800:] if rg.exists() else "(no log)") + "\n```\n")
write(lines)

for label, spec, loss, prefix, extra in EXPERIMENTS:
    if spec == "cwt_phase":
        log("waiting for cwt_phase cache ...")
        while len(list(PHASE_CACHE.glob("*.npy"))) < 25000:
            time.sleep(60)
    log(f"=== {label} ===")
    try:
        train_5fold(spec, loss, prefix, extra)
        out = add_check(prefix)
    except Exception as e:
        out = f"FAILED: {e}"
    robust = "ROBUST+" in out
    flag = "  ⚠️ **ROBUST+ → SUBMIT-WORTHY, verify with gate_decisive + build submission**" if robust else ""
    log(out.strip().splitlines()[-1] if out.strip() else "(no output)")
    lines.append(f"## {label}{flag}\n```\n{out.strip()}\n```\n")
    write(lines)

lines.append("\n**DONE — re-invoke me to queue the next wave toward 0.985** "
             "(connectivity→MaxViT, contrastive-SSL, new backbones, stacking).\n")
write(lines)
(REPO / "_logs_/OVERNIGHT_DONE").touch()
log("ALL OVERNIGHT EXPERIMENTS DONE")
