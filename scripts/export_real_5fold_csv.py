"""Export REAL 5-fold final report as CSV files."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
PRED_DIR = REPO / "neuromm26_results/predictions"
LOG_DIR = REPO / "neuromm26_results/logs"
OUT_DIR = REPO / "neuromm26_results/csv_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("muku_fold__resnet18", "muku raw resnet18"),
    ("muku_fold__tf_efficientnet_b0_ns_jft_in1k", "muku raw effv2s_b0"),
    ("muku_fold__convnext_pico_d1_in1k", "muku raw convnext_pico"),
    ("spec_cwt_fold__resnet18", "CWT-Morlet resnet18"),
    ("spec_cwt_fold__tf_efficientnet_b0_ns_jft_in1k", "CWT-Morlet effv2s_b0"),
    ("spec_cwt_fold__convnext_pico_d1_in1k", "CWT-Morlet convnext_pico"),
    ("concat_cwt_fold__convnext_tiny_fb_in22k_ft_in1k_384", "V2 ConcatCWT convnext_tiny"),
]

# ===== 1. Per-fold per-model AUPRC =====
print("Writing per-fold table...")
per_fold_rows = []
for prefix, name in MODELS:
    row = {"model": name}
    for f in range(5):
        log_path = LOG_DIR / f"{prefix}__fold{f}__seed0.log"
        v = ""
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                if "DONE best_auprc" in line:
                    v = line.split("best_auprc=")[1].split()[0]
                    break
        row[f"fold_{f}"] = v
    # Compute mean/std
    vals = [float(row[f"fold_{f}"]) for f in range(5) if row[f"fold_{f}"]]
    if vals:
        row["mean"] = f"{np.mean(vals):.4f}"
        row["std"] = f"{np.std(vals):.4f}"
    per_fold_rows.append(row)

with (OUT_DIR / "real_5fold_per_model_per_fold.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "mean", "std"])
    w.writeheader()
    w.writerows(per_fold_rows)
print(f"  → {OUT_DIR / 'real_5fold_per_model_per_fold.csv'}")


# ===== 2. Per-model OOF AUPRC (concatenated 25426) =====
print("Computing OOF per model...")
SEEDS_SUFFIX = "__seed0_oof.npz"

def load_model_oof(prefix):
    all_sids, all_logits, all_labels = [], [], []
    for f in range(5):
        path = PRED_DIR / f"{prefix}__fold{f}{SEEDS_SUFFIX}"
        if not path.exists():
            return None, None, None
        data = np.load(path)
        all_sids.append(data["sample_ids"])
        all_logits.append(data["logits"])
        all_labels.append(data["labels"])
    return np.concatenate(all_sids), np.concatenate(all_logits), np.concatenate(all_labels)

model_oof = {}
ref_sids = None
oof_rows = []
for prefix, name in MODELS:
    sids, logits, labels = load_model_oof(prefix)
    if sids is None:
        continue
    if ref_sids is None:
        ref_sids = sids
    else:
        if not np.array_equal(sids, ref_sids):
            idx_map = {s: i for i, s in enumerate(sids)}
            perm = np.array([idx_map[s] for s in ref_sids])
            logits = logits[perm]
            labels = labels[perm]
    m = compute_binary_classification_metrics(torch.from_numpy(logits), torch.from_numpy(labels))
    model_oof[name] = {"logits": logits, "labels": labels}
    oof_rows.append({
        "model": name,
        "OOF_AUPRC": f"{m['auprc']:.4f}",
        "OOF_F1@0.5": f"{m['binary_f1']:.4f}",
        "OOF_accuracy": f"{m['accuracy']:.4f}",
        "OOF_macro_f1": f"{m['macro_f1']:.4f}",
        "OOF_balanced_acc": f"{m['balanced_accuracy']:.4f}",
    })

with (OUT_DIR / "real_5fold_per_model_OOF.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "OOF_AUPRC", "OOF_F1@0.5", "OOF_accuracy", "OOF_macro_f1", "OOF_balanced_acc"])
    w.writeheader()
    w.writerows(oof_rows)
print(f"  → {OUT_DIR / 'real_5fold_per_model_OOF.csv'}")


# ===== 3. Ensemble combinations =====
print("Ensemble combinations...")
ref_y = torch.from_numpy(list(model_oof.values())[0]["labels"])

def sweep_thr(probs, Y):
    best = {"f1": -1, "thr": 0.5, "p": 0, "r": 0}
    for thr in np.arange(0.05, 0.95 + 1e-9, 0.01):
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (Y == 1)).sum())
        fp = int(((pred == 1) & (Y == 0)).sum())
        fn = int(((pred == 0) & (Y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": f1, "p": p, "r": r}
    return best


def p_at_recall(probs, Y, target=0.70):
    n_pos = int(Y.sum())
    order = np.argsort(-probs)
    sl = Y[order]
    tp_cum = np.cumsum(sl)
    recall = tp_cum / n_pos
    precision = tp_cum / np.maximum(np.arange(1, len(Y) + 1), 1)
    above = np.where(recall >= target)[0]
    return float(precision[above[0]]) if len(above) > 0 else 0.0


def ens_row(name, model_names):
    L = [torch.from_numpy(model_oof[n]["logits"]) for n in model_names if n in model_oof]
    if not L:
        return None
    avg = torch.stack(L).mean(dim=0)
    m = compute_binary_classification_metrics(avg, ref_y)
    probs = torch.sigmoid(avg).numpy()
    Y = ref_y.numpy().astype(int)
    thr = sweep_thr(probs, Y)
    ps70 = p_at_recall(probs, Y, 0.70)
    ps80 = p_at_recall(probs, Y, 0.80)
    return {
        "ensemble": name,
        "N_models": len(L),
        "AUPRC": f"{m['auprc']:.4f}",
        "F1@0.5": f"{m['binary_f1']:.4f}",
        "F1@best_thr": f"{thr['f1']:.4f}",
        "best_thr": f"{thr['thr']:.2f}",
        "precision@best_thr": f"{thr['p']:.4f}",
        "recall@best_thr": f"{thr['r']:.4f}",
        "P@S70": f"{ps70:.4f}",
        "P@S80": f"{ps80:.4f}",
        "accuracy": f"{m['accuracy']:.4f}",
    }


muku_names = ["muku raw resnet18", "muku raw effv2s_b0", "muku raw convnext_pico"]
cwt_names = ["CWT-Morlet resnet18", "CWT-Morlet effv2s_b0", "CWT-Morlet convnext_pico"]
v2_names = ["V2 ConcatCWT convnext_tiny"]

combos = [
    ("muku raw (3 backbones)", muku_names),
    ("CWT-Morlet (3 backbones)", cwt_names),
    ("V2 ConcatCWT alone", v2_names),
    ("muku + CWT (6 models)", muku_names + cwt_names),
    ("muku + V2 (4 models)", muku_names + v2_names),
    ("CWT + V2 (4 models)", cwt_names + v2_names),
    ("ALL 7 models (muku+CWT+V2)", muku_names + cwt_names + v2_names),
]
ens_rows = [ens_row(n, ms) for n, ms in combos]
ens_rows = [r for r in ens_rows if r]

with (OUT_DIR / "real_5fold_ensemble_combinations.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ens_rows[0].keys()))
    w.writeheader()
    w.writerows(ens_rows)
print(f"  → {OUT_DIR / 'real_5fold_ensemble_combinations.csv'}")


# ===== 4. Fake vs Real comparison =====
print("Fake vs Real comparison...")
FAKE_PER_FOLD = {
    "muku raw resnet18":          [0.9874, 0.8639, 0.9668, 0.9762, 0.9838],
    "muku raw effv2s_b0":         [0.9872, 0.8646, 0.9807, 0.9822, 0.9973],
    "muku raw convnext_pico":     [None,   0.8707, 0.9725, None,   0.9971],
    "CWT-Morlet resnet18":        [0.9767, 0.8687, 0.9831, 0.9828, 0.9994],
    "CWT-Morlet effv2s_b0":       [0.9804, 0.8800, 0.9889, 0.9734, 0.9926],
    "CWT-Morlet convnext_pico":   [0.9862, None,   None,   0.9730, 0.9951],
    "V2 ConcatCWT convnext_tiny": [0.9904, 0.9106, 0.9860, 0.9703, 0.9921],
}

cmp_rows = []
for row in per_fold_rows:
    name = row["model"]
    real_vals = [float(row[f"fold_{f}"]) for f in range(5) if row[f"fold_{f}"]]
    fake_vals = [v for v in FAKE_PER_FOLD.get(name, []) if v is not None]
    real_mean = np.mean(real_vals) if real_vals else None
    fake_mean = np.mean(fake_vals) if fake_vals else None
    inflation = fake_mean - real_mean if (real_mean and fake_mean) else None
    cmp_rows.append({
        "model": name,
        "fake_fold_mean": f"{fake_mean:.4f}" if fake_mean is not None else "",
        "real_fold_mean": f"{real_mean:.4f}" if real_mean is not None else "",
        "leak_inflation": f"{inflation:+.4f}" if inflation is not None else "",
    })

with (OUT_DIR / "fake_vs_real_fold_comparison.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "fake_fold_mean", "real_fold_mean", "leak_inflation"])
    w.writeheader()
    w.writerows(cmp_rows)
print(f"  → {OUT_DIR / 'fake_vs_real_fold_comparison.csv'}")


# ===== 5. Summary stats =====
print("\nSummary (best of each):")
best_auprc = max(ens_rows, key=lambda r: float(r["AUPRC"]))
best_f1 = max(ens_rows, key=lambda r: float(r["F1@best_thr"]))
best_p70 = max(ens_rows, key=lambda r: float(r["P@S70"]))
print(f"  Best AUPRC:    {best_auprc['ensemble']} = {best_auprc['AUPRC']}")
print(f"  Best F1@thr:   {best_f1['ensemble']} = {best_f1['F1@best_thr']} @ thr={best_f1['best_thr']}")
print(f"  Best P@S70:    {best_p70['ensemble']} = {best_p70['P@S70']}")

print("\nAll 4 CSV files written to:")
print(f"  {OUT_DIR}/")
