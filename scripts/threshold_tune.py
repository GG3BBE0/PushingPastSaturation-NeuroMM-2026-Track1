"""Search F1-optimal classification threshold on val for binary tasks (T1, T2).

The default 0.5 threshold may not be optimal under class imbalance
(positive_ratio ≈ 9.6%). This script loads the ensembled (4-seed) probabilities
on val and sweeps thresholds in [0.05, 0.95] to find the one that maximizes
binary F1, then reports metrics at:
  - default 0.5
  - best-F1 threshold
  - best-balanced-accuracy threshold (for reference)

Usage:
    python scripts/threshold_tune.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets import EEGFeatureDataset, NeuroMMMultimodalFeatureDataset
from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.models import build_legacy_eeg_model
from neuromm26_baseline.models.multimodal_late_fusion import EEGVideoLateFusion


REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
MANIFEST = REPO / "neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv"
EEG_FEATURE_ROOT = REPO / "neuromm26_datasets/processed/features/eeg"
VIDEO_FEATURE_ROOT = REPO / "neuromm26_datasets/processed/features/video"
CKPT_DIR = REPO / "neuromm26_results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (0, 1, 2, 3)


def collect_probs_from_ckpts(ckpt_paths, build_model_fn, loader):
    """Load each ckpt, run val, return averaged probabilities and labels."""
    all_logits = []
    labels = None
    for cp in ckpt_paths:
        state = torch.load(cp, map_location=DEVICE, weights_only=False)
        model = build_model_fn().to(DEVICE)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        seed_logits = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                logits = model(batch).view(-1).cpu()
                seed_logits.append(logits)
                if labels is None or len(labels) < 5128:
                    pass  # we collect labels in second pass below
        all_logits.append(torch.cat(seed_logits))
    # labels (one pass)
    lb = []
    for batch in loader:
        lb.append(batch["label"].view(-1).float())
    labels = torch.cat(lb)
    avg_logits = torch.stack(all_logits).mean(dim=0)
    return torch.sigmoid(avg_logits).numpy(), labels.numpy().astype(int)


def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "threshold": thr, "f1": f1, "precision": precision, "recall": recall,
        "specificity": specificity, "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def sweep(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    return [metrics_at_threshold(probs, labels, float(t)) for t in thresholds]


def report_task(name: str, probs: np.ndarray, labels: np.ndarray):
    print(f"\n========== Threshold tuning — {name} ==========")
    thresholds = np.arange(0.05, 0.95 + 1e-9, 0.01)
    rows = sweep(probs, labels, thresholds)
    by_f1 = max(rows, key=lambda r: r["f1"])
    by_bacc = max(rows, key=lambda r: r["balanced_accuracy"])
    at05 = metrics_at_threshold(probs, labels, 0.5)
    print(f"  threshold=0.50 (default)  F1={at05['f1']:.4f}  P={at05['precision']:.4f}  R={at05['recall']:.4f}  bAcc={at05['balanced_accuracy']:.4f}")
    print(f"  best-F1 @ {by_f1['threshold']:.2f}        F1={by_f1['f1']:.4f}  P={by_f1['precision']:.4f}  R={by_f1['recall']:.4f}  bAcc={by_f1['balanced_accuracy']:.4f}")
    print(f"  best-bAcc @ {by_bacc['threshold']:.2f}      F1={by_bacc['f1']:.4f}  P={by_bacc['precision']:.4f}  R={by_bacc['recall']:.4f}  bAcc={by_bacc['balanced_accuracy']:.4f}")
    print(f"  → F1 lift from default: {by_f1['f1'] - at05['f1']:+.4f}")
    return {"default": at05, "best_f1": by_f1, "best_bacc": by_bacc}


def main():
    results = {}

    # Task 1 — tcnet_eeg
    ds = EEGFeatureDataset(
        split_csv=None, manifest_csv=str(MANIFEST), split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        preload_in_memory=True, target_shape=(26, 2000),
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    paths = [CKPT_DIR / f"neuromm26_train_eeg_only__legacy-tcnet_eeg__seed{s}__lr0.0005/best.pt" for s in SEEDS]
    probs, labels = collect_probs_from_ckpts(paths, lambda: build_legacy_eeg_model("tcnet_eeg"), loader)
    results["T1"] = report_task("Task 1 (tcnet_eeg ensemble)", probs, labels)

    # Task 2 — eegnet + timesformer-k400
    ds = NeuroMMMultimodalFeatureDataset(
        manifest_csv=str(MANIFEST), split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        video_feature_root=str(VIDEO_FEATURE_ROOT),
        video_feature_name="timesformer-k400",
        preload_in_memory=True, target_shape=(26, 2000), require_video_feature=True,
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    sample_npy = next((VIDEO_FEATURE_ROOT / "timesformer-k400").glob("*.npy"))
    feat_dim = np.load(sample_npy).shape[-1]
    paths = [CKPT_DIR / f"eeg_video_fusion__eegnet__timesformer-k400__seed{s}__lr0.0005/best.pt" for s in SEEDS]

    def build_t2():
        return EEGVideoLateFusion(eeg_model_name="eegnet", video_feature_dim=feat_dim,
                                  video_hidden_dim=256, video_dropout=0.2)
    probs, labels = collect_probs_from_ckpts(paths, build_t2, loader)
    results["T2"] = report_task("Task 2 (eegnet+timesformer-k400 ensemble)", probs, labels)

    out = REPO / "neuromm26_results/metrics/threshold_tune_summary.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
