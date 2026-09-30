"""Test-time augmentation evaluator.

For each val sample, run forward pass K times with different EEG augmentations,
then average logits before computing metrics.

Augmentations (EEG only — video features are pre-extracted and not augmented):
  - identity
  - gaussian noise (sigma=0.01)
  - time shift (±25 timesteps via torch.roll)
  - channel-wise scaling (per-channel factor uniform in [0.95, 1.05])
  - amplitude scaling (global factor uniform in [0.95, 1.05])

Usage:
    python scripts/tta_eval.py --task T1
    python scripts/tta_eval.py --task T2
    python scripts/tta_eval.py --task T3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets import EEGFeatureDataset, NeuroMMMultimodalFeatureDataset
from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.datasets.task3_datasets import Task3MultimodalFeatureDataset
from neuromm26_baseline.models import build_legacy_eeg_model
from neuromm26_baseline.models.multimodal_late_fusion import EEGVideoLateFusion
from neuromm26_baseline.models.multiclass_late_fusion import EEGVideoLateFusion5Class
from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics
from neuromm26_baseline.utils.multiclass_metrics import compute_multiclass_metrics


REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
MANIFEST = REPO / "neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv"
EEG_FEATURE_ROOT = REPO / "neuromm26_datasets/processed/features/eeg"
VIDEO_FEATURE_ROOT = REPO / "neuromm26_datasets/processed/features/video"
CKPT_DIR = REPO / "neuromm26_results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (0, 1, 2, 3)


def make_augs():
    """Return list of (name, fn) where fn maps an EEG tensor (B, C, T) → (B, C, T)."""
    def identity(x):
        return x

    def gaussian(x, sigma=0.01):
        return x + torch.randn_like(x) * sigma

    def time_shift(x, max_shift=25):
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        return torch.roll(x, shifts=shift, dims=-1)

    def channel_scale(x, low=0.95, high=1.05):
        scale = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) * (high - low) + low
        return x * scale

    def amp_scale(x, low=0.95, high=1.05):
        factor = float(torch.rand(1).item()) * (high - low) + low
        return x * factor

    return [
        ("identity", identity),
        ("gaussian", gaussian),
        ("time_shift", time_shift),
        ("channel_scale", channel_scale),
        ("amp_scale", amp_scale),
    ]


def collect_logits_with_aug(model, loader, augs, num_classes: int | None) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Run forward pass K times (one per aug) and average logits per sample."""
    model.eval()
    accum_logits = None
    labels_all = []
    sids_all = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            eeg_orig = batch["eeg"].clone()
            per_aug_logits = []
            for _, aug_fn in augs:
                batch["eeg"] = aug_fn(eeg_orig)
                logits = model(batch)
                if num_classes is None:
                    logits = logits.view(-1)
                else:
                    logits = logits.view(-1, num_classes)
                per_aug_logits.append(logits.cpu())
            avg = torch.stack(per_aug_logits).mean(dim=0)
            if accum_logits is None:
                accum_logits = [avg]
            else:
                accum_logits.append(avg)
            labels_all.append(batch["label"].cpu())
            sids_all.extend(batch["sample_id"])
    return sids_all, torch.cat(accum_logits), torch.cat(labels_all)


def detect_video_feature_dim(name: str) -> int:
    root = VIDEO_FEATURE_ROOT / name
    arr = np.load(next(root.glob("*.npy")))
    return int(arr.shape[-1]) if arr.ndim > 1 else int(arr.shape[0])


# ---- task-specific builders ----
def build_task1_dataset():
    return EEGFeatureDataset(
        split_csv=None, manifest_csv=str(MANIFEST), split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        preload_in_memory=True, target_shape=(26, 2000),
    )


def build_task2_dataset(video_name: str):
    return NeuroMMMultimodalFeatureDataset(
        manifest_csv=str(MANIFEST), split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        video_feature_root=str(VIDEO_FEATURE_ROOT),
        video_feature_name=video_name,
        preload_in_memory=True, target_shape=(26, 2000), require_video_feature=True,
    )


def build_task3_dataset(video_name: str):
    return Task3MultimodalFeatureDataset(
        manifest_csv=str(MANIFEST), split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        video_feature_root=str(VIDEO_FEATURE_ROOT),
        video_feature_name=video_name,
        target_shape=(26, 2000),
    )


def run_t1():
    print("\n========== TTA — Task 1 (tcnet_eeg) ==========")
    ds = build_task1_dataset()
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    augs = make_augs()
    print(f"  Augmentations: {[n for n, _ in augs]}")

    aligned_logits = []
    label_ref = None
    for s in SEEDS:
        state = torch.load(
            CKPT_DIR / f"neuromm26_train_eeg_only__legacy-tcnet_eeg__seed{s}__lr0.0005/best.pt",
            map_location=DEVICE, weights_only=False)
        model = build_legacy_eeg_model("tcnet_eeg").to(DEVICE)
        model.load_state_dict(state["model_state_dict"])
        _, logits, labels = collect_logits_with_aug(model, loader, augs, num_classes=None)
        label_ref = labels
        aligned_logits.append(logits)
        m = compute_binary_classification_metrics(logits, labels)
        print(f"  seed={s} TTA  auprc={m['auprc']:.4f}  f1={m['binary_f1']:.4f}")
    ens = torch.stack(aligned_logits).mean(dim=0)
    em = compute_binary_classification_metrics(ens, label_ref)
    print(f"  TTA + 4-seed ensemble  auprc={em['auprc']:.4f}  f1={em['binary_f1']:.4f}  acc={em['accuracy']:.4f}")
    return em


def run_t2():
    print("\n========== TTA — Task 2 (eegnet+timesformer-k400) ==========")
    ds = build_task2_dataset("timesformer-k400")
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    augs = make_augs()
    print(f"  Augmentations: {[n for n, _ in augs]}")

    feat_dim = detect_video_feature_dim("timesformer-k400")
    aligned_logits = []
    label_ref = None
    for s in SEEDS:
        state = torch.load(
            CKPT_DIR / f"eeg_video_fusion__eegnet__timesformer-k400__seed{s}__lr0.0005/best.pt",
            map_location=DEVICE, weights_only=False)
        model = EEGVideoLateFusion(
            eeg_model_name="eegnet", video_feature_dim=feat_dim,
            video_hidden_dim=state.get("video_hidden_dim", 256), video_dropout=0.2,
        ).to(DEVICE)
        model.load_state_dict(state["model_state_dict"])
        _, logits, labels = collect_logits_with_aug(model, loader, augs, num_classes=None)
        label_ref = labels
        aligned_logits.append(logits)
        m = compute_binary_classification_metrics(logits, labels)
        print(f"  seed={s} TTA  auprc={m['auprc']:.4f}  f1={m['binary_f1']:.4f}")
    ens = torch.stack(aligned_logits).mean(dim=0)
    em = compute_binary_classification_metrics(ens, label_ref)
    print(f"  TTA + 4-seed ensemble  auprc={em['auprc']:.4f}  f1={em['binary_f1']:.4f}  acc={em['accuracy']:.4f}")
    return em


def run_t3():
    print("\n========== TTA — Task 3 (convnext_large+videomae-large) ==========")
    ds = build_task3_dataset("videomae-large")
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    augs = make_augs()
    print(f"  Augmentations: {[n for n, _ in augs]}")

    feat_dim = detect_video_feature_dim("videomae-large")
    aligned_logits = []
    label_ref = None
    for s in SEEDS:
        state = torch.load(
            CKPT_DIR / f"task3_fusion__convnext_large_eeg__videomae-large__seed{s}__lr0.0005/best.pt",
            map_location=DEVICE, weights_only=False)
        model = EEGVideoLateFusion5Class(
            eeg_model_name="convnext_large_eeg", video_feature_dim=feat_dim,
            video_hidden_dim=state.get("video_hidden_dim", 256), video_dropout=0.2,
            num_classes=state.get("num_classes", 5),
        ).to(DEVICE)
        model.load_state_dict(state["model_state_dict"])
        _, logits, labels = collect_logits_with_aug(model, loader, augs, num_classes=5)
        label_ref = labels
        aligned_logits.append(logits)
        m = compute_multiclass_metrics(logits, labels, num_classes=5)
        print(f"  seed={s} TTA  wf1={m['weighted_f1']:.4f}  mf1={m['macro_f1']:.4f}")
    ens = torch.stack(aligned_logits).mean(dim=0)
    em = compute_multiclass_metrics(ens, label_ref, num_classes=5)
    print(f"  TTA + 4-seed ensemble  wf1={em['weighted_f1']:.4f}  mf1={em['macro_f1']:.4f}  acc={em['accuracy']:.4f}")
    return em


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["T1", "T2", "T3", "ALL"], default="ALL")
    args = parser.parse_args()

    results = {}
    if args.task in ("T1", "ALL"):
        results["T1"] = run_t1()
    if args.task in ("T2", "ALL"):
        results["T2"] = run_t2()
    if args.task in ("T3", "ALL"):
        results["T3"] = run_t3()

    out = REPO / "neuromm26_results/metrics/tta_val_summary.json"
    out.write_text(json.dumps({k: dict(v) for k, v in results.items()}, indent=2, default=float))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
