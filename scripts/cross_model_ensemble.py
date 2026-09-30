"""Cross-model 12-ckpt ensemble per task (3 architectures × 4 seeds).

For each task it loads 12 checkpoints from 3 different model architectures,
runs val inference, sample-aligns by sample_id, then logit-averages.

For T3 different model variants use different video backbones, so we build
a separate val dataset per video feature.
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


def detect_video_feature_dim(name: str) -> int:
    arr = np.load(next((VIDEO_FEATURE_ROOT / name).glob("*.npy")))
    return int(arr.shape[-1]) if arr.ndim > 1 else int(arr.shape[0])


def t1_dataset_loader():
    ds = EEGFeatureDataset(
        split_csv=None, manifest_csv=str(MANIFEST), split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        preload_in_memory=True, target_shape=(26, 2000),
    )
    return DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=neuromm_collate)


def t2_dataset_loader(video_name: str):
    ds = NeuroMMMultimodalFeatureDataset(
        manifest_csv=str(MANIFEST), split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        video_feature_root=str(VIDEO_FEATURE_ROOT),
        video_feature_name=video_name,
        preload_in_memory=True, target_shape=(26, 2000), require_video_feature=True,
    )
    return DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=neuromm_collate)


def t3_dataset_loader(video_name: str):
    ds = Task3MultimodalFeatureDataset(
        manifest_csv=str(MANIFEST), split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        video_feature_root=str(VIDEO_FEATURE_ROOT),
        video_feature_name=video_name,
        target_shape=(26, 2000),
    )
    return DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=neuromm_collate)


def run_inference_binary(model, loader):
    model.eval()
    logits, labels, sids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            l = model(batch).view(-1).cpu()
            logits.append(l)
            labels.append(batch["label"].view(-1).float().cpu())
            sids.extend(batch["sample_id"])
    return sids, torch.cat(logits), torch.cat(labels)


def run_inference_multiclass(model, loader, num_classes=5):
    model.eval()
    logits, labels, sids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            l = model(batch).view(-1, num_classes).cpu()
            logits.append(l)
            labels.append(batch["label"].view(-1).long().cpu())
            sids.extend(batch["sample_id"])
    return sids, torch.cat(logits), torch.cat(labels)


def align_logits(logits: torch.Tensor, sids: list[str], ref_sids: list[str]) -> torch.Tensor:
    """Reorder logits so they match ref_sids order."""
    if sids == ref_sids:
        return logits
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}
    perm = [sid_to_idx[s] for s in ref_sids]
    return logits[perm]


# ============================================================
# Task 1
# ============================================================
def task1():
    print("\n========== Task 1: 3 models × 4 seeds = 12 ckpts ==========")
    loader = t1_dataset_loader()
    configs = [
        ("legacy/tcnet_eeg", "tcnet_eeg", "neuromm26_train_eeg_only__legacy-tcnet_eeg"),
        ("legacy/efficientnet_v2_s_eeg", "efficientnet_v2_s_eeg", "neuromm26_train_eeg_only__legacy-efficientnet_v2_s_eeg"),
        ("legacy/mobilenet_v3_large_eeg", "mobilenet_v3_large_eeg", "neuromm26_train_eeg_only__legacy-mobilenet_v3_large_eeg"),
    ]
    aligned = []
    ref_sids, ref_labels = None, None
    per_model = {}
    for full, legacy_name, prefix in configs:
        model_logits = []
        for s in SEEDS:
            path = CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt"
            state = torch.load(path, map_location=DEVICE, weights_only=False)
            m = build_legacy_eeg_model(legacy_name).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, logits, labels = run_inference_binary(m, loader)
            if ref_sids is None:
                ref_sids, ref_labels = sids, labels
            else:
                logits = align_logits(logits, sids, ref_sids)
            model_logits.append(logits)
            aligned.append(logits)
        m4 = torch.stack(model_logits).mean(dim=0)
        m4_metric = compute_binary_classification_metrics(m4, ref_labels)
        per_model[full] = m4_metric
        print(f"  {full:50s} 4-seed ens auprc={m4_metric['auprc']:.4f}  f1={m4_metric['binary_f1']:.4f}")

    full_ens = torch.stack(aligned).mean(dim=0)
    fm = compute_binary_classification_metrics(full_ens, ref_labels)
    print(f"\n  → CROSS-MODEL 12-ckpt  auprc={fm['auprc']:.4f}  f1={fm['binary_f1']:.4f}  acc={fm['accuracy']:.4f}")
    return {"per_model": per_model, "cross_model": fm}


# ============================================================
# Task 2 — all 3 use timesformer-k400 video features
# ============================================================
def task2():
    print("\n========== Task 2: 3 models × 4 seeds = 12 ckpts (all video=timesformer-k400) ==========")
    loader = t2_dataset_loader("timesformer-k400")
    feat_dim = detect_video_feature_dim("timesformer-k400")
    configs = [
        ("fusion/eegnet+timesformer-k400", "eegnet", "eeg_video_fusion__eegnet__timesformer-k400"),
        ("fusion/tcnet_eeg+timesformer-k400", "tcnet_eeg", "eeg_video_fusion__tcnet_eeg__timesformer-k400"),
        ("fusion/efficientnet_v2_s_eeg+timesformer-k400", "efficientnet_v2_s_eeg",
         "eeg_video_fusion__efficientnet_v2_s_eeg__timesformer-k400"),
    ]
    aligned = []
    ref_sids, ref_labels = None, None
    per_model = {}
    for full, eeg_name, prefix in configs:
        model_logits = []
        for s in SEEDS:
            path = CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt"
            state = torch.load(path, map_location=DEVICE, weights_only=False)
            m = EEGVideoLateFusion(
                eeg_model_name=eeg_name, video_feature_dim=feat_dim,
                video_hidden_dim=state.get("video_hidden_dim", 256), video_dropout=0.2,
            ).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, logits, labels = run_inference_binary(m, loader)
            if ref_sids is None:
                ref_sids, ref_labels = sids, labels
            else:
                logits = align_logits(logits, sids, ref_sids)
            model_logits.append(logits)
            aligned.append(logits)
        m4 = torch.stack(model_logits).mean(dim=0)
        m4_metric = compute_binary_classification_metrics(m4, ref_labels)
        per_model[full] = m4_metric
        print(f"  {full:60s} 4-seed ens auprc={m4_metric['auprc']:.4f}  f1={m4_metric['binary_f1']:.4f}")

    full_ens = torch.stack(aligned).mean(dim=0)
    fm = compute_binary_classification_metrics(full_ens, ref_labels)
    print(f"\n  → CROSS-MODEL 12-ckpt  auprc={fm['auprc']:.4f}  f1={fm['binary_f1']:.4f}  acc={fm['accuracy']:.4f}")
    return {"per_model": per_model, "cross_model": fm}


# ============================================================
# Task 3 — different video backbones, need per-model dataset
# ============================================================
def task3():
    print("\n========== Task 3: 3 models × 4 seeds = 12 ckpts (different video backbones) ==========")
    configs = [
        ("task3_fusion/convnext_large_eeg+videomae-large", "convnext_large_eeg", "videomae-large",
         "task3_fusion__convnext_large_eeg__videomae-large"),
        ("task3_fusion/convnext_large_eeg+dinov2-base", "convnext_large_eeg", "dinov2-base",
         "task3_fusion__convnext_large_eeg__dinov2-base"),
        ("task3_fusion/convnext_tiny_eeg+siglip-base", "convnext_tiny_eeg", "siglip-base",
         "task3_fusion__convnext_tiny_eeg__siglip-base"),
    ]
    aligned = []
    ref_sids, ref_labels = None, None
    per_model = {}
    for full, eeg_name, video_name, prefix in configs:
        loader = t3_dataset_loader(video_name)
        feat_dim = detect_video_feature_dim(video_name)
        model_logits = []
        for s in SEEDS:
            path = CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt"
            state = torch.load(path, map_location=DEVICE, weights_only=False)
            m = EEGVideoLateFusion5Class(
                eeg_model_name=eeg_name, video_feature_dim=feat_dim,
                video_hidden_dim=state.get("video_hidden_dim", 256), video_dropout=0.2,
                num_classes=state.get("num_classes", 5),
            ).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, logits, labels = run_inference_multiclass(m, loader, num_classes=5)
            if ref_sids is None:
                ref_sids, ref_labels = sids, labels
            else:
                logits = align_logits(logits, sids, ref_sids)
            model_logits.append(logits)
            aligned.append(logits)
        m4 = torch.stack(model_logits).mean(dim=0)
        m4_metric = compute_multiclass_metrics(m4, ref_labels, num_classes=5)
        per_model[full] = m4_metric
        print(f"  {full:60s} 4-seed ens wf1={m4_metric['weighted_f1']:.4f}  mf1={m4_metric['macro_f1']:.4f}")

    full_ens = torch.stack(aligned).mean(dim=0)
    fm = compute_multiclass_metrics(full_ens, ref_labels, num_classes=5)
    print(f"\n  → CROSS-MODEL 12-ckpt  wf1={fm['weighted_f1']:.4f}  mf1={fm['macro_f1']:.4f}  acc={fm['accuracy']:.4f}")
    return {"per_model": per_model, "cross_model": fm}


def main():
    results = {
        "T1": task1(),
        "T2": task2(),
        "T3": task3(),
    }

    print("\n=========== CROSS-MODEL ENSEMBLE SUMMARY ===========")
    for tk, r in results.items():
        metric = "weighted_f1" if tk == "T3" else "auprc"
        best_single = max(r["per_model"].values(), key=lambda m: m[metric])[metric]
        ens = r["cross_model"][metric]
        print(f"{tk}  best single-model 4-seed ens: {best_single:.4f}   →   12-ckpt cross-model ens: {ens:.4f}   Δ {ens-best_single:+.4f}")

    out = REPO / "neuromm26_results/metrics/cross_model_ensemble.json"
    out.write_text(json.dumps({
        tk: {
            "per_model": {k: dict(v) for k, v in r["per_model"].items()},
            "cross_model": dict(r["cross_model"]),
        } for tk, r in results.items()
    }, indent=2, default=float))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
