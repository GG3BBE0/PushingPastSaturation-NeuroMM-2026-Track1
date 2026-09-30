"""Final tuning on top of cross-model ensemble:
  1. Threshold sweep for binary tasks (T1, T2) — searches optimal F1 threshold.
  2. T3 leave-one-out — try ensemble without the weakest model.
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


def feat_dim(name): return int(np.load(next((VIDEO_FEATURE_ROOT / name).glob("*.npy"))).shape[-1])


def collect_logits_binary(model, loader):
    model.eval()
    L, Y, S = [], [], []
    with torch.no_grad():
        for b in loader:
            b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            L.append(model(b).view(-1).cpu())
            Y.append(b["label"].view(-1).float().cpu())
            S.extend(b["sample_id"])
    return S, torch.cat(L), torch.cat(Y)


def collect_logits_mc(model, loader, K=5):
    model.eval()
    L, Y, S = [], [], []
    with torch.no_grad():
        for b in loader:
            b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            L.append(model(b).view(-1, K).cpu())
            Y.append(b["label"].view(-1).long().cpu())
            S.extend(b["sample_id"])
    return S, torch.cat(L), torch.cat(Y)


def align(logits, sids, ref_sids):
    if sids == ref_sids: return logits
    idx = {s: i for i, s in enumerate(sids)}
    return logits[[idx[s] for s in ref_sids]]


def sweep_threshold(probs, labels):
    best = {"f1": -1}
    rows = []
    for thr in np.arange(0.05, 0.95 + 1e-9, 0.01):
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        rows.append({"thr": float(thr), "f1": f1, "p": p, "r": r,
                     "acc": (tp + tn) / (tp + tn + fp + fn)})
        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": f1, "p": p, "r": r,
                    "acc": (tp + tn) / (tp + tn + fp + fn)}
    return best, rows


def task1_probs():
    print("\n--- Re-running T1 cross-model 12-ckpt inference for threshold tune ---")
    ds = EEGFeatureDataset(split_csv=None, manifest_csv=str(MANIFEST), split="val",
                           eeg_feature_root=str(EEG_FEATURE_ROOT),
                           preload_in_memory=True, target_shape=(26, 2000))
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    configs = [
        ("tcnet_eeg", "neuromm26_train_eeg_only__legacy-tcnet_eeg"),
        ("efficientnet_v2_s_eeg", "neuromm26_train_eeg_only__legacy-efficientnet_v2_s_eeg"),
        ("mobilenet_v3_large_eeg", "neuromm26_train_eeg_only__legacy-mobilenet_v3_large_eeg"),
    ]
    aligned = []
    ref_sids, ref_y = None, None
    for legacy, prefix in configs:
        for s in SEEDS:
            sd = torch.load(CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt",
                            map_location=DEVICE, weights_only=False)
            m = build_legacy_eeg_model(legacy).to(DEVICE)
            m.load_state_dict(sd["model_state_dict"])
            sids, lg, y = collect_logits_binary(m, loader)
            if ref_sids is None:
                ref_sids, ref_y = sids, y
            else:
                lg = align(lg, sids, ref_sids)
            aligned.append(lg)
    ens_logits = torch.stack(aligned).mean(dim=0)
    return torch.sigmoid(ens_logits).numpy(), ref_y.numpy().astype(int)


def task2_probs():
    print("\n--- Re-running T2 cross-model 12-ckpt inference for threshold tune ---")
    ds = NeuroMMMultimodalFeatureDataset(
        manifest_csv=str(MANIFEST), split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        video_feature_root=str(VIDEO_FEATURE_ROOT),
        video_feature_name="timesformer-k400",
        preload_in_memory=True, target_shape=(26, 2000), require_video_feature=True,
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    fd = feat_dim("timesformer-k400")
    configs = [
        ("eegnet", "eeg_video_fusion__eegnet__timesformer-k400"),
        ("tcnet_eeg", "eeg_video_fusion__tcnet_eeg__timesformer-k400"),
        ("efficientnet_v2_s_eeg", "eeg_video_fusion__efficientnet_v2_s_eeg__timesformer-k400"),
    ]
    aligned = []
    ref_sids, ref_y = None, None
    for eeg, prefix in configs:
        for s in SEEDS:
            sd = torch.load(CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt",
                            map_location=DEVICE, weights_only=False)
            m = EEGVideoLateFusion(eeg_model_name=eeg, video_feature_dim=fd,
                                   video_hidden_dim=sd.get("video_hidden_dim", 256),
                                   video_dropout=0.2).to(DEVICE)
            m.load_state_dict(sd["model_state_dict"])
            sids, lg, y = collect_logits_binary(m, loader)
            if ref_sids is None:
                ref_sids, ref_y = sids, y
            else:
                lg = align(lg, sids, ref_sids)
            aligned.append(lg)
    ens_logits = torch.stack(aligned).mean(dim=0)
    return torch.sigmoid(ens_logits).numpy(), ref_y.numpy().astype(int)


def task3_variants():
    print("\n--- T3 ensemble variants (with/without weakest model) ---")
    configs = [
        ("cn_large+videomae-large", "convnext_large_eeg", "videomae-large",
         "task3_fusion__convnext_large_eeg__videomae-large"),
        ("cn_large+dinov2-base", "convnext_large_eeg", "dinov2-base",
         "task3_fusion__convnext_large_eeg__dinov2-base"),
        ("cn_tiny+siglip-base", "convnext_tiny_eeg", "siglip-base",
         "task3_fusion__convnext_tiny_eeg__siglip-base"),
    ]
    per_cfg = {}
    ref_sids, ref_y = None, None
    for tag, eeg, vid, prefix in configs:
        ds = Task3MultimodalFeatureDataset(
            manifest_csv=str(MANIFEST), split="val",
            eeg_feature_root=str(EEG_FEATURE_ROOT),
            video_feature_root=str(VIDEO_FEATURE_ROOT),
            video_feature_name=vid, target_shape=(26, 2000),
        )
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
        fd = feat_dim(vid)
        seed_logits = []
        for s in SEEDS:
            sd = torch.load(CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt",
                            map_location=DEVICE, weights_only=False)
            m = EEGVideoLateFusion5Class(eeg_model_name=eeg, video_feature_dim=fd,
                                         video_hidden_dim=sd.get("video_hidden_dim", 256),
                                         video_dropout=0.2, num_classes=5).to(DEVICE)
            m.load_state_dict(sd["model_state_dict"])
            sids, lg, y = collect_logits_mc(m, loader, K=5)
            if ref_sids is None:
                ref_sids, ref_y = sids, y
            else:
                lg = align(lg, sids, ref_sids)
            seed_logits.append(lg)
        per_cfg[tag] = torch.stack(seed_logits).mean(dim=0)

    def ens(*tags):
        L = torch.stack([per_cfg[t] for t in tags]).mean(dim=0)
        return compute_multiclass_metrics(L, ref_y, num_classes=5)

    print("\n  Variants (sorted by weighted_f1):")
    variants = {
        "all-3 (large+vmae-l, large+dv2, tiny+sg)": ens(
            "cn_large+videomae-large", "cn_large+dinov2-base", "cn_tiny+siglip-base"),
        "top-2 (large+vmae-l, large+dv2)": ens(
            "cn_large+videomae-large", "cn_large+dinov2-base"),
        "best single (large+dv2)": compute_multiclass_metrics(
            per_cfg["cn_large+dinov2-base"], ref_y, num_classes=5),
    }
    for name, m in sorted(variants.items(), key=lambda kv: -kv[1]["weighted_f1"]):
        print(f"    {name:55s}  wf1={m['weighted_f1']:.4f}  mf1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}")
    return variants


def main():
    results = {}

    # Task 1 threshold
    p1, y1 = task1_probs()
    default1 = {"thr": 0.5, "f1": 0.0}
    default1 = {"thr": 0.5}
    default_pred = (p1 >= 0.5).astype(int)
    default1["f1"] = (2 * ((default_pred == 1) & (y1 == 1)).sum() /
                     max(default_pred.sum() + y1.sum(), 1))
    best1, _ = sweep_threshold(p1, y1)
    print(f"\n  T1 default thr=0.50 F1={default1['f1']:.4f}")
    print(f"  T1 best    thr={best1['thr']:.2f} F1={best1['f1']:.4f}  P={best1['p']:.4f}  R={best1['r']:.4f}  acc={best1['acc']:.4f}")
    print(f"  T1 Δ F1 = {best1['f1']-default1['f1']:+.4f}")
    results["T1"] = {"default": default1, "best": best1}

    # Task 2 threshold
    p2, y2 = task2_probs()
    default_pred = (p2 >= 0.5).astype(int)
    default2 = {"thr": 0.5}
    default2["f1"] = (2 * ((default_pred == 1) & (y2 == 1)).sum() /
                     max(default_pred.sum() + y2.sum(), 1))
    best2, _ = sweep_threshold(p2, y2)
    print(f"\n  T2 default thr=0.50 F1={default2['f1']:.4f}")
    print(f"  T2 best    thr={best2['thr']:.2f} F1={best2['f1']:.4f}  P={best2['p']:.4f}  R={best2['r']:.4f}  acc={best2['acc']:.4f}")
    print(f"  T2 Δ F1 = {best2['f1']-default2['f1']:+.4f}")
    results["T2"] = {"default": default2, "best": best2}

    # Task 3 variants
    t3v = task3_variants()
    results["T3"] = {k: dict(v) for k, v in t3v.items()}

    out = REPO / "neuromm26_results/metrics/final_tuning.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
