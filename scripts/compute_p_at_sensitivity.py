"""Compute Precision@Sensitivity=70% (P@S70) for T1 and T2 ensembles.

P@S70: fix sensitivity (recall) at 0.70, report precision at that operating point.
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
from neuromm26_baseline.models.muku_eegnet import MukuEEGNet
from neuromm26_baseline.models.muku_eeg_video import MukuEEGVideoNet

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
MANIFEST = REPO / "neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv"
EEG_ROOT = REPO / "neuromm26_datasets/processed/features/eeg"
VID_ROOT = REPO / "neuromm26_datasets/processed/features/video"
CKPT_DIR = REPO / "neuromm26_results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (0, 1, 2, 3)


def precision_at_sensitivity(probs: np.ndarray, labels: np.ndarray, target_recall: float = 0.70) -> dict:
    """Find threshold where recall ≥ target_recall, return precision at that threshold."""
    n_pos = int(labels.sum())
    if n_pos == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0}

    # Sort by prob descending
    order = np.argsort(-probs)
    sorted_labels = labels[order]
    sorted_probs = probs[order]

    cum_tp = np.cumsum(sorted_labels)
    cum_fp = np.cumsum(1 - sorted_labels)
    recall = cum_tp / n_pos
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)

    # Find smallest k where recall[k] >= target
    above = np.where(recall >= target_recall)[0]
    if len(above) == 0:
        return {"threshold": sorted_probs[-1], "precision": precision[-1], "recall": recall[-1]}
    k = above[0]
    return {
        "threshold": float(sorted_probs[k]),
        "precision": float(precision[k]),
        "recall": float(recall[k]),
        "tp_at_threshold": int(cum_tp[k]),
        "fp_at_threshold": int(cum_fp[k]),
    }


@torch.no_grad()
def coll_bin(model, loader):
    model.eval()
    L, Y, S = [], [], []
    for b in loader:
        b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        out = model(b)
        L.append((out["main"] if isinstance(out, dict) else out).view(-1).cpu())
        Y.append(b["label"].view(-1).float().cpu())
        S.extend(b["sample_id"])
    return S, torch.cat(L), torch.cat(Y)


def align(L, sids, ref):
    if sids == ref: return L
    idx = {s: i for i, s in enumerate(sids)}
    return L[[idx[s] for s in ref]]


def t1_ensemble():
    """T1: combine legacy 12 + muku 12 = 24 ckpt ensemble."""
    print("\n========== T1: P@S70 ==========")
    ds = EEGFeatureDataset(split_csv=None, manifest_csv=str(MANIFEST), split="val",
                          eeg_feature_root=str(EEG_ROOT), preload_in_memory=True,
                          target_shape=(26, 2000))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)

    legacy = [
        ("tcnet_eeg", "neuromm26_train_eeg_only__legacy-tcnet_eeg"),
        ("efficientnet_v2_s_eeg", "neuromm26_train_eeg_only__legacy-efficientnet_v2_s_eeg"),
        ("mobilenet_v3_large_eeg", "neuromm26_train_eeg_only__legacy-mobilenet_v3_large_eeg"),
    ]
    muku = [
        ("resnet18", "muku__resnet18__seed{s}__lr0.0005_5e-05"),
        ("tf_efficientnet_b0.ns_jft_in1k", "muku__tf_efficientnet_b0_ns_jft_in1k__seed{s}__lr0.0005_5e-05"),
        ("convnext_pico.d1_in1k", "muku__convnext_pico_d1_in1k__seed{s}__lr0.0005_5e-05"),
    ]

    all_legacy, all_muku, ref, ref_y = [], [], None, None

    for name, prefix in legacy:
        for s in SEEDS:
            cp = CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt"
            if not cp.exists(): continue
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = build_legacy_eeg_model(name).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, Y = coll_bin(m, loader)
            if ref is None: ref, ref_y = sids, Y
            else: L = align(L, sids, ref)
            all_legacy.append(L)

    for bb, pat in muku:
        for s in SEEDS:
            cp = CKPT_DIR / pat.format(s=s) / "best.pt"
            if not cp.exists(): continue
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = MukuEEGNet(num_classes=1, backbone=bb, in_chans=26, n_freqs=8, fs=500,
                           pretrained=False).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, _ = coll_bin(m, loader)
            L = align(L, sids, ref)
            all_muku.append(L)

    Y_np = ref_y.numpy().astype(int)

    def eval_set(label, logits_list):
        if not logits_list:
            print(f"  {label}: no ckpts")
            return None
        ens = torch.stack(logits_list).mean(dim=0)
        probs = torch.sigmoid(ens).numpy()
        ps = precision_at_sensitivity(probs, Y_np, 0.70)
        # Also some other sensitivity levels
        ps80 = precision_at_sensitivity(probs, Y_np, 0.80)
        ps60 = precision_at_sensitivity(probs, Y_np, 0.60)
        print(f"  {label:30s} (N={len(logits_list)})")
        print(f"    P@S60 = {ps60['precision']:.4f} (thr {ps60['threshold']:.3f}, recall {ps60['recall']:.4f})")
        print(f"    P@S70 = {ps['precision']:.4f} (thr {ps['threshold']:.3f}, recall {ps['recall']:.4f})")
        print(f"    P@S80 = {ps80['precision']:.4f} (thr {ps80['threshold']:.3f}, recall {ps80['recall']:.4f})")
        return {"P@S60": ps60, "P@S70": ps, "P@S80": ps80}

    res = {}
    res["legacy_12"] = eval_set("Legacy 12-ckpt", all_legacy)
    res["muku_12"] = eval_set("Muku 12-ckpt", all_muku)
    res["combined_24"] = eval_set("Combined 24-ckpt", all_legacy + all_muku)
    return res


def t2_ensemble():
    """T2: muku 12 only (best for T2)."""
    print("\n========== T2: P@S70 ==========")
    video_name = "timesformer-k400"
    arr = np.load(next((VID_ROOT / video_name).glob("*.npy")))
    video_dim = int(arr.shape[-1]) if arr.ndim > 1 else int(arr.shape[0])

    ds = NeuroMMMultimodalFeatureDataset(
        manifest_csv=str(MANIFEST), split="val",
        eeg_feature_root=str(EEG_ROOT), video_feature_root=str(VID_ROOT),
        video_feature_name=video_name, preload_in_memory=True,
        target_shape=(26, 2000), require_video_feature=True,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)

    muku = [
        ("resnet18", "muku_t2__resnet18__timesformer-k400__seed{s}__lr0.0005_5e-05"),
        ("tf_efficientnet_b0.ns_jft_in1k", "muku_t2__tf_efficientnet_b0_ns_jft_in1k__timesformer-k400__seed{s}__lr0.0005_5e-05"),
        ("convnext_pico.d1_in1k", "muku_t2__convnext_pico_d1_in1k__timesformer-k400__seed{s}__lr0.0005_5e-05"),
    ]
    legacy = [
        ("eegnet", "eeg_video_fusion__eegnet__timesformer-k400"),
        ("tcnet_eeg", "eeg_video_fusion__tcnet_eeg__timesformer-k400"),
        ("efficientnet_v2_s_eeg", "eeg_video_fusion__efficientnet_v2_s_eeg__timesformer-k400"),
    ]

    all_muku, all_legacy, ref, ref_y = [], [], None, None

    for bb, pat in muku:
        for s in SEEDS:
            cp = CKPT_DIR / pat.format(s=s) / "best.pt"
            if not cp.exists(): continue
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = MukuEEGVideoNet(num_classes=1, backbone=bb, in_chans=26, n_freqs=8, fs=500,
                                video_feat_dim=video_dim, video_hidden=state.get("video_hidden", 256),
                                video_dropout=0.2, pretrained=False).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, Y = coll_bin(m, loader)
            if ref is None: ref, ref_y = sids, Y
            else: L = align(L, sids, ref)
            all_muku.append(L)

    for eeg, prefix in legacy:
        for s in SEEDS:
            cp = CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt"
            if not cp.exists(): continue
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = EEGVideoLateFusion(eeg_model_name=eeg, video_feature_dim=video_dim,
                                   video_hidden_dim=state.get("video_hidden_dim", 256),
                                   video_dropout=0.2).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, _ = coll_bin(m, loader)
            L = align(L, sids, ref)
            all_legacy.append(L)

    Y_np = ref_y.numpy().astype(int)

    def eval_set(label, logits_list):
        if not logits_list:
            print(f"  {label}: no ckpts")
            return None
        ens = torch.stack(logits_list).mean(dim=0)
        probs = torch.sigmoid(ens).numpy()
        ps = precision_at_sensitivity(probs, Y_np, 0.70)
        ps80 = precision_at_sensitivity(probs, Y_np, 0.80)
        ps60 = precision_at_sensitivity(probs, Y_np, 0.60)
        print(f"  {label:30s} (N={len(logits_list)})")
        print(f"    P@S60 = {ps60['precision']:.4f} (thr {ps60['threshold']:.3f}, recall {ps60['recall']:.4f})")
        print(f"    P@S70 = {ps['precision']:.4f} (thr {ps['threshold']:.3f}, recall {ps['recall']:.4f})")
        print(f"    P@S80 = {ps80['precision']:.4f} (thr {ps80['threshold']:.3f}, recall {ps80['recall']:.4f})")
        return {"P@S60": ps60, "P@S70": ps, "P@S80": ps80}

    res = {}
    res["muku_12"] = eval_set("Muku 12-ckpt (best for T2)", all_muku)
    res["legacy_12"] = eval_set("Legacy 12-ckpt", all_legacy)
    res["combined_24"] = eval_set("Combined 24-ckpt", all_muku + all_legacy)
    return res


def main():
    out = {"T1": t1_ensemble(), "T2": t2_ensemble()}
    op = REPO / "neuromm26_results/metrics/p_at_sensitivity.json"
    op.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {op}")


if __name__ == "__main__":
    main()
