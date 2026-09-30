"""Evaluate muku models: per-backbone 4-seed ensemble + cross-backbone 12-ckpt ensemble.

Then COMBINE muku 12-ckpt ensemble with the previous cross-model 12-ckpt baseline
to see if we beat the 0.8224 AUPRC mark.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets import EEGFeatureDataset
from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.models import build_legacy_eeg_model
from neuromm26_baseline.models.muku_eegnet import MukuEEGNet
from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
MANIFEST = REPO / "neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv"
EEG_FEATURE_ROOT = REPO / "neuromm26_datasets/processed/features/eeg"
CKPT_DIR = REPO / "neuromm26_results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (0, 1, 2, 3)

MUKU_BACKBONES = [
    ("resnet18", "muku__resnet18__seed{seed}__lr0.0005_5e-05"),
    ("tf_efficientnet_b0.ns_jft_in1k", "muku__tf_efficientnet_b0_ns_jft_in1k__seed{seed}__lr0.0005_5e-05"),
    ("convnext_pico.d1_in1k", "muku__convnext_pico_d1_in1k__seed{seed}__lr0.0005_5e-05"),
]


def make_loader():
    ds = EEGFeatureDataset(split_csv=None, manifest_csv=str(MANIFEST), split="val",
                          eeg_feature_root=str(EEG_FEATURE_ROOT), preload_in_memory=True,
                          target_shape=(26, 2000))
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)


@torch.no_grad()
def collect_binary_logits(model, loader):
    model.eval()
    L, Y, S = [], [], []
    for b in loader:
        b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        out = model(b)
        if isinstance(out, dict):
            logits = out["main"].view(-1).cpu()
        else:
            logits = out.view(-1).cpu()
        L.append(logits)
        Y.append(b["label"].view(-1).float().cpu())
        S.extend(b["sample_id"])
    return S, torch.cat(L), torch.cat(Y)


def align(logits, sids, ref_sids):
    if sids == ref_sids: return logits
    idx = {s: i for i, s in enumerate(sids)}
    return logits[[idx[s] for s in ref_sids]]


def main():
    loader = make_loader()

    # 1) Per-backbone 4-seed ensembles
    print("=" * 60)
    print("Per-backbone muku 4-seed ensemble (val AUPRC):")
    all_muku_logits = []
    ref_sids, ref_y = None, None
    per_backbone = {}

    for backbone, ckpt_pattern in MUKU_BACKBONES:
        seed_logits = []
        for s in SEEDS:
            ckpt_path = CKPT_DIR / ckpt_pattern.format(seed=s) / "best.pt"
            if not ckpt_path.exists():
                print(f"  SKIP {backbone} seed={s}: ckpt missing {ckpt_path}")
                continue
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model = MukuEEGNet(num_classes=1, backbone=backbone, in_chans=26, n_freqs=8, fs=500).to(DEVICE)
            model.load_state_dict(state["model_state_dict"])
            sids, logits, labels = collect_binary_logits(model, loader)
            if ref_sids is None:
                ref_sids, ref_y = sids, labels
            else:
                logits = align(logits, sids, ref_sids)
            seed_logits.append(logits)
            all_muku_logits.append(logits)
        if seed_logits:
            ens = torch.stack(seed_logits).mean(dim=0)
            m = compute_binary_classification_metrics(ens, ref_y)
            per_backbone[backbone] = m
            print(f"  {backbone:50s}  N={len(seed_logits)}  auprc={m['auprc']:.4f}  f1={m['binary_f1']:.4f}  acc={m['accuracy']:.4f}")

    # 2) Cross-backbone muku ensemble (up to 12 ckpts)
    print(f"\nCross-backbone muku ensemble ({len(all_muku_logits)} ckpts):")
    if all_muku_logits:
        cross_muku = torch.stack(all_muku_logits).mean(dim=0)
        m = compute_binary_classification_metrics(cross_muku, ref_y)
        print(f"  CROSS-BACKBONE muku  auprc={m['auprc']:.4f}  f1={m['binary_f1']:.4f}  acc={m['accuracy']:.4f}")
        cross_muku_m = m
    else:
        cross_muku = None
        cross_muku_m = None

    # 3) Load previous cross-model 12-ckpt ensemble (tcnet + effv2s + mbn) and combine
    print(f"\nCombining muku ensemble with previous cross-model (tcnet + effv2s + mbn):")
    legacy_configs = [
        ("tcnet_eeg", "neuromm26_train_eeg_only__legacy-tcnet_eeg"),
        ("efficientnet_v2_s_eeg", "neuromm26_train_eeg_only__legacy-efficientnet_v2_s_eeg"),
        ("mobilenet_v3_large_eeg", "neuromm26_train_eeg_only__legacy-mobilenet_v3_large_eeg"),
    ]
    legacy_logits = []
    for legacy_name, prefix in legacy_configs:
        for s in SEEDS:
            path = CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt"
            if not path.exists():
                continue
            state = torch.load(path, map_location=DEVICE, weights_only=False)
            model = build_legacy_eeg_model(legacy_name).to(DEVICE)
            model.load_state_dict(state["model_state_dict"])
            sids, logits, labels = collect_binary_logits(model, loader)
            logits = align(logits, sids, ref_sids)
            legacy_logits.append(logits)
    print(f"  Legacy 12-ckpt loaded: {len(legacy_logits)}")

    if legacy_logits:
        legacy_ens = torch.stack(legacy_logits).mean(dim=0)
        m_legacy = compute_binary_classification_metrics(legacy_ens, ref_y)
        print(f"  Legacy cross-model ens (reference)  auprc={m_legacy['auprc']:.4f}  f1={m_legacy['binary_f1']:.4f}")

        # Combined: muku + legacy (all logit averaging)
        if cross_muku is not None:
            combined = torch.stack(all_muku_logits + legacy_logits).mean(dim=0)
            m_combined = compute_binary_classification_metrics(combined, ref_y)
            print(f"\n  COMBINED muku+legacy ({len(all_muku_logits) + len(legacy_logits)} ckpts)")
            print(f"    auprc={m_combined['auprc']:.4f}  f1={m_combined['binary_f1']:.4f}  acc={m_combined['accuracy']:.4f}")

    out = {
        "per_backbone": {k: dict(v) for k, v in per_backbone.items()},
        "cross_backbone_muku": dict(cross_muku_m) if cross_muku_m else None,
        "legacy_cross_model": dict(m_legacy) if legacy_logits else None,
        "combined_muku_plus_legacy": dict(m_combined) if (legacy_logits and cross_muku is not None) else None,
    }
    out_path = REPO / "neuromm26_results/metrics/muku_combined_ensemble.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
