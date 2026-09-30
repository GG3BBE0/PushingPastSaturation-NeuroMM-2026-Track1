"""Phase A: evaluate STFT 4-seed ensemble + combine with existing T1 ensemble."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets import EEGFeatureDataset
from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.datasets.multiband_dataset import MultiBandFeatureDataset
from neuromm26_baseline.datasets.spec_dataset import SpecFeatureDataset
from neuromm26_baseline.models import build_legacy_eeg_model
from neuromm26_baseline.models.muku_eegnet import MukuEEGNet
from neuromm26_baseline.models.spec_cnn import SpecCNN
from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
MANIFEST = REPO / "neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv"
EEG_ROOT = REPO / "neuromm26_datasets/processed/features/eeg"
MB_ROOT = REPO / "neuromm26_datasets/processed/features/multiband"
STFT_ROOT = REPO / "neuromm26_datasets/processed/features/stft"
CKPT_DIR = REPO / "neuromm26_results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (0, 1, 2, 3)


@torch.no_grad()
def coll(model, loader, key="eeg"):
    model.eval()
    L, Y, S = [], [], []
    for b in loader:
        b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        if key == "eeg":
            out = model({"eeg": b["eeg"]})
        elif key == "spec":
            out = model({"spec": b["spec"]})
        logits = out["main"] if isinstance(out, dict) else out
        L.append(logits.view(-1).cpu())
        Y.append(b["label"].view(-1).float().cpu())
        S.extend(b["sample_id"])
    return S, torch.cat(L), torch.cat(Y)


def align(L, sids, ref):
    if sids == ref: return L
    idx = {s: i for i, s in enumerate(sids)}
    return L[[idx[s] for s in ref]]


def main():
    # ---- STFT 4 ckpt ----
    print("=" * 60)
    print("Phase A — STFT SpecCNN 4-seed ensemble")
    print("=" * 60)
    spec_ds = SpecFeatureDataset(spec_root=str(STFT_ROOT), manifest_csv=str(MANIFEST),
                                  split="val", preload_in_memory=True)
    spec_loader = DataLoader(spec_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)

    stft_logits = []
    ref_sids = None; ref_y = None
    for s in SEEDS:
        cp = CKPT_DIR / f"spec_stft__resnet18__seed{s}__lr0.0005_5e-05/best.pt"
        state = torch.load(cp, map_location=DEVICE, weights_only=False)
        m = SpecCNN(num_classes=1, backbone="resnet18", in_chans=26, pretrained=False).to(DEVICE)
        m.load_state_dict(state["model_state_dict"])
        sids, L, Y = coll(m, spec_loader, key="spec")
        if ref_sids is None: ref_sids, ref_y = sids, Y
        else: L = align(L, sids, ref_sids)
        stft_logits.append(L)
        sm = compute_binary_classification_metrics(L, ref_y)
        print(f"  seed {s}: auprc = {sm['auprc']:.4f}  f1 = {sm['binary_f1']:.4f}")
    stft_ens = torch.stack(stft_logits).mean(dim=0)
    stft_m = compute_binary_classification_metrics(stft_ens, ref_y)
    print(f"\n  4-seed ensemble: auprc = {stft_m['auprc']:.4f}  f1 = {stft_m['binary_f1']:.4f}")

    # ---- Existing T1 ensemble (24 ckpt) ----
    print("\n" + "=" * 60)
    print("Existing T1 ensemble + Phase C")
    print("=" * 60)
    eeg_ds = EEGFeatureDataset(split_csv=None, manifest_csv=str(MANIFEST), split="val",
                               eeg_feature_root=str(EEG_ROOT), preload_in_memory=True,
                               target_shape=(26, 2000))
    eeg_loader = DataLoader(eeg_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)

    raw_logits = []
    for legacy, prefix in [("tcnet_eeg", "neuromm26_train_eeg_only__legacy-tcnet_eeg"),
                            ("efficientnet_v2_s_eeg", "neuromm26_train_eeg_only__legacy-efficientnet_v2_s_eeg"),
                            ("mobilenet_v3_large_eeg", "neuromm26_train_eeg_only__legacy-mobilenet_v3_large_eeg")]:
        for s in SEEDS:
            cp = CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = build_legacy_eeg_model(legacy).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, _ = coll(m, eeg_loader, key="eeg")
            L = align(L, sids, ref_sids)
            raw_logits.append(L)
    for bb, pat in [("resnet18", "muku__resnet18__seed{s}__lr0.0005_5e-05"),
                     ("tf_efficientnet_b0.ns_jft_in1k", "muku__tf_efficientnet_b0_ns_jft_in1k__seed{s}__lr0.0005_5e-05"),
                     ("convnext_pico.d1_in1k", "muku__convnext_pico_d1_in1k__seed{s}__lr0.0005_5e-05")]:
        for s in SEEDS:
            cp = CKPT_DIR / pat.format(s=s) / "best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = MukuEEGNet(num_classes=1, backbone=bb, in_chans=26, n_freqs=8, fs=500,
                           pretrained=False).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, _ = coll(m, eeg_loader, key="eeg")
            L = align(L, sids, ref_sids)
            raw_logits.append(L)

    # Phase C multi-band 4 ckpt
    mb_ds = MultiBandFeatureDataset(multiband_root=str(MB_ROOT), manifest_csv=str(MANIFEST),
                                    split="val", preload_in_memory=True, target_shape=(130, 2000))
    mb_loader = DataLoader(mb_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    mb_logits = []
    for s in SEEDS:
        cp = CKPT_DIR / f"muku_mb__resnet18__seed{s}__lr0.0005_5e-05/best.pt"
        state = torch.load(cp, map_location=DEVICE, weights_only=False)
        m = MukuEEGNet(num_classes=1, backbone="resnet18", in_chans=130, n_freqs=8, fs=500,
                       pretrained=False).to(DEVICE)
        m.load_state_dict(state["model_state_dict"])
        sids, L, _ = coll(m, mb_loader, key="eeg")
        L = align(L, sids, ref_sids)
        mb_logits.append(L)

    existing_24 = torch.stack(raw_logits).mean(dim=0)
    existing_28 = torch.stack(raw_logits + mb_logits).mean(dim=0)
    ex24_m = compute_binary_classification_metrics(existing_24, ref_y)
    ex28_m = compute_binary_classification_metrics(existing_28, ref_y)
    print(f"  24-ckpt (legacy+muku): auprc = {ex24_m['auprc']:.4f}  f1 = {ex24_m['binary_f1']:.4f}")
    print(f"  28-ckpt (+phase C):    auprc = {ex28_m['auprc']:.4f}  f1 = {ex28_m['binary_f1']:.4f}")

    # ---- Combined 28 + STFT 4 = 32 ----
    print("\n" + "=" * 60)
    print("Combined: 28 + Phase A 4 = 32 ckpt")
    print("=" * 60)
    combined_32 = torch.stack(raw_logits + mb_logits + stft_logits).mean(dim=0)
    cm32 = compute_binary_classification_metrics(combined_32, ref_y)
    print(f"  32-ckpt combined: auprc = {cm32['auprc']:.4f}  f1 = {cm32['binary_f1']:.4f}")
    print(f"  ΔAUPRC vs 28-ckpt: {cm32['auprc'] - ex28_m['auprc']:+.4f}")
    print(f"  ΔF1 vs 28-ckpt:    {cm32['binary_f1'] - ex28_m['binary_f1']:+.4f}")
    print(f"  ΔAUPRC vs 24-ckpt: {cm32['auprc'] - ex24_m['auprc']:+.4f}")

    if cm32["auprc"] > ex28_m["auprc"] + 0.005:
        verdict = "✓ Phase A HELPED (Δ > 0.005)"
    elif cm32["auprc"] > ex28_m["auprc"]:
        verdict = "~ Phase A marginal"
    else:
        verdict = "✗ Phase A HURT ensemble"
    print(f"  Verdict: {verdict}")

    out = {
        "phase_a_per_seed_auprc": [
            float(compute_binary_classification_metrics(L, ref_y)["auprc"]) for L in stft_logits
        ],
        "phase_a_ensemble": dict(stft_m),
        "existing_24": dict(ex24_m),
        "existing_28_with_C": dict(ex28_m),
        "combined_32": dict(cm32),
        "delta_vs_28": float(cm32["auprc"] - ex28_m["auprc"]),
    }
    op = REPO / "neuromm26_results/metrics/phaseA_eval.json"
    op.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {op}")


if __name__ == "__main__":
    main()
