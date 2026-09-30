"""Phase C: evaluate multi-band 4-seed ensemble + combine with existing T1 ensemble."""

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
from neuromm26_baseline.models import build_legacy_eeg_model
from neuromm26_baseline.models.muku_eegnet import MukuEEGNet
from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
MANIFEST = REPO / "neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv"
EEG_ROOT = REPO / "neuromm26_datasets/processed/features/eeg"
MB_ROOT = REPO / "neuromm26_datasets/processed/features/multiband"
CKPT_DIR = REPO / "neuromm26_results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (0, 1, 2, 3)


@torch.no_grad()
def coll_bin(model, loader, key="eeg"):
    model.eval()
    L, Y, S = [], [], []
    for b in loader:
        b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        out = model({"eeg": b[key]} if key != "eeg" else {"eeg": b["eeg"]})
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
    # ---- Multi-band 4 ckpt inference ----
    mb_ds = MultiBandFeatureDataset(multiband_root=str(MB_ROOT), manifest_csv=str(MANIFEST),
                                    split="val", preload_in_memory=True, target_shape=(130, 2000))
    mb_loader = DataLoader(mb_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)

    print("=" * 60)
    print("Phase C — Multi-band 4-seed ensemble")
    print("=" * 60)
    mb_logits = []
    ref_sids = None; ref_y = None
    for s in SEEDS:
        cp = CKPT_DIR / f"muku_mb__resnet18__seed{s}__lr0.0005_5e-05/best.pt"
        state = torch.load(cp, map_location=DEVICE, weights_only=False)
        m = MukuEEGNet(num_classes=1, backbone="resnet18", in_chans=130,
                       n_freqs=8, fs=500, pretrained=False).to(DEVICE)
        m.load_state_dict(state["model_state_dict"])
        sids, L, Y = coll_bin(m, mb_loader)
        if ref_sids is None: ref_sids, ref_y = sids, Y
        else: L = align(L, sids, ref_sids)
        mb_logits.append(L)
        single_m = compute_binary_classification_metrics(L, ref_y)
        print(f"  seed {s}: val_auprc = {single_m['auprc']:.4f}  f1 = {single_m['binary_f1']:.4f}")
    mb_ens = torch.stack(mb_logits).mean(dim=0)
    mb_m = compute_binary_classification_metrics(mb_ens, ref_y)
    print(f"\n  4-seed ensemble: auprc = {mb_m['auprc']:.4f}  f1 = {mb_m['binary_f1']:.4f}")

    # ---- Existing T1 ensemble inference (need to recompute on val set) ----
    print("\n" + "=" * 60)
    print("Existing T1 ensemble (muku raw + legacy)")
    print("=" * 60)
    eeg_ds = EEGFeatureDataset(split_csv=None, manifest_csv=str(MANIFEST), split="val",
                               eeg_feature_root=str(EEG_ROOT), preload_in_memory=True,
                               target_shape=(26, 2000))
    eeg_loader = DataLoader(eeg_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)

    raw_logits = []  # combined legacy + muku raw
    # Legacy 12 ckpt
    for legacy, prefix in [("tcnet_eeg", "neuromm26_train_eeg_only__legacy-tcnet_eeg"),
                            ("efficientnet_v2_s_eeg", "neuromm26_train_eeg_only__legacy-efficientnet_v2_s_eeg"),
                            ("mobilenet_v3_large_eeg", "neuromm26_train_eeg_only__legacy-mobilenet_v3_large_eeg")]:
        for s in SEEDS:
            cp = CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = build_legacy_eeg_model(legacy).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, _ = coll_bin(m, eeg_loader)
            L = align(L, sids, ref_sids)
            raw_logits.append(L)
    # Muku raw 12 ckpt
    for bb, pat in [("resnet18", "muku__resnet18__seed{s}__lr0.0005_5e-05"),
                     ("tf_efficientnet_b0.ns_jft_in1k", "muku__tf_efficientnet_b0_ns_jft_in1k__seed{s}__lr0.0005_5e-05"),
                     ("convnext_pico.d1_in1k", "muku__convnext_pico_d1_in1k__seed{s}__lr0.0005_5e-05")]:
        for s in SEEDS:
            cp = CKPT_DIR / pat.format(s=s) / "best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = MukuEEGNet(num_classes=1, backbone=bb, in_chans=26, n_freqs=8, fs=500,
                           pretrained=False).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, _ = coll_bin(m, eeg_loader)
            L = align(L, sids, ref_sids)
            raw_logits.append(L)
    print(f"  Loaded {len(raw_logits)} ckpts (12 legacy + 12 muku raw)")
    existing_ens = torch.stack(raw_logits).mean(dim=0)
    ex_m = compute_binary_classification_metrics(existing_ens, ref_y)
    print(f"  Existing 24-ckpt ensemble: auprc = {ex_m['auprc']:.4f}  f1 = {ex_m['binary_f1']:.4f}")

    # ---- Combined ----
    print("\n" + "=" * 60)
    print("Combined: existing 24 + Phase C 4 = 28 ckpt")
    print("=" * 60)
    combined = torch.stack(raw_logits + mb_logits).mean(dim=0)
    cm = compute_binary_classification_metrics(combined, ref_y)
    print(f"  28-ckpt combined: auprc = {cm['auprc']:.4f}  f1 = {cm['binary_f1']:.4f}")
    print(f"  ΔAUPRC vs existing 24-ckpt: {cm['auprc'] - ex_m['auprc']:+.4f}")
    print(f"  ΔF1     vs existing 24-ckpt: {cm['binary_f1'] - ex_m['binary_f1']:+.4f}")

    # Verdict
    if cm["auprc"] > ex_m["auprc"] + 0.005:
        verdict = "✓ Phase C HELPED ensemble (ΔAUPRC > 0.005)"
    elif cm["auprc"] > ex_m["auprc"]:
        verdict = "~ Phase C marginal (0 < Δ < 0.005)"
    else:
        verdict = "✗ Phase C HURT ensemble (Δ ≤ 0)"
    print(f"\n  Verdict: {verdict}")

    out = {
        "phase_c_per_seed": [
            float(compute_binary_classification_metrics(L, ref_y)["auprc"]) for L in mb_logits
        ],
        "phase_c_ensemble": dict(mb_m),
        "existing_ensemble": dict(ex_m),
        "combined_28": dict(cm),
        "delta_auprc": float(cm["auprc"] - ex_m["auprc"]),
    }
    op = REPO / "neuromm26_results/metrics/phaseC_eval.json"
    op.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {op}")


if __name__ == "__main__":
    main()
