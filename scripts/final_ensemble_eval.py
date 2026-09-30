"""Final ensemble eval combining all phases (legacy + muku raw + Phase C/A/B)."""

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
CWT_ROOT = REPO / "neuromm26_datasets/processed/features/cwt"
CKPT_DIR = REPO / "neuromm26_results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (0, 1, 2, 3)


@torch.no_grad()
def coll(model, loader, key="eeg"):
    model.eval()
    L, Y, S = [], [], []
    for b in loader:
        b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        out = model({"eeg": b["eeg"]} if key == "eeg" else {"spec": b["spec"]})
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
    eeg_ds = EEGFeatureDataset(split_csv=None, manifest_csv=str(MANIFEST), split="val",
                               eeg_feature_root=str(EEG_ROOT), preload_in_memory=True,
                               target_shape=(26, 2000))
    eeg_loader = DataLoader(eeg_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)

    ref_sids = None; ref_y = None
    groups = {}

    # ---- Legacy 12 ----
    legacy = []
    for name, prefix in [("tcnet_eeg", "neuromm26_train_eeg_only__legacy-tcnet_eeg"),
                          ("efficientnet_v2_s_eeg", "neuromm26_train_eeg_only__legacy-efficientnet_v2_s_eeg"),
                          ("mobilenet_v3_large_eeg", "neuromm26_train_eeg_only__legacy-mobilenet_v3_large_eeg")]:
        for s in SEEDS:
            cp = CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = build_legacy_eeg_model(name).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, Y = coll(m, eeg_loader, "eeg")
            if ref_sids is None: ref_sids, ref_y = sids, Y
            else: L = align(L, sids, ref_sids)
            legacy.append(L)
    groups["legacy_12"] = legacy
    print(f"Legacy 12 loaded")

    # ---- Muku raw 12 ----
    muku_raw = []
    for bb, pat in [("resnet18", "muku__resnet18__seed{s}__lr0.0005_5e-05"),
                     ("tf_efficientnet_b0.ns_jft_in1k", "muku__tf_efficientnet_b0_ns_jft_in1k__seed{s}__lr0.0005_5e-05"),
                     ("convnext_pico.d1_in1k", "muku__convnext_pico_d1_in1k__seed{s}__lr0.0005_5e-05")]:
        for s in SEEDS:
            cp = CKPT_DIR / pat.format(s=s) / "best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = MukuEEGNet(num_classes=1, backbone=bb, in_chans=26, n_freqs=8, fs=500,
                           pretrained=False).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, _ = coll(m, eeg_loader, "eeg")
            L = align(L, sids, ref_sids)
            muku_raw.append(L)
    groups["muku_raw_12"] = muku_raw
    print(f"Muku raw 12 loaded")

    # ---- Phase C: multi-band 4 ----
    mb_ds = MultiBandFeatureDataset(multiband_root=str(MB_ROOT), manifest_csv=str(MANIFEST),
                                    split="val", preload_in_memory=True, target_shape=(130, 2000))
    mb_loader = DataLoader(mb_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    mb = []
    for s in SEEDS:
        cp = CKPT_DIR / f"muku_mb__resnet18__seed{s}__lr0.0005_5e-05/best.pt"
        state = torch.load(cp, map_location=DEVICE, weights_only=False)
        m = MukuEEGNet(num_classes=1, backbone="resnet18", in_chans=130, n_freqs=8, fs=500,
                       pretrained=False).to(DEVICE)
        m.load_state_dict(state["model_state_dict"])
        sids, L, _ = coll(m, mb_loader, "eeg")
        L = align(L, sids, ref_sids)
        mb.append(L)
    groups["phaseC_multiband_4"] = mb
    print(f"Phase C multiband 4 loaded")

    # ---- Phase A: STFT 4 ----
    stft_ds = SpecFeatureDataset(spec_root=str(STFT_ROOT), manifest_csv=str(MANIFEST),
                                  split="val", preload_in_memory=True)
    stft_loader = DataLoader(stft_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    stft = []
    for s in SEEDS:
        cp = CKPT_DIR / f"spec_stft__resnet18__seed{s}__lr0.0005_5e-05/best.pt"
        state = torch.load(cp, map_location=DEVICE, weights_only=False)
        m = SpecCNN(num_classes=1, backbone="resnet18", in_chans=26, pretrained=False).to(DEVICE)
        m.load_state_dict(state["model_state_dict"])
        sids, L, _ = coll(m, stft_loader, "spec")
        L = align(L, sids, ref_sids)
        stft.append(L)
    groups["phaseA_stft_4"] = stft
    print(f"Phase A STFT 4 loaded")

    # ---- Phase B: CWT 4 ----
    cwt_ds = SpecFeatureDataset(spec_root=str(CWT_ROOT), manifest_csv=str(MANIFEST),
                                 split="val", preload_in_memory=True)
    cwt_loader = DataLoader(cwt_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    cwt = []
    for s in SEEDS:
        cp = CKPT_DIR / f"spec_cwt__resnet18__seed{s}__lr0.0005_5e-05/best.pt"
        state = torch.load(cp, map_location=DEVICE, weights_only=False)
        m = SpecCNN(num_classes=1, backbone="resnet18", in_chans=26, pretrained=False).to(DEVICE)
        m.load_state_dict(state["model_state_dict"])
        sids, L, _ = coll(m, cwt_loader, "spec")
        L = align(L, sids, ref_sids)
        cwt.append(L)
    groups["phaseB_cwt_4"] = cwt
    print(f"Phase B CWT 4 loaded")

    # ---- Evaluate combinations ----
    print("\n" + "=" * 70)
    print("ENSEMBLE COMBINATIONS")
    print("=" * 70)

    def ens(name, *grps):
        all_L = []
        for g in grps: all_L.extend(g)
        e = torch.stack(all_L).mean(dim=0)
        m = compute_binary_classification_metrics(e, ref_y)
        n = len(all_L)
        print(f"  {name:60s} N={n:3d}  auprc={m['auprc']:.4f}  f1={m['binary_f1']:.4f}  acc={m['accuracy']:.4f}")
        return m

    results = {}
    print("\n--- Individual groups (4-seed ensemble each) ---")
    for k, g in groups.items():
        results[k] = ens(k, g)

    print("\n--- Cumulative additions ---")
    results["legacy_only"] = results["legacy_12"]
    results["muku_only"] = results["muku_raw_12"]
    results["legacy+muku_24"] = ens("legacy+muku_24", groups["legacy_12"], groups["muku_raw_12"])
    results["+phaseC_28"] = ens("+phaseC_28", groups["legacy_12"], groups["muku_raw_12"], groups["phaseC_multiband_4"])
    results["+phaseA_32"] = ens("+phaseA_32", groups["legacy_12"], groups["muku_raw_12"],
                                groups["phaseC_multiband_4"], groups["phaseA_stft_4"])
    results["+phaseB_36"] = ens("+phaseB_36", groups["legacy_12"], groups["muku_raw_12"],
                                groups["phaseC_multiband_4"], groups["phaseA_stft_4"], groups["phaseB_cwt_4"])

    print("\n--- Alternative: skip Phase A (HURT in standalone), try B+C only ---")
    results["legacy+muku+B_28"] = ens("legacy+muku+B_28", groups["legacy_12"], groups["muku_raw_12"], groups["phaseB_cwt_4"])
    results["legacy+muku+B+C_32"] = ens("legacy+muku+B+C_32", groups["legacy_12"], groups["muku_raw_12"],
                                        groups["phaseC_multiband_4"], groups["phaseB_cwt_4"])
    results["muku+B_only_16"] = ens("muku+B_only_16", groups["muku_raw_12"], groups["phaseB_cwt_4"])
    results["muku+B+C_20"] = ens("muku+B+C_20", groups["muku_raw_12"], groups["phaseC_multiband_4"], groups["phaseB_cwt_4"])

    print("\n=== Best ensemble found ===")
    best_name = max(results.keys(), key=lambda k: results[k]["auprc"])
    print(f"  {best_name}  auprc={results[best_name]['auprc']:.4f}  f1={results[best_name]['binary_f1']:.4f}")

    out = {k: dict(v) for k, v in results.items()}
    op = REPO / "neuromm26_results/metrics/final_ensemble_eval.json"
    op.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {op}")


if __name__ == "__main__":
    main()
