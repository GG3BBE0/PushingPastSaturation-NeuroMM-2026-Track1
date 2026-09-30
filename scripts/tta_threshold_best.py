"""TTA + threshold tuning on best ensemble (muku raw 12 + CWT 3bb 12 = 24).

For TTA, we apply augmentations only to the muku raw 12 (the spec models can't
easily be TTA'd because spec is precomputed; we'd need to recompute STFT/CWT
per augmented sample). So we TTA only the raw EEG portion and combine with
the existing CWT logits.
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
from neuromm26_baseline.datasets.spec_dataset import SpecFeatureDataset
from neuromm26_baseline.models import build_legacy_eeg_model
from neuromm26_baseline.models.muku_eegnet import MukuEEGNet
from neuromm26_baseline.models.spec_cnn import SpecCNN
from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
MANIFEST = REPO / "neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv"
EEG_ROOT = REPO / "neuromm26_datasets/processed/features/eeg"
CWT_ROOT = REPO / "neuromm26_datasets/processed/features/cwt"
CKPT_DIR = REPO / "neuromm26_results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (0, 1, 2, 3)


def make_augs():
    def identity(x): return x
    def gaussian(x): return x + torch.randn_like(x) * 0.01
    def time_shift_pos(x):
        return torch.roll(x, shifts=int(torch.randint(20, 60, (1,)).item()), dims=-1)
    def time_shift_neg(x):
        return torch.roll(x, shifts=-int(torch.randint(20, 60, (1,)).item()), dims=-1)
    def channel_scale(x):
        scale = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) * 0.1 + 0.95
        return x * scale
    return [("id", identity), ("gauss", gaussian),
            ("shift+", time_shift_pos), ("shift-", time_shift_neg),
            ("ch_scale", channel_scale)]


@torch.no_grad()
def tta_logits(model, loader, augs, key="eeg"):
    model.eval()
    accum, labels, sids = [], [], []
    for b in loader:
        b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        if key == "eeg":
            orig = b["eeg"].clone()
            per_aug = []
            for _, fn in augs:
                b["eeg"] = fn(orig)
                out = model({"eeg": b["eeg"]})
                per_aug.append((out["main"] if isinstance(out, dict) else out).view(-1).cpu())
            accum.append(torch.stack(per_aug).mean(dim=0))
        else:
            out = model({"spec": b["spec"]})
            accum.append((out["main"] if isinstance(out, dict) else out).view(-1).cpu())
        labels.append(b["label"].view(-1).float().cpu())
        sids.extend(b["sample_id"])
    return sids, torch.cat(accum), torch.cat(labels)


@torch.no_grad()
def plain_logits(model, loader, key="eeg"):
    model.eval()
    L, Y, S = [], [], []
    for b in loader:
        b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        out = model({"eeg": b["eeg"]} if key == "eeg" else {"spec": b["spec"]})
        L.append((out["main"] if isinstance(out, dict) else out).view(-1).cpu())
        Y.append(b["label"].view(-1).float().cpu())
        S.extend(b["sample_id"])
    return S, torch.cat(L), torch.cat(Y)


def align(L, sids, ref):
    if sids == ref: return L
    idx = {s: i for i, s in enumerate(sids)}
    return L[[idx[s] for s in ref]]


def sweep_threshold(probs, labels):
    best = {"f1": -1, "thr": 0.5}
    for thr in np.arange(0.05, 0.95 + 1e-9, 0.01):
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": f1, "p": p, "r": r}
    return best


def precision_at_sensitivity(probs, labels, target_recall=0.70):
    n_pos = int(labels.sum())
    if n_pos == 0: return None
    order = np.argsort(-probs)
    sl = labels[order]
    sp = probs[order]
    tp = np.cumsum(sl)
    fp = np.cumsum(1 - sl)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    above = np.where(recall >= target_recall)[0]
    if len(above) == 0: return None
    k = above[0]
    return {"thr": float(sp[k]), "precision": float(precision[k]), "recall": float(recall[k])}


def main():
    eeg_ds = EEGFeatureDataset(split_csv=None, manifest_csv=str(MANIFEST), split="val",
                               eeg_feature_root=str(EEG_ROOT), preload_in_memory=True,
                               target_shape=(26, 2000))
    eeg_loader = DataLoader(eeg_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    cwt_ds = SpecFeatureDataset(spec_root=str(CWT_ROOT), manifest_csv=str(MANIFEST),
                                 split="val", preload_in_memory=True)
    cwt_loader = DataLoader(cwt_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)

    augs = make_augs()
    ref_sids = None; ref_y = None

    # ---- Muku raw 12 (plain + TTA) ----
    print("Muku raw 12 (plain + TTA) ...")
    muku_plain, muku_tta = [], []
    for bb, pat in [("resnet18", "muku__resnet18__seed{s}__lr0.0005_5e-05"),
                     ("tf_efficientnet_b0.ns_jft_in1k", "muku__tf_efficientnet_b0_ns_jft_in1k__seed{s}__lr0.0005_5e-05"),
                     ("convnext_pico.d1_in1k", "muku__convnext_pico_d1_in1k__seed{s}__lr0.0005_5e-05")]:
        for s in SEEDS:
            cp = CKPT_DIR / pat.format(s=s) / "best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = MukuEEGNet(num_classes=1, backbone=bb, in_chans=26, n_freqs=8, fs=500,
                           pretrained=False).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            # plain
            sids_p, lp, Y = plain_logits(m, eeg_loader, "eeg")
            if ref_sids is None: ref_sids, ref_y = sids_p, Y
            else: lp = align(lp, sids_p, ref_sids)
            muku_plain.append(lp)
            # TTA
            sids_t, lt, _ = tta_logits(m, eeg_loader, augs, "eeg")
            lt = align(lt, sids_t, ref_sids)
            muku_tta.append(lt)

    # ---- CWT 3bb 12 (plain only — spec precomputed) ----
    print("CWT 3bb 12 (plain) ...")
    cwt_plain = []
    for bb, pat in [("resnet18", "spec_cwt__resnet18__seed{s}__lr0.0005_5e-05"),
                     ("tf_efficientnet_b0.ns_jft_in1k", "spec_cwt__tf_efficientnet_b0_ns_jft_in1k__seed{s}__lr0.0005_5e-05"),
                     ("convnext_pico.d1_in1k", "spec_cwt__convnext_pico_d1_in1k__seed{s}__lr0.0005_5e-05")]:
        for s in SEEDS:
            cp = CKPT_DIR / pat.format(s=s) / "best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = SpecCNN(num_classes=1, backbone=bb, in_chans=26, pretrained=False).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, _ = plain_logits(m, cwt_loader, "spec")
            L = align(L, sids, ref_sids)
            cwt_plain.append(L)

    # ---- Legacy 12 (plain) for F1-optimized config ----
    print("Legacy 12 (plain) ...")
    legacy_plain = []
    for name, prefix in [("tcnet_eeg", "neuromm26_train_eeg_only__legacy-tcnet_eeg"),
                          ("efficientnet_v2_s_eeg", "neuromm26_train_eeg_only__legacy-efficientnet_v2_s_eeg"),
                          ("mobilenet_v3_large_eeg", "neuromm26_train_eeg_only__legacy-mobilenet_v3_large_eeg")]:
        for s in SEEDS:
            cp = CKPT_DIR / f"{prefix}__seed{s}__lr0.0005/best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = build_legacy_eeg_model(name).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, _ = plain_logits(m, eeg_loader, "eeg")
            L = align(L, sids, ref_sids)
            legacy_plain.append(L)

    Y_np = ref_y.numpy().astype(int)

    def eval_combo(name, logits_list):
        ens = torch.stack(logits_list).mean(dim=0)
        m = compute_binary_classification_metrics(ens, ref_y)
        probs = torch.sigmoid(ens).numpy()
        thr = sweep_threshold(probs, Y_np)
        ps70 = precision_at_sensitivity(probs, Y_np, 0.70)
        print(f"  {name:50s}  auprc={m['auprc']:.4f}  f1@0.5={m['binary_f1']:.4f}  f1@thr={thr['f1']:.4f}@{thr['thr']:.2f}  P@S70={ps70['precision']:.4f}")
        return {"metrics": dict(m), "best_thr": thr, "P@S70": ps70}

    print("\n===== Results =====")
    res = {}
    res["muku_plain"] = eval_combo("muku raw 12 plain", muku_plain)
    res["muku_TTA"] = eval_combo("muku raw 12 TTA", muku_tta)
    res["cwt_plain"] = eval_combo("CWT 3bb 12 plain", cwt_plain)
    res["best_AUPRC_plain"] = eval_combo("BEST: muku+CWT 24 plain", muku_plain + cwt_plain)
    res["best_AUPRC_TTA"] = eval_combo("BEST: muku TTA + CWT 24", muku_tta + cwt_plain)
    res["best_F1_plain"] = eval_combo("F1: legacy+muku+CWT 36 plain", legacy_plain + muku_plain + cwt_plain)
    res["best_F1_TTA"] = eval_combo("F1: legacy+muku TTA+CWT 36", legacy_plain + muku_tta + cwt_plain)

    op = REPO / "neuromm26_results/metrics/tta_threshold_best.json"
    op.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nSaved: {op}")


if __name__ == "__main__":
    main()
