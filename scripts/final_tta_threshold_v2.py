"""TTA + threshold tune on FINAL T1 28-ckpt and T2 16-ckpt winners (post-V2)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets import EEGFeatureDataset, NeuroMMMultimodalFeatureDataset
from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.datasets.spec_dataset import SpecFeatureDataset
from neuromm26_baseline.datasets.spec_video_dataset import SpecVideoFeatureDataset
from neuromm26_baseline.models import build_legacy_eeg_model
from neuromm26_baseline.models.multimodal_late_fusion import EEGVideoLateFusion
from neuromm26_baseline.models.muku_eegnet import MukuEEGNet
from neuromm26_baseline.models.muku_eeg_video import MukuEEGVideoNet
from neuromm26_baseline.models.spec_cnn import SpecCNN
from neuromm26_baseline.models.spec_cnn_concat import ConcatSpecCNN
from neuromm26_baseline.models.spec_cnn_concat_video import ConcatSpecVideoCNN
from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
MANIFEST = REPO / "neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv"
EEG_ROOT = REPO / "neuromm26_datasets/processed/features/eeg"
VID_ROOT = REPO / "neuromm26_datasets/processed/features/video"
CWT_ROOT = REPO / "neuromm26_datasets/processed/features/cwt"
CKPT_DIR = REPO / "neuromm26_results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (0, 1, 2, 3)


def fd(name):
    return int(np.load(next((VID_ROOT / name).glob("*.npy"))).shape[-1])


def make_augs_eeg():
    def identity(x): return x
    def gaussian(x): return x + torch.randn_like(x) * 0.01
    def shift_pos(x): return torch.roll(x, shifts=int(torch.randint(20, 50, (1,)).item()), dims=-1)
    def shift_neg(x): return torch.roll(x, shifts=-int(torch.randint(20, 50, (1,)).item()), dims=-1)
    def ch_scale(x):
        scale = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) * 0.1 + 0.95
        return x * scale
    return [identity, gaussian, shift_pos, shift_neg, ch_scale]


@torch.no_grad()
def coll_bin_plain(model, loader, key="eeg"):
    model.eval()
    L, Y, S = [], [], []
    for b in loader:
        b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        out = model(b)
        logits = out["main"] if isinstance(out, dict) else out
        L.append(logits.view(-1).cpu())
        Y.append(b["label"].view(-1).float().cpu())
        S.extend(b["sample_id"])
    return S, torch.cat(L), torch.cat(Y)


@torch.no_grad()
def coll_bin_tta_eeg(model, loader, augs):
    """TTA on raw EEG (for muku raw models)."""
    model.eval()
    accum, labels, sids = [], [], []
    for b in loader:
        b = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        orig_eeg = b["eeg"].clone()
        per_aug = []
        for fn in augs:
            b["eeg"] = fn(orig_eeg)
            out = model(b)
            logits = out["main"] if isinstance(out, dict) else out
            per_aug.append(logits.view(-1).cpu())
        accum.append(torch.stack(per_aug).mean(dim=0))
        labels.append(b["label"].view(-1).float().cpu())
        sids.extend(b["sample_id"])
    return sids, torch.cat(accum), torch.cat(labels)


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


def precision_at_recall(probs, labels, target=0.70):
    n_pos = int(labels.sum())
    if n_pos == 0: return None
    order = np.argsort(-probs)
    sl = labels[order]; sp = probs[order]
    tp = np.cumsum(sl); fp = np.cumsum(1 - sl)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    above = np.where(recall >= target)[0]
    if len(above) == 0: return None
    k = above[0]
    return {"thr": float(sp[k]), "precision": float(precision[k]), "recall": float(recall[k])}


def run_T1():
    print("\n" + "=" * 70)
    print("T1 FINAL — muku raw (12) + CWT-Morlet (12) + V2-Concat (4) = 28")
    print("=" * 70)
    augs = make_augs_eeg()

    eeg_ds = EEGFeatureDataset(split_csv=None, manifest_csv=str(MANIFEST), split="val",
                               eeg_feature_root=str(EEG_ROOT), preload_in_memory=True,
                               target_shape=(26, 2000))
    eeg_loader = DataLoader(eeg_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    cwt_ds = SpecFeatureDataset(spec_root=str(CWT_ROOT), manifest_csv=str(MANIFEST),
                                 split="val", preload_in_memory=True)
    cwt_loader = DataLoader(cwt_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)

    ref_sids, ref_y = None, None

    # Muku raw 12 - both plain and TTA
    muku_plain, muku_tta = [], []
    for bb, pat in [
        ("resnet18", "muku__resnet18__seed{s}__lr0.0005_5e-05"),
        ("tf_efficientnet_b0.ns_jft_in1k", "muku__tf_efficientnet_b0_ns_jft_in1k__seed{s}__lr0.0005_5e-05"),
        ("convnext_pico.d1_in1k", "muku__convnext_pico_d1_in1k__seed{s}__lr0.0005_5e-05"),
    ]:
        for s in SEEDS:
            cp = CKPT_DIR / pat.format(s=s) / "best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = MukuEEGNet(num_classes=1, backbone=bb, in_chans=26, n_freqs=8, fs=500,
                           pretrained=False).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids_p, lp, Y = coll_bin_plain(m, eeg_loader)
            if ref_sids is None: ref_sids, ref_y = sids_p, Y
            else: lp = align(lp, sids_p, ref_sids)
            muku_plain.append(lp)
            sids_t, lt, _ = coll_bin_tta_eeg(m, eeg_loader, augs)
            lt = align(lt, sids_t, ref_sids)
            muku_tta.append(lt)

    # CWT-Morlet 12 (plain only — can't easily TTA on cached spec)
    cwt = []
    for bb, pat in [
        ("resnet18", "spec_cwt__resnet18__seed{s}__lr0.0005_5e-05"),
        ("tf_efficientnet_b0.ns_jft_in1k", "spec_cwt__tf_efficientnet_b0_ns_jft_in1k__seed{s}__lr0.0005_5e-05"),
        ("convnext_pico.d1_in1k", "spec_cwt__convnext_pico_d1_in1k__seed{s}__lr0.0005_5e-05"),
    ]:
        for s in SEEDS:
            cp = CKPT_DIR / pat.format(s=s) / "best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = SpecCNN(num_classes=1, backbone=bb, in_chans=26, pretrained=False).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids, L, _ = coll_bin_plain(m, cwt_loader)
            L = align(L, sids, ref_sids)
            cwt.append(L)

    # V2 ConcatCWT 4
    v2 = []
    for s in SEEDS:
        cp = CKPT_DIR / f"concat_cwt__convnext_tiny_fb_in22k_ft_in1k_384__seed{s}__lr0.0005_5e-05/best.pt"
        state = torch.load(cp, map_location=DEVICE, weights_only=False)
        m = ConcatSpecCNN(num_classes=1, backbone="convnext_tiny.fb_in22k_ft_in1k_384",
                          target_size=384, pretrained=False).to(DEVICE)
        m.load_state_dict(state["model_state_dict"])
        sids, L, _ = coll_bin_plain(m, cwt_loader)
        L = align(L, sids, ref_sids)
        v2.append(L)

    Y_np = ref_y.numpy().astype(int)

    def eval_combo(name, logits_list):
        ens = torch.stack(logits_list).mean(dim=0)
        m = compute_binary_classification_metrics(ens, ref_y)
        probs = torch.sigmoid(ens).numpy()
        thr = sweep_threshold(probs, Y_np)
        ps70 = precision_at_recall(probs, Y_np, 0.70)
        print(f"  {name:45s}  AUPRC={m['auprc']:.4f}  F1@0.5={m['binary_f1']:.4f}  "
              f"F1@thr={thr['f1']:.4f}@{thr['thr']:.2f}  P@S70={ps70['precision']:.4f}")
        return {"metrics": dict(m), "best_thr": thr, "P@S70": ps70}

    results = {}
    results["plain_28"] = eval_combo("PLAIN: muku+CWT+V2 (28)", muku_plain + cwt + v2)
    results["TTA_28"] = eval_combo("TTA: muku TTA + CWT + V2 (28)", muku_tta + cwt + v2)
    return results


def run_T2():
    print("\n" + "=" * 70)
    print("T2 FINAL — muku (12) + V2 (4) = 16")
    print("=" * 70)
    augs = make_augs_eeg()
    video_dim = fd("timesformer-k400")

    mm_ds = NeuroMMMultimodalFeatureDataset(
        manifest_csv=str(MANIFEST), split="val",
        eeg_feature_root=str(EEG_ROOT), video_feature_root=str(VID_ROOT),
        video_feature_name="timesformer-k400", preload_in_memory=True,
        target_shape=(26, 2000), require_video_feature=True,
    )
    mm_loader = DataLoader(mm_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)
    sv_ds = SpecVideoFeatureDataset(
        spec_root=str(CWT_ROOT), video_feature_root=str(VID_ROOT),
        video_feature_name="timesformer-k400", manifest_csv=str(MANIFEST),
        split="val", preload_in_memory=True, label_mode="binary",
    )
    sv_loader = DataLoader(sv_ds, batch_size=48, shuffle=False, num_workers=0, collate_fn=neuromm_collate)

    ref_sids, ref_y = None, None
    muku_plain, muku_tta = [], []

    for bb, pat in [
        ("resnet18", "muku_t2__resnet18__timesformer-k400__seed{s}__lr0.0005_5e-05"),
        ("tf_efficientnet_b0.ns_jft_in1k", "muku_t2__tf_efficientnet_b0_ns_jft_in1k__timesformer-k400__seed{s}__lr0.0005_5e-05"),
        ("convnext_pico.d1_in1k", "muku_t2__convnext_pico_d1_in1k__timesformer-k400__seed{s}__lr0.0005_5e-05"),
    ]:
        for s in SEEDS:
            cp = CKPT_DIR / pat.format(s=s) / "best.pt"
            state = torch.load(cp, map_location=DEVICE, weights_only=False)
            m = MukuEEGVideoNet(num_classes=1, backbone=bb, in_chans=26, n_freqs=8, fs=500,
                                video_feat_dim=video_dim, video_hidden=state.get("video_hidden", 256),
                                video_dropout=0.2, pretrained=False).to(DEVICE)
            m.load_state_dict(state["model_state_dict"])
            sids_p, lp, Y = coll_bin_plain(m, mm_loader)
            if ref_sids is None: ref_sids, ref_y = sids_p, Y
            else: lp = align(lp, sids_p, ref_sids)
            muku_plain.append(lp)
            sids_t, lt, _ = coll_bin_tta_eeg(m, mm_loader, augs)
            lt = align(lt, sids_t, ref_sids)
            muku_tta.append(lt)

    # V2 4
    v2 = []
    for s in SEEDS:
        cp = CKPT_DIR / f"t2v2__cwt__convnext_tiny_fb_in22k_ft_in1k_384__timesformer-k400__seed{s}__lr0.0005_5e-05/best.pt"
        state = torch.load(cp, map_location=DEVICE, weights_only=False)
        m = ConcatSpecVideoCNN(num_classes=1, backbone="convnext_tiny.fb_in22k_ft_in1k_384",
                               target_size=384, video_feat_dim=video_dim,
                               video_hidden=state.get("video_hidden", 256),
                               video_dropout=0.2, pretrained=False).to(DEVICE)
        m.load_state_dict(state["model_state_dict"])
        sids, L, _ = coll_bin_plain(m, sv_loader)
        L = align(L, sids, ref_sids)
        v2.append(L)

    Y_np = ref_y.numpy().astype(int)

    def eval_combo(name, logits_list):
        ens = torch.stack(logits_list).mean(dim=0)
        m = compute_binary_classification_metrics(ens, ref_y)
        probs = torch.sigmoid(ens).numpy()
        thr = sweep_threshold(probs, Y_np)
        ps70 = precision_at_recall(probs, Y_np, 0.70)
        print(f"  {name:45s}  AUPRC={m['auprc']:.4f}  F1@0.5={m['binary_f1']:.4f}  "
              f"F1@thr={thr['f1']:.4f}@{thr['thr']:.2f}  P@S70={ps70['precision']:.4f}")
        return {"metrics": dict(m), "best_thr": thr, "P@S70": ps70}

    results = {}
    results["plain_16"] = eval_combo("PLAIN: muku+V2 (16)", muku_plain + v2)
    results["TTA_16"] = eval_combo("TTA: muku TTA + V2 (16)", muku_tta + v2)
    return results


def main():
    res = {"T1": run_T1(), "T2": run_T2()}
    op = REPO / "neuromm26_results/metrics/final_tta_threshold_v2.json"
    op.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nSaved: {op}")


if __name__ == "__main__":
    main()
