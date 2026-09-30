"""TTA + threshold tuning on combined muku + legacy ensemble.

Runs after muku_ensemble_eval.py to extract additional gain.
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


def make_augs():
    """5 TTA augmentations."""
    def identity(x): return x
    def gaussian(x): return x + torch.randn_like(x) * 0.01
    def time_shift(x):
        shift = int(torch.randint(-25, 26, (1,)).item())
        return torch.roll(x, shifts=shift, dims=-1)
    def channel_scale(x):
        scale = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) * 0.1 + 0.95
        return x * scale
    def amp(x):
        return x * (float(torch.rand(1).item()) * 0.1 + 0.95)
    return [("identity", identity), ("gaussian", gaussian),
            ("time_shift", time_shift), ("channel_scale", channel_scale),
            ("amp", amp)]


def make_loader():
    ds = EEGFeatureDataset(split_csv=None, manifest_csv=str(MANIFEST), split="val",
                          eeg_feature_root=str(EEG_FEATURE_ROOT), preload_in_memory=True,
                          target_shape=(26, 2000))
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=neuromm_collate)


@torch.no_grad()
def tta_logits(model, loader, augs):
    """Run K augmentations, average logits per sample."""
    model.eval()
    accum, labels, sids = [], [], []
    for batch in loader:
        batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        orig = batch["eeg"].clone()
        per_aug = []
        for _, fn in augs:
            batch["eeg"] = fn(orig)
            out = model(batch)
            logits = out["main"] if isinstance(out, dict) else out
            per_aug.append(logits.view(-1).cpu())
        accum.append(torch.stack(per_aug).mean(dim=0))
        labels.append(batch["label"].view(-1).float().cpu())
        sids.extend(batch["sample_id"])
    return sids, torch.cat(accum), torch.cat(labels)


def align(logits, sids, ref_sids):
    if sids == ref_sids: return logits
    idx = {s: i for i, s in enumerate(sids)}
    return logits[[idx[s] for s in ref_sids]]


MUKU_BACKBONES = [
    ("resnet18", "muku__resnet18__seed{seed}__lr0.0005_5e-05"),
    ("tf_efficientnet_b0.ns_jft_in1k", "muku__tf_efficientnet_b0_ns_jft_in1k__seed{seed}__lr0.0005_5e-05"),
    ("convnext_pico.d1_in1k", "muku__convnext_pico_d1_in1k__seed{seed}__lr0.0005_5e-05"),
]

LEGACY = [
    ("tcnet_eeg", "neuromm26_train_eeg_only__legacy-tcnet_eeg"),
    ("efficientnet_v2_s_eeg", "neuromm26_train_eeg_only__legacy-efficientnet_v2_s_eeg"),
    ("mobilenet_v3_large_eeg", "neuromm26_train_eeg_only__legacy-mobilenet_v3_large_eeg"),
]


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


def main():
    loader = make_loader()
    augs = make_augs()

    # Plain (no TTA) + TTA logits
    all_muku_plain, all_muku_tta = [], []
    ref_sids, ref_y = None, None

    print("Muku models (plain + TTA) ...")
    for backbone, ckpt_pattern in MUKU_BACKBONES:
        for s in SEEDS:
            ckpt = CKPT_DIR / ckpt_pattern.format(seed=s) / "best.pt"
            if not ckpt.exists():
                print(f"  miss {backbone} s{s}")
                continue
            state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            model = MukuEEGNet(num_classes=1, backbone=backbone, in_chans=26, n_freqs=8, fs=500).to(DEVICE)
            model.load_state_dict(state["model_state_dict"])
            # plain
            sids_p, logits_p, y = tta_logits(model, loader, [augs[0]])  # identity only
            # TTA
            sids_t, logits_t, _ = tta_logits(model, loader, augs)
            if ref_sids is None: ref_sids, ref_y = sids_p, y
            logits_p = align(logits_p, sids_p, ref_sids)
            logits_t = align(logits_t, sids_t, ref_sids)
            all_muku_plain.append(logits_p)
            all_muku_tta.append(logits_t)
        print(f"  {backbone} done, {len(all_muku_plain)} ckpts so far")

    # Legacy logits (plain, no TTA — already evaluated before)
    print("Legacy models ...")
    all_legacy = []
    for name, prefix in LEGACY:
        for s in SEEDS:
            ckpt = CKPT_DIR / f"{prefix}__seed{s}__lr0.0005" / "best.pt"
            if not ckpt.exists(): continue
            state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            model = build_legacy_eeg_model(name).to(DEVICE)
            model.load_state_dict(state["model_state_dict"])
            sids, logits, _ = tta_logits(model, loader, [augs[0]])
            logits = align(logits, sids, ref_sids)
            all_legacy.append(logits)
    print(f"  legacy {len(all_legacy)} ckpts")

    # Ensemble combinations
    print("\n===== Ensemble combinations =====")
    def metrics_of(L):
        ens = torch.stack(L).mean(dim=0)
        m = compute_binary_classification_metrics(ens, ref_y)
        probs = torch.sigmoid(ens).numpy()
        thr = sweep_threshold(probs, ref_y.numpy().astype(int))
        return m, thr

    results = {}
    if all_muku_plain:
        m, t = metrics_of(all_muku_plain)
        results["muku_plain_12ckpt"] = {"metrics": dict(m), "best_thr": t}
        print(f"muku_plain_12ckpt           auprc={m['auprc']:.4f}  f1@0.5={m['binary_f1']:.4f}  f1@thr={t['f1']:.4f}@{t['thr']:.2f}")

    if all_muku_tta:
        m, t = metrics_of(all_muku_tta)
        results["muku_TTA_12ckpt"] = {"metrics": dict(m), "best_thr": t}
        print(f"muku_TTA_12ckpt             auprc={m['auprc']:.4f}  f1@0.5={m['binary_f1']:.4f}  f1@thr={t['f1']:.4f}@{t['thr']:.2f}")

    if all_legacy:
        m, t = metrics_of(all_legacy)
        results["legacy_12ckpt"] = {"metrics": dict(m), "best_thr": t}
        print(f"legacy_12ckpt (ref)         auprc={m['auprc']:.4f}  f1@0.5={m['binary_f1']:.4f}  f1@thr={t['f1']:.4f}@{t['thr']:.2f}")

    if all_muku_plain and all_legacy:
        m, t = metrics_of(all_muku_plain + all_legacy)
        results["combined_24ckpt"] = {"metrics": dict(m), "best_thr": t}
        print(f"COMBINED_24ckpt             auprc={m['auprc']:.4f}  f1@0.5={m['binary_f1']:.4f}  f1@thr={t['f1']:.4f}@{t['thr']:.2f}")

    if all_muku_tta and all_legacy:
        m, t = metrics_of(all_muku_tta + all_legacy)
        results["combined_TTA_24ckpt"] = {"metrics": dict(m), "best_thr": t}
        print(f"COMBINED_TTA_24ckpt         auprc={m['auprc']:.4f}  f1@0.5={m['binary_f1']:.4f}  f1@thr={t['f1']:.4f}@{t['thr']:.2f}")

    out_path = REPO / "neuromm26_results/metrics/muku_tta_threshold.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
