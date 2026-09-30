"""4-seed logit-averaging ensemble evaluation on val for T1/T2/T3.

For each task it:
  1. Reconstructs the same architecture used in training
  2. Loads each seed's best.pt
  3. Runs val inference, collects logits per sample_id
  4. Averages logits across 4 seeds (sample-aligned)
  5. Computes the canonical metric on the averaged logits
  6. Prints single-seed vs ensemble comparison

Run from repo root:
    python scripts/ensemble_eval.py
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


def collect_binary_logits(model, loader) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list, labels_list, sid_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            logits = model(batch).view(-1).cpu()
            logits_list.append(logits)
            labels_list.append(batch["label"].view(-1).float().cpu())
            sid_list.extend(batch["sample_id"])
    return sid_list, torch.cat(logits_list), torch.cat(labels_list)


def collect_multiclass_logits(model, loader, num_classes: int) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list, labels_list, sid_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            logits = model(batch).view(-1, num_classes).cpu()
            logits_list.append(logits)
            labels_list.append(batch["label"].view(-1).long().cpu())
            sid_list.extend(batch["sample_id"])
    return sid_list, torch.cat(logits_list), torch.cat(labels_list)


def detect_video_feature_dim(name: str) -> int:
    root = VIDEO_FEATURE_ROOT / name
    sample = next(root.glob("*.npy"))
    arr = np.load(sample)
    return int(arr.shape[-1]) if arr.ndim > 1 else int(arr.shape[0])


# ============================================================
# Task 1
# ============================================================
def run_task1() -> dict:
    print("\n========== Task 1: legacy/tcnet_eeg (EEG-only binary) ==========")
    ds = EEGFeatureDataset(
        split_csv=None,
        manifest_csv=str(MANIFEST),
        split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        preload_in_memory=True,
        target_shape=(26, 2000),
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0,
                        collate_fn=neuromm_collate)

    per_seed = {}
    sid_ref = None
    label_ref = None
    aligned_logits = []
    for s in SEEDS:
        ckpt_path = CKPT_DIR / f"neuromm26_train_eeg_only__legacy-tcnet_eeg__seed{s}__lr0.0005/best.pt"
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = build_legacy_eeg_model("tcnet_eeg").to(DEVICE)
        model.load_state_dict(state["model_state_dict"])
        sids, logits, labels = collect_binary_logits(model, loader)
        if sid_ref is None:
            sid_ref, label_ref = sids, labels
        else:
            assert sids == sid_ref, "sample order changed between seeds"
        aligned_logits.append(logits)
        m = compute_binary_classification_metrics(logits, labels)
        per_seed[s] = m
        print(f"  seed={s}  auprc={m['auprc']:.4f}  f1={m['binary_f1']:.4f}  acc={m['accuracy']:.4f}")

    mean_auprc = float(np.mean([per_seed[s]["auprc"] for s in SEEDS]))
    std_auprc = float(np.std([per_seed[s]["auprc"] for s in SEEDS]))

    # Ensemble: mean of logits then sigmoid (mathematically same as mean of sigmoids? No, different — pick logit mean.)
    ens_logits = torch.stack(aligned_logits, dim=0).mean(dim=0)
    ens_m = compute_binary_classification_metrics(ens_logits, label_ref)
    print(f"\n  mean of 4 seeds   auprc = {mean_auprc:.4f} ± {std_auprc:.4f}")
    print(f"  ENSEMBLE (logit avg)  auprc = {ens_m['auprc']:.4f}  f1 = {ens_m['binary_f1']:.4f}  acc = {ens_m['accuracy']:.4f}")
    print(f"  → gain vs mean   ΔAUPRC = {ens_m['auprc']-mean_auprc:+.4f}")
    return {"task": "T1", "per_seed": per_seed, "mean": mean_auprc, "std": std_auprc, "ensemble": ens_m}


# ============================================================
# Task 2
# ============================================================
def run_task2() -> dict:
    print("\n========== Task 2: fusion/eegnet+timesformer-k400 (EEG+Video binary) ==========")
    feat_dim = detect_video_feature_dim("timesformer-k400")
    ds = NeuroMMMultimodalFeatureDataset(
        manifest_csv=str(MANIFEST),
        split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        video_feature_root=str(VIDEO_FEATURE_ROOT),
        video_feature_name="timesformer-k400",
        preload_in_memory=True,
        target_shape=(26, 2000),
        require_video_feature=True,
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0,
                        collate_fn=neuromm_collate)

    per_seed = {}
    sid_ref = None
    label_ref = None
    aligned_logits = []
    for s in SEEDS:
        ckpt_path = CKPT_DIR / f"eeg_video_fusion__eegnet__timesformer-k400__seed{s}__lr0.0005/best.pt"
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = EEGVideoLateFusion(
            eeg_model_name="eegnet",
            video_feature_dim=feat_dim,
            video_hidden_dim=state.get("video_hidden_dim", 256),
            video_dropout=0.2,
        ).to(DEVICE)
        model.load_state_dict(state["model_state_dict"])
        sids, logits, labels = collect_binary_logits(model, loader)
        if sid_ref is None:
            sid_ref, label_ref = sids, labels
        else:
            assert sids == sid_ref
        aligned_logits.append(logits)
        m = compute_binary_classification_metrics(logits, labels)
        per_seed[s] = m
        print(f"  seed={s}  auprc={m['auprc']:.4f}  f1={m['binary_f1']:.4f}  acc={m['accuracy']:.4f}")

    mean_auprc = float(np.mean([per_seed[s]["auprc"] for s in SEEDS]))
    std_auprc = float(np.std([per_seed[s]["auprc"] for s in SEEDS]))
    ens_logits = torch.stack(aligned_logits, dim=0).mean(dim=0)
    ens_m = compute_binary_classification_metrics(ens_logits, label_ref)
    print(f"\n  mean of 4 seeds   auprc = {mean_auprc:.4f} ± {std_auprc:.4f}")
    print(f"  ENSEMBLE (logit avg)  auprc = {ens_m['auprc']:.4f}  f1 = {ens_m['binary_f1']:.4f}  acc = {ens_m['accuracy']:.4f}")
    print(f"  → gain vs mean   ΔAUPRC = {ens_m['auprc']-mean_auprc:+.4f}")
    return {"task": "T2", "per_seed": per_seed, "mean": mean_auprc, "std": std_auprc, "ensemble": ens_m}


# ============================================================
# Task 3
# ============================================================
def run_task3() -> dict:
    print("\n========== Task 3: task3_fusion/convnext_large_eeg+videomae-large (5-class) ==========")
    feat_dim = detect_video_feature_dim("videomae-large")
    ds = Task3MultimodalFeatureDataset(
        manifest_csv=str(MANIFEST),
        split="val",
        eeg_feature_root=str(EEG_FEATURE_ROOT),
        video_feature_root=str(VIDEO_FEATURE_ROOT),
        video_feature_name="videomae-large",
        target_shape=(26, 2000),
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0,
                        collate_fn=neuromm_collate)

    per_seed = {}
    sid_ref = None
    label_ref = None
    aligned_logits = []
    for s in SEEDS:
        ckpt_path = CKPT_DIR / f"task3_fusion__convnext_large_eeg__videomae-large__seed{s}__lr0.0005/best.pt"
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = EEGVideoLateFusion5Class(
            eeg_model_name="convnext_large_eeg",
            video_feature_dim=feat_dim,
            video_hidden_dim=state.get("video_hidden_dim", 256),
            video_dropout=0.2,
            num_classes=state.get("num_classes", 5),
        ).to(DEVICE)
        model.load_state_dict(state["model_state_dict"])
        sids, logits, labels = collect_multiclass_logits(model, loader, num_classes=5)
        if sid_ref is None:
            sid_ref, label_ref = sids, labels
        else:
            assert sids == sid_ref
        aligned_logits.append(logits)
        m = compute_multiclass_metrics(logits, labels, num_classes=5)
        per_seed[s] = m
        print(f"  seed={s}  wf1={m['weighted_f1']:.4f}  mf1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}")

    mean_wf1 = float(np.mean([per_seed[s]["weighted_f1"] for s in SEEDS]))
    std_wf1 = float(np.std([per_seed[s]["weighted_f1"] for s in SEEDS]))
    ens_logits = torch.stack(aligned_logits, dim=0).mean(dim=0)
    ens_m = compute_multiclass_metrics(ens_logits, label_ref, num_classes=5)
    print(f"\n  mean of 4 seeds   wf1 = {mean_wf1:.4f} ± {std_wf1:.4f}")
    print(f"  ENSEMBLE (logit avg)  wf1 = {ens_m['weighted_f1']:.4f}  mf1 = {ens_m['macro_f1']:.4f}  acc = {ens_m['accuracy']:.4f}")
    print(f"  → gain vs mean   ΔwF1 = {ens_m['weighted_f1']-mean_wf1:+.4f}")
    return {"task": "T3", "per_seed": per_seed, "mean": mean_wf1, "std": std_wf1, "ensemble": ens_m}


def main() -> None:
    summary = {
        "T1": run_task1(),
        "T2": run_task2(),
        "T3": run_task3(),
    }
    print("\n\n=========== SUMMARY ===========")
    print(f"{'Task':<6} {'metric':<12} {'mean±std':<22} {'ENSEMBLE':<10} {'Δ':<10}")
    for tk, r in summary.items():
        if tk == "T3":
            metric = "weighted_f1"
            mean_v = r["mean"]; ens_v = r["ensemble"]["weighted_f1"]
        else:
            metric = "auprc"
            mean_v = r["mean"]; ens_v = r["ensemble"]["auprc"]
        print(f"{tk:<6} {metric:<12} {mean_v:.4f}±{r['std']:.4f}     {ens_v:.4f}    {ens_v-mean_v:+.4f}")

    # save JSON
    out = REPO / "neuromm26_results/metrics/ensemble_val_summary.json"
    out.write_text(json.dumps({
        tk: {
            "per_seed": {str(s): v for s, v in r["per_seed"].items()},
            "mean": r["mean"], "std": r["std"], "ensemble": r["ensemble"],
        } for tk, r in summary.items()
    }, indent=2, default=float))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
