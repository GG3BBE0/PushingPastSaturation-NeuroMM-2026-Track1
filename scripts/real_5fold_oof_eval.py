"""OOF ensemble eval on REAL 5-fold (fold_df_fixed) outputs.

Each fold trainer saves an OOF .npz: sample_ids, logits, labels.
We concat 5 folds → full 25,426-sample OOF predictions per model.
Then compute per-model OOF AUPRC + ensemble combinations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
PRED_DIR = REPO / "neuromm26_results/predictions"
SEEDS_SUFFIX = "__seed0_oof.npz"

MODELS = [
    ("muku_fold__resnet18", "muku raw resnet18"),
    ("muku_fold__tf_efficientnet_b0_ns_jft_in1k", "muku raw effv2s_b0"),
    ("muku_fold__convnext_pico_d1_in1k", "muku raw convnext_pico"),
    ("spec_cwt_fold__resnet18", "CWT-Morlet resnet18"),
    ("spec_cwt_fold__tf_efficientnet_b0_ns_jft_in1k", "CWT-Morlet effv2s_b0"),
    ("spec_cwt_fold__convnext_pico_d1_in1k", "CWT-Morlet convnext_pico"),
    ("concat_cwt_fold__convnext_tiny_fb_in22k_ft_in1k_384", "V2 ConcatCWT convnext_tiny"),
]


def load_model_oof(prefix):
    """Concat 5 fold val predictions → full OOF tensor (25426 samples)."""
    all_sids, all_logits, all_labels = [], [], []
    missing = []
    for f in range(5):
        path = PRED_DIR / f"{prefix}__fold{f}{SEEDS_SUFFIX}"
        if not path.exists():
            missing.append(f)
            continue
        data = np.load(path)
        all_sids.append(data["sample_ids"])
        all_logits.append(data["logits"])
        all_labels.append(data["labels"])
    if missing:
        return None, None, None, missing
    sids = np.concatenate(all_sids)
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    return sids, logits, labels, []


def main():
    print("=" * 70)
    print("REAL 5-fold OOF AUPRC (patient-disjoint)")
    print("=" * 70)

    model_oof = {}
    ref_sids = None
    for prefix, name in MODELS:
        sids, logits, labels, missing = load_model_oof(prefix)
        if missing:
            print(f"  ⚠ {name}: missing folds {missing}, skip")
            continue
        if ref_sids is None:
            ref_sids = sids
        else:
            # Align: reorder logits to match ref_sids order
            if not np.array_equal(sids, ref_sids):
                idx_map = {s: i for i, s in enumerate(sids)}
                perm = np.array([idx_map[s] for s in ref_sids])
                logits = logits[perm]
                labels = labels[perm]
        m = compute_binary_classification_metrics(torch.from_numpy(logits), torch.from_numpy(labels))
        model_oof[name] = {"logits": logits, "labels": labels, "metrics": dict(m)}
        print(f"  {name:42s} N={len(sids):5d}  AUPRC={m['auprc']:.4f}  F1@0.5={m['binary_f1']:.4f}")

    print()
    print("=" * 70)
    print("Ensemble combinations")
    print("=" * 70)

    ref_y = torch.from_numpy(list(model_oof.values())[0]["labels"])

    def ens(name, model_names):
        L = [torch.from_numpy(model_oof[n]["logits"]) for n in model_names if n in model_oof]
        if not L:
            return None
        avg = torch.stack(L).mean(dim=0)
        m = compute_binary_classification_metrics(avg, ref_y)
        # threshold tune
        probs = torch.sigmoid(avg).numpy()
        Y = ref_y.numpy().astype(int)
        best_f1 = -1; best_thr = 0.5; best_p = 0; best_r = 0
        for thr in np.arange(0.05, 0.95 + 1e-9, 0.01):
            pred = (probs >= thr).astype(int)
            tp = int(((pred == 1) & (Y == 1)).sum())
            fp = int(((pred == 1) & (Y == 0)).sum())
            fn = int(((pred == 0) & (Y == 1)).sum())
            p = tp / (tp + fp) if tp + fp else 0
            r = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * p * r / (p + r) if p + r else 0
            if f1 > best_f1:
                best_f1 = f1; best_thr = thr; best_p = p; best_r = r
        # P@S70
        n_pos = int(Y.sum())
        order = np.argsort(-probs)
        sl = Y[order]
        tp_cum = np.cumsum(sl)
        recall = tp_cum / n_pos
        precision = tp_cum / np.maximum(np.arange(1, len(Y) + 1), 1)
        above70 = np.where(recall >= 0.70)[0]
        p70 = float(precision[above70[0]]) if len(above70) > 0 else None
        print(f"  {name:60s}  N_models={len(L):2d}  AUPRC={m['auprc']:.4f}  "
              f"F1@0.5={m['binary_f1']:.4f}  F1@best={best_f1:.4f}@{best_thr:.2f}  P@S70={p70:.4f}")
        return {"metrics": dict(m), "best_thr": {"thr": float(best_thr), "f1": float(best_f1), "p": best_p, "r": best_r},
                "P@S70": p70}

    print("\n--- Individual model OOF (4-seed equiv = 5-fold) ---")
    res = {}
    for name in model_oof:
        res[name] = None  # individual already in model_oof
    print()

    print("--- Group ensembles ---")
    muku_names = ["muku raw resnet18", "muku raw effv2s_b0", "muku raw convnext_pico"]
    cwt_names = ["CWT-Morlet resnet18", "CWT-Morlet effv2s_b0", "CWT-Morlet convnext_pico"]
    v2_names = ["V2 ConcatCWT convnext_tiny"]

    res["muku_3bb"] = ens("muku raw (3 backbones)", muku_names)
    res["cwt_3bb"] = ens("CWT-Morlet (3 backbones)", cwt_names)
    res["v2"] = ens("V2 ConcatCWT alone", v2_names)
    res["muku+cwt"] = ens("muku + CWT (6 models)", muku_names + cwt_names)
    res["muku+v2"] = ens("muku + V2 (4 models)", muku_names + v2_names)
    res["cwt+v2"] = ens("CWT + V2 (4 models)", cwt_names + v2_names)
    res["all_7"] = ens("ALL 7 models (muku+CWT+V2)", muku_names + cwt_names + v2_names)

    # Save
    out_path = REPO / "neuromm26_results/metrics/real_5fold_oof_eval.json"
    save_data = {
        "per_model": {k: v["metrics"] for k, v in model_oof.items()},
        "ensembles": {k: v for k, v in res.items() if v is not None},
    }
    out_path.write_text(json.dumps(save_data, indent=2, default=float))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
