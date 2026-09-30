"""Stacking + Weighted-ensemble search on the existing 7-model real 5-fold OOF.

Reads OOF logits per (model, fold) from neuromm26_real_5fold_result/predictions/,
joins by sample_id, and produces:
  - simple_mean: current baseline (averaging sigmoid(logits))
  - weighted_mean: scipy SLSQP search over w ∈ Δ^7
  - stack_lr: LogisticRegression meta-learner (fold-aware OOF)
  - stack_lgb: LightGBM meta-learner (fold-aware OOF)

Outputs:
  neuromm26_real_5fold_result/csv_reports/stacking_weighted_search.csv
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
PRED_DIR = REPO / "neuromm26_real_5fold_result/predictions"
OUT_DIR = REPO / "neuromm26_real_5fold_result/csv_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FOLD_CSV = REPO / "fold_df_fixed.csv"

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
    """Concat all 5 folds into one big (N, ) array per (sample_ids, logits, labels)."""
    sids_all, logits_all, labels_all = [], [], []
    for f in range(5):
        path = PRED_DIR / f"{prefix}__fold{f}__seed0_oof.npz"
        d = np.load(path)
        sids_all.append(d["sample_ids"])
        logits_all.append(d["logits"])
        labels_all.append(d["labels"])
    return (np.concatenate(sids_all),
            np.concatenate(logits_all).astype(np.float64),
            np.concatenate(labels_all).astype(np.int32))


def main():
    print("=== Load OOF for 7 models ===")
    ref_sids = None
    ref_labels = None
    X_logits = []
    for prefix, name in MODELS:
        sids, logits, labels = load_model_oof(prefix)
        if ref_sids is None:
            ref_sids = sids
            ref_labels = labels
            X_logits.append(logits)
        else:
            if np.array_equal(sids, ref_sids):
                X_logits.append(logits)
            else:
                idx_map = {s: i for i, s in enumerate(sids)}
                perm = np.array([idx_map[s] for s in ref_sids])
                X_logits.append(logits[perm])
                assert np.array_equal(labels[perm], ref_labels)
        print(f"  {name}: N={len(sids)}")
    X = np.stack(X_logits, axis=1)               # (N, M) M=7
    y = ref_labels
    N, M = X.shape
    print(f"Total: N={N} M={M} pos_ratio={y.mean():.4f}")

    # Load fold assignments
    fold_of_sid: dict[str, int] = {}
    with FOLD_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            fold_of_sid[r["sample_id"]] = int(r["fold"])
    folds = np.array([fold_of_sid[s] for s in ref_sids], dtype=np.int32)
    print(f"Fold distribution: {np.bincount(folds)}")

    P = 1.0 / (1.0 + np.exp(-X))                 # (N, M) probabilities
    rows = []
    from sklearn.metrics import average_precision_score

    # ----- 1a. simple mean (prob space) -----
    avg_p = P.mean(axis=1)
    auprc_p = average_precision_score(y, avg_p)
    rows.append({"method": "simple_mean (prob space)", "AUPRC": f"{auprc_p:.4f}", "F1@0.5": "—"})
    print(f"\n[simple_mean prob ] AUPRC={auprc_p:.4f}")

    # ----- 1b. simple mean (logit space) → sigmoid -----
    avg_logit = X.mean(axis=1)
    p_logitavg = 1.0 / (1.0 + np.exp(-avg_logit))
    auprc_l = average_precision_score(y, p_logitavg)
    rows.append({"method": "simple_mean (logit space)", "AUPRC": f"{auprc_l:.4f}", "F1@0.5": "—"})
    print(f"[simple_mean logit] AUPRC={auprc_l:.4f}  (this is what export_real_5fold_csv.py reported)")
    base_auprc = max(auprc_p, auprc_l)

    # ----- 2a. weighted mean over probs -----
    def neg_auprc(w, P, y):
        w = np.maximum(w, 0); s = w.sum()
        w = w / s if s > 0 else np.ones_like(w) / len(w)
        return -average_precision_score(y, P @ w)
    w0 = np.ones(M) / M
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0, 1)] * M
    res = minimize(neg_auprc, w0, args=(P, y), method="SLSQP",
                   bounds=bnds, constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    w_best_p = np.maximum(res.x, 0); w_best_p /= w_best_p.sum()
    p_w_p = P @ w_best_p
    auprc_w_p = average_precision_score(y, p_w_p)
    print(f"\n[weighted prob ] AUPRC={auprc_w_p:.4f}  weights={[f'{w:.2f}' for w in w_best_p]}")
    rows.append({"method": "weighted_mean (prob space)", "AUPRC": f"{auprc_w_p:.4f}", "F1@0.5": "—"})

    # ----- 2b. weighted mean over logits → sigmoid -----
    def neg_auprc_logit(w, X, y):
        w = np.maximum(w, 0); s = w.sum()
        w = w / s if s > 0 else np.ones_like(w) / len(w)
        logits = X @ w
        return -average_precision_score(y, logits)  # AUPRC invariant under sigmoid (monotone)
    res2 = minimize(neg_auprc_logit, w0, args=(X, y), method="SLSQP",
                    bounds=bnds, constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    w_best_l = np.maximum(res2.x, 0); w_best_l /= w_best_l.sum()
    p_w_l = 1.0 / (1.0 + np.exp(-(X @ w_best_l)))
    auprc_w_l = average_precision_score(y, p_w_l)
    print(f"[weighted logit] AUPRC={auprc_w_l:.4f}  weights={[f'{w:.2f}' for w in w_best_l]}")
    rows.append({"method": "weighted_mean (logit space)", "AUPRC": f"{auprc_w_l:.4f}", "F1@0.5": "—"})

    # Use best weighted for downstream
    if auprc_w_l >= auprc_w_p:
        p_w, w_best, w_space = p_w_l, w_best_l, "logit"
        auprc_w = auprc_w_l
    else:
        p_w, w_best, w_space = p_w_p, w_best_p, "prob"
        auprc_w = auprc_w_p

    weights_row = {"method": f"  weights ({w_space} space)", "AUPRC": "—", "F1@0.5": "—"}
    for (_, name), w in zip(MODELS, w_best):
        weights_row[name] = f"{w:.3f}"
    rows.append(weights_row)

    # ----- 3. stack_lr (fold-aware OOF) -----
    print("\n=== Stacking: LogisticRegression (fold-aware OOF) ===")
    stack_oof_lr = np.zeros(N, dtype=np.float64)
    for k in range(5):
        tr = folds != k; va = folds == k
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(X[tr], y[tr])
        stack_oof_lr[va] = lr.predict_proba(X[va])[:, 1]
    auprc_lr = average_precision_score(y, stack_oof_lr)
    print(f"  AUPRC = {auprc_lr:.4f}  (Δ vs simple_mean = {auprc_lr - base_auprc:+.4f})")
    rows.append({"method": "stack_lr (logistic, 5-fold CV)",
                 "AUPRC": f"{auprc_lr:.4f}", "F1@0.5": "—"})

    # ----- 4. stack_lgb (fold-aware OOF) -----
    print("\n=== Stacking: LightGBM (fold-aware OOF) ===")
    stack_oof_lgb = np.zeros(N, dtype=np.float64)
    lgb_params = dict(
        objective="binary", learning_rate=0.05, num_leaves=15,
        min_data_in_leaf=64, feature_fraction=0.9, bagging_fraction=0.8,
        bagging_freq=5, verbose=-1, n_estimators=400,
    )
    for k in range(5):
        tr = folds != k; va = folds == k
        # Use logits AND probs as features
        feat_tr = np.concatenate([X[tr], P[tr]], axis=1)
        feat_va = np.concatenate([X[va], P[va]], axis=1)
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(feat_tr, y[tr],
                  eval_set=[(feat_va, y[va])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        stack_oof_lgb[va] = model.predict_proba(feat_va)[:, 1]
    auprc_lgb = average_precision_score(y, stack_oof_lgb)
    print(f"  AUPRC = {auprc_lgb:.4f}  (Δ vs simple_mean = {auprc_lgb - base_auprc:+.4f})")
    rows.append({"method": "stack_lgb (LightGBM, 5-fold CV)",
                 "AUPRC": f"{auprc_lgb:.4f}", "F1@0.5": "—"})

    # ----- 5. stack_lr with probs (alternative feature space) -----
    print("\n=== Stacking: LogisticRegression on probs ===")
    stack_oof_lr2 = np.zeros(N, dtype=np.float64)
    for k in range(5):
        tr = folds != k; va = folds == k
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(P[tr], y[tr])
        stack_oof_lr2[va] = lr.predict_proba(P[va])[:, 1]
    auprc_lr2 = average_precision_score(y, stack_oof_lr2)
    print(f"  AUPRC = {auprc_lr2:.4f}")
    rows.append({"method": "stack_lr (probs input)",
                 "AUPRC": f"{auprc_lr2:.4f}", "F1@0.5": "—"})

    # ----- Threshold tuning for the best -----
    print("\n=== F1 threshold tuning for best method ===")
    candidates = {
        "simple_mean_prob": avg_p,
        "simple_mean_logit": p_logitavg,
        "weighted_mean": p_w,
        "stack_lr_logits": stack_oof_lr,
        "stack_lr_probs": stack_oof_lr2,
        "stack_lgb": stack_oof_lgb,
    }
    best_name = max(candidates, key=lambda k: average_precision_score(y, candidates[k]))
    probs = candidates[best_name]
    print(f"  Best AUPRC method: {best_name}")
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.96, 0.01):
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p_ = tp / (tp + fp) if tp + fp else 0
        r_ = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    print(f"  F1@best_thr = {best_f1:.4f} @ thr={best_thr:.2f}")
    rows.append({"method": f"BEST ({best_name}) F1@best_thr",
                 "AUPRC": f"{average_precision_score(y, probs):.4f}",
                 "F1@0.5": f"{best_f1:.4f} @ thr={best_thr:.2f}"})

    # Write CSV
    fieldnames = ["method", "AUPRC", "F1@0.5"] + [n for _, n in MODELS]
    with (OUT_DIR / "stacking_weighted_search.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            for col in fieldnames:
                r.setdefault(col, "")
            w.writerow(r)
    print(f"\nResult CSV: {OUT_DIR / 'stacking_weighted_search.csv'}")

    # Also dump weights as JSON
    summary = {
        "simple_mean_AUPRC": float(base_auprc),
        "weighted_mean_AUPRC": float(auprc_w),
        "weighted_weights": {name: float(w) for (_, name), w in zip(MODELS, w_best)},
        "stack_lr_logits_AUPRC": float(auprc_lr),
        "stack_lr_probs_AUPRC": float(auprc_lr2),
        "stack_lgb_AUPRC": float(auprc_lgb),
        "best_method": best_name,
        "best_f1": float(best_f1),
        "best_thr": float(best_thr),
    }
    (OUT_DIR / "stacking_weighted_search.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON: {OUT_DIR / 'stacking_weighted_search.json'}")


if __name__ == "__main__":
    main()
