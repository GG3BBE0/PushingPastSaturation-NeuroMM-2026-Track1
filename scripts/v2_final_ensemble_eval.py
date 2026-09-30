"""V2-augmented final ensemble eval.

Once muku V2 5-fold sweep is done, this script:
  1. Loads the V2 OOF preds (3 backbones × 5 folds = 15 npz)
  2. Combines with the existing 7-model real 5-fold ensemble (now 10 models)
  3. Computes:
       - simple_mean (logit space, our best so far)
       - weighted_mean (SLSQP in logit space)
       - stack_lr (fold-aware OOF, may now help with V2 diversity)
       - stack_lgb (fold-aware OOF)
  4. Threshold tunes for F1, P@S70, P@S80
  5. Writes CSV report

Output: neuromm26_real_5fold_result/csv_reports/v2_ensemble_final.csv
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
PRED_REAL = REPO / "neuromm26_real_5fold_result/predictions"
PRED_V2 = REPO / "neuromm26_results/predictions"           # V2 lives here until moved
OUT_DIR = REPO / "neuromm26_real_5fold_result/csv_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FOLD_CSV = REPO / "fold_df_fixed.csv"

# Existing 7 models (real 5-fold)
MODELS_BASE = [
    ("muku_fold__resnet18", "muku raw resnet18", PRED_REAL),
    ("muku_fold__tf_efficientnet_b0_ns_jft_in1k", "muku raw effv2s_b0", PRED_REAL),
    ("muku_fold__convnext_pico_d1_in1k", "muku raw convnext_pico", PRED_REAL),
    ("spec_cwt_fold__resnet18", "CWT-Morlet resnet18", PRED_REAL),
    ("spec_cwt_fold__tf_efficientnet_b0_ns_jft_in1k", "CWT-Morlet effv2s_b0", PRED_REAL),
    ("spec_cwt_fold__convnext_pico_d1_in1k", "CWT-Morlet convnext_pico", PRED_REAL),
    ("concat_cwt_fold__convnext_tiny_fb_in22k_ft_in1k_384", "V2 ConcatCWT convnext_tiny", PRED_REAL),
]

# 3 V2 muku superlet
MODELS_V2 = [
    ("muku_v2_superlet_fold__resnet18", "muku V2 superlet resnet18", PRED_V2),
    ("muku_v2_superlet_fold__tf_efficientnet_b0_ns_jft_in1k", "muku V2 superlet effv2s_b0", PRED_V2),
    ("muku_v2_superlet_fold__convnext_pico_d1_in1k", "muku V2 superlet convnext_pico", PRED_V2),
]

# 3 V3 muku superlet+TCN
MODELS_V3 = [
    ("muku_v3_superlet_fold__tf_efficientnet_b0_ns_jft_in1k", "muku V3 effv2s_b0", PRED_V2),
    ("muku_v3_superlet_fold__convnext_pico_d1_in1k", "muku V3 convnext_pico", PRED_V2),
    ("muku_v3_superlet_fold__mobilenetv3_large_100_ra_in1k", "muku V3 mobilenetv3_large", PRED_V2),
]

# 2 standalone legacy
MODELS_LEGACY = [
    ("legacy_tcnet_eeg_fold", "legacy tcnet_eeg", PRED_V2),
    ("legacy_mobilenet_v3_large_eeg_fold", "legacy mobilenet_v3_large_eeg", PRED_V2),
]

ALL_MODELS = MODELS_BASE + MODELS_V2 + MODELS_V3 + MODELS_LEGACY


def load_model_oof(prefix, root):
    s, l, y = [], [], []
    for f in range(5):
        p = root / f"{prefix}__fold{f}__seed0_oof.npz"
        if not p.exists():
            return None, None, None
        d = np.load(p)
        s.append(d["sample_ids"]); l.append(d["logits"]); y.append(d["labels"])
    return np.concatenate(s), np.concatenate(l).astype(np.float64), np.concatenate(y).astype(np.int32)


def sweep_thr(probs, y):
    best = {"f1": -1, "thr": 0.5, "p": 0, "r": 0}
    for thr in np.arange(0.05, 0.95 + 1e-9, 0.01):
        pred = (probs >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": f1, "p": p, "r": r}
    return best


def p_at_recall(probs, y, target=0.70):
    n_pos = int(y.sum())
    order = np.argsort(-probs)
    sl = y[order]
    tp_cum = np.cumsum(sl)
    recall = tp_cum / n_pos
    precision = tp_cum / np.maximum(np.arange(1, len(y) + 1), 1)
    above = np.where(recall >= target)[0]
    return float(precision[above[0]]) if len(above) > 0 else 0.0


def main():
    # Load all available models
    ref_sids, ref_y = None, None
    X_logits, names = [], []
    missing = []
    for prefix, name, root in ALL_MODELS:
        sids, logits, y = load_model_oof(prefix, root)
        if sids is None:
            missing.append(name); continue
        if ref_sids is None:
            ref_sids = sids; ref_y = y; X_logits.append(logits)
        else:
            if np.array_equal(sids, ref_sids):
                X_logits.append(logits)
            else:
                idx_map = {s: i for i, s in enumerate(sids)}
                perm = np.array([idx_map[s] for s in ref_sids])
                X_logits.append(logits[perm])
        names.append(name)
        print(f"  ✓ {name}: N={len(sids)}")
    if missing:
        print(f"  MISSING (skipped): {missing}")
    X = np.stack(X_logits, axis=1)
    P = 1.0 / (1.0 + np.exp(-X))
    M = X.shape[1]
    print(f"Total models loaded: {M}  N={len(ref_y)}")

    # Load folds for stacking CV
    fold_of_sid = {}
    with FOLD_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            fold_of_sid[r["sample_id"]] = int(r["fold"])
    folds = np.array([fold_of_sid[s] for s in ref_sids], dtype=np.int32)

    rows = []

    # ----- 1. Simple mean (logit) -----
    p_simple = 1.0 / (1.0 + np.exp(-X.mean(axis=1)))
    auprc = average_precision_score(ref_y, p_simple)
    thr = sweep_thr(p_simple, ref_y)
    p70 = p_at_recall(p_simple, ref_y, 0.70)
    p80 = p_at_recall(p_simple, ref_y, 0.80)
    rows.append({"method": f"simple_mean (logit, {M} models)", "AUPRC": f"{auprc:.4f}",
                 "F1@best": f"{thr['f1']:.4f}", "thr": f"{thr['thr']:.2f}",
                 "P@S70": f"{p70:.4f}", "P@S80": f"{p80:.4f}"})

    # ----- 2. Weighted (Nelder-Mead is more reliable than SLSQP on this surface) -----
    def neg_auprc_logit(w, X, y):
        w = np.maximum(w, 0); s = w.sum()
        w = w / s if s > 0 else np.ones_like(w) / len(w)
        return -average_precision_score(y, X @ w)
    # SLSQP from uniform tends to get stuck — use Nelder-Mead from a slightly
    # perturbed start instead.
    rng = np.random.default_rng(42)
    best_w, best_auprc_w = None, -1.0
    for trial in range(5):
        w0 = np.ones(M) / M + 0.05 * rng.standard_normal(M)
        w0 = np.maximum(w0, 1e-3); w0 /= w0.sum()
        res = minimize(neg_auprc_logit, w0, args=(X, ref_y), method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-7})
        w_t = np.maximum(res.x, 0); w_t /= w_t.sum()
        a_t = average_precision_score(ref_y, X @ w_t)
        if a_t > best_auprc_w:
            best_auprc_w = a_t; best_w = w_t
    w_best = best_w
    p_w = 1.0 / (1.0 + np.exp(-(X @ w_best)))
    auprc_w = best_auprc_w
    thr = sweep_thr(p_w, ref_y)
    rows.append({"method": "weighted_mean (logit, Nelder-Mead × 5)", "AUPRC": f"{auprc_w:.4f}",
                 "F1@best": f"{thr['f1']:.4f}", "thr": f"{thr['thr']:.2f}",
                 "P@S70": f"{p_at_recall(p_w, ref_y, 0.70):.4f}",
                 "P@S80": f"{p_at_recall(p_w, ref_y, 0.80):.4f}"})
    wrow = {"method": "  weights"}
    for name, w in zip(names, w_best):
        wrow[name] = f"{w:.3f}"
    rows.append(wrow)

    # ----- 3. Stack LR (logit features) -----
    stack_lr = np.zeros(len(ref_y))
    for k in range(5):
        tr = folds != k; va = folds == k
        m = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        m.fit(X[tr], ref_y[tr])
        stack_lr[va] = m.predict_proba(X[va])[:, 1]
    auprc_lr = average_precision_score(ref_y, stack_lr)
    thr = sweep_thr(stack_lr, ref_y)
    rows.append({"method": "stack_LR (logits, fold-CV)", "AUPRC": f"{auprc_lr:.4f}",
                 "F1@best": f"{thr['f1']:.4f}", "thr": f"{thr['thr']:.2f}",
                 "P@S70": f"{p_at_recall(stack_lr, ref_y, 0.70):.4f}",
                 "P@S80": f"{p_at_recall(stack_lr, ref_y, 0.80):.4f}"})

    # ----- 4. Stack LightGBM (logits + probs) -----
    stack_lgb = np.zeros(len(ref_y))
    feat = np.concatenate([X, P], axis=1)
    lgb_params = dict(objective="binary", learning_rate=0.05, num_leaves=15,
                      min_data_in_leaf=64, feature_fraction=0.9, bagging_fraction=0.8,
                      bagging_freq=5, verbose=-1, n_estimators=400)
    for k in range(5):
        tr = folds != k; va = folds == k
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(feat[tr], ref_y[tr],
              eval_set=[(feat[va], ref_y[va])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        stack_lgb[va] = m.predict_proba(feat[va])[:, 1]
    auprc_lgb = average_precision_score(ref_y, stack_lgb)
    thr = sweep_thr(stack_lgb, ref_y)
    rows.append({"method": "stack_LGB (logits+probs, fold-CV)", "AUPRC": f"{auprc_lgb:.4f}",
                 "F1@best": f"{thr['f1']:.4f}", "thr": f"{thr['thr']:.2f}",
                 "P@S70": f"{p_at_recall(stack_lgb, ref_y, 0.70):.4f}",
                 "P@S80": f"{p_at_recall(stack_lgb, ref_y, 0.80):.4f}"})

    # Write CSV
    fieldnames = ["method", "AUPRC", "F1@best", "thr", "P@S70", "P@S80"] + names
    with (OUT_DIR / "v3_ensemble_final.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            for k in fieldnames:
                r.setdefault(k, "")
            w.writerow(r)
    print(f"\nResult: {OUT_DIR / 'v3_ensemble_final.csv'}")
    print(f"Summary:")
    for r in rows:
        if "method" in r and r.get("AUPRC"):
            print(f"  {r['method']:50s}  AUPRC={r['AUPRC']}  F1@best={r.get('F1@best','')}  P@S70={r.get('P@S70','')}")


if __name__ == "__main__":
    main()
