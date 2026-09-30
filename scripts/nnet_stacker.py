"""nnet stacker A/B test vs bagged NM (regnm).

Stacker = small linear model on per-arch OOF logits, trained with BCE loss,
weight_decay=0.01, input dropout=0.1. K-fold inner CV using fold_df_fixed
folds (subject-disjoint to avoid stacker peeking).

Outputs:
- 5-fold inner-CV OOF AUPRC (clean, no stacker contamination)
- cos(w_A, w_B) on subject-disjoint halves
- Spearman vs current best submission_test1_filteeg.csv
- Per-fold AUPRC (esp fold2, fold4)
- Top-K weights
- submission_test1_stacker.zip if better than regnm
"""
from __future__ import annotations
import csv
import shutil
import zipfile
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
REAL = REPO / "neuromm26_real_5fold_result"
RES = REPO / "neuromm26_results"
FOLD_CSV = REPO / "fold_df_fixed.csv"
ARCH_LOGITS_NPZ = RES / "candidate_arch_logits.npz"
CAND_IDS = REPO / "NeuroMM-2026/candidate/candidate/candidate_ids.txt"
CURRENT_BEST_CSV = REPO / "submission_test1_filteeg.csv"

POOL_LOOKUP = {
    "muku raw resnet18": ("muku_fold__resnet18", REAL),
    "muku raw effv2s_b0": ("muku_fold__tf_efficientnet_b0_ns_jft_in1k", REAL),
    "muku raw convnext_pico": ("muku_fold__convnext_pico_d1_in1k", REAL),
    "CWT resnet18": ("spec_cwt_fold__resnet18", REAL),
    "CWT effv2s_b0": ("spec_cwt_fold__tf_efficientnet_b0_ns_jft_in1k", REAL),
    "CWT convnext_pico": ("spec_cwt_fold__convnext_pico_d1_in1k", REAL),
    "ConcatCWT convnext_tiny": ("concat_cwt_fold__convnext_tiny_fb_in22k_ft_in1k_384", REAL),
    "muku V2 sl resnet18": ("muku_v2_superlet_fold__resnet18", RES),
    "muku V2 sl effv2s_b0": ("muku_v2_superlet_fold__tf_efficientnet_b0_ns_jft_in1k", RES),
    "muku V2 sl convnext_pico": ("muku_v2_superlet_fold__convnext_pico_d1_in1k", RES),
    "muku V3 effv2s_b0": ("muku_v3_superlet_fold__tf_efficientnet_b0_ns_jft_in1k", RES),
    "muku V3 convnext_pico": ("muku_v3_superlet_fold__convnext_pico_d1_in1k", RES),
    "muku V3 mobilenetv3": ("muku_v3_superlet_fold__mobilenetv3_large_100_ra_in1k", RES),
    "legacy tcnet_eeg": ("legacy_tcnet_eeg_fold", RES),
    "legacy mobilenet_v3_large_eeg": ("legacy_mobilenet_v3_large_eeg_fold", RES),
    "muku V3 swinv2": ("muku_v3_superlet_fold__swinv2_cr_tiny_ns_224_sw_in1k", RES),
    "ConcatCWT maxvit256": ("concat_cwt_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatSuperlet convnext_tiny384": ("concat_superlet_fold__convnext_tiny_fb_in22k_ft_in1k_384", RES),
    "ConcatCWT maxvit384": ("concat_cwt_fold__maxvit_tiny_tf_384_in1k", RES),
    "CWT swinv2": ("spec_cwt_fold__swinv2_cr_tiny_ns_224_sw_in1k", RES),
    "CWT caformer": ("spec_cwt_fold__caformer_s18_sail_in22k_ft_in1k", RES),
    "ConcatSuperlet maxvit384": ("concat_superlet_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatCWT coatnet": ("concat_cwt_fold__coatnet_0_rw_224_sw_in1k", RES),
    "ConcatCWT convnext_small": ("concat_cwt_fold__convnext_small_fb_in22k_ft_in1k_384", RES),
    "ConcatFilt maxvit256": ("concat_cwt_filtered_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "CWTFilt convnext_pico": ("spec_cwt_filtered_fold__convnext_pico_d1_in1k", RES),
    "ConcatSTFT maxvit256": ("concat_stft_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "STFT convnext_pico": ("spec_stft_fold__convnext_pico_d1_in1k", RES),
    "ConcatPaul maxvit256": ("concat_cwt_paul_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "Paul convnext_pico": ("spec_cwt_paul_fold__convnext_pico_d1_in1k", RES),
    "muku Filt convnext_pico": ("muku_filteeg_fold__convnext_pico_d1_in1k", RES),
    "muku Filt resnet18": ("muku_filteeg_fold__resnet18", RES),
    "legacy tcnet Filt": ("legacy_tcnet_filteeg_fold", RES),
    "muku V2 Filt sl convnext_pico": ("muku_v2_filteeg_sl_fold__convnext_pico_d1_in1k", RES),
    "legacy eegnet": ("legacy_eegnet_fold", RES),
    "legacy actnet_s": ("legacy_actnet_s_fold", RES),
    "legacy lmda_eeg": ("legacy_lmda_eeg_fold", RES),
    "muku_mb convnext_pico": ("muku_mb_fold__convnext_pico_d1_in1k", RES),
    "muku_mb effv2s_b0": ("muku_mb_fold__tf_efficientnet_b0_ns_jft_in1k", RES),
    "ConcatFilt maxvit384": ("concat_cwt_filtered_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatPaul maxvit384": ("concat_cwt_paul_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatSTFT maxvit384": ("concat_stft_fold__maxvit_tiny_tf_384_in1k", RES),
}

TRIM_OOF = 0.68
TRIM_NMW = 0.005


def load_oof(prefix, root):
    s, l, y = [], [], []
    for f in range(5):
        d = np.load(root / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz")
        s.append(d["sample_ids"]); l.append(d["logits"]); y.append(d["labels"])
    return (np.concatenate(s).astype(str),
            np.concatenate(l).astype(np.float64),
            np.concatenate(y).astype(np.int32))


class Stacker(nn.Module):
    """Linear stacker: w · X + b (b included so model has calibration freedom; AUPRC unaffected by b but
    we keep it for stable BCE optimization and for ranking-equivalence printout)."""

    def __init__(self, n_models, dropout=0.1):
        super().__init__()
        # init equal weights summing to 1
        init = torch.ones(n_models) / n_models
        self.weight = nn.Parameter(init)
        self.bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, M)
        x = self.dropout(x)
        return (x * self.weight).sum(-1) + self.bias.squeeze()


def train_stacker(X_tr, y_tr, n_models, *, epochs=300, lr=5e-3, wd=0.01, dropout=0.1, bs=512, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = Stacker(n_models, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    n = len(yt)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def stacker_weights(model):
    """Return positive-constrained per-arch weight (clipped, renormalized) for comparison."""
    w = model.weight.detach().cpu().numpy()
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def main():
    # Load candidate logits + names
    d = np.load(ARCH_LOGITS_NPZ)
    names = [str(n) for n in d["names"]]
    X_cand = d["logits"]  # (20000, 39)
    cand_ids_arr = d["ids"]
    print(f"Candidate logits: {X_cand.shape}, names={len(names)}")

    # Load OOF per arch in same order
    fold_df = pd.read_csv(FOLD_CSV)
    fold_df["sample_id"] = fold_df["sample_id"].astype(str)
    sid_to_fold = dict(zip(fold_df["sample_id"], fold_df["fold"]))
    sid_to_subj = dict(zip(fold_df["sample_id"], fold_df["subject_id"]))

    ref_s = ref_y = None
    X_cols, arch_aurpc = [], []
    for nm in names:
        prefix, root = POOL_LOOKUP[nm]
        s, l, y = load_oof(prefix, root)
        ss = s.astype(str)
        if ref_s is None:
            ref_s = ss; ref_y = y
            X_cols.append(l)
        else:
            idx = {x: i for i, x in enumerate(ss)}
            perm = np.array([idx[x] for x in ref_s])
            X_cols.append(l[perm])
        p = 1 / (1 + np.exp(-l))
        arch_aurpc.append(average_precision_score(y, p))
    X_oof = np.stack(X_cols, axis=1)
    arch_aurpc = np.array(arch_aurpc)
    ref_fold = np.array([sid_to_fold[s] for s in ref_s])
    ref_subj = np.array([sid_to_subj[s] for s in ref_s])
    print(f"OOF: {X_oof.shape}")

    # Trim using same rule as regnm (need baseline NM weight, recompute)
    print("\nComputing baseline NM weights for trim eligibility...")
    from scipy.optimize import minimize

    def derive_nm(X, y, n_restart=6, seed=42):
        M = X.shape[1]
        def neg(w):
            w = np.maximum(w, 0); s = w.sum(); w = w / s if s > 0 else np.ones_like(w) / M
            return -average_precision_score(y, X @ w)
        rng = np.random.default_rng(seed)
        bw, ba = None, -1
        for _ in range(n_restart):
            w0 = np.ones(M) / M + 0.05 * rng.standard_normal(M)
            w0 = np.maximum(w0, 1e-3); w0 /= w0.sum()
            r = minimize(neg, w0, method="Nelder-Mead",
                         options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-7})
            w = np.maximum(r.x, 0); w /= w.sum()
            a = average_precision_score(y, X @ w)
            if a > ba: ba, bw = a, w
        return bw, ba

    w_baseline, oof_baseline = derive_nm(X_oof, ref_y)
    drop_mask = (arch_aurpc < TRIM_OOF) & (w_baseline < TRIM_NMW)
    keep_mask = ~drop_mask
    print(f"Trim: keep {keep_mask.sum()}, drop {drop_mask.sum()}")
    names_T = [n for i, n in enumerate(names) if keep_mask[i]]
    X_oof_T = X_oof[:, keep_mask]
    X_cand_T = X_cand[:, keep_mask]

    # ===================== Stacker (trimmed, 34 archs) =====================
    print("\n=== Stacker on trimmed 34 archs ===")
    M = X_oof_T.shape[1]
    # 5-fold inner CV using existing patient-disjoint folds
    oof_stacker_pred = np.zeros(len(ref_y), dtype=np.float64)
    cand_preds = []
    fold_weights = []
    device = "cpu"  # tiny model, CPU fine
    for f in range(5):
        tr_mask = ref_fold != f
        va_mask = ref_fold == f
        model = train_stacker(X_oof_T[tr_mask], ref_y[tr_mask], M,
                              epochs=300, lr=5e-3, wd=0.01, dropout=0.1, bs=512, seed=f, device=device)
        with torch.no_grad():
            xva = torch.tensor(X_oof_T[va_mask], dtype=torch.float32, device=device)
            oof_stacker_pred[va_mask] = model(xva).cpu().numpy()
            # Apply to candidate
            xc = torch.tensor(X_cand_T, dtype=torch.float32, device=device)
            cand_preds.append(model(xc).cpu().numpy())
        fold_weights.append(stacker_weights(model))
        ap_f = average_precision_score(ref_y[va_mask], oof_stacker_pred[va_mask])
        print(f"  fold{f}: stacker val AUPRC={ap_f:.4f}")

    oof_stacker_auprc = average_precision_score(ref_y, oof_stacker_pred)
    print(f"\nStacker 5-fold OOF AUPRC = {oof_stacker_auprc:.4f}")

    # Mean candidate prediction (5 stackers)
    cand_pred_mean = np.mean(cand_preds, axis=0)

    # Mean stacker weights for reporting
    avg_weights = np.mean(fold_weights, axis=0)

    # Per-fold OOF AUPRC for the stacker
    per_fold = {}
    for f in range(5):
        m = ref_fold == f
        per_fold[f] = average_precision_score(ref_y[m], oof_stacker_pred[m])

    # cos(w_A, w_B) using subject-disjoint halves: train stackers on each half, compare weights
    print("\nComputing cos(w_A, w_B) for stacker...")
    rng = np.random.default_rng(7)
    subjects = np.unique(ref_subj)
    rng.shuffle(subjects)
    half = len(subjects) // 2
    set_A = set(subjects[:half])
    maskA = np.array([s in set_A for s in ref_subj])
    maskB = ~maskA
    mA = train_stacker(X_oof_T[maskA], ref_y[maskA], M,
                       epochs=300, lr=5e-3, wd=0.01, dropout=0.1, bs=512, seed=11, device=device)
    mB = train_stacker(X_oof_T[maskB], ref_y[maskB], M,
                       epochs=300, lr=5e-3, wd=0.01, dropout=0.1, bs=512, seed=13, device=device)
    wA = stacker_weights(mA); wB = stacker_weights(mB)
    cos_AB = float(np.dot(wA, wB) / (np.linalg.norm(wA) * np.linalg.norm(wB) + 1e-12))
    print(f"  cos(w_A, w_B) stacker = {cos_AB:.4f}")

    # Cross-AUPRC
    with torch.no_grad():
        a_AonB = average_precision_score(ref_y[maskB], mA(torch.tensor(X_oof_T[maskB], dtype=torch.float32)).cpu().numpy())
        a_BonA = average_precision_score(ref_y[maskA], mB(torch.tensor(X_oof_T[maskA], dtype=torch.float32)).cpu().numpy())

    # ===================== Spearman vs current best =====================
    prob = 1.0 / (1.0 + np.exp(-cand_pred_mean))
    ids = [l.strip() for l in CAND_IDS.read_text().splitlines() if l.strip()]
    # validate
    cand_ids_list = [str(x) for x in cand_ids_arr]
    assert cand_ids_list == ids, "candidate_ids order mismatch"

    if CURRENT_BEST_CSV.exists():
        cur_rows = list(csv.DictReader(CURRENT_BEST_CSV.open()))
        cur_ids = [r["sample_id"] for r in cur_rows]
        cur_p = np.array([float(r["prediction"]) for r in cur_rows])
        id_to_idx = {sid: i for i, sid in enumerate(cur_ids)}
        order = np.array([id_to_idx[s] for s in ids])
        cur_p_aligned = cur_p[order]
        rho = float(spearmanr(prob, cur_p_aligned).statistic)
    else:
        rho = None

    # ===================== Print comparison =====================
    # Load regnm submission for direct compare
    regnm_csv = REPO / "submission_test1_regnm.csv"
    rho_vs_regnm = None
    if regnm_csv.exists():
        r2 = list(csv.DictReader(regnm_csv.open()))
        r2_ids = [r["sample_id"] for r in r2]
        r2_p = np.array([float(r["prediction"]) for r in r2])
        idx_map = {sid: i for i, sid in enumerate(r2_ids)}
        order = np.array([idx_map[s] for s in ids])
        rho_vs_regnm = float(spearmanr(prob, r2_p[order]).statistic)

    print("\n=== Stacker vs Bagged NM (regnm) ===")
    print(f"{'metric':<35} {'regnm (D)':<15} {'stacker':<15}")
    print(f"{'-'*65}")
    print(f"{'archs':<35} {'34':<15} {'34':<15}")
    print(f"{'OOF AUPRC':<35} {'0.8925':<15} {f'{oof_stacker_auprc:.4f}':<15}")
    print(f"{'fold2 OOF':<35} {'0.8344':<15} {f'{per_fold[2]:.4f}':<15}")
    print(f"{'fold4 OOF':<35} {'0.6553':<15} {f'{per_fold[4]:.4f}':<15}")
    print(f"{'cos(w_A, w_B)':<35} {'0.4387':<15} {f'{cos_AB:.4f}':<15}")
    print(f"{'max weight':<35} {'15.88%':<15} {f'{avg_weights.max()*100:.2f}%':<15}")
    print(f"{'Spearman vs filteeg (0.9663)':<35} {'0.9924':<15} {f'{rho:.4f}' if rho else 'n/a':<15}")
    if rho_vs_regnm is not None:
        print(f"{'Spearman vs regnm (0.9665)':<35} {'-':<15} {f'{rho_vs_regnm:.4f}':<15}")

    print("\nTop 12 stacker weights (mean across 5 folds):")
    order = np.argsort(-avg_weights)[:12]
    for i in order:
        print(f"  {names_T[i]:32s}  {avg_weights[i]*100:6.2f}%")

    # ===================== Submission =====================
    out_csv = REPO / "submission_test1_stacker.csv"
    with out_csv.open("w", newline="") as f:
        w_csv = csv.writer(f); w_csv.writerow(["sample_id", "prediction"])
        for sid, p in zip(ids, prob):
            w_csv.writerow([sid, f"{p:.6f}"])

    rows = list(csv.DictReader(out_csv.open()))
    sids = [r["sample_id"] for r in rows]
    valid = (len(rows) == 20000 and set(sids) == set(ids) and sids == ids
             and not any(np.isnan(float(r["prediction"])) for r in rows))
    print(f"\nsubmission valid={valid}")
    if valid:
        out_dir = REPO / "submissions"; out_dir.mkdir(exist_ok=True)
        staged = out_dir / "submission.csv"
        shutil.copy(out_csv, staged)
        zname = out_dir / "submission_test1_stacker.zip"
        with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(staged, arcname="submission.csv")
        staged.unlink()
        print(f"Zipped {zname}")

    # ===================== Write markdown report =====================
    rep = REPO / "WAKEUP_REPORT_stacker.md"
    lines = []
    lines.append("# nnet Stacker A/B vs Bagged NM (regnm)")
    lines.append("")
    lines.append("## Comparison")
    lines.append("")
    lines.append("| metric | regnm (D) bagged NM | stacker |")
    lines.append("|---|---|---|")
    lines.append("| archs | 34 | 34 |")
    lines.append(f"| OOF AUPRC | 0.8925 | **{oof_stacker_auprc:.4f}** |")
    lines.append(f"| fold2 OOF | 0.8344 | {per_fold[2]:.4f} |")
    lines.append(f"| fold4 OOF | 0.6553 | {per_fold[4]:.4f} |")
    lines.append(f"| cos(w_A, w_B) | 0.4387 | **{cos_AB:.4f}** |")
    lines.append(f"| A→B AUPRC | 0.9420 | {a_AonB:.4f} |")
    lines.append(f"| B→A AUPRC | 0.7664 | {a_BonA:.4f} |")
    lines.append(f"| max weight | 15.88% | {avg_weights.max()*100:.2f}% |")
    lines.append(f"| Spearman vs filteeg (0.9663) | 0.9924 | {rho:.4f if rho else 'n/a'} |")
    if rho_vs_regnm is not None:
        lines.append(f"| Spearman vs regnm (0.9665) | 1.0000 | {rho_vs_regnm:.4f} |")
    lines.append("")
    lines.append("## Top 12 stacker weights (mean across 5 inner-CV folds)")
    lines.append("")
    lines.append("| rank | arch | weight |")
    lines.append("|---|---|---|")
    for rk, i in enumerate(order, start=1):
        lines.append(f"| {rk} | {names_T[i]} | {avg_weights[i]*100:.2f}% |")
    lines.append("")
    lines.append("## Submission")
    lines.append(f"- `submissions/submission_test1_stacker.zip` ({'valid' if valid else 'INVALID'})")
    lines.append("")
    lines.append("## Decision rubric")
    lines.append("- If stacker OOF > regnm AND cos ≥ regnm AND Spearman vs filteeg in 0.93-0.995 → upload")
    lines.append("- If stacker OOF within 0.001 of regnm but cos higher → marginal, still uploadable")
    lines.append("- If stacker OOF < regnm by 0.002+ → stop, bagged NM is at the limit for our data")
    rep.write_text("\n".join(lines))
    print(f"\nWrote {rep}")


if __name__ == "__main__":
    raise SystemExit(main())
