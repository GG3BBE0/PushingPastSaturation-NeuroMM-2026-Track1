"""Trim weak archs + Bagged-Capped NM weight regularization.

Compares 4 configs on the 39-arch pool:
  A. baseline (current NM, no trim)
  B. trim only
  C. trim + bagged NM
  D. trim + bagged + capped (recommended)

For each: report OOF AUPRC, cos(w_A, w_B), top weights.
Then write submission for config D using existing candidate_arch_logits.npz
(no re-inference; just apply new weights to saved 20000x39 logits).
"""
from __future__ import annotations
import csv
import shutil
import zipfile
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
REAL = REPO / "neuromm26_real_5fold_result"
RES = REPO / "neuromm26_results"
FOLD_CSV = REPO / "fold_df_fixed.csv"
ARCH_LOGITS_NPZ = RES / "candidate_arch_logits.npz"
CAND_IDS = REPO / "NeuroMM-2026/candidate/candidate/candidate_ids.txt"

# Sync with predict_candidate_full_pool.py POOL: name -> (prefix, root)
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
    # NCHC B-plan
    "ConcatFilt maxvit384": ("concat_cwt_filtered_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatPaul maxvit384": ("concat_cwt_paul_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatSTFT maxvit384": ("concat_stft_fold__maxvit_tiny_tf_384_in1k", RES),
    # Heavy aug
    "ConcatSuperlet heavyaug384": ("concat_superlet_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatPaul heavyaug256": ("concat_cwt_paul_heavyaug_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatCWT heavyaug256": ("concat_cwt_heavyaug_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatPaul focalheavy256": ("concat_cwt_paul_focalheavy_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatCWT focalheavy256": ("concat_cwt_focalheavy_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatCWT heavyaug384": ("concat_cwt_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
    # NCHC heavy-aug @384 (batch 2026-06-02)
    "ConcatPaul heavyaug384": ("concat_cwt_paul_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatFilt heavyaug384": ("concat_cwt_filtered_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatSuperlet maxvit_base heavyaug384": ("concat_superlet_heavyaug_fold__maxvit_base_tf_384_in1k", RES),
    # NCHC orthogonal paradigms (honest-OOF gate winners)
    "GNN orthopara": ("eeg_gnn_fold__gcn", RES),
    "Spec3D superlet": ("spec3d_superlet_scratch_fold__r3d_18", RES),
    # in-domain SSL-pretrained EEG transformer (orthogonal, raw waveform)
    "NeuroMAE v1": ("neuromae_fold__v1", RES),
    # consensus-justified label correction (relabel 121 missed-spikes + down-weight 66) + matched A/B baseline
    "ConcatCWT relabel256": ("concat_cwt_relabel_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatCWT relbase256": ("concat_cwt_relbase_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    # expand-corrected top arches (only if A/B clears the gate)
    "ConcatPaul relabel256": ("concat_cwt_paul_relabel_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatSTFT relabel256": ("concat_stft_relabel_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    # AUPRC-surrogate (pairwise ranking loss) retrains
    "ConcatCWT rank256": ("concat_cwt_rank_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatPaul rank256": ("concat_cwt_paul_rank_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    # pseudo-label retrains (real+pseudo candidates, down-weighted)
    "ConcatCWT pseudo256": ("concat_cwt_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatPaul pseudo256": ("concat_cwt_paul_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatFilt pseudo256": ("concat_cwt_filtered_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatSTFT pseudo256": ("concat_stft_pseudo_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatSuperlet pseudo384": ("concat_superlet_pseudo_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatCWT pseudo384": ("concat_cwt_pseudo_fold__maxvit_tiny_tf_384_in1k", RES),
    "muku raw pseudo convnext_pico": ("muku_pseudo_fold__convnext_pico_d1_in1k", RES),
    "muku raw pseudo resnet18": ("muku_pseudo_fold__resnet18", RES),
    "muku Filt pseudo convnext_pico": ("muku_filteeg_pseudo_fold__convnext_pico_d1_in1k", RES),
    # round-2 pseudo (re-selected from pseudo3, the cleaner source)
    "ConcatCWT pseudo_r2 256": ("concat_cwt_pseudo_r2_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatPaul pseudo_r2 256": ("concat_cwt_paul_pseudo_r2_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    # NCHC 6-arch pseudo (round-1 labels, bigger-batch retrain; _nchc to not clobber local)
    "ConcatCWT pseudo_nchc 256": ("concat_cwt_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatPaul pseudo_nchc 256": ("concat_cwt_paul_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatFilt pseudo_nchc 256": ("concat_cwt_filtered_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatSTFT pseudo_nchc 256": ("concat_stft_pseudo_nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatSuperlet pseudo_nchc 384": ("concat_superlet_pseudo_nchc_fold__maxvit_tiny_tf_384_in1k", RES),
    "ConcatCWT pseudo_nchc 384": ("concat_cwt_pseudo_nchc_fold__maxvit_tiny_tf_384_in1k", RES),
    # NCHC candidate-specialist (round-1 spec labels, bigger batch + superlet384 + muku)
    "ConcatCWT specnchc 256": ("concat_cwt_specnchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatPaul specnchc 256": ("concat_cwt_paul_specnchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatFilt specnchc 256": ("concat_cwt_filtered_specnchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatSuperlet specnchc 384": ("concat_superlet_specnchc_fold__maxvit_tiny_tf_384_in1k", RES),
    "muku specnchc convnext": ("muku_specnchc_fold__convnext_pico_d1_in1k", RES),
    # NCHC Noisy-Student SOFT-label distillation (teacher soft probs on candidates; decorrelated blend members)
    "ConcatCWT soft 256": ("concat_cwt_soft_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatPaul soft 256": ("concat_cwt_paul_soft_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatFilt soft 256": ("concat_cwt_filtered_soft_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatSuperlet soft 384": ("concat_superlet_soft_fold__maxvit_tiny_tf_384_in1k", RES),
    # candidate-specialist LOCAL (11336 candidates labeled by nchc_n2filt, weight 1.0; -> public 0.9818)
    "ConcatCWT spec 256": ("concat_cwt_spec_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatPaul spec 256": ("concat_cwt_paul_spec_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatFilt spec 256": ("concat_cwt_filtered_spec_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    # round-3 aggressive pseudo (source=nchc_n2filt 0.9778, HI=0.80, weight 0.5; local bs)
    "ConcatCWT pseudo_r3 256": ("concat_cwt_pseudo_r3_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatPaul pseudo_r3 256": ("concat_cwt_paul_pseudo_r3_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatFilt pseudo_r3 256": ("concat_cwt_filtered_pseudo_r3_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    # NCHC pseudo2: aggressive r3 @ NCHC bigger-batch + honest muku raw
    "ConcatCWT r3nchc 256": ("concat_cwt_pseudo_r3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatPaul r3nchc 256": ("concat_cwt_paul_pseudo_r3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "ConcatFilt r3nchc 256": ("concat_cwt_filtered_pseudo_r3nchc_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    "muku raw pseudo_nchc convnext_pico": ("muku_pseudo_nchc_fold__convnext_pico_d1_in1k", RES),
    "muku raw pseudo_nchc resnet18": ("muku_pseudo_nchc_fold__resnet18", RES),
}

TRIM_CUTOFF_OOF = 0.68     # arch dropped if OOF < this
TRIM_CUTOFF_NMW = 0.005    # AND current baseline NM weight < this (two-dim rule)
NM_RESTARTS = 6
BAG_N = 50
CAP_DEFAULT = 0.20
CURRENT_BEST_CSV = REPO / "submission_test1_filteeg.csv"  # for final spearman sanity


def _common_seeds_oof(prefix, root):
    """Seeds whose OOF npz exists for ALL 5 folds (intersection). Guarantees no fold mixes a
    different seed-count -> safe to use mid-training (partial seed1 -> falls back to {0})."""
    pred = root / "predictions"
    per_fold = []
    for f in range(5):
        seeds = set()
        for fp in pred.glob(f"{prefix}__fold{f}__seed*_oof.npz"):
            try: seeds.add(int(fp.name.rsplit("__seed", 1)[1].split("_oof")[0]))
            except (IndexError, ValueError): pass
        per_fold.append(seeds)
    return sorted(set.intersection(*per_fold)) if all(per_fold) else []


def load_oof(prefix, root):
    """5-fold OOF, seed-averaged WITHIN each fold over all seeds common to all folds.
    seed0-only -> identical to before (mean of 1 = itself)."""
    seeds = _common_seeds_oof(prefix, root)
    if not seeds:
        raise FileNotFoundError(f"no OOF (common across folds) for {prefix} under {root}")
    pred = root / "predictions"
    s, l, y = [], [], []
    for f in range(5):
        d0 = np.load(pred / f"{prefix}__fold{f}__seed{seeds[0]}_oof.npz")
        sids = d0["sample_ids"].astype(str)
        acc = np.zeros(len(sids), dtype=np.float64)
        for sd in seeds:
            d = np.load(pred / f"{prefix}__fold{f}__seed{sd}_oof.npz")
            ss = d["sample_ids"].astype(str)
            lg = d["logits"].astype(np.float64)
            acc += lg if np.array_equal(ss, sids) else lg[np.array([{x: i for i, x in enumerate(ss)}[x] for x in sids])]
        s.append(d0["sample_ids"]); l.append(acc / len(seeds)); y.append(d0["labels"])
    return np.concatenate(s), np.concatenate(l).astype(np.float64), np.concatenate(y).astype(np.int32)


def derive_nm(X, y, n_restarts=NM_RESTARTS, seed=42):
    M = X.shape[1]
    def neg(w):
        w = np.maximum(w, 0); sm = w.sum(); w = w / sm if sm > 0 else np.ones_like(w) / M
        return -average_precision_score(y, X @ w)
    rng = np.random.default_rng(seed)
    best_w, best_a = None, -1
    for _ in range(n_restarts):
        w0 = np.ones(M) / M + 0.05 * rng.standard_normal(M)
        w0 = np.maximum(w0, 1e-3); w0 /= w0.sum()
        r = minimize(neg, w0, method="Nelder-Mead",
                     options={"maxiter": 30000, "xatol": 1e-7, "fatol": 1e-7})
        w = np.maximum(r.x, 0); w /= w.sum()
        a = average_precision_score(y, X @ w)
        if a > best_a:
            best_a, best_w = a, w
    return best_w, best_a


def derive_bagged_nm(X, y, n_iter=BAG_N, n_restarts_per=2, seed=0):
    """Bootstrap-bag NM: sample with replacement, NM on each, average weights."""
    M = X.shape[1]
    rng = np.random.default_rng(seed)
    weights_list = []
    for it in range(n_iter):
        idx = rng.integers(0, len(y), size=len(y))
        Xi, yi = X[idx], y[idx]
        # ensure positives present
        if yi.sum() < 2: continue

        def neg(w):
            w = np.maximum(w, 0); sm = w.sum(); w = w / sm if sm > 0 else np.ones_like(w) / M
            return -average_precision_score(yi, Xi @ w)
        best_w_inner, best_a_inner = None, -1
        for r in range(n_restarts_per):
            w0 = np.ones(M) / M + 0.05 * rng.standard_normal(M)
            w0 = np.maximum(w0, 1e-3); w0 /= w0.sum()
            res = minimize(neg, w0, method="Nelder-Mead",
                           options={"maxiter": 8000, "xatol": 1e-6, "fatol": 1e-6})
            w = np.maximum(res.x, 0); w /= w.sum()
            a = average_precision_score(yi, Xi @ w)
            if a > best_a_inner:
                best_a_inner, best_w_inner = a, w
        weights_list.append(best_w_inner)
    avg_w = np.mean(weights_list, axis=0)
    avg_w = np.maximum(avg_w, 0); avg_w /= avg_w.sum()
    return avg_w


def apply_cap(w, cap):
    if cap is None: return w
    # iterative: clip > cap, renormalize remaining; clip again if anything still exceeds
    w = w.copy()
    for _ in range(20):
        over = w > cap
        if not over.any(): break
        excess = (w[over] - cap).sum()
        w[over] = cap
        rem_mask = ~over
        rem_sum = w[rem_mask].sum()
        if rem_sum <= 0: break
        w[rem_mask] += excess * (w[rem_mask] / rem_sum)
    w = np.minimum(w, cap)
    s = w.sum()
    if s > 0:
        w /= s
    return w


def main():
    # ---- Load candidate arch logits ----
    d = np.load(ARCH_LOGITS_NPZ)
    names = [str(n) for n in d["names"]]
    X_cand = d["logits"]  # (20000, 39)
    cand_ids_arr = d["ids"]
    print(f"Loaded candidate arch logits: {X_cand.shape}, names={len(names)}")
    assert len(names) == X_cand.shape[1], "name count != logit columns"

    # ---- Load OOF for each arch in order ----
    fold_df = pd.read_csv(FOLD_CSV)
    sid_to_subj = dict(zip(fold_df["sample_id"].astype(str), fold_df["subject_id"]))

    ref_s = ref_y = None
    X_oof_cols = []
    arch_oof_auprc = []
    for nm in names:
        prefix, root = POOL_LOOKUP[nm]
        s, l, y = load_oof(prefix, root)
        ss = s.astype(str)
        if ref_s is None:
            ref_s = ss; ref_y = y
            X_oof_cols.append(l)
        else:
            idx = {x: i for i, x in enumerate(ss)}
            perm = np.array([idx[x] for x in ref_s])
            X_oof_cols.append(l[perm])
        # per-arch OOF AUPRC
        p = 1 / (1 + np.exp(-l))
        arch_oof_auprc.append(average_precision_score(y, p))
    X_oof = np.stack(X_oof_cols, axis=1)  # (N, M)
    arch_oof_auprc = np.array(arch_oof_auprc)
    print(f"OOF matrix: {X_oof.shape}")

    # Subject map for Check 1 split
    ref_subj = np.array([sid_to_subj.get(x, "?") for x in ref_s])

    # ---- Per-fold OOF AUPRC (need for fold2/4 worst-fold metric later) ----
    sid_fold = dict(zip(fold_df["sample_id"].astype(str), fold_df["fold"].astype(int)))
    ref_fold = np.array([sid_fold.get(x, -1) for x in ref_s])
    fold_masks = {f: (ref_fold == f) for f in range(5)}

    # ---- Compute baseline 39-arch NM weights FIRST to use in two-dim trim rule ----
    print("\n[A] baseline NM (no trim, no reg)")
    wA, aA = derive_nm(X_oof, ref_y)
    arch_baseline_nmw = wA  # per-arch NM weight under current 39-pool

    # ---- Trim: two-dim rule = OOF < cutoff_oof AND baseline NM weight < cutoff_nmw ----
    # An arch is "safe to drop" only if BOTH are low (weak AND unused).
    # Avoids killing arch like muku V3 mobilenetv3 (OOF~0.76 but real NM contribution).
    drop_mask = (arch_oof_auprc < TRIM_CUTOFF_OOF) & (arch_baseline_nmw < TRIM_CUTOFF_NMW)
    keep_mask = ~drop_mask
    print(f"\nTrim rule: drop if (OOF < {TRIM_CUTOFF_OOF}) AND (current NM weight < {TRIM_CUTOFF_NMW})")
    print(f"  → keep {keep_mask.sum()} / drop {drop_mask.sum()}")
    dropped = [names[i] for i in range(len(names)) if drop_mask[i]]
    print("Dropped archs (verified both OOF low AND NM≈0):")
    for nm in dropped:
        i = names.index(nm)
        print(f"  - {nm:32s} OOF={arch_oof_auprc[i]:.4f}  baseline NM={arch_baseline_nmw[i]*100:.3f}%")

    # Sanity print: borderline archs that pass on NM weight criteria despite low OOF
    print("\nKept despite low OOF (NM still uses them, possible diversity):")
    for i in range(len(names)):
        if arch_oof_auprc[i] < 0.75 and arch_baseline_nmw[i] >= TRIM_CUTOFF_NMW:
            print(f"  + {names[i]:32s} OOF={arch_oof_auprc[i]:.4f}  baseline NM={arch_baseline_nmw[i]*100:.3f}%")

    # ---- 4 configs ----
    configs = {}
    configs["A_baseline"] = {"w": wA, "names": names, "oof": aA, "Xc": X_cand, "X_oof": X_oof}

    # B. trim only
    print("[B] trim only")
    names_T = [n for i, n in enumerate(names) if keep_mask[i]]
    X_oof_T = X_oof[:, keep_mask]
    X_cand_T = X_cand[:, keep_mask]
    # Alignment safety: names order in trim mask must match column order in both matrices
    assert len(names_T) == X_oof_T.shape[1] == X_cand_T.shape[1], "trim mask alignment broken"
    wB, aB = derive_nm(X_oof_T, ref_y)
    configs["B_trim"] = {"w": wB, "names": names_T, "oof": aB, "Xc": X_cand_T, "X_oof": X_oof_T}

    # C. trim + bagged
    print("[C] trim + bagged")
    wC = derive_bagged_nm(X_oof_T, ref_y, n_iter=BAG_N)
    aC = average_precision_score(ref_y, X_oof_T @ wC)
    configs["C_trim_bagged"] = {"w": wC, "names": names_T, "oof": aC, "Xc": X_cand_T, "X_oof": X_oof_T}

    # D. trim + bagged + cap 0.20
    print("[D] trim + bagged + cap 0.20")
    wD = apply_cap(wC, CAP_DEFAULT)
    aD = average_precision_score(ref_y, X_oof_T @ wD)
    configs["D_trim_bagged_cap"] = {"w": wD, "names": names_T, "oof": aD, "Xc": X_cand_T, "X_oof": X_oof_T}

    # ---- Check 1 cos for each config ----
    def check1_cos(X_oof_local, ref_y_local, names_local, weights_full):
        rng = np.random.default_rng(7)
        subjects = np.unique(ref_subj)
        rng.shuffle(subjects)
        half = len(subjects) // 2
        half_A_set = set(subjects[:half])
        maskA = np.array([s in half_A_set for s in ref_subj])
        maskB = ~maskA
        wA_, _ = derive_nm(X_oof_local[maskA], ref_y_local[maskA])
        wB_, _ = derive_nm(X_oof_local[maskB], ref_y_local[maskB])
        cos = float(np.dot(wA_, wB_) / (np.linalg.norm(wA_) * np.linalg.norm(wB_) + 1e-12))
        a_AonB = average_precision_score(ref_y_local[maskB], X_oof_local[maskB] @ wA_)
        a_BonA = average_precision_score(ref_y_local[maskA], X_oof_local[maskA] @ wB_)
        a_fullonA = average_precision_score(ref_y_local[maskA], X_oof_local[maskA] @ weights_full)
        a_fullonB = average_precision_score(ref_y_local[maskB], X_oof_local[maskB] @ weights_full)
        return cos, a_AonB, a_BonA, a_fullonA, a_fullonB

    print("\n=== Comparing configs ===")
    summary = []
    header = f"{'cfg':<20} {'archs':<6} {'OOF':<8} {'f2 OOF':<8} {'f4 OOF':<8} {'cos(A,B)':<10} {'max w':<8}"
    print(header)
    for key, cfg in configs.items():
        X_l = cfg["X_oof"]
        names_l = cfg["names"]
        cos, ab, ba, fA, fB = check1_cos(X_l, ref_y, names_l, cfg["w"])
        cfg["cos"] = cos
        cfg["a_AonB"] = ab; cfg["a_BonA"] = ba
        # fold2 + fold4 OOF AUPRC (the historically-hard folds)
        f2 = average_precision_score(ref_y[fold_masks[2]], X_l[fold_masks[2]] @ cfg["w"])
        f4 = average_precision_score(ref_y[fold_masks[4]], X_l[fold_masks[4]] @ cfg["w"])
        cfg["f2_oof"] = f2; cfg["f4_oof"] = f4
        max_w = float(cfg["w"].max())
        cfg["max_w"] = max_w
        summary.append((key, len(names_l), cfg["oof"], f2, f4, cos, max_w))
        print(f"{key:<20} {len(names_l):<6} {cfg['oof']:.4f}  {f2:.4f}  {f4:.4f}  {cos:.4f}    {max_w:.4f}")

    # ---- Generate submission for D ----
    print("\n=== Generating submission for config D ===")
    cfgD = configs["D_trim_bagged_cap"]
    final_logit = cfgD["Xc"] @ cfgD["w"]
    prob = 1.0 / (1.0 + np.exp(-final_logit))

    ids = [l.strip() for l in CAND_IDS.read_text().splitlines() if l.strip()]
    out_csv = REPO / "submission_test1_regnm_v5.csv"
    with out_csv.open("w", newline="") as f:
        w_csv = csv.writer(f); w_csv.writerow(["sample_id", "prediction"])
        # candidate_arch_logits.npz stored ids in order — use them
        cand_ids_list = list(cand_ids_arr)
        for sid, p in zip(cand_ids_list, prob):
            w_csv.writerow([str(sid), f"{p:.6f}"])
    # validate vs candidate_ids.txt order/content
    rows = list(csv.DictReader(out_csv.open()))
    sids = [r["sample_id"] for r in rows]
    ok = (len(rows) == 20000 and set(sids) == set(ids) and sids == ids
          and not any(np.isnan(float(r["prediction"])) for r in rows))
    print(f"valid={ok}")
    if not ok:
        print("FATAL: validation failed"); return 1

    # zip
    out_dir = REPO / "submissions"
    out_dir.mkdir(exist_ok=True)
    staged = out_dir / "submission.csv"
    shutil.copy(out_csv, staged)
    zname = out_dir / "submission_test1_regnm_v5.zip"
    with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(staged, arcname="submission.csv")
    staged.unlink()
    print(f"Zipped {zname}")

    # ---- Final sanity: Spearman vs current best uploaded submission ----
    from scipy.stats import spearmanr
    if CURRENT_BEST_CSV.exists():
        cur_rows = list(csv.DictReader(CURRENT_BEST_CSV.open()))
        cur_ids = [r["sample_id"] for r in cur_rows]
        cur_p = np.array([float(r["prediction"]) for r in cur_rows])
        # align by id
        id_to_idx = {sid: i for i, sid in enumerate(cur_ids)}
        order = np.array([id_to_idx[s] for s in ids])
        cur_p_aligned = cur_p[order]
        rho = spearmanr(prob, cur_p_aligned).statistic
        print(f"\nSpearman ρ(submission_test1_regnm vs submission_test1_filteeg) = {rho:.4f}")
        if rho > 0.995:
            print("  ⚠️ ρ > 0.995 → almost identical to current best, regularization barely changed predictions")
        elif rho < 0.93:
            print("  ⚠️ ρ < 0.93 → big structural change vs current best; trim/cap may have killed useful signal")
        else:
            print("  ✅ ρ in healthy range 0.93–0.995 (real but bounded change)")
    else:
        rho = None
        print(f"\n(skip Spearman: {CURRENT_BEST_CSV} not found)")

    # ---- Top-weight tables ----
    def print_top(cfg, label, k=12):
        print(f"\n--- {label} top-{k} NM weights (OOF {cfg['oof']:.4f}, cos {cfg.get('cos','-')}) ---")
        order = np.argsort(-cfg["w"])[:k]
        for i in order:
            print(f"  {cfg['names'][i]:32s}  {cfg['w'][i]*100:6.2f}%")

    print_top(configs["A_baseline"], "A baseline")
    print_top(configs["D_trim_bagged_cap"], "D trim+bagged+cap")

    # ---- Write summary markdown ----
    rep = REPO / "WAKEUP_REPORT_regnm_v2.md"
    lines = []
    lines.append("# Regularized NM + Trim — Report")
    lines.append("")
    lines.append(f"Trim cutoff OOF >= {TRIM_CUTOFF} → dropped {len(dropped)} archs:")
    for nm in dropped:
        i = names.index(nm)
        lines.append(f"- {nm} (OOF {arch_oof_auprc[i]:.4f})")
    lines.append("")
    lines.append(f"NM regularization: bagged N={BAG_N}, cap={CAP_DEFAULT}")
    lines.append("")
    lines.append("## Config comparison")
    lines.append("")
    lines.append("| config | archs | OOF | fold2 OOF | fold4 OOF | cos(w_A,w_B) | max weight |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, M, oof, f2, f4, cos, mw in summary:
        lines.append(f"| {key} | {M} | {oof:.4f} | {f2:.4f} | {f4:.4f} | {cos:.4f} | {mw*100:.2f}% |")
    lines.append("")
    if rho is not None:
        lines.append(f"**Spearman ρ(regnm vs current best filteeg.csv) = {rho:.4f}** "
                     f"({'identical' if rho > 0.995 else 'too-different' if rho < 0.93 else 'healthy'})")
        lines.append("")
    lines.append("## Top 12 NM weights — config D (trim + bagged + cap 0.20)")
    lines.append("")
    lines.append("| rank | arch | NM weight |")
    lines.append("|---|---|---|")
    order = np.argsort(-cfgD["w"])[:12]
    for rk, i in enumerate(order, start=1):
        lines.append(f"| {rk} | {cfgD['names'][i]} | {cfgD['w'][i]*100:.2f}% |")
    lines.append("")
    lines.append("## Submission")
    lines.append(f"- `submissions/submission_test1_regnm.zip` (config D)")
    lines.append("")
    lines.append("## Decision rubric")
    lines.append("- If config D OOF within 0.001 of baseline AND cos improves significantly → safe to upload")
    lines.append("- Else: try cap 0.25 (less aggressive) or trim cutoff 0.70")
    rep.write_text("\n".join(lines))
    print(f"\nWrote {rep}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
