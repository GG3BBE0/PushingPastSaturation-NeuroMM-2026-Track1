"""Settle whether cos(w_A,w_B) is a reliable signal or split-noise.

Same trimmed pool, N different random subject-half splits, derive NM on each
half, compute cos. Report the distribution. If cos varies wildly across splits,
it is NOT a usable private-LB risk signal.
"""
from __future__ import annotations
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

# Import the same POOL_LOOKUP + trim from regnm script for consistency
import importlib.util
spec = importlib.util.spec_from_file_location("regnm", REPO / "scripts/regularized_nm_submission.py")

# Just re-derive minimal pieces here to avoid running regnm's main()
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
    "ConcatSuperlet heavyaug384": ("concat_superlet_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
}
TRIM_OOF, TRIM_NMW = 0.68, 0.005


def load_oof(prefix, root):
    s, l, y = [], [], []
    for f in range(5):
        d = np.load(root / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz")
        s.append(d["sample_ids"]); l.append(d["logits"]); y.append(d["labels"])
    return np.concatenate(s).astype(str), np.concatenate(l).astype(np.float64), np.concatenate(y).astype(np.int32)


def derive_nm(X, y, n_restart=1, seed=42):
    """SLSQP with sum=1 + bounds [0,1] constraint — far faster than Nelder-Mead.
    AUPRC is non-smooth so we optimize a smooth surrogate (logistic NLL) then the
    weights still rank-correlate; good enough for a stability sweep."""
    M = X.shape[1]
    Xs = X  # (N, M)
    def neg_auprc(w):
        w = np.clip(w, 0, None); s = w.sum(); w = w / s if s > 0 else np.ones_like(w)/M
        return -average_precision_score(y, Xs @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * M
    rng = np.random.default_rng(seed)
    bw, ba = None, -1
    for _ in range(n_restart):
        w0 = np.ones(M)/M + 0.02 * rng.standard_normal(M)
        w0 = np.clip(w0, 1e-3, None); w0 /= w0.sum()
        r = minimize(neg_auprc, w0, method="SLSQP", bounds=bnds, constraints=cons,
                     options={"maxiter": 200, "ftol": 1e-7})
        w = np.clip(r.x, 0, None); w /= w.sum()
        a = average_precision_score(y, Xs @ w)
        if a > ba: ba, bw = a, w
    return bw


def main():
    names = list(POOL_LOOKUP.keys())
    fold_df = pd.read_csv(FOLD_CSV); fold_df["sample_id"] = fold_df["sample_id"].astype(str)
    sid_subj = dict(zip(fold_df["sample_id"], fold_df["subject_id"]))

    ref_s = ref_y = None; cols = []; auprc = []
    avail = []
    for nm in names:
        prefix, root = POOL_LOOKUP[nm]
        try:
            s, l, y = load_oof(prefix, root)
        except FileNotFoundError:
            continue
        avail.append(nm)
        ss = s.astype(str)
        if ref_s is None:
            ref_s, ref_y = ss, y; cols.append(l)
        else:
            idx = {x: i for i, x in enumerate(ss)}
            cols.append(l[np.array([idx[x] for x in ref_s])])
        auprc.append(average_precision_score(y, 1/(1+np.exp(-l))))
    X = np.stack(cols, axis=1); auprc = np.array(auprc)
    print(f"Loaded {len(avail)} archs, OOF {X.shape}")

    # baseline NM for trim
    w_base = derive_nm(X, ref_y, n_restart=2)
    keep = ~((auprc < TRIM_OOF) & (w_base < TRIM_NMW))
    Xt = X[:, keep]
    print(f"Trimmed to {keep.sum()} archs", flush=True)

    ref_subj = np.array([sid_subj[s] for s in ref_s])
    subjects = np.unique(ref_subj)

    # ---- cos across many random splits ----
    cos_vals = []
    print(f"\nComputing cos over 12 random subject splits (SLSQP)...", flush=True)
    for seed in range(12):
        rng = np.random.default_rng(seed)
        sh = subjects.copy(); rng.shuffle(sh)
        half = len(sh) // 2
        setA = set(sh[:half])
        mA = np.array([s in setA for s in ref_subj]); mB = ~mA
        wA = derive_nm(Xt[mA], ref_y[mA], n_restart=1, seed=100+seed)
        wB = derive_nm(Xt[mB], ref_y[mB], n_restart=1, seed=200+seed)
        cos = float(np.dot(wA, wB) / (np.linalg.norm(wA)*np.linalg.norm(wB)+1e-12))
        cos_vals.append(cos)
        print(f"  split seed {seed:2d}: cos = {cos:.4f}", flush=True)

    cos_vals = np.array(cos_vals)
    print(f"\n=== cos(w_A,w_B) stability over 15 splits ===")
    print(f"  min   = {cos_vals.min():.4f}")
    print(f"  max   = {cos_vals.max():.4f}")
    print(f"  mean  = {cos_vals.mean():.4f}")
    print(f"  std   = {cos_vals.std():.4f}")
    print(f"  range = {cos_vals.max()-cos_vals.min():.4f}")
    verdict = "NOISE (unreliable signal)" if cos_vals.std() > 0.08 else "fairly stable"
    print(f"\n  VERDICT: cos is {verdict}")
    print(f"  → std {cos_vals.std():.3f}; if > 0.08, single cos number is not a usable private-LB signal.")


if __name__ == "__main__":
    main()
