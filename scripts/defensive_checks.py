"""Defensive checks 1-4 for the current 33-arch T1 ensemble.

Check 1: NM weight stability on OOF halves (subject-disjoint split)
Check 2: Per-arch OOF AUPRC vs NM weight — find mismatches
Check 3: Fold-wise per-arch AUPRC variance
Check 4: Spearman correlation across past submission CSVs

Writes report: neuromm26_results/defensive_checks_report.md
"""
from __future__ import annotations
import csv
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
REAL = REPO / "neuromm26_real_5fold_result"
RES = REPO / "neuromm26_results"
FOLD_CSV = REPO / "fold_df_fixed.csv"
REPORT = RES / "defensive_checks_report.md"

# Mirror of POOL in predict_candidate_full_pool.py (name, prefix, root)
POOL = [
    ("muku raw resnet18", "muku_fold__resnet18", REAL),
    ("muku raw effv2s_b0", "muku_fold__tf_efficientnet_b0_ns_jft_in1k", REAL),
    ("muku raw convnext_pico", "muku_fold__convnext_pico_d1_in1k", REAL),
    ("CWT resnet18", "spec_cwt_fold__resnet18", REAL),
    ("CWT effv2s_b0", "spec_cwt_fold__tf_efficientnet_b0_ns_jft_in1k", REAL),
    ("CWT convnext_pico", "spec_cwt_fold__convnext_pico_d1_in1k", REAL),
    ("ConcatCWT convnext_tiny", "concat_cwt_fold__convnext_tiny_fb_in22k_ft_in1k_384", REAL),
    ("muku V2 sl resnet18", "muku_v2_superlet_fold__resnet18", RES),
    ("muku V2 sl effv2s_b0", "muku_v2_superlet_fold__tf_efficientnet_b0_ns_jft_in1k", RES),
    ("muku V2 sl convnext_pico", "muku_v2_superlet_fold__convnext_pico_d1_in1k", RES),
    ("muku V3 effv2s_b0", "muku_v3_superlet_fold__tf_efficientnet_b0_ns_jft_in1k", RES),
    ("muku V3 convnext_pico", "muku_v3_superlet_fold__convnext_pico_d1_in1k", RES),
    ("muku V3 mobilenetv3", "muku_v3_superlet_fold__mobilenetv3_large_100_ra_in1k", RES),
    ("legacy tcnet_eeg", "legacy_tcnet_eeg_fold", RES),
    ("legacy mobilenet_v3_large_eeg", "legacy_mobilenet_v3_large_eeg_fold", RES),
    ("muku V3 swinv2", "muku_v3_superlet_fold__swinv2_cr_tiny_ns_224_sw_in1k", RES),
    ("ConcatCWT maxvit256", "concat_cwt_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    ("ConcatSuperlet convnext_tiny384", "concat_superlet_fold__convnext_tiny_fb_in22k_ft_in1k_384", RES),
    ("ConcatCWT maxvit384", "concat_cwt_fold__maxvit_tiny_tf_384_in1k", RES),
    ("CWT swinv2", "spec_cwt_fold__swinv2_cr_tiny_ns_224_sw_in1k", RES),
    ("CWT caformer", "spec_cwt_fold__caformer_s18_sail_in22k_ft_in1k", RES),
    ("ConcatSuperlet maxvit384", "concat_superlet_fold__maxvit_tiny_tf_384_in1k", RES),
    ("ConcatCWT coatnet", "concat_cwt_fold__coatnet_0_rw_224_sw_in1k", RES),
    ("ConcatCWT convnext_small", "concat_cwt_fold__convnext_small_fb_in22k_ft_in1k_384", RES),
    ("ConcatFilt maxvit256", "concat_cwt_filtered_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    ("CWTFilt convnext_pico", "spec_cwt_filtered_fold__convnext_pico_d1_in1k", RES),
    ("ConcatSTFT maxvit256", "concat_stft_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    ("STFT convnext_pico", "spec_stft_fold__convnext_pico_d1_in1k", RES),
    ("ConcatPaul maxvit256", "concat_cwt_paul_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    ("Paul convnext_pico", "spec_cwt_paul_fold__convnext_pico_d1_in1k", RES),
    ("muku Filt convnext_pico", "muku_filteeg_fold__convnext_pico_d1_in1k", RES),
    ("muku Filt resnet18", "muku_filteeg_fold__resnet18", RES),
    ("legacy tcnet Filt", "legacy_tcnet_filteeg_fold", RES),
    ("muku V2 Filt sl convnext_pico", "muku_v2_filteeg_sl_fold__convnext_pico_d1_in1k", RES),
    # ---- Axis-1 wave additions (sync with predict_candidate_full_pool.py POOL) ----
    ("legacy eegnet",   "legacy_eegnet_fold",   RES),
    ("legacy actnet_s", "legacy_actnet_s_fold", RES),
    ("legacy lmda_eeg", "legacy_lmda_eeg_fold", RES),
    ("muku_mb convnext_pico", "muku_mb_fold__convnext_pico_d1_in1k",          RES),
    ("muku_mb effv2s_b0",     "muku_mb_fold__tf_efficientnet_b0_ns_jft_in1k", RES),
    # NCHC B-plan
    ("ConcatFilt maxvit384", "concat_cwt_filtered_fold__maxvit_tiny_tf_384_in1k", RES),
    ("ConcatPaul maxvit384", "concat_cwt_paul_fold__maxvit_tiny_tf_384_in1k",     RES),
    ("ConcatSTFT maxvit384", "concat_stft_fold__maxvit_tiny_tf_384_in1k",         RES),
    # Heavy aug
    ("ConcatSuperlet heavyaug384", "concat_superlet_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES),
    ("ConcatPaul heavyaug256", "concat_cwt_paul_heavyaug_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    ("ConcatCWT heavyaug256", "concat_cwt_heavyaug_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    ("ConcatPaul focalheavy256", "concat_cwt_paul_focalheavy_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
]


def have_full_set(prefix, root, seed=0):
    for f in range(5):
        if not (root / "checkpoints" / f"{prefix}__fold{f}__seed{seed}" / "best.pt").exists(): return False
        if not (root / "predictions" / f"{prefix}__fold{f}__seed{seed}_oof.npz").exists(): return False
    return True


def load_oof(prefix, root):
    s, l, y, fold_ids = [], [], [], []
    for f in range(5):
        d = np.load(root / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz")
        s.append(d["sample_ids"]); l.append(d["logits"]); y.append(d["labels"])
        fold_ids.append(np.full(len(d["sample_ids"]), f, dtype=np.int8))
    return (np.concatenate(s), np.concatenate(l).astype(np.float64),
            np.concatenate(y).astype(np.int32), np.concatenate(fold_ids))


def derive_weights(X, y, seeds=6):
    """X: (N, M), y: (N,) → (M,) NM weights + best AUPRC."""
    M = X.shape[1]
    def neg(w):
        w = np.maximum(w, 0); sm = w.sum(); w = w / sm if sm > 0 else np.ones_like(w) / M
        return -average_precision_score(y, X @ w)
    rng = np.random.default_rng(42)
    best_w, best_a = None, -1
    for _ in range(seeds):
        w0 = np.ones(M) / M + 0.05 * rng.standard_normal(M)
        w0 = np.maximum(w0, 1e-3); w0 /= w0.sum()
        r = minimize(neg, w0, method="Nelder-Mead", options={"maxiter": 30000, "xatol": 1e-7, "fatol": 1e-7})
        w = np.maximum(r.x, 0); w /= w.sum()
        a = average_precision_score(y, X @ w)
        if a > best_a: best_a, best_w = a, w
    return best_w, best_a


def main():
    # ---- 1. Discover available archs and build OOF matrix ----
    avail = []
    for name, prefix, root in POOL:
        if have_full_set(prefix, root):
            avail.append({"name": name, "prefix": prefix, "root": root})
    print(f"Available archs: {len(avail)}")

    fold_df = pd.read_csv(FOLD_CSV)
    sid_to_subj = dict(zip(fold_df["sample_id"].astype(str), fold_df["subject_id"]))

    # Reference sample_ids + labels from first arch
    s0, l0, y0, f0 = load_oof(avail[0]["prefix"], avail[0]["root"])
    ref_s = s0.astype(str)
    ref_y = y0
    ref_fold = f0
    ref_subj = np.array([sid_to_subj.get(s, "?") for s in ref_s])

    # Permute every arch to match ref order
    X = np.zeros((len(ref_s), len(avail)), dtype=np.float64)
    arch_aurpc_per_fold = np.zeros((len(avail), 5), dtype=np.float64)
    arch_overall = np.zeros(len(avail), dtype=np.float64)
    for j, e in enumerate(avail):
        s, l, y, fids = load_oof(e["prefix"], e["root"])
        idx = {ss: i for i, ss in enumerate(s.astype(str))}
        perm = np.array([idx[ss] for ss in ref_s])
        l_p = l[perm]
        X[:, j] = l_p
        for f in range(5):
            mask = ref_fold == f
            if mask.sum() > 0:
                arch_aurpc_per_fold[j, f] = average_precision_score(ref_y[mask], 1/(1+np.exp(-l_p[mask])))
        arch_overall[j] = average_precision_score(ref_y, 1/(1+np.exp(-l_p)))

    # ---- Full-pool NM weights for reference ----
    print("\nDeriving full-pool NM weights (reference)...")
    w_full, oof_full = derive_weights(X, ref_y)
    print(f"Full ensemble OOF = {oof_full:.4f}")

    # ===== CHECK 1: NM weight stability on OOF halves =====
    # Split by SUBJECT (not random), so check is subject-disjoint inside the existing 5 folds
    print("\nCheck 1: NM weight stability across subject-disjoint OOF halves...")
    rng = np.random.default_rng(7)
    subjects = np.unique(ref_subj)
    rng.shuffle(subjects)
    half = len(subjects) // 2
    half_A = set(subjects[:half]); half_B = set(subjects[half:])
    maskA = np.array([s in half_A for s in ref_subj])
    maskB = ~maskA

    w_A, a_A = derive_weights(X[maskA], ref_y[maskA])
    w_B, a_B = derive_weights(X[maskB], ref_y[maskB])
    cos_sim = float(np.dot(w_A, w_B) / (np.linalg.norm(w_A) * np.linalg.norm(w_B) + 1e-12))
    l1_diff = float(np.abs(w_A - w_B).sum())
    spearman_w = spearmanr(w_A, w_B).statistic

    # Cross-evaluation: weight from A applied to B, and vice versa
    a_A_on_B = average_precision_score(ref_y[maskB], X[maskB] @ w_A)
    a_B_on_A = average_precision_score(ref_y[maskA], X[maskA] @ w_B)
    a_full_on_A = average_precision_score(ref_y[maskA], X[maskA] @ w_full)
    a_full_on_B = average_precision_score(ref_y[maskB], X[maskB] @ w_full)

    # Sample-level ensemble logit stability
    pred_full = X @ w_full
    pred_A_w = X @ w_A
    pred_B_w = X @ w_B
    spearman_AB = spearmanr(pred_A_w, pred_B_w).statistic

    # ===== CHECK 2: Per-arch OOF vs NM weight =====
    print("Check 2: Per-arch OOF AUPRC vs NM weight mismatches...")
    table2 = sorted(zip(range(len(avail)), avail, arch_overall, w_full), key=lambda r: -r[3])
    # Find suspicious: high weight but low OOF, low weight but high OOF
    arr_w = np.array([w_full[i] for i, _, _, _ in table2])
    arr_o = np.array([o for _, _, o, _ in table2])
    rk_w = np.argsort(-arr_w); rk_o = np.argsort(-arr_o)
    rank_pos_w = np.zeros(len(arr_w), int); rank_pos_o = np.zeros_like(rank_pos_w)
    for r, idx in enumerate(rk_w): rank_pos_w[idx] = r
    for r, idx in enumerate(rk_o): rank_pos_o[idx] = r
    rank_gap = rank_pos_w - rank_pos_o   # positive = OOF-strong but low NM weight

    # ===== CHECK 3: Per-arch per-fold AUPRC variance =====
    print("Check 3: Per-arch per-fold AUPRC variance...")
    fold_std = arch_aurpc_per_fold.std(axis=1)
    fold_range = arch_aurpc_per_fold.max(axis=1) - arch_aurpc_per_fold.min(axis=1)

    # ===== CHECK 4: Submission Spearman correlation =====
    print("Check 4: Past submission Spearman correlation...")
    sub_dir = REPO
    sub_files = sorted(sub_dir.glob("submission_test1_*.csv"))
    sub_preds = {}
    sub_ids_ref = None
    for sf in sub_files:
        # Skip smean variants vs main to reduce clutter? keep both.
        rows = list(csv.DictReader(open(sf)))
        if len(rows) != 20000: continue
        ids = np.array([r["sample_id"] for r in rows])
        preds = np.array([float(r["prediction"]) for r in rows], dtype=np.float64)
        if sub_ids_ref is None:
            sub_ids_ref = ids
            order = np.arange(len(ids))
        else:
            # Align by id
            id_to_i = {sid: i for i, sid in enumerate(ids)}
            order = np.array([id_to_i[sid] for sid in sub_ids_ref])
        sub_preds[sf.stem] = preds[order]
    names = sorted(sub_preds.keys())
    n_s = len(names)
    corr = np.zeros((n_s, n_s), dtype=np.float64)
    for i in range(n_s):
        for j in range(n_s):
            if i == j: corr[i, j] = 1.0
            else:
                corr[i, j] = spearmanr(sub_preds[names[i]], sub_preds[names[j]]).statistic

    # ===== Write report =====
    lines = []
    lines.append("# Defensive Checks Report — T1 33-arch Ensemble")
    lines.append("")
    lines.append(f"Generated for {len(avail)} available archs, OOF samples = {len(ref_s)}")
    lines.append(f"Full ensemble OOF AUPRC = **{oof_full:.4f}**")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Check 1
    lines.append("## Check 1 — NM weight stability (subject-disjoint OOF halves)")
    lines.append("")
    lines.append(f"- Half A: {maskA.sum()} samples ({len(half_A)} subjects), Half B: {maskB.sum()} samples ({len(half_B)} subjects)")
    lines.append(f"- AUPRC(half A) = {a_A:.4f}, AUPRC(half B) = {a_B:.4f}")
    lines.append(f"- AUPRC(full weights → half A) = {a_full_on_A:.4f}")
    lines.append(f"- AUPRC(full weights → half B) = {a_full_on_B:.4f}")
    lines.append(f"- AUPRC(half-A weights → half B) = {a_A_on_B:.4f}  ← cross-val")
    lines.append(f"- AUPRC(half-B weights → half A) = {a_B_on_A:.4f}  ← cross-val")
    lines.append(f"- **Cosine similarity** w_A vs w_B = **{cos_sim:.4f}**")
    lines.append(f"- L1 diff |w_A - w_B| = {l1_diff:.3f}")
    lines.append(f"- Spearman ρ(weights) = {spearman_w:.4f}")
    lines.append(f"- Spearman ρ(ensemble logits A vs B over all samples) = {spearman_AB:.4f}")
    lines.append("")
    lines.append("**Interpretation:**")
    lines.append(f"- cos(w_A, w_B) ≥ 0.95 → weights stable; ≥ 0.85 borderline; < 0.85 → overfit-OOF risk")
    lines.append(f"- cross-AUPRC (apply A's weights to B) within 0.005 of (apply full to B) → low overfit")
    lines.append("")
    lines.append("**Top-10 weight delta (A vs B):**")
    lines.append("| arch | w_A | w_B | Δ |")
    lines.append("|---|---|---|---|")
    deltas = sorted([(avail[i]["name"], float(w_A[i]), float(w_B[i]), float(w_A[i]-w_B[i])) for i in range(len(avail))],
                    key=lambda r: -abs(r[3]))
    for nm, a, b, d in deltas[:10]:
        lines.append(f"| {nm} | {a:.4f} | {b:.4f} | {d:+.4f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Check 2
    lines.append("## Check 2 — Per-arch OOF AUPRC vs NM weight")
    lines.append("")
    lines.append("| rank (NM) | arch | OOF AUPRC | NM weight | OOF rank | rank gap |")
    lines.append("|---|---|---|---|---|---|")
    for idx, e, o, w in table2:
        wr = rank_pos_w[list(table2).index((idx, e, o, w))]
        or_ = rank_pos_o[list(table2).index((idx, e, o, w))]
        g = wr - or_
        flag = ""
        if w > 0.05 and o < 0.78:
            flag = " ⚠️ high weight, weak OOF"
        if w < 0.005 and o > 0.83:
            flag = " ⚠️ strong OOF, ~0 weight (redundant?)"
        lines.append(f"| {wr+1} | {e['name']} | {o:.4f} | {w*100:5.2f}% | {or_+1} | {g:+d}{flag} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Check 3
    lines.append("## Check 3 — Per-arch per-fold AUPRC variance")
    lines.append("")
    lines.append("| arch | OOF | fold0 | fold1 | fold2 | fold3 | fold4 | std | range |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    order3 = np.argsort(-fold_std)
    for j in order3[:15]:  # top 15 by variance
        f_a = arch_aurpc_per_fold[j]
        lines.append(f"| {avail[j]['name']} | {arch_overall[j]:.4f} | {f_a[0]:.4f} | {f_a[1]:.4f} | {f_a[2]:.4f} | {f_a[3]:.4f} | {f_a[4]:.4f} | {fold_std[j]:.4f} | {fold_range[j]:.4f} |")
    lines.append("")
    lines.append(f"_(showing 15 highest-std archs out of {len(avail)})_")
    lines.append("")
    lines.append("**Interpretation:** range > 0.05 → fold-unstable arch, possibly noisy contributor.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Check 4
    lines.append("## Check 4 — Submission Spearman correlation")
    lines.append("")
    lines.append(f"Compared {n_s} submission CSVs.")
    lines.append("")
    short = [n.replace("submission_test1_", "") for n in names]
    lines.append("| | " + " | ".join(short) + " |")
    lines.append("|---|" + "|".join(["---"] * n_s) + "|")
    for i in range(n_s):
        row = [short[i]]
        for j in range(n_s):
            row.append(f"{corr[i,j]:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("**Interpretation:**")
    lines.append("- Pairs with ρ > 0.995 → essentially same prediction, public delta is noise")
    lines.append("- Pairs with ρ < 0.95 → structurally different, public delta reflects real change")
    lines.append("- Especially check filteeg vs specviews (the latest gain) and vs fullpool_final")
    lines.append("")
    # Spotlight: filteeg vs specviews vs fullpool_final
    spotlight = ["filteeg", "specviews", "fullpool_final"]
    lines.append("**Key pairs (latest progression):**")
    for a in spotlight:
        for b in spotlight:
            if a >= b: continue
            ai = next((i for i, n in enumerate(short) if n == a), None)
            bi = next((i for i, n in enumerate(short) if n == b), None)
            if ai is not None and bi is not None:
                lines.append(f"- ρ({a}, {b}) = **{corr[ai,bi]:.4f}**")
    lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT}")
    print(f"\n=== SUMMARY ===")
    print(f"Check 1: cos(w_A, w_B) = {cos_sim:.4f}  ({'STABLE' if cos_sim > 0.95 else 'BORDERLINE' if cos_sim > 0.85 else 'OVERFIT-RISK'})")
    print(f"Check 1: AUPRC(A→B) = {a_A_on_B:.4f}, AUPRC(full→B) = {a_full_on_B:.4f}, gap = {a_full_on_B - a_A_on_B:+.4f}")
    print(f"Check 2: max rank gap = {abs(rank_gap).max()} positions  (small=consistent)")
    print(f"Check 3: max fold-AUPRC range = {fold_range.max():.4f}  ({avail[fold_range.argmax()]['name']})")
    print(f"Check 4: filteeg vs specviews ρ = " + (
        f"{corr[short.index('filteeg'), short.index('specviews')]:.4f}"
        if 'filteeg' in short and 'specviews' in short else "n/a"))


if __name__ == "__main__":
    main()
