"""Deep analysis of why fold4 has consistently low OOF AUPRC across models.

Sections:
A. Per-fold class balance + subject distribution
B. AUPRC vs class-balance-normalized lift (does fold4 look better when normalized?)
C. Per-subject failure analysis in fold4 (which subjects do models miss?)
D. Signal-level fold comparison (EEG amplitude / spectrum stats)
E. Label-type composition per fold (which subtypes over/under in fold4?)
F. Cross-fold subject characteristics (sample counts, positivity rate distribution)
"""
from __future__ import annotations
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
REAL = REPO / "neuromm26_real_5fold_result"
RES = REPO / "neuromm26_results"
FOLD_CSV = REPO / "fold_df_fixed.csv"
EEG_ROOT = REPO / "neuromm26_datasets/processed/features/eeg"
REPORT = REPO / "FOLD4_ANALYSIS.md"

# Top NM-weight archs to use as proxies for "the ensemble"
TOP_ARCHS = [
    ("ConcatFilt maxvit256",         "concat_cwt_filtered_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    ("ConcatSuperlet maxvit384",     "concat_superlet_fold__maxvit_tiny_tf_384_in1k", RES),
    ("ConcatPaul maxvit256",         "concat_cwt_paul_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
    ("muku Filt convnext_pico",      "muku_filteeg_fold__convnext_pico_d1_in1k", RES),
    ("ConcatCWT maxvit384",          "concat_cwt_fold__maxvit_tiny_tf_384_in1k", RES),
    ("muku raw convnext_pico",       "muku_fold__convnext_pico_d1_in1k", REAL),
    ("muku raw resnet18",            "muku_fold__resnet18", REAL),
    ("ConcatCWT maxvit256",          "concat_cwt_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES),
]


def load_oof(prefix, root):
    s, l, y = [], [], []
    for f in range(5):
        d = np.load(root / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz")
        s.append(d["sample_ids"]); l.append(d["logits"]); y.append(d["labels"])
    return (np.concatenate(s).astype(str),
            np.concatenate(l).astype(np.float64),
            np.concatenate(y).astype(np.int32))


def main():
    df = pd.read_csv(FOLD_CSV)
    df["sample_id"] = df["sample_id"].astype(str)
    sid_to_fold = dict(zip(df["sample_id"], df["fold"]))
    sid_to_subj = dict(zip(df["sample_id"], df["subject_id"]))
    sid_to_label = dict(zip(df["sample_id"], df["label"]))
    sid_to_ltype = dict(zip(df["sample_id"], df["label_type"]))

    # ===== SECTION A =====
    A = df.groupby("fold").agg(
        n=("sample_id", "count"),
        n_pos=("label", "sum"),
        n_subj=("subject_id", "nunique"),
    )
    A["pos_rate"] = A["n_pos"] / A["n"]
    A["random_baseline_auprc"] = A["pos_rate"]
    print("\n=== A. Per-fold class balance ===")
    print(A)

    # ===== SECTION B: AUPRC + lift =====
    print("\n=== B. Per-arch per-fold AUPRC + lift ===")
    rows = []
    arch_oofs = {}
    for name, prefix, root in TOP_ARCHS:
        s, l, y = load_oof(prefix, root)
        p = 1 / (1 + np.exp(-l))
        arch_oofs[name] = (s, p, y)
        per_fold = {}
        for f in range(5):
            mask = np.array([sid_to_fold[ss] == f for ss in s])
            if mask.sum() > 0:
                ap = average_precision_score(y[mask], p[mask])
                base = y[mask].mean()
                lift = ap / base if base > 0 else 0
                per_fold[f] = (ap, lift)
        rows.append((name, per_fold))
        print(f"{name}:")
        for f in range(5):
            ap, lift = per_fold[f]
            print(f"  fold{f}: AUPRC={ap:.4f}  baseline={A['pos_rate'][f]:.4f}  lift={lift:.2f}x")

    # Build ensemble OOF (mean over top archs)
    ref_s = list(arch_oofs.values())[0][0]
    ref_y = list(arch_oofs.values())[0][2]
    P_mat = []
    for name, (s, p, y) in arch_oofs.items():
        idx = {ss: i for i, ss in enumerate(s)}
        perm = np.array([idx[x] for x in ref_s])
        P_mat.append(p[perm])
    P_mean = np.mean(P_mat, axis=0)
    print("\n  Ensemble (mean of top archs):")
    ens_per_fold = {}
    for f in range(5):
        mask = np.array([sid_to_fold[ss] == f for ss in ref_s])
        if mask.sum() > 0:
            ap = average_precision_score(ref_y[mask], P_mean[mask])
            base = ref_y[mask].mean()
            lift = ap / base if base > 0 else 0
            ens_per_fold[f] = (ap, lift)
            print(f"  fold{f}: AUPRC={ap:.4f}  baseline={base:.4f}  lift={lift:.2f}x")

    # ===== SECTION C: Per-subject failure analysis in fold4 =====
    print("\n=== C. Per-subject ensemble error in fold4 ===")
    fold4_mask = np.array([sid_to_fold[ss] == 4 for ss in ref_s])
    fold4_sids = ref_s[fold4_mask]
    fold4_y = ref_y[fold4_mask]
    fold4_p = P_mean[fold4_mask]
    fold4_subj = np.array([sid_to_subj[ss] for ss in fold4_sids])

    subj_rows = []
    for sj in np.unique(fold4_subj):
        m = fold4_subj == sj
        if m.sum() == 0: continue
        n = int(m.sum()); np_ = int(fold4_y[m].sum())
        mean_p = float(fold4_p[m].mean())
        if np_ >= 1 and np_ < n:
            ap = average_precision_score(fold4_y[m], fold4_p[m])
            lift = ap / (np_ / n)
        else:
            ap, lift = float("nan"), float("nan")
        # mean ensemble prob of positives vs negatives within this subject
        pos_p = float(fold4_p[m & (fold4_y == 1)].mean()) if (fold4_y[m] == 1).any() else float("nan")
        neg_p = float(fold4_p[m & (fold4_y == 0)].mean()) if (fold4_y[m] == 0).any() else float("nan")
        subj_rows.append((sj, n, np_, n - np_, np_ / n, mean_p, pos_p, neg_p, ap, lift))
    subj_df = pd.DataFrame(subj_rows, columns=[
        "subject", "n", "n_pos", "n_neg", "pos_rate", "mean_prob",
        "pos_mean_prob", "neg_mean_prob", "subj_AUPRC", "subj_lift",
    ]).sort_values("subj_AUPRC", ascending=True)
    print(subj_df.to_string(index=False, float_format="%.4f"))

    # Identify systematic-fail subjects (pos_mean_prob LOW or below neg_mean_prob)
    bad = subj_df[
        (~subj_df["pos_mean_prob"].isna()) &
        (subj_df["pos_mean_prob"] < subj_df["neg_mean_prob"])
    ]
    if len(bad):
        print("\n  ⚠️ Subjects where positives score LOWER than negatives (model is INVERTED on these):")
        print(bad[["subject", "n_pos", "n_neg", "pos_mean_prob", "neg_mean_prob"]].to_string(index=False))

    # ===== SECTION D: Signal stats per fold =====
    print("\n=== D. EEG signal stats per fold (sample 200 per fold) ===")
    sig_rows = []
    for f in range(5):
        fold_sids = df.loc[df["fold"] == f, "sample_id"].sample(min(200, (df["fold"]==f).sum()), random_state=7).tolist()
        amps, energies = [], []
        for sid in fold_sids:
            try:
                wave = np.load(EEG_ROOT / f"{sid}.npy")  # (29,2000)
                # use first 23 channels (EEG only, skip heart/muscle)
                eeg = wave[:23]
                amps.append(float(np.abs(eeg).mean()))
                energies.append(float((eeg ** 2).mean()))
            except Exception:
                pass
        sig_rows.append((f, len(amps),
                         float(np.mean(amps)), float(np.std(amps)),
                         float(np.mean(energies)), float(np.std(energies))))
    sig_df = pd.DataFrame(sig_rows, columns=[
        "fold", "n_samples", "mean_amp", "std_amp", "mean_energy", "std_energy"])
    print(sig_df.to_string(index=False))

    # ===== SECTION E: label_type per fold =====
    print("\n=== E. label_type per fold ===")
    lt = df.groupby(["fold", "label_type"]).size().unstack(fill_value=0)
    lt["total_pos"] = lt[[c for c in lt.columns if c != 0]].sum(axis=1)
    lt["pos_rate"] = lt["total_pos"] / lt.sum(axis=1)
    print(lt)

    # ===== SECTION F: Subject pos-rate distribution per fold =====
    print("\n=== F. Subject-level positivity rate distribution per fold ===")
    for f in range(5):
        fold_sub = df[df["fold"] == f].groupby("subject_id")["label"].agg(["count", "sum"])
        fold_sub.columns = ["n", "n_pos"]
        fold_sub["pos_rate"] = fold_sub["n_pos"] / fold_sub["n"]
        print(f"\nfold{f}: {len(fold_sub)} subjects")
        print(f"  per-subject pos_rate: min={fold_sub['pos_rate'].min():.3f}  "
              f"max={fold_sub['pos_rate'].max():.3f}  "
              f"median={fold_sub['pos_rate'].median():.3f}  "
              f"#subj with 0 positives = {(fold_sub['n_pos']==0).sum()}  "
              f"#subj with n_pos < 5 = {(fold_sub['n_pos']<5).sum()}")

    # ===== WRITE REPORT =====
    lines = []
    lines.append("# Fold4 Deep Analysis — why is fold4 consistently low?")
    lines.append("")
    lines.append("## A. Per-fold class balance (root cause)")
    lines.append("")
    lines.append("| fold | n | n_pos | pos_rate | random_baseline_AUPRC |")
    lines.append("|---|---|---|---|---|")
    for f in range(5):
        lines.append(f"| {f} | {A['n'][f]} | {A['n_pos'][f]} | {A['pos_rate'][f]:.4f} | {A['pos_rate'][f]:.4f} |")
    lines.append("")
    lines.append("**Key fact:** fold4 positivity rate **5.8%** vs others 9-13%. "
                 "AUPRC's random baseline = positivity rate → fold4's baseline is half of others. "
                 "→ Same-quality ranker scores lower AUPRC on fold4 purely from class imbalance.")
    lines.append("")
    lines.append("## B. AUPRC normalized by class balance (lift)")
    lines.append("")
    lines.append("| arch | f0 AUPRC | f0 lift | f4 AUPRC | f4 lift | lift Δ |")
    lines.append("|---|---|---|---|---|---|")
    for name, per_fold in rows:
        ap0, l0 = per_fold[0]; ap4, l4 = per_fold[4]
        lines.append(f"| {name} | {ap0:.4f} | {l0:.2f}x | {ap4:.4f} | {l4:.2f}x | {l4-l0:+.2f}x |")
    ap0e, l0e = ens_per_fold[0]; ap4e, l4e = ens_per_fold[4]
    lines.append(f"| **ensemble (mean)** | {ap0e:.4f} | {l0e:.2f}x | {ap4e:.4f} | {l4e:.2f}x | {l4e-l0e:+.2f}x |")
    lines.append("")
    lines.append("**Interpretation:** if lift Δ > 0 → model is actually *better* in ranking on fold4 (just looks bad due to baseline).")
    lines.append("")
    lines.append("## C. Per-subject ensemble error in fold4")
    lines.append("")
    lines.append("Sorted by ascending subj_AUPRC. NaN = subject has only one class (e.g., all negatives).")
    lines.append("")
    lines.append("| subject | n | n_pos | n_neg | pos_rate | mean_prob | pos_prob | neg_prob | AUPRC | lift |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in subj_df.iterrows():
        sap = "—" if pd.isna(r["subj_AUPRC"]) else f"{r['subj_AUPRC']:.4f}"
        sli = "—" if pd.isna(r["subj_lift"]) else f"{r['subj_lift']:.2f}"
        spp = "—" if pd.isna(r["pos_mean_prob"]) else f"{r['pos_mean_prob']:.3f}"
        snp = "—" if pd.isna(r["neg_mean_prob"]) else f"{r['neg_mean_prob']:.3f}"
        lines.append(f"| {r['subject']} | {int(r['n'])} | {int(r['n_pos'])} | {int(r['n_neg'])} | "
                     f"{r['pos_rate']:.3f} | {r['mean_prob']:.3f} | {spp} | {snp} | {sap} | {sli} |")
    if len(bad):
        lines.append("")
        lines.append("**⚠️ Inverted subjects (pos prob < neg prob):**")
        for _, r in bad.iterrows():
            lines.append(f"- `{r['subject']}` n_pos={int(r['n_pos'])} n_neg={int(r['n_neg'])} "
                         f"pos_prob={r['pos_mean_prob']:.3f} neg_prob={r['neg_mean_prob']:.3f}")
    lines.append("")
    lines.append("## D. EEG signal stats per fold")
    lines.append("")
    lines.append("| fold | n_sampled | mean_amp | std_amp | mean_energy | std_energy |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in sig_df.iterrows():
        lines.append(f"| {int(r['fold'])} | {int(r['n_samples'])} | "
                     f"{r['mean_amp']:.4f} | {r['std_amp']:.4f} | "
                     f"{r['mean_energy']:.2f} | {r['std_energy']:.2f} |")
    lines.append("")
    lines.append("## E. label_type composition per fold")
    lines.append("")
    lt_str = lt.to_string()
    lines.append("```")
    lines.append(lt_str)
    lines.append("```")
    lines.append("")
    lines.append("## F. Per-subject positivity rate (sparsity check)")
    lines.append("")
    lines.append("| fold | n_subj | min pos_rate | max pos_rate | median | #subj 0 pos | #subj <5 pos |")
    lines.append("|---|---|---|---|---|---|---|")
    for f in range(5):
        fold_sub = df[df["fold"] == f].groupby("subject_id")["label"].agg(["count", "sum"])
        fold_sub.columns = ["n", "n_pos"]
        fold_sub["pos_rate"] = fold_sub["n_pos"] / fold_sub["n"]
        lines.append(f"| {f} | {len(fold_sub)} | {fold_sub['pos_rate'].min():.3f} | "
                     f"{fold_sub['pos_rate'].max():.3f} | {fold_sub['pos_rate'].median():.3f} | "
                     f"{(fold_sub['n_pos']==0).sum()} | {(fold_sub['n_pos']<5).sum()} |")
    lines.append("")
    lines.append("## Synthesis")
    lines.append("")
    lines.append("1. fold4 positivity rate 5.8% (others 9-13%) → AUPRC random baseline ~half → expected lower AUPRC")
    lines.append("2. If lift Δ ≥ 0 across models, fold4 is NOT a 'broken' fold — model quality is comparable")
    lines.append("3. If specific subjects show inverted prob → label noise or subject-specific distribution shift")
    lines.append("4. Action items will depend on findings — see report")

    REPORT.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT}")


if __name__ == "__main__":
    main()
