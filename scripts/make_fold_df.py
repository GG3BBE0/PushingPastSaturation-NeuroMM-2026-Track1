"""Generate a patient-disjoint 5-fold split for NeuroMM-2026.

Replacement for the buggy `split_fold.ipynb`.

Bug in the original: it called
    df_label5 = df[["subject_id", "label_5"]].drop_duplicates()
which keeps every (subject_id, label_5) pair, so subjects with multiple
label_5 values produce multiple rows. StratifiedKFold then assigns each
(subject, label_5) pair independently, and the subsequent
`merge(on="subject_id")` duplicates samples across folds. Result: same
sample appears in train AND val of the same fold → patient leak.

This script:
  1. Reads `neuromm2026_train_val_patient_split.csv`.
  2. Extracts `label_5` from sample_id (suffix after "__").
  3. Collapses to ONE row per subject_id, using max(label_5) as the
     stratification key (so positive subtypes drive the stratification).
  4. Runs StratifiedGroupKFold(5) with groups=subject_id, guaranteeing
     each subject lives in exactly one fold.
  5. Merges fold back to sample-level (1 row per sample, no duplication).
  6. Runs hard assertions + prints a summary so you can sanity-check.

Usage:
    python scripts/make_fold_df.py
    python scripts/make_fold_df.py --seed 1111
    python scripts/make_fold_df.py --out fold_df.csv     # overwrite original

Output schema (sample-level):
    sample_id, label, label_type, subject_id, label_5, fold
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
DEFAULT_MANIFEST = REPO / "NeuroMM-2026/annotations/neuromm2026_train_val_patient_split.csv"
DEFAULT_OUT = REPO / "fold_df_fixed.csv"


def build_fold_df(manifest: Path, n_splits: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(manifest)
    df.columns = [c.strip() for c in df.columns]

    # label_5 = digit after "__" suffix; e.g. "DA001003_0_2000_500__3" -> 3
    df["label_5"] = df["sample_id"].apply(lambda x: int(x.split("__")[1]))

    # ---- patient-level table (1 row per subject) ----
    # Use max(label_5) so positive subtypes drive stratification
    # (a subject with any positive subtype is more interesting than a pure negative).
    df_subj = (
        df.groupby("subject_id", as_index=False)["label_5"]
        .max()
        .rename(columns={"label_5": "strat_key"})
    )
    assert df_subj["subject_id"].is_unique, "subject_id must be unique after groupby"

    # ---- StratifiedGroupKFold (groups=subject_id ensures patient-disjoint folds) ----
    df_subj["fold"] = -1
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(
        skf.split(df_subj, df_subj["strat_key"], df_subj["subject_id"])
    ):
        df_subj.loc[val_idx, "fold"] = fold_idx
    assert (df_subj["fold"] >= 0).all(), "every subject must receive a fold"

    # ---- merge fold back to sample level (1-to-1) ----
    sample_fold = df_subj[["subject_id", "fold"]]
    out = df.merge(sample_fold, on="subject_id", how="left")
    # drop the split column from the official manifest if present;
    # the canonical train/val info is now encoded by `fold`.
    if "split" in out.columns:
        out = out.drop(columns=["split"])
    return out, df_subj.drop(columns=["strat_key"])


def verify(out: pd.DataFrame, df_subj: pd.DataFrame, manifest: Path) -> None:
    src = pd.read_csv(manifest)

    # 1. Same row count as source manifest (no duplication)
    assert len(out) == len(src), (
        f"row count mismatch: out={len(out)} vs source={len(src)} → duplication bug"
    )

    # 2. Every sample appears in exactly 1 fold
    per_sample = out.groupby("sample_id")["fold"].nunique()
    assert (per_sample == 1).all(), (
        f"{(per_sample > 1).sum()} samples appear in multiple folds → leak"
    )

    # 3. Every subject in exactly 1 fold (patient-disjoint)
    per_subj = out.groupby("subject_id")["fold"].nunique()
    assert (per_subj == 1).all(), (
        f"{(per_subj > 1).sum()} subjects span multiple folds → patient leak"
    )

    # 4. Every fold is non-empty
    assert out["fold"].nunique() == df_subj["fold"].nunique() == len(df_subj["fold"].unique()), (
        "some folds are empty"
    )


def print_summary(out: pd.DataFrame, df_subj: pd.DataFrame) -> None:
    n_subj = df_subj["subject_id"].nunique()
    n_samp = len(out)
    print(f"\n=== fold_df summary ===")
    print(f"total subjects : {n_subj}")
    print(f"total samples  : {n_samp}")
    print(f"folds          : {sorted(out['fold'].unique().tolist())}")

    print(f"\n=== per-fold subject / sample counts ===")
    fold_subj = out.groupby("fold")["subject_id"].nunique().rename("n_subjects")
    fold_samp = out.groupby("fold").size().rename("n_samples")
    fold_pos = (
        out.groupby("fold")["label"].mean().mul(100).round(2).rename("pos_pct")
    )
    summary = pd.concat([fold_subj, fold_samp, fold_pos], axis=1)
    print(summary.to_string())

    print(f"\n=== per-fold label_5 distribution (val side of each fold) ===")
    pivot = (
        out.groupby(["fold", "label_5"]).size().unstack(fill_value=0)
    )
    print(pivot.to_string())

    print(f"\n=== subject-level stratification (max label_5 per subject) per fold ===")
    subj_strat = (
        out.groupby("subject_id")
        .agg(fold=("fold", "first"), max_label_5=("label_5", "max"))
        .reset_index()
    )
    print(
        subj_strat.groupby(["fold", "max_label_5"]).size().unstack(fill_value=0).to_string()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    print(f"Reading: {args.manifest}")
    out, df_subj = build_fold_df(args.manifest, args.n_splits, args.seed)
    verify(out, df_subj, args.manifest)
    print_summary(out, df_subj)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
