"""Generate final summary of muku-NeuroMM experiments.

Reads all relevant JSON outputs and prints a clean table for reporting.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parent.parent))
METRICS = REPO / "neuromm26_results/metrics"


def load(name):
    p = METRICS / name
    if p.exists():
        return json.loads(p.read_text())
    return None


def fmt(v, k="auprc"):
    if isinstance(v, dict) and k in v:
        return f"{v[k]:.4f}"
    if isinstance(v, dict) and "metrics" in v:
        return f"{v['metrics'].get(k, 0):.4f}"
    return "-"


def main():
    print("=" * 80)
    print("NeuroMM-2026 T1 (binary EEG-only) — muku-NeuroMM final summary")
    print("=" * 80)

    print("\n--- Stage 1: baseline reproduction (4 seeds × 30 epoch) ---")
    print(f"  legacy/tcnet_eeg        4-seed mean         val AUPRC: 0.7264 ± 0.034")
    print(f"  legacy/efficientnet_v2_s_eeg 4-seed mean     val AUPRC: 0.7387 ± 0.002")
    print(f"  legacy/mobilenet_v3_large_eeg 4-seed mean    val AUPRC: 0.7223 ± 0.015")

    print("\n--- Stage 2: 4-seed ensemble (per-model) ---")
    es = load("ensemble_val_summary.json")
    if es and "T1" in es:
        print(f"  tcnet_eeg ensemble                    val AUPRC: {es['T1']['ensemble']['auprc']:.4f}")

    print("\n--- Stage 3: cross-model 12-ckpt legacy ensemble ---")
    cm = load("cross_model_ensemble.json")
    if cm and "T1" in cm:
        m = cm["T1"]["cross_model"]
        print(f"  legacy 12-ckpt (tcnet+effv2s+mbn)      val AUPRC: {m.get('auprc', 0):.4f}  binary_f1: {m.get('binary_f1', 0):.4f}")

    print("\n--- Stage 4: muku-NeuroMM additions ---")
    mc = load("muku_combined_ensemble.json")
    if mc:
        if mc.get("per_backbone"):
            print("  muku per-backbone 4-seed ensembles:")
            for k, v in mc["per_backbone"].items():
                print(f"    {k:50s}  auprc: {v.get('auprc', 0):.4f}  f1: {v.get('binary_f1', 0):.4f}")
        if mc.get("cross_backbone_muku"):
            v = mc["cross_backbone_muku"]
            print(f"  muku cross-backbone 12-ckpt ensemble    auprc: {v.get('auprc', 0):.4f}  f1: {v.get('binary_f1', 0):.4f}")
        if mc.get("combined_muku_plus_legacy"):
            v = mc["combined_muku_plus_legacy"]
            print(f"  COMBINED muku+legacy 24-ckpt            auprc: {v.get('auprc', 0):.4f}  f1: {v.get('binary_f1', 0):.4f}")

    print("\n--- Stage 5: TTA + threshold tuning ---")
    tt = load("muku_tta_threshold.json")
    if tt:
        for k, v in tt.items():
            m = v.get("metrics", {})
            t = v.get("best_thr", {})
            print(f"  {k:30s}  auprc: {m.get('auprc', 0):.4f}  f1@0.5: {m.get('binary_f1', 0):.4f}  f1@best_thr: {t.get('f1', 0):.4f}@{t.get('thr', 0.5):.2f}")

    print("\n--- Final: best AUPRC achieved ---")
    best = 0
    best_label = ""
    if cm and "T1" in cm:
        v = cm["T1"]["cross_model"]["auprc"]
        if v > best: best, best_label = v, "legacy 12-ckpt"
    if mc and mc.get("cross_backbone_muku"):
        v = mc["cross_backbone_muku"]["auprc"]
        if v > best: best, best_label = v, "muku 12-ckpt"
    if mc and mc.get("combined_muku_plus_legacy"):
        v = mc["combined_muku_plus_legacy"]["auprc"]
        if v > best: best, best_label = v, "combined 24-ckpt"
    if tt:
        for k, v in tt.items():
            a = v.get("metrics", {}).get("auprc", 0)
            if a > best: best, best_label = a, k
    print(f"  BEST val AUPRC: {best:.4f}  ({best_label})")
    print(f"  vs baseline (tcnet single 0.7264):   +{(best - 0.7264) * 100:.2f}%")
    print(f"  vs legacy 12-ckpt (0.8224):           {'+' if best > 0.8224 else ''}{(best - 0.8224) * 100:+.2f}%")


if __name__ == "__main__":
    main()
