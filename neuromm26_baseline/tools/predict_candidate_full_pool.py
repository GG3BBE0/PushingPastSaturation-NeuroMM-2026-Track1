"""Comprehensive Task-1 submission from the FULL model pool (original 15 + new diverse).

Auto-discovers which architectures have a complete 5-fold set (ckpts + OOF),
derives Nelder-Mead weights from the available OOF, runs candidate inference
(avg 5 folds per arch), weighted-combines, writes submission.

Robust for unattended use: any architecture missing ckpts/OOF is skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from neuromm26_baseline.models.muku_eegnet import MukuEEGNet
from neuromm26_baseline.models.muku_eegnet_v2 import MukuEEGNetV2
from neuromm26_baseline.models.muku_eegnet_v3 import MukuEEGNetV3
from neuromm26_baseline.models.spec_cnn import SpecCNN
from neuromm26_baseline.models.spec_cnn_concat import ConcatSpecCNN
from neuromm26_baseline.models.legacy.registry import build_legacy_eeg_model

REPO = Path(os.environ.get("NEUROMM_REPO", Path(__file__).resolve().parents[2]))
REAL = REPO / "neuromm26_real_5fold_result"
RES = REPO / "neuromm26_results"
FOLD_CSV = REPO / "fold_df_fixed.csv"

# name, exp_prefix, root, kind, backbone, feature, target_size
POOL = [
    ("muku raw resnet18", "muku_fold__resnet18", REAL, "mukuv1", "resnet18", "eeg", 224),
    ("muku raw effv2s_b0", "muku_fold__tf_efficientnet_b0_ns_jft_in1k", REAL, "mukuv1", "tf_efficientnet_b0.ns_jft_in1k", "eeg", 224),
    ("muku raw convnext_pico", "muku_fold__convnext_pico_d1_in1k", REAL, "mukuv1", "convnext_pico.d1_in1k", "eeg", 224),
    ("CWT resnet18", "spec_cwt_fold__resnet18", REAL, "speccnn", "resnet18", "cwt", 224),
    ("CWT effv2s_b0", "spec_cwt_fold__tf_efficientnet_b0_ns_jft_in1k", REAL, "speccnn", "tf_efficientnet_b0.ns_jft_in1k", "cwt", 224),
    ("CWT convnext_pico", "spec_cwt_fold__convnext_pico_d1_in1k", REAL, "speccnn", "convnext_pico.d1_in1k", "cwt", 224),
    ("ConcatCWT convnext_tiny", "concat_cwt_fold__convnext_tiny_fb_in22k_ft_in1k_384", REAL, "concat", "convnext_tiny.fb_in22k_ft_in1k_384", "cwt", 384),
    ("muku V2 sl resnet18", "muku_v2_superlet_fold__resnet18", RES, "mukuv2", "resnet18", "superlet_eeg", 224),
    ("muku V2 sl effv2s_b0", "muku_v2_superlet_fold__tf_efficientnet_b0_ns_jft_in1k", RES, "mukuv2", "tf_efficientnet_b0.ns_jft_in1k", "superlet_eeg", 224),
    ("muku V2 sl convnext_pico", "muku_v2_superlet_fold__convnext_pico_d1_in1k", RES, "mukuv2", "convnext_pico.d1_in1k", "superlet_eeg", 224),
    ("muku V3 effv2s_b0", "muku_v3_superlet_fold__tf_efficientnet_b0_ns_jft_in1k", RES, "mukuv3", "tf_efficientnet_b0.ns_jft_in1k", "superlet_eeg", 224),
    ("muku V3 convnext_pico", "muku_v3_superlet_fold__convnext_pico_d1_in1k", RES, "mukuv3", "convnext_pico.d1_in1k", "superlet_eeg", 224),
    ("muku V3 mobilenetv3", "muku_v3_superlet_fold__mobilenetv3_large_100_ra_in1k", RES, "mukuv3", "mobilenetv3_large_100.ra_in1k", "superlet_eeg", 224),
    ("legacy tcnet_eeg", "legacy_tcnet_eeg_fold", RES, "legacy", "tcnet_eeg", "eeg", 224),
    ("legacy mobilenet_v3_large_eeg", "legacy_mobilenet_v3_large_eeg_fold", RES, "legacy", "mobilenet_v3_large_eeg", "eeg", 224),
    # ---- new diverse (wave 1) ----
    ("muku V3 swinv2", "muku_v3_superlet_fold__swinv2_cr_tiny_ns_224_sw_in1k", RES, "mukuv3", "swinv2_cr_tiny_ns_224.sw_in1k", "superlet_eeg", 224),
    ("ConcatCWT maxvit256", "concat_cwt_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES, "concat", "maxvit_rmlp_tiny_rw_256.sw_in1k", "cwt", 256),
    # ConcatSuperlet convnext: re-included. OOF +0.006 is the trustworthy CV signal;
    # earlier test−0.006 was a public-LB-driven removal (private-LB risk).
    ("ConcatSuperlet convnext_tiny384", "concat_superlet_fold__convnext_tiny_fb_in22k_ft_in1k_384", RES, "concat", "convnext_tiny.fb_in22k_ft_in1k_384", "superlet", 384),
    # ---- new diverse (wave 2) ----
    ("ConcatCWT maxvit384", "concat_cwt_fold__maxvit_tiny_tf_384_in1k", RES, "concat", "maxvit_tiny_tf_384.in1k", "cwt", 384),
    ("CWT swinv2", "spec_cwt_fold__swinv2_cr_tiny_ns_224_sw_in1k", RES, "speccnn", "swinv2_cr_tiny_ns_224.sw_in1k", "cwt", 224),
    ("CWT caformer", "spec_cwt_fold__caformer_s18_sail_in22k_ft_in1k", RES, "speccnn", "caformer_s18.sail_in22k_ft_in1k", "cwt", 224),
    # ---- new diverse (wave 4) ----
    ("ConcatSuperlet maxvit384", "concat_superlet_fold__maxvit_tiny_tf_384_in1k", RES, "concat", "maxvit_tiny_tf_384.in1k", "superlet", 384),
    ("ConcatCWT coatnet", "concat_cwt_fold__coatnet_0_rw_224_sw_in1k", RES, "concat", "coatnet_0_rw_224.sw_in1k", "cwt", 224),
    ("ConcatCWT convnext_small", "concat_cwt_fold__convnext_small_fb_in22k_ft_in1k_384", RES, "concat", "convnext_small.fb_in22k_ft_in1k_384", "cwt", 384),
    # ---- new diverse (wave 5, GPU 1-2) ----
    ("ConcatCWT maxvit_small384", "concat_cwt_fold__maxvit_small_tf_384_in1k", RES, "concat", "maxvit_small_tf_384.in1k", "cwt", 384),
    ("ConcatCWT coatnet2", "concat_cwt_fold__coatnet_rmlp_2_rw_224_sw_in12k_ft_in1k", RES, "concat", "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", "cwt", 224),
    # ---- filtered CWT: weak single-model (-0.08) BUT ensemble OOF +0.0073 (diversity). Re-included; test will tell. ----
    ("ConcatFilt maxvit256", "concat_cwt_filtered_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES, "concat", "maxvit_rmlp_tiny_rw_256.sw_in1k", "cwt_filtered", 256),
    ("CWTFilt convnext_pico", "spec_cwt_filtered_fold__convnext_pico_d1_in1k", RES, "speccnn", "convnext_pico.d1_in1k", "cwt_filtered", 224),
    # ---- new spec views (STFT, Paul wavelet) — full-info, diverse basis ----
    ("ConcatSTFT maxvit256", "concat_stft_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES, "concat", "maxvit_rmlp_tiny_rw_256.sw_in1k", "stft", 256),
    ("STFT convnext_pico", "spec_stft_fold__convnext_pico_d1_in1k", RES, "speccnn", "convnext_pico.d1_in1k", "stft", 224),
    ("ConcatPaul maxvit256", "concat_cwt_paul_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES, "concat", "maxvit_rmlp_tiny_rw_256.sw_in1k", "cwt_paul", 256),
    ("Paul convnext_pico", "spec_cwt_paul_fold__convnext_pico_d1_in1k", RES, "speccnn", "convnext_pico.d1_in1k", "cwt_paul", 224),
    # ---- filtered RAW EEG (no spec); muku/legacy time-domain models ----
    ("muku Filt convnext_pico", "muku_filteeg_fold__convnext_pico_d1_in1k", RES, "mukuv1", "convnext_pico.d1_in1k", "filt_eeg", 224),
    ("muku Filt resnet18", "muku_filteeg_fold__resnet18", RES, "mukuv1", "resnet18", "filt_eeg", 224),
    ("legacy tcnet Filt", "legacy_tcnet_filteeg_fold", RES, "legacy", "tcnet_eeg", "filt_eeg", 224),
    ("muku V2 Filt sl convnext_pico", "muku_v2_filteeg_sl_fold__convnext_pico_d1_in1k", RES, "mukuv2", "convnext_pico.d1_in1k", "filt_superlet_eeg", 224),
    # ---- Axis-1 wave: 1D time-domain (zero new infra; legacy registry) ----
    ("legacy eegnet",   "legacy_eegnet_fold",   RES, "legacy", "eegnet",   "eeg", 224),
    ("legacy actnet_s", "legacy_actnet_s_fold", RES, "legacy", "actnet_s", "eeg", 224),
    ("legacy lmda_eeg", "legacy_lmda_eeg_fold", RES, "legacy", "lmda_eeg", "eeg", 224),
    # ---- Axis-1 wave: multiband (130-ch raw+δ/θ/α/β bandpass time-domain) ----
    ("muku_mb convnext_pico", "muku_mb_fold__convnext_pico_d1_in1k",          RES, "mukumb", "convnext_pico.d1_in1k",          "multiband", 224),
    ("muku_mb effv2s_b0",     "muku_mb_fold__tf_efficientnet_b0_ns_jft_in1k", RES, "mukumb", "tf_efficientnet_b0.ns_jft_in1k", "multiband", 224),
    # ---- NCHC B-plan: maxvit_tiny @384 on new spec basis (filt / paul / stft) ----
    ("ConcatFilt maxvit384", "concat_cwt_filtered_fold__maxvit_tiny_tf_384_in1k", RES, "concat", "maxvit_tiny_tf_384.in1k", "cwt_filtered", 384),
    ("ConcatPaul maxvit384", "concat_cwt_paul_fold__maxvit_tiny_tf_384_in1k",     RES, "concat", "maxvit_tiny_tf_384.in1k", "cwt_paul",     384),
    ("ConcatSTFT maxvit384", "concat_stft_fold__maxvit_tiny_tf_384_in1k",         RES, "concat", "maxvit_tiny_tf_384.in1k", "stft",         384),
    # ---- Heavy aug ablation: ConcatSuperlet maxvit384 retrained with heavy SpecAug ----
    ("ConcatSuperlet heavyaug384", "concat_superlet_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES, "concat", "maxvit_tiny_tf_384.in1k", "superlet", 384),
    # ---- NCHC heavy aug rollout @256 (Paul +0.0080, CWT +0.0036; Filt skipped: -0.003) ----
    ("ConcatPaul heavyaug256", "concat_cwt_paul_heavyaug_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES, "concat", "maxvit_rmlp_tiny_rw_256.sw_in1k", "cwt_paul", 256),
    ("ConcatCWT heavyaug256",  "concat_cwt_heavyaug_fold__maxvit_rmlp_tiny_rw_256_sw_in1k",      RES, "concat", "maxvit_rmlp_tiny_rw_256.sw_in1k", "cwt",      256),
    # ---- focal + heavy aug (superadditive: single-arch +0.0227 OOF, biggest single-arch gain) ----
    ("ConcatPaul focalheavy256", "concat_cwt_paul_focalheavy_fold__maxvit_rmlp_tiny_rw_256_sw_in1k", RES, "concat", "maxvit_rmlp_tiny_rw_256.sw_in1k", "cwt_paul", 256),
    ("ConcatCWT focalheavy256",  "concat_cwt_focalheavy_fold__maxvit_rmlp_tiny_rw_256_sw_in1k",      RES, "concat", "maxvit_rmlp_tiny_rw_256.sw_in1k", "cwt",      256),
    # ---- NCHC batch-2 heavy aug @384 (ConcatCWT: 5/5 fold win vs BCE@384, fold4 +0.0692) ----
    ("ConcatCWT heavyaug384", "concat_cwt_heavyaug_fold__maxvit_tiny_tf_384_in1k", RES, "concat", "maxvit_tiny_tf_384.in1k", "cwt", 384),
    # ---- NCHC orthogonal paradigms (2026-06-02): build()/POOL_LOOKUP kept for re-test, but
    # EXCLUDED from the active pool — did NOT robustly clear the honest gate. Multi-seed paired
    # Δ vs v5-best straddled 0 (GNN meanΔ +0.0005, Spec3D +0.0042, both sign-flip across seeds;
    # honest-OOF noise floor ~±0.005-0.01 swamps them). GNN underfits (OOF 0.547); Spec3D (0.80)
    # is a superlet-CNN substitute (corr 0.69). Re-enable only if a STRONGER orthogonal variant
    # is trained. See scripts/gate_decisive.py.
    # ("GNN orthopara", "eeg_gnn_fold__gcn", RES, "gnn", "gcn", "eeg", 2000),
    # ("Spec3D superlet", "spec3d_superlet_scratch_fold__r3d_18", RES, "spec3d", "r3d_18", "superlet", 256),
    # ---- NeuroMAE: in-domain SSL-pretrained EEG transformer (orthogonal, raw waveform).
    # feature "filt_eeg" rides the existing infer_arch filt path (batch["eeg"]=(B,26,2000)).
    # COMMENTED until it clears honest_oof + gate_decisive (meanΔ>+0.005, no sign-flip).
    # ("NeuroMAE v1", "neuromae_fold__v1", RES, "neuromae", "vit", "filt_eeg", 2000),
]


def build(kind, bb, tsize, device):
    if kind == "mukuv1":
        m = MukuEEGNet(num_classes=1, backbone=bb, in_chans=26, n_freqs=8, fs=500, pretrained=False)
    elif kind == "speccnn":
        m = SpecCNN(num_classes=1, backbone=bb, in_chans=26, target_size=tsize, pretrained=False)
    elif kind == "concat":
        m = ConcatSpecCNN(num_classes=1, backbone=bb, target_size=tsize, pretrained=False)
    elif kind == "mukuv2":
        m = MukuEEGNetV2(num_classes=1, backbone=bb, in_chans=26, n_freqs=8, fs=500, pretrained=False)
    elif kind == "mukuv3":
        m = MukuEEGNetV3(num_classes=1, backbone=bb, in_chans=26, n_freqs=8, fs=500, tcn_layers=8,
                         target_size=tsize, target_size_spec=tsize, pretrained=False)
    elif kind == "mukumb":
        m = MukuEEGNet(num_classes=1, backbone=bb, in_chans=130, n_freqs=8, fs=500, pretrained=False)
    elif kind == "legacy":
        m = build_legacy_eeg_model(bb)
    else:
        raise ValueError(kind)
    return m.to(device).eval()


def normalize_29_to_26(wave):
    wave = wave.astype(np.float32, copy=True)
    if wave.shape[0] >= 29:
        wave[:23] = wave[:23] / 1e-3
        wave[23:] = wave[23:] * 1e-2
        heart = wave[23] - wave[24]; m1 = wave[25] - wave[26]; m2 = wave[27] - wave[28]
        wave = np.concatenate([wave[:23], np.stack([heart, m1, m2], 0)], 0)
    out = np.zeros((26, 2000), dtype=np.float32)
    c, t = wave.shape
    out[:min(c, 26), :min(t, 2000)] = wave[:min(c, 26), :min(t, 2000)]
    return out


# Filtered raw-EEG pipeline (identical to scripts/precompute_filt_eeg.py): used for
# features "filt_eeg" and "filt_superlet_eeg" (EEG branch only is filtered).
from scipy.signal import butter as _butter, filtfilt as _filtfilt, iirnotch as _iirnotch
_FS = 500
_N_EEG = 23
_FILT_BP_B, _FILT_BP_A = _butter(4, [0.5, 70.0], btype="band", fs=_FS)
_FILT_NOTCH_B, _FILT_NOTCH_A = _iirnotch(50.0, Q=30.0, fs=_FS)

def filt_eeg_signal(wave_26x2000):
    """Bandpass 0.5-70 Hz + notch 50 Hz + CAR(23 EEG) + robust per-channel norm."""
    x = wave_26x2000.astype(np.float64)
    x = _filtfilt(_FILT_BP_B, _FILT_BP_A, x, axis=-1)
    x = _filtfilt(_FILT_NOTCH_B, _FILT_NOTCH_A, x, axis=-1)
    car = x[:_N_EEG].mean(axis=0, keepdims=True)
    x[:_N_EEG] = x[:_N_EEG] - car
    med = np.median(x, axis=-1, keepdims=True)
    q75 = np.percentile(x, 75, axis=-1, keepdims=True)
    q25 = np.percentile(x, 25, axis=-1, keepdims=True)
    iqr = (q75 - q25)
    x = (x - med) / (iqr + 1e-6)
    return x.astype(np.float32)


def have_full_set(prefix, root):
    for f in range(5):
        if not (root / "checkpoints" / f"{prefix}__fold{f}__seed0" / "best.pt").exists():
            return False
        if not (root / "predictions" / f"{prefix}__fold{f}__seed0_oof.npz").exists():
            return False
    return True


def _common_seeds_oof(prefix, root):
    pred = root / "predictions"
    per_fold = []
    for f in range(5):
        seeds = set()
        for fp in pred.glob(f"{prefix}__fold{f}__seed*_oof.npz"):
            try: seeds.add(int(fp.name.rsplit("__seed", 1)[1].split("_oof")[0]))
            except (IndexError, ValueError): pass
        per_fold.append(seeds)
    return sorted(set.intersection(*per_fold)) if all(per_fold) else []


def _common_seeds_ckpt(prefix, root):
    """Seeds whose best.pt exists for ALL 5 folds (intersection). seed0-only -> [0]."""
    ck = root / "checkpoints"
    per_fold = []
    for f in range(5):
        seeds = set()
        for d in ck.glob(f"{prefix}__fold{f}__seed*"):
            if (d / "best.pt").exists():
                try: seeds.add(int(d.name.rsplit("__seed", 1)[1]))
                except (IndexError, ValueError): pass
        per_fold.append(seeds)
    return sorted(set.intersection(*per_fold)) if all(per_fold) else []


def load_oof(prefix, root):
    """5-fold OOF, seed-averaged within each fold over seeds common to all folds (seed0-only=identical)."""
    seeds = _common_seeds_oof(prefix, root) or [0]
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


def derive_weights(avail):
    ref_s, ref_y, X = None, None, []
    for e in avail:
        s, l, y = load_oof(e["prefix"], e["root"])
        if ref_s is None:
            ref_s, ref_y = s, y; X.append(l)
        else:
            idx = {ss: i for i, ss in enumerate(s)}
            perm = np.array([idx[ss] for ss in ref_s])
            X.append(l[perm])
    X = np.stack(X, axis=1); M = X.shape[1]

    def neg(w):
        w = np.maximum(w, 0); sm = w.sum(); w = w / sm if sm > 0 else np.ones_like(w) / len(w)
        return -average_precision_score(ref_y, X @ w)
    rng = np.random.default_rng(42)
    best_w, best_a = None, -1
    for _ in range(6):
        w0 = np.ones(M) / M + 0.05 * rng.standard_normal(M)
        w0 = np.maximum(w0, 1e-3); w0 /= w0.sum()
        r = minimize(neg, w0, method="Nelder-Mead", options={"maxiter": 30000, "xatol": 1e-7, "fatol": 1e-7})
        w = np.maximum(r.x, 0); w /= w.sum()
        a = average_precision_score(ref_y, X @ w)
        if a > best_a:
            best_a, best_w = a, w
    return best_w, best_a


@torch.no_grad()
def infer_arch(e, ids, caches, device, bs=96):
    eeg, cwt, sl, cwtfilt, stft, paul, filteeg, multiband = caches
    feat = e["feature"]
    n = len(ids); sumlog = np.zeros(n, dtype=np.float64); nf = 0
    seeds = _common_seeds_ckpt(e["prefix"], e["root"]) or [0]

    def forward_ckpt(cp):
        model = build(e["kind"], e["backbone"], e["tsize"], device)
        st = torch.load(cp, map_location=device, weights_only=False)
        model.load_state_dict(st["model_state_dict"])
        fl = np.zeros(n, dtype=np.float64)
        for i0 in range(0, n, bs):
            bids = ids[i0:i0 + bs]
            batch = {}
            # EEG branch (raw or filtered or multiband)
            if feat in ("eeg", "superlet_eeg"):
                batch["eeg"] = torch.stack([eeg[s] for s in bids]).to(device)
            elif feat in ("filt_eeg", "filt_superlet_eeg"):
                batch["eeg"] = torch.stack([filteeg[s] for s in bids]).to(device)
            elif feat == "multiband":
                batch["eeg"] = torch.stack([multiband[s] for s in bids]).to(device)
            # Spec branch
            if feat == "cwt":
                batch["spec"] = torch.stack([cwt[s].float() for s in bids]).to(device)
            elif feat in ("superlet", "superlet_eeg", "filt_superlet_eeg"):
                batch["spec"] = torch.stack([sl[s].float() for s in bids]).to(device)
            elif feat == "cwt_filtered":
                batch["spec"] = torch.stack([cwtfilt[s].float() for s in bids]).to(device)
            elif feat == "stft":
                batch["spec"] = torch.stack([stft[s].float() for s in bids]).to(device)
            elif feat == "cwt_paul":
                batch["spec"] = torch.stack([paul[s].float() for s in bids]).to(device)
            with torch.amp.autocast("cuda", enabled=True):
                out = model(batch)
            o = out["main"] if isinstance(out, dict) else out
            fl[i0:i0 + len(bids)] = o.view(-1).float().cpu().numpy()
        del model; torch.cuda.empty_cache()
        return fl

    # seed-average WITHIN each fold (over seeds common to all folds), then average across folds
    for f in range(5):
        fold_acc = np.zeros(n, dtype=np.float64); ns = 0
        for sd in seeds:
            cp = e["root"] / "checkpoints" / f"{e['prefix']}__fold{f}__seed{sd}" / "best.pt"
            if not cp.exists():
                continue
            fold_acc += forward_ckpt(cp); ns += 1
        if ns > 0:
            sumlog += fold_acc / ns; nf += 1
    return sumlog / max(nf, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", default="NeuroMM-2026/candidate/candidate")
    ap.add_argument("--out", default="submission_test1_fullpool.csv")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=96)
    args = ap.parse_args()
    device = torch.device(args.device)
    cand = Path(args.candidate_dir)
    ids = [l.strip() for l in (cand / "candidate_ids.txt").read_text().splitlines() if l.strip()]
    print(f"Candidate ids: {len(ids)}")

    # discover available architectures
    avail = []
    for (name, prefix, root, kind, bb, feat, tsize) in POOL:
        if have_full_set(prefix, root):
            avail.append({"name": name, "prefix": prefix, "root": root, "kind": kind,
                          "backbone": bb, "feature": feat, "tsize": tsize})
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ skip (incomplete): {name}")
    print(f"Available architectures: {len(avail)}")

    print("\n=== Deriving Nelder-Mead weights from OOF ===")
    weights, oof_auprc = derive_weights(avail)
    for e, w in zip(avail, weights):
        print(f"  {e['name']:32s} {w:.4f}")
    print(f"OOF ensemble AUPRC = {oof_auprc:.4f}")

    # caches
    print("\n=== Loading candidate features ===")
    need_eeg = any(e["feature"] in ("eeg", "superlet_eeg") for e in avail)
    need_cwt = any(e["feature"] == "cwt" for e in avail)
    need_sl = any(e["feature"] in ("superlet", "superlet_eeg", "filt_superlet_eeg") for e in avail)
    need_cwtfilt = any(e["feature"] == "cwt_filtered" for e in avail)
    need_stft = any(e["feature"] == "stft" for e in avail)
    need_paul = any(e["feature"] == "cwt_paul" for e in avail)
    need_filteeg = any(e["feature"] in ("filt_eeg", "filt_superlet_eeg") for e in avail)
    need_multiband = any(e["feature"] == "multiband" for e in avail)
    eeg, cwt, sl, cwtfilt, stft, paul, filteeg, multiband = {}, {}, {}, {}, {}, {}, {}, {}
    if need_cwtfilt:
        for sid in tqdm(ids, desc="cwt_filtered"):
            cwtfilt[sid] = torch.from_numpy(np.load(cand / "cwt_filtered" / f"{sid}.npy"))
    if need_stft:
        for sid in tqdm(ids, desc="stft"):
            stft[sid] = torch.from_numpy(np.load(cand / "stft" / f"{sid}.npy"))
    if need_paul:
        for sid in tqdm(ids, desc="cwt_paul"):
            paul[sid] = torch.from_numpy(np.load(cand / "cwt_paul" / f"{sid}.npy"))
    if need_eeg:
        for sid in tqdm(ids, desc="eeg"):
            eeg[sid] = torch.from_numpy(normalize_29_to_26(np.load(cand / "eeg" / f"{sid}.npy")))
    if need_cwt:
        for sid in tqdm(ids, desc="cwt"):
            cwt[sid] = torch.from_numpy(np.load(cand / "cwt" / f"{sid}.npy"))
    if need_sl:
        for sid in tqdm(ids, desc="superlet"):
            sl[sid] = torch.from_numpy(np.load(cand / "superlet" / f"{sid}.npy"))
    if need_filteeg:
        for sid in tqdm(ids, desc="filt_eeg"):
            wave = normalize_29_to_26(np.load(cand / "eeg" / f"{sid}.npy"))
            filteeg[sid] = torch.from_numpy(filt_eeg_signal(wave))
    if need_multiband:
        for sid in tqdm(ids, desc="multiband"):
            arr = np.load(cand / "multiband" / f"{sid}.npy").astype(np.float32)
            multiband[sid] = torch.from_numpy(arr)
    caches = (eeg, cwt, sl, cwtfilt, stft, paul, filteeg, multiband)

    print("\n=== Per-arch inference ===")
    X = []
    for e in avail:
        print(f"[{e['name']}] feature={e['feature']}")
        X.append(infer_arch(e, ids, caches, device, args.batch_size))
    X = np.stack(X, axis=1)
    final_logit = X @ weights
    prob = 1.0 / (1.0 + np.exp(-final_logit))

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "prediction"])
        for sid, p in zip(ids, prob):
            w.writerow([sid, f"{p:.6f}"])
    print(f"\nWrote {out} ({len(ids)} rows)  range min={prob.min():.4f} mean={prob.mean():.4f} max={prob.max():.4f}")
    print(f"Pool size: {len(avail)} architectures, OOF AUPRC={oof_auprc:.4f}")

    # ---- Save per-arch candidate logits for offline re-combination ----
    names = [e["name"] for e in avail]
    np.savez(REPO / "neuromm26_results/candidate_arch_logits.npz",
             names=np.array(names), logits=X, ids=np.array(ids))
    print(f"Saved per-arch logits: {X.shape}")

    # ---- Also produce a robust SIMPLE-MEAN of individually-strong archs (OOF>0.76) ----
    import zipfile
    strong_idx, strong_names = [], []
    for i, e in enumerate(avail):
        _, l, y = load_oof(e["prefix"], e["root"])
        a = average_precision_score(y, 1 / (1 + np.exp(-l)))
        if a > 0.76:
            strong_idx.append(i); strong_names.append(f"{e['name']}({a:.3f})")
    if strong_idx:
        smean_logit = X[:, strong_idx].mean(axis=1)
        smean_prob = 1 / (1 + np.exp(-smean_logit))
        sm_csv = Path(str(out).replace(".csv", "_smean.csv"))
        with sm_csv.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["sample_id", "prediction"])
            for sid, p in zip(ids, smean_prob):
                w.writerow([sid, f"{p:.6f}"])
        subdir = REPO / "submissions"; subdir.mkdir(exist_ok=True)
        tmp = subdir / "submission.csv"
        import shutil
        shutil.copy(sm_csv, tmp)
        zname = subdir / (sm_csv.stem + ".zip")
        with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(tmp, arcname="submission.csv")
        tmp.unlink()
        print(f"SIMPLE-MEAN strong ({len(strong_idx)} archs) -> {zname}")
        print(f"  strong archs: {strong_names}")


if __name__ == "__main__":
    raise SystemExit(main())
