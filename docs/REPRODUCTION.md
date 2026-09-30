# Reproduction notes — NeuroMM-2026 Track-1 (NMM-Basic-IED), 0.9846

Binary spike/IED detection, EEG-only, metric AUPRC, 5-fold patient-disjoint CV, candidate/test = a fixed 20000 unlabeled samples.
Final submission = `submissions/submission_test1_specnchc.zip` (leaderboard **0.9846**). This file covers: folder structure → full pipeline
(Stage 0→5, how 0.9846 is produced step by step) → per-file description → reproduction steps.

---

## 0. Folder structure

```
<repo root>/
├── README.md                       project overview
├── docs/REPRODUCTION.md            this file (structure + pipeline + per-file + reproduction)
├── docs/checkpoints_manifest.tsv   weight prefix → root → stage mapping
├── requirements.txt  setup.py      environment (Python 3.14, PyTorch cu130)
├── fold_df_fixed.csv               (from HF) 5-fold patient-disjoint CV split (25426 rows)
├── fold_df_pseudo.csv              (from HF) Stage 3 round-1 pseudo-label training table  (+ pseudo_sids.txt)
├── fold_df_pseudo_spec.csv         (from HF) Stage 4 candidate-specialist training table, 11336 candidates (+ pseudo_sids_spec.txt)
├── neuromm26_baseline/             Python package: models / datasets / tools / utils
├── scripts/                        pipeline scripts: precompute / dispatcher / build / gate / analysis
├── submissions/                    submission_test1_specnchc.zip — final scored submission (leaderboard 0.9846)
│
├── neuromm26_results/              ◀── weights (RES root) ★large, uploaded separately to HuggingFace
│   ├── candidate_arch_logits.npz       each arch's logits on the 20000 candidates (ensemble input)
│   ├── checkpoints/<prefix>__fold<f>__seed0/best.pt
│   └── predictions/<prefix>__fold<f>__seed0_oof.npz   (5-fold OOF = ensemble-weight input)
└── neuromm26_real_5fold_result/    ◀── weights (REAL root) ★large, uploaded separately to HuggingFace
    ├── checkpoints/...  └── predictions/...
```
> **The weight directories (`neuromm26_results/` + `neuromm26_real_5fold_result/`, ~24GB total) are hosted separately on HuggingFace: [`GG3BBE0/NeuroMM-T1-weights`](https://huggingface.co/datasets/GG3BBE0/NeuroMM-T1-weights) (dataset repo).**
> Code + docs + data (the rest, ~tens of MB) is the main package. To restore the two weight directories into the repo root:
> ```bash
> hf download GG3BBE0/NeuroMM-T1-weights --repo-type dataset --local-dir .
> ```

---

## 1. Score ladder (method correspondence)
```
0.9247 15-arch → 0.9446 fullpool(23) → 0.9605 +STFT/Paul(29) → 0.9663 +filt_eeg(33)
0.9711 regnm_v3(48-arch NM ensemble)                     ← best before pseudo-labeling / round-1 teacher
── below: transductive self-training on the released candidates ──
0.9778 nchc_n2filt(NCHC pseudo, 3-arch swap)             ← zero-extra-risk floor final / Stage4 teacher
0.9818 spec(candidate-specialist local, 11336 candidates @ weight 1.0)
0.9846 specnchc(NCHC 5-arch specialist)                  ← BEST final
```
Total +0.0135 over the "saturated" 0.9711. Two finals were selected: 0.9846 + 0.9778.

---

## 2. Full pipeline (how 0.9846 is produced end to end)

### Stage 0 — EEG preprocessing & features
- Window 4s@500Hz=2000 samples; **29→26 channels** (`normalize_29_to_26`: EEG ch0-22 ×1e3; physiological ch23-28 ×1e-2 then differenced → ECG/EMG, 2 channels).
- **7 cached representations** (`scripts/precompute_*.py` → `neuromm26_datasets/processed/features/`):
  `eeg` (raw), `filt_eeg` (butter(4,[0.5,70]) + iirnotch(50,Q30), both filtfilt + CAR (23 EEG) + robust norm),
  `cwt` (Morlet, 64 freqs 0.5–60, time 2000→256 cv2.resize, log1p), `cwt_paul` (Paul m=4), `cwt_filtered` (CWT after filtering),
  `superlet` (Morlet multi-cycle 3–7, geometric mean over 5 orders), `stft` (n_fft128/hop32).
- **ConcatSpec stacking** (`spec_cnn_concat.py`): `(B,26,F,T)`→`(B,1,26·F,T)`→ resize 256/384 → ImageNet 2-D backbone.

### Stage 1 — 48-arch model zoo (each 5-fold)
- **ConcatSpecCNN (main force)**: MaxViT `maxvit_rmlp_tiny_rw_256`@256 / `maxvit_tiny_tf_384`@384 are the aces (+convnext/coatnet/caformer), features covering cwt/filtered/superlet/stft, including heavyaug, focalheavy variants.
- **muku type (from HMS) V1/V2/V3** (learned-bandpass time-domain; V2 +spectrogram branch; V3 +TCN), **SpecCNN** (single spectrogram), **legacy 1-D** (tcnet/mobilenet/eegnet/actnet/lmda), **muku_mb** (multiband).
- **Recipe**: two-stage (freeze backbone 10ep → unfreeze 30ep); LR ConcatSpec 7e-4/7e-5, muku 1e-3/1e-4, others 8e-4/8e-5; optimizer Adan + OneCycleLR + AMP + grad clip 5; loss defaults BCE+ls0.05+pos_weight, **key lever = Focal(γ=2)+pos_weight** (focalheavy series); augmentation SpecAug/mixup, **HEAVY version** (time×2/freq×2/channel-drop×2 prob0.7 + mixup0.7) targets the hard fold4. seed 0.
- Output: OOF `{prefix}__fold{f}__seed0_oof.npz` (sample_ids/logits/labels) + ckpt `best.pt`, placed in two roots (REAL=muku-raw+some CWT, 6 archs; RES=the other 42).

### Stage 2 — Nelder-Mead ensemble → 0.9711
`regnm_real.py` + `regularized_nm_submission.py`, **weights derived on OOF AUPRC only**:
`derive_nm` baseline → **trim** (single-model OOF<0.68 **and** weight<0.005) → **bagged NM** (bootstrap n_iter=40) → `cap 0.20`.
Running NM requires `OMP_NUM_THREADS=1`. = regnm_v3 → OOF 0.8974 → **leaderboard 0.9711** (round-1 teacher).

### Stage 3 — Pseudo-label self-training → 0.9778 (real-label honest OOF as the brake)
1. `build_pseudo_dataset.py`: teacher=v3; probabilities uncalibrated → relative thresholds **HI=0.90 positive / LO=0.330 negative**, middle discarded; selected candidates → hard labels, symlinked into train, `fold=-1` (always train, never val → OOF still evaluated on real labels only); `--noise-weight 0.4` (down-weighted).
2. Local retrain → 0.9769; **NCHC large-batch retrain** (`dispatcher_nchc_pseudo.py`+`B_nchc_pseudo.sb`).
3. `build_nchc_submission.py`: cwt/paul/filt maxvit256 swapped to pseudo versions → trim+bagged+cap → **n2filt = leaderboard 0.9778** (floor final + Stage4 teacher). Pseudo-labeling then hits a wall at ~0.978.

### Stage 4 — Candidate-specialist (breakthrough) → 0.9818 → 0.9846
The wall is broken by "**quantity + full weight**" (more data wins), not a more aggressive threshold.
1. `build_pseudo_dataset_iter.py nchc_n2filt.zip spec 0.60 35`: teacher=n2filt; **HI=0.60 / LO=bottom 35%** → **11336 candidates**; `fold=-1` → `fold_df_pseudo_spec.csv`.
2. **Key counterintuitive point: `--noise-weight 1.0` (full trust)**, not 0.4.
3. Local retrain of 3 concats@256 → **0.9818**; **NCHC retrain of 5 specialists** (`dispatcher_nchc_specialist.py`+`B_nchc_specialist.sb`, prefix `*_specnchc`):

   | specialist | feature | backbone | tsize | bs | LR1/2 |
   |---|---|---|---|---|---|
   | ConcatCWT/Paul/Filt specnchc | cwt/cwt_paul/cwt_filtered | maxvit_rmlp_tiny_rw_256 | 256 | 128 | 1e-3/1e-4 |
   | ConcatSuperlet specnchc | superlet | maxvit_tiny_tf_384 | 384 | 96 | 9e-4/9e-5 |
   | muku specnchc | eeg | convnext_pico | 224 | 256 | 2e-3/2e-4 |

   all `--loss focal --focal-gamma 2.0`, 10+30 ep, `--noise-sids pseudo_sids_spec.txt --noise-weight 1.0`. Single-model real-label OOF all rise (+0.024~0.044) = genuine generalization.

### Stage 5 — Final build → 0.9846
`build_specnchc_submission.py`: 48 canon → swap the 5 to `*_specnchc` → `derive_nm` → trim (drop 5) → keep 43 → `derive_bagged_nm(40)` → `cap0.20` → apply weights to candidate logits → sigmoid → zip.
- 43 members, weights Σ=1.0, top 5 (specnchc) ≈ 64%; `ConcatSuperlet specnchc 384` hits cap 0.20.
- Ensemble in-sample OOF ≈ **0.9268~0.9275** → **leaderboard 0.9846**.
- Validation: `honest_oof.py` (honest nested-OOF) + `build_private_sim.py` (private-safety simulation: evaluate on a held-out fold's real labels, confirming self-training is genuine generalization, not gaming).

---

## 3. Per-file description (★=0.9846 core, ◦=training/analysis supporting evidence)

**`neuromm26_baseline/`**
- models: ★`spec_cnn_concat.py` (ConcatSpec), ★`muku_eegnet{,_v2,_v3}.py` (muku type V1/V2/V3), ◦`spec_cnn.py` (single spectrogram); legacy: `registry.py`+`tcnet/cnn_eeg/eegnet/actnet/lmda.py` (1-D networks).
- datasets: ★`fold_datasets.py` (5-fold, fold=-1=always train), ★`muku_dataset.py` (augmentation), `multiband_dataset.py`, `spec_dataset.py`, ★`collate_fn.py`.
- tools: ★ 6 `train_*_fold.py` (+`train_muku_multiband_fold.py`) + ★`predict_candidate_full_pool.py` (candidate inference + NM).
- utils: ★`adan.py`/`metrics.py`/`seed.py` + `io/logger/config/visualization`.

**`scripts/` by stage**
- **Stage 0**: ★`precompute_{filt_eeg,cwt,cwt_paul,cwt_filtered,superlet_gpu,stft}.py` (+`cwtphase/multiband`).
- **Stage 1 training dispatchers**: ◦`dispatcher_{5fold,diverse*,filtered,filt_eeg,heavyaug*,focalheavy_*,clean_cwt,muku_v2,muku_v3,tcnet,specviews}.py`; NCHC ◦`B_{maxvit384,heavyaug*,nchc_focal*,nchc_heavyaug384}.sb`+corresponding dispatchers.
- **Stage 2**: ★`regularized_nm_submission.py` (core NM), ★`regnm_real.py`.
- **Stage 3**: ★`build_pseudo_dataset{,_iter,_r2}.py`, ◦`dispatcher_pseudo_*.py`, ★`dispatcher_nchc_pseudo.py`+`B_nchc_pseudo.sb`, ★`build_nchc_submission.py`, ★`dump_ensemble_teacher.py`, `build_pseudo_submission.py`/`build_upweight.py`.
- **Stage 4**: ★`dispatcher_specialist.py`, ★`dispatcher_nchc_specialist.py`+`B_nchc_specialist.sb`, ★`build_spec_submission.py` (0.9818), ★`build_specnchc_submission.py` (**0.9846**), ★`{nchc_one_logit,muku_candidate_logit,append_arch_to_npz,nchc_merge_logits}.py`; `dispatcher_specialist2/3*`+`_nchc_specialist2/3` (later iterations).
- **Stage 5 validation**: ★`honest_oof.py`, ★`gate_spec.py`/`gate_specnchc.py`, ★`build_private_sim.py`+`analyze_private_sim*.py`+`dispatcher_private_sim*.py`+`symlink_sim_features.py`, ★`print_specnchc_weights.py`; `gate_*`/`honest_add_check`/`nm_noise_robust`/`cos_stability_check`.
- **Extraction/regeneration**: ◦`extract_nchc_*.py`, `regen_oof_*.py`.
- **Analysis/evaluation (supporting)**: ◦`*_ensemble_eval.py`, `stacking_and_weighted/nnet_stacker/cross_model_ensemble.py`, `tta_*/threshold_*/final_tuning.py` (ultimately unused), `fold4_{analysis,label_audit}.py`/`missed_spike_consensus.py`, `make_fold_df/export_real_5fold_csv/analyze_channel_order.py`, `defensive_checks/validate_filteeg_zip/zip_filteeg.py`.

**Weights / submissions**
- ★`neuromm26_results/` (RES) + ★`neuromm26_real_5fold_result/` (REAL): checkpoints + predictions (OOF) + `candidate_arch_logits.npz`. **★Hosted separately on HuggingFace: [`GG3BBE0/NeuroMM-T1-weights`](https://huggingface.co/datasets/GG3BBE0/NeuroMM-T1-weights).**
- ★`submissions/`: `submission_test1_specnchc.zip` — the final scored submission (leaderboard **0.9846**).

---

## 4. Reproduction

### Environment
```
conda create -n neuromm26-baseline python=3.14 && conda activate neuromm26-baseline
cd PushingPastSaturation-NeuroMM-2026-Track1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt && pip install -e .
```
The scripts' `REPO` auto-resolves to the repo root (or `export NEUROMM_REPO=<this folder>`).
**First download the two weight directories from HuggingFace into the repo root:**
```bash
hf download GG3BBE0/NeuroMM-T1-weights --repo-type dataset --local-dir .
```

### Path A — rebuild 0.9846 directly from the cache
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/build_specnchc_submission.py
python scripts/print_specnchc_weights.py
```
> The shipped `submissions/submission_test1_specnchc.zip` **is the exact scored file**; a rerun, because NM is non-convex, matches it at Spearman 0.9998 / same score (not bit-identical). This package's ckpt 5-fold ensemble OOF = 0.9275 (= build OOF, verified).

### Path B — full retrain (Stage 0→5)
`precompute_*.py` (features, requires regenerating the >100GB cache from the HuggingFace dataset) → `dispatcher_*.py` (train the 48 zoo) → `regnm_real.py` (0.9711) → `build_pseudo_dataset.py`+`build_nchc_submission.py` (regenerates `submissions/submission_test1_nchc_n2filt.zip`, 0.9778) → `build_pseudo_dataset_iter.py submissions/submission_test1_nchc_n2filt.zip spec 0.60 35`+`dispatcher_nchc_specialist.py` (specialist) → `build_specnchc_submission.py` (0.9846). (Only the final `submission_test1_specnchc.zip` is shipped; the 0.9711/0.9778 milestones above are regenerated en route.)

### Notes
- **Not shipped with the package**: raw EEG + the 6 spectrogram caches + candidate features (>100GB) → regenerate from the HuggingFace dataset via `precompute_*.py` (Path A doesn't need them; Path B step 0 does).
- **NCHC scripts** (`dispatcher_nchc_*.py` / `B_nchc_*.sb`) carry the original cluster paths (`WORK_ROOT`/`REPO_DIR`, overridable via environment variable)
