<div align="center">

# 🥇 Pushing Past Saturation
### An EEG Spike-Detection System for NeuroMM-2026 Track-1

**Rank-1 solution** · NeuroMM-2026 Grand Challenge, Track-1 (NMM-Basic-IED) · **ACM Multimedia 2026**

[![Paper](https://img.shields.io/badge/Paper-ACM_MM_2026-b31b1b?style=flat-square)](paper/PushingPastSaturation_NeuroMM2026_Track1.pdf)
[![AUPRC](https://img.shields.io/badge/Test_AUPRC-0.9846-gold?style=flat-square)](#-results)
[![Rank](https://img.shields.io/badge/Leaderboard-🥇_1st-gold?style=flat-square)](#-results)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)](#-quick-start)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11_cu130-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#-quick-start)
[![Weights](https://img.shields.io/badge/🤗_Weights-NeuroMM--T1--weights-blue?style=flat-square)](https://huggingface.co/datasets/GG3BBE0/NeuroMM-T1-weights)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*Ming-Chun Chiang · Kuan-Chuan Peng · Bo-Yun Yu · Jun-Wei Hsieh*

National Yang Ming Chiao Tung University · Mitsubishi Electric Research Laboratories

</div>

---

## 📌 TL;DR

Binary detection of **interictal epileptiform discharges (IEDs, "spikes")** in 4-second scalp-EEG
windows, scored by **AUPRC**. Our system combines domain-aware preprocessing, **five
time–frequency representations**, ImageNet-pretrained vision backbones, and a **regularized
Nelder–Mead ensemble**, reaching an inductive **0.9711** — where adding more architectures stops
helping. A pseudo-labelling warm-up lifts this to **0.9778**, and our **candidate-specialist
self-training** — many confident candidates, committed at **full weight**, audited by a
**candidate-excluded real-label CV gate** — reaches **0.9846, rank 1**.

> **The lesson:** on a saturated EEG benchmark, broad model diversity is not enough. The decisive
> lever is *quantity × commitment* of pseudo-labelled candidates, gated by validation that never
> sees a pseudo-label.

## 🏆 Results

| Stage | Leaderboard AUPRC | Regime |
|:---|:---:|:---:|
| Regularized Nelder–Mead ensemble (43 members) | 0.9711 | inductive (saturated) |
| + pseudo-labelling warm-up | 0.9778 | transductive |
| + candidate specialist (full weight) | 0.9818 | transductive |
| **+ teacher iteration (final)** | **0.9846** | **transductive · 🥇 rank 1** |

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/score_ladder_dark.png">
  <img src="assets/score_ladder_light.png" width="100%" alt="Score ladder: 0.9247 → 0.9711 inductive, then 0.9778 → 0.9818 → 0.9846 transductive">
</picture>
</div>

Each inductive step adds less than the previous one (+0.0199 → +0.0159 → +0.0058 → +0.0048);
further same-family architectures inflated in-sample CV without moving the leaderboard. The
transductive stage adds **+0.0135** without any new architecture. Re-deriving the ensemble weights
under eight bagging seeds keeps the leaderboard within 0.9842–0.9852, so the late steps are not
optimizer noise.

## 🧩 The task

| | |
|:---|:---|
| **Input** | 4-s windows, 2000 samples @ 500 Hz, 29 channels (EEG + ECG/EMG) |
| **Label** | spike (IED) vs. non-spike |
| **Training data** | 25,426 labelled windows, **patient-disjoint 5-fold CV**, 9.9 % positive (5.8 %–13.0 % per fold) |
| **Scored set** | 20,000 released unlabelled candidates |
| **Metric** | AUPRC |

The same spike looks very different depending on the time–frequency basis. Rather than commit to
one, we feed all of them to the model pool:

<div align="center">
<img src="assets/spike_representations.png" width="92%" alt="One IED across CWT, Paul wavelet, superlet, band-passed CWT, STFT and the stacked model input">
</div>

## 🧠 Architecture

<div align="center">
<img src="assets/architecture.png" width="100%" alt="System data-flow">
</div>

- **Preprocessing** — 29 → 26 channels (scaled EEG + three bipolar ECG/EMG derivations). A
  band-passed branch adds a 0.5–70 Hz zero-phase Butterworth, 50 Hz notch, EEG-only common-average
  reference and robust `(x − median)/IQR` normalization.
- **Five time–frequency representations** — Morlet CWT, Paul-wavelet CWT, superlet (geometric mean
  of |CWT| over 3–7 cycles, batched-FFT GPU kernel), STFT, and CWT of the band-passed signal
  (BP-CWT).
- **Stacked-spectrogram CNN** — the `26 × F × T` map is stacked along frequency into one image,
  resized to 256² or 384², and fed to an ImageNet-pretrained `timm` backbone (MaxViT, ConvNeXt, …),
  following the channel-stacking scheme of a public HMS solution.
- **Multi-view time-domain model** — a learnable temporal-convolution filter bank (DW block)
  whose output is reordered into channel-major and frequency-major views, each with its own
  backbone; a simplified re-implementation of a public HMS design. Focal loss on the main head,
  BCE on auxiliary heads.
- **Regularized Nelder–Mead ensemble** — derivative-free AUPRC maximization over simplex weights
  (gradient/SLSQP solvers collapse to near-uniform), stabilized by *two-dimensional trimming*
  (drop only members that are both weak **and** near-zero weight), *bagging* over bootstrap
  resamples, and a *0.20 per-member cap*.
- **Candidate-specialist self-training** — the ensemble teacher pseudo-labels confident candidates
  (high tail → positive, low tail → negative, uncertain middle discarded); specialists are retrained
  and re-ensembled, and each round is accepted only if real-label CV improves.

### The `fold = −1` real-label gate

Every pseudo-labelled candidate is tagged `fold = −1`: always in training, **never in any
validation fold**. The CV that accepts or rejects a self-training round is therefore computed
entirely on real labels — if pseudo-label errors damage transferable decision structure, they show
up as a CV *drop*, not a spurious gain. This is what makes full-weight training affordable.

## ⚙️ How it works

| # | Stage | What happens | Code |
|:-:|:---|:---|:---|
| 0 | **Features** | Montage + filtering; cache the 7 inputs (raw / band-passed EEG, CWT, Paul, BP-CWT, superlet, STFT) | `scripts/precompute_*.py` |
| 1 | **Model pool** | Stacked-spectrogram CNNs × representations × {256, 384} px, multi-view models, legacy 1-D CNNs; two-stage fine-tuning (frozen 10 ep → unfrozen 30 ep), Adan + OneCycle + AMP | `neuromm26_baseline/tools/train_*_fold.py`, `scripts/dispatcher_*.py` |
| 2 | **NM ensemble → 0.9711** | Trim → bagged Nelder–Mead (40 bootstraps) → cap 0.20, weights fit on out-of-fold AUPRC | `scripts/regularized_nm_submission.py`, `scripts/regnm_real.py` |
| 3 | **Pseudo-label warm-up → 0.9778** | Few high-threshold candidates, down-weighted 0.4–0.5, `fold = −1` | `scripts/build_pseudo_dataset.py`, `scripts/build_nchc_submission.py` |
| 4 | **Candidate specialists → 0.9846** | 11,336 candidates (4,336 pos / 7,000 neg) at weight **1.0**; retrain 5 specialists (CWT / Paul / BP-CWT @256, superlet @384, multi-view) with focal loss | `scripts/build_pseudo_dataset_iter.py`, `scripts/dispatcher_nchc_specialist.py` |
| 5 | **Final build + checks** | Swap specialists in, re-derive NM weights (43 members), write the submission; nested-CV and held-out-fold replay | `scripts/build_specnchc_submission.py`, `scripts/honest_oof.py`, `scripts/build_private_sim.py` |

## 🔬 What made the difference

### Quantity × commitment beats conservative pseudo-labelling

The pseudo-label warm-up and the candidate specialist are the *same* procedure with opposite
hyper-parameters. Few, down-weighted pseudo-labels (2,695–3,438 candidates, weight 0.4–0.5) stall at
0.978 at every threshold we tried and give inconsistent per-model gains; many candidates at full
weight (11,336 at threshold 0.60, weight 1.0) lift every specialist.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/ablation_commitment_dark.png">
  <img src="assets/ablation_commitment_light.png" width="100%" alt="Down-weighted pseudo-label rounds give mixed gains; full-weight candidate specialists improve every representation">
</picture>
</div>

At the ensemble level, swapping in the full-weight specialists raises **nested** out-of-fold AUPRC
by **+0.0386** over the inductive baseline — positive on all five seeds (+0.032 to +0.043) and
computed on real folds that never contain candidates. We read this for *direction*, not magnitude.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/specialist_gain_dark.png">
  <img src="assets/specialist_gain_light.png" width="100%" alt="Per-specialist single-model gain from +0.036 to +0.080">
</picture>
</div>

### The specialists carry the final ensemble

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/ensemble_weights_dark.png">
  <img src="assets/ensemble_weights_light.png" width="100%" alt="The five specialists carry 0.649 of the ensemble weight">
</picture>
</div>

### More findings

- **Augmentation is representation-dependent.** Heavy SpecAugment helps superlet@384 (+0.0167),
  Paul@256 (+0.008) and CWT@256 (+0.0036) but *hurts* BP-CWT@256 (−0.003; its input is already
  denoised). This — not representation superiority — explains the superlet member's large weight.
- **No single representation dominates.** At 384 px: Paul 0.858, superlet 0.849, CWT 0.841, with
  heavily overlapping bootstrap intervals.
- **Hard targets beat soft targets at saturation.** A Noisy-Student-style soft-target variant gains
  only +0.0020 nested CV, inside the ±0.005–0.01 noise floor.

### Held-out-fold generalization check

Because the leaderboard gain is in-sample by design, we replay the whole procedure on labelled data:

```mermaid
flowchart LR
    A["Teacher trained on<br/>folds ≠ k"] --> B["Pseudo-label fold-k windows<br/>(≥ HI / ≤ LO)"]
    B --> C["Inject at fold = −1,<br/>retrain specialist"]
    C --> D["Evaluate on fold-k<br/><b>true</b> labels"]
```

Δ (specialist − canonical) is positive on **4 / 5 folds** for both CWT (mean +0.0318) and the
multi-view model (mean +0.0547); fold 4 is the only non-positive fold in both. A stronger teacher
yields a larger gain (+0.0318 vs +0.0022 for a weak single-model teacher). We read this as
**suggestive of partial transfer, not proof** that the gain is free of candidate-specific adaptation.

## 🚀 Quick start

```bash
# 1. Environment (exact versions pinned in requirements.txt / environment.yml)
conda create -n neuromm26-baseline python=3.14 && conda activate neuromm26-baseline
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt && pip install -e .

# 2. Download checkpoints, out-of-fold predictions, candidate logits and the CV/pseudo-label
#    tables (~30 GB on disk) into the repo root
hf download GG3BBE0/NeuroMM-T1-weights --repo-type dataset --local-dir .

# 3. Rebuild the 0.9846 submission from the cached logits (CPU only)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/build_specnchc_submission.py
python scripts/print_specnchc_weights.py      # 43 members, specialists ≈ 0.65 of the weight
```

`submissions/submission_test1_specnchc.zip` is the **exact file scored at 0.9846**. Because
Nelder–Mead is non-convex, a rebuild matches it at Spearman 0.9998 with the same score, not
bit-for-bit. `OMP_NUM_THREADS=1` is required for the NM search.

## 📦 Model checkpoints

Checkpoints are too large for git and live on 🤗 Hugging Face:
**[`GG3BBE0/NeuroMM-T1-weights`](https://huggingface.co/datasets/GG3BBE0/NeuroMM-T1-weights)**
(dataset repo). The download restores two directories next to the code:

| Directory | Contents |
|:---|:---|
| `neuromm26_results/` | `checkpoints/<prefix>__fold<f>__seed0/best.pt`, `predictions/<prefix>__fold<f>__seed0_oof.npz`, `candidate_arch_logits.npz` (per-arch logits on the 20,000 candidates — the ensemble input) |
| `neuromm26_real_5fold_result/` | same layout for the remaining multi-view / CWT members |

The same download also places the labelled data tables in the repo root, where the scripts expect
them: `fold_df_fixed.csv` (5-fold patient-disjoint split, 25,426 windows), `fold_df_pseudo.csv` +
`pseudo_sids.txt` (Stage-3 table) and `fold_df_pseudo_spec.csv` + `pseudo_sids_spec.txt` (Stage-4
table, 11,336 candidates). They carry training labels, so they are kept out of this git repository.

Every weight prefix is mapped to its root directory and pipeline stage (`S1_basezoo`, `S3_n2filt`,
`S4_specnchc`, …) in [`docs/checkpoints_manifest.tsv`](docs/checkpoints_manifest.tsv).

## 📁 Repository layout

<details>
<summary><b>Click to expand</b></summary>

```
neuromm26_baseline/            Python package (pip install -e .)
  models/
    spec_cnn_concat.py         ★ stacked-spectrogram CNN (ConcatSpec)
    muku_eegnet{,_v2,_v3}.py   ★ multi-view time-domain models
    spec_cnn.py                single-spectrogram CNN
    legacy/                    1-D EEG CNNs (EEGNet-style, TCNet, MobileNet, ...)
  datasets/                    fold datasets (fold = −1 ⇒ always train), augmentation, collate
  losses/                      focal / BCE losses
  tools/
    train_*_fold.py            ★ per-fold trainers for every model family
    predict_candidate_full_pool.py   ★ candidate inference for the whole pool
  utils/                       Adan optimizer, AUPRC metrics, seeding, logging
scripts/                       pipeline scripts, by stage (see docs/REPRODUCTION.md §3)
  precompute_*.py              Stage 0  feature caches
  dispatcher_*.py, B_*.sb      Stage 1/3/4  training launchers (local GPU / SLURM)
  regularized_nm_submission.py Stage 2  regularized Nelder–Mead ensemble
  build_pseudo_dataset*.py     Stage 3/4  pseudo-label & candidate tables
  build_specnchc_submission.py Stage 5  ★ builds the 0.9846 submission
  honest_oof.py, gate_*.py,    validation: nested CV, admission gates,
  build_private_sim.py         held-out-fold replay
submissions/                   submission_test1_specnchc.zip — the scored 0.9846 file
docs/
  REPRODUCTION.md              stage-by-stage technical notes and per-file map
  checkpoints_manifest.tsv     weight prefix → root → stage
assets/                        README figures
paper/                         camera-ready paper (PDF)

# Restored by `hf download` (not tracked in git):
neuromm26_results/             checkpoints, OOF predictions, candidate logits
neuromm26_real_5fold_result/   same, for the remaining members
fold_df_*.csv, pseudo_sids*.txt  CV split and pseudo-label training tables
```

</details>

Scripts resolve the repository root from their own location (or `$NEUROMM_REPO`), and the data
tables must sit at the root because the scripts reference them there.

## 🔁 Full reproduction (retrain from scratch)

<details>
<summary><b>Click to expand</b></summary>

1. **Features** — obtain the NeuroMM-2026 Track-1 data from the organizers and cache all
   representations (> 100 GB): `python scripts/precompute_{filt_eeg,cwt,cwt_paul,cwt_filtered,superlet_gpu,stft}.py`
2. **Model pool** — train every family on the 5 folds via the `scripts/dispatcher_*.py` launchers
   (each wraps `neuromm26_baseline/tools/train_*_fold.py`).
3. **Inductive ensemble (0.9711)** — `python scripts/regnm_real.py`
4. **Pseudo-label warm-up (0.9778)** — `python scripts/build_pseudo_dataset.py`, retrain the three
   @256 carriers (`scripts/dispatcher_nchc_pseudo.py`), then `python scripts/build_nchc_submission.py`
   (writes `submissions/submission_test1_nchc_n2filt.zip`).
5. **Candidate specialists** —
   `python scripts/build_pseudo_dataset_iter.py submissions/submission_test1_nchc_n2filt.zip spec 0.60 35`,
   then train the five specialists with `scripts/dispatcher_nchc_specialist.py`
   (`--loss focal --focal-gamma 2.0 --noise-weight 1.0`).
6. **Final ensemble (0.9846)** — `python scripts/build_specnchc_submission.py`

Specialist recipes (all 10 + 30 epochs, focal γ = 2):

| Specialist | Input | Backbone | Size | Batch | LR (frozen / unfrozen) |
|:---|:---|:---|:-:|:-:|:-:|
| CWT / Paul / BP-CWT | `cwt`, `cwt_paul`, `cwt_filtered` | `maxvit_rmlp_tiny_rw_256` | 256 | 128 | 1e-3 / 1e-4 |
| Superlet | `superlet` | `maxvit_tiny_tf_384` | 384 | 96 | 9e-4 / 9e-5 |
| Multi-view | raw EEG | `convnext_pico` | 224 | 256 | 2e-3 / 2e-4 |

Inductive models were trained on RTX 3090s and the final specialists on H100s; the ensemble rebuild
runs on CPU. The `B_*.sb` SLURM scripts contain `<your-account>` / `<your-project-id>` /
`<your-email>` placeholders to fill in for your cluster. Stage-by-stage details are in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

</details>

## ⚖️ Scope and rules

- The released candidates were used as **unlabelled** training data, which the challenge permits;
  **no hidden labels were used anywhere**.
- The final regime is **transductive by design**: specialists train on pseudo-labelled members of
  the scored pool (~57 % of it), so 0.9846 carries an in-sample component. Out-of-sample transfer is
  assessed separately by the held-out-fold replay above.
- The recipe needs the scored set to exist before training — true for challenges and retrospective
  review of an archived corpus, not for prospective monitoring, where test-time adaptation is the
  analogue.
- The raw EEG is **not redistributed** here; obtain it from the NeuroMM-2026 organizers.
- Code is released under the [MIT License](LICENSE).

## 🙏 Acknowledgements

The stacked-spectrogram CNN follows the channel-stacking scheme of a public HMS Harmful Brain
Activity Classification solution (suguuuuu), and the multi-view model is a simplified
re-implementation of another (muku). Backbones come from
[`timm`](https://github.com/huggingface/pytorch-image-models). Thanks to the NeuroMM-2026 organizers
for the challenge and data.

The paper appears in the Proceedings of the 34th ACM International Conference on Multimedia
(MM '26), DOI [10.1145/3767308.3837695](https://doi.org/10.1145/3767308.3837695) (active once the
proceedings are published); the camera-ready PDF is in [`paper/`](paper/).
