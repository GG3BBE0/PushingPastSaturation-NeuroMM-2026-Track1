# Pushing Past Saturation: An EEG Spike-Detection System for NeuroMM-2026 Track-1

### 🏆 ACM MM 2026 · Grand Challenge Paper &nbsp;|&nbsp; 🥇 Rank 1, NeuroMM-2026 Track-1 (NMM-Basic-IED)

<p align="left">
  <img src="https://img.shields.io/badge/Venue-ACM%20MM%202026-blue" alt="Venue">
  <img src="https://img.shields.io/badge/Track--1%20AUPRC-0.9846-success" alt="AUPRC">
  <img src="https://img.shields.io/badge/Rank-1st%20Place-gold" alt="Rank">
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-orange" alt="Code">
</p>

> **Abstract:** We report the rank-1 system for NeuroMM-2026 Track-1, binary EEG spike detection
> scored by AUPRC. The system combines domain-aware preprocessing, five time–frequency
> representations, ImageNet-pretrained vision backbones adapted from prior competition solutions, and
> a regularized Nelder–Mead ensemble, reaching an inductive score of 0.9711. A conventional
> pseudo-labelling warm-up improves this to 0.9778, after which our candidate-specialist
> self-training procedure adds high-confidence released candidates at full weight and reaches
> **0.9846**. Because this final regime is transductive and in-sample by design, we evaluate it with
> a held-out-fold replay, which suggests partial transfer while not excluding candidate-specific
> memorization. The main lesson is that, on this saturated EEG benchmark, broad model diversity is
> insufficient; the decisive lever is many confident pseudo-labelled candidates, full commitment, and
> candidate-excluded real-label validation.

## 🔑 Highlights

- **Breaking a saturated benchmark.** A 43-member inductive ensemble plateaus at 0.9711; the
  transductive stage lifts it to **0.9846** (+0.0135), enough for 1st place.
- **Quantity over confidence.** Committing **many** pseudo-labelled candidates (11,336 at a
  permissive threshold) at **full weight** beats the conventional recipe of few, high-threshold,
  down-weighted pseudo-labels — which stalls at 0.978 regardless of threshold.
- **The `fold=-1` real-label gate.** Every pseudo-labelled candidate is always in training and never
  in a validation fold, so cross-validation stays computed purely on real labels and any confirmation
  bias shows up as a CV *drop* rather than a spurious gain.
- **A generalization check, not just a leaderboard number.** A held-out-fold replay repeats the whole
  transductive procedure on labelled folds the teacher never saw, and is positive on 4/5 folds.

## 📢 News

- **[Jul 2026]** Our paper has been accepted to the **ACM MM 2026 main conference proceedings** as a
  Grand Challenge paper! 🎉
- **[Jun 2026]** Our system ranked **1st** on NeuroMM-2026 Track-1 with an AUPRC of **0.9846**.
- **[Coming Soon]** The complete PyTorch implementation will be released here. Stay tuned!

## ⏳ Code Release Schedule (TODO)

We are currently organizing the code and will release it step by step:

- [ ] Release EEG preprocessing and time–frequency representation scripts (Morlet/band-passed CWT,
      Paul wavelet, superlet, STFT).
- [ ] Release the diverse model pool and per-fold training scripts.
- [ ] Release the regularized Nelder–Mead ensemble (two-dimensional trimming, bagging, capping).
- [ ] Release the candidate-specialist transductive self-training pipeline and the `fold=-1` gate.
- [ ] Provide evaluation scripts to reproduce the 0.9846 submission.
- [ ] Upload pre-trained weights.

## 📌 Citation

If you find our work or this repository useful, please consider citing our paper:

```bibtex
@inproceedings{chiang2026pushing,
  title={Pushing Past Saturation: An EEG Spike-Detection System for NeuroMM-2026 Track-1},
  author={Chiang, Ming-Chun},
  booktitle={Proceedings of the ACM International Conference on Multimedia (ACM MM)},
  year={2026}
}
```

## 📮 Contact

Ming-Chun Chiang — College of Artificial Intelligence, National Yang Ming Chiao Tung University,
Tainan, Taiwan.
