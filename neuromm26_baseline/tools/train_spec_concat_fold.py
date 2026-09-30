"""ConcatSpecCNN (V2) trainer with 5-fold CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from neuromm26_baseline.datasets.collate_fn import neuromm_collate
from neuromm26_baseline.datasets.fold_datasets import FoldSpecDataset
from neuromm26_baseline.models.spec_cnn_concat import ConcatSpecCNN
from neuromm26_baseline.utils.adan import Adan
from neuromm26_baseline.utils.io import ensure_dir
from neuromm26_baseline.utils.logger import get_logger
from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics
from neuromm26_baseline.utils.seed import build_torch_generator, set_seed


def bce_smoothed(logits, target, smoothing=0.05, pos_weight=None):
    smoothed = target * (1 - smoothing) + (1 - target) * smoothing
    return Fnn.binary_cross_entropy_with_logits(logits, smoothed, pos_weight=pos_weight)


def focal_loss(logits, target, gamma=2.0, smoothing=0.05, pos_weight=None):
    """Binary focal loss (T1). Down-weights easy samples by (1-p_t)^gamma so the
    model focuses on hard/ambiguous samples (the pos/neg-inseparable kind the
    per-subject diagnosis flagged). pos_weight optional — focal already handles
    imbalance via the modulating term, so combining the two can over-weight
    positives; the pilot tests gamma 2.0 with and without pos_weight."""
    if smoothing > 0:
        target = target * (1 - smoothing) + (1 - target) * smoothing
    ce = Fnn.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    return (ce * (1 - p_t).clamp(min=1e-6) ** gamma).mean()


def gce_loss(logits, target, q=0.7, pos_weight=None):
    """Generalized Cross Entropy (Zhang & Sabuncu 2018) — robust to label noise.
    L = (1 - p_t^q) / q. q->0 approaches CE (full noise sensitivity); q=1 is MAE
    (max robustness). Targets the DA00103D (fold4) sample-level label noise instead
    of fitting it. pos_weight up-weights positives (GCE has no built-in imbalance
    handling)."""
    p = torch.sigmoid(logits)
    p_t = (p * target + (1 - p) * (1 - target)).clamp(min=1e-6, max=1.0)
    loss = (1.0 - p_t ** q) / q
    if pos_weight is not None:
        loss = loss * (1.0 + (pos_weight - 1.0) * target)
    return loss.mean()


def rank_loss(logits, target, margin=1.0, bce_mix=0.1, smoothing=0.05, pos_weight=None):
    """Pairwise ranking (AUPRC-surrogate): for every (pos, neg) pair in the batch, push
    pos_logit above neg_logit by `margin` via softplus. Directly optimizes RANKING (what AUPRC
    scores) instead of calibrated BCE. A small BCE term keeps logits bounded. Genuinely different
    objective -> different error structure than the BCE/focal pool."""
    z = logits.view(-1); t = target.view(-1)
    pos = z[t > 0.5]; neg = z[t < 0.5]
    base = bce_smoothed(z, t, smoothing, pos_weight)
    if pos.numel() == 0 or neg.numel() == 0:
        return base
    diff = neg.view(1, -1) - pos.view(-1, 1) + margin   # want neg - pos + margin <= 0
    rank = Fnn.softplus(diff).mean()
    return rank + bce_mix * base


def per_sample_loss(logits, target, loss_name, focal_gamma, gce_q, smoothing, pos_weight):
    """Per-sample (reduction='none') version of the chosen loss, for noise down-weighting."""
    t = target
    if loss_name == "gce":
        p = torch.sigmoid(logits)
        p_t = (p * t + (1 - p) * (1 - t)).clamp(min=1e-6, max=1.0)
        ls = (1.0 - p_t ** gce_q) / gce_q
        if pos_weight is not None:
            ls = ls * (1.0 + (pos_weight - 1.0) * t)
        return ls
    ts = t * (1 - smoothing) + (1 - t) * smoothing if smoothing > 0 else t
    ce = Fnn.binary_cross_entropy_with_logits(logits, ts, pos_weight=pos_weight, reduction="none")
    if loss_name == "focal":
        p = torch.sigmoid(logits)
        p_t = p * t + (1 - p) * (1 - t)
        return ce * (1 - p_t).clamp(min=1e-6) ** focal_gamma
    return ce


def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


class SpecAug:
    """Time/Freq/Channel masking on spec input (C, H=freq, W=time).

    `c` enables channel masking on dim 0 (the EEG channel axis of the spec stack).
    `n_t / n_f / n_c` apply multiple masks per sample (heavy-aug mode uses >1).
    """

    def __init__(self, t=(0, 12), f=(0, 8), c=0, n_t=1, n_f=1, n_c=1, prob=0.5):
        self.t = t; self.f = f; self.c = c
        self.n_t = n_t; self.n_f = n_f; self.n_c = n_c; self.p = prob

    def __call__(self, x):
        # Clone once (cheap); modify in place afterwards
        x = x.clone()
        for _ in range(self.n_t):
            if np.random.random() < self.p and self.t[1] > 0:
                tl = np.random.randint(*self.t)
                if tl > 0:
                    ts = np.random.randint(0, max(1, x.shape[-1] - tl))
                    x[..., ts:ts + tl] = 0
        for _ in range(self.n_f):
            if np.random.random() < self.p and self.f[1] > 0:
                fl = np.random.randint(*self.f)
                if fl > 0:
                    fs = np.random.randint(0, max(1, x.shape[-2] - fl))
                    x[..., fs:fs + fl, :] = 0
        for _ in range(self.n_c):
            if self.c > 0 and np.random.random() < self.p:
                n_drop = np.random.randint(1, self.c + 1)
                ch_idx = np.random.choice(x.shape[0], min(n_drop, x.shape[0]), replace=False)
                x[ch_idx, :, :] = 0
        return x


class AugWrap(torch.utils.data.Dataset):
    def __init__(self, base, transform=None):
        self.base = base; self.transform = transform
    def __len__(self): return len(self.base)
    def __getattr__(self, n): return getattr(self.base, n)
    def __getitem__(self, i):
        s = self.base[i]
        if self.transform is not None and isinstance(s.get("spec"), torch.Tensor):
            s["spec"] = self.transform(s["spec"])
        return s


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    L, Y, S = [], [], []
    for b in loader:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        out = model({"spec": b["spec"]})
        L.append(out["main"].view(-1).cpu())
        Y.append(b["label"].view(-1).float().cpu())
        S.extend(b["sample_id"])
    L = torch.cat(L); Y = torch.cat(Y)
    return compute_binary_classification_metrics(L, Y), L, Y, S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-type", choices=["cwt", "cwt_paul", "stft", "superlet", "cwt_filtered", "cwt_phase"], default="cwt")
    parser.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k_384")
    parser.add_argument("--target-size", type=int, default=384)
    parser.add_argument("--fold-csv", default="fold_df_fixed.csv")
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage1-lr", type=float, default=7e-4)
    parser.add_argument("--stage2-lr", type=float, default=7e-5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--smoothing", type=float, default=0.05)
    parser.add_argument("--loss", choices=["bce", "focal", "gce", "rank"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--rank-margin", type=float, default=1.0, help="pairwise ranking margin (--loss rank)")
    parser.add_argument("--gce-q", type=float, default=0.7, help="GCE q in (0,1]; q->0=CE, q=1=MAE (more noise-robust)")
    parser.add_argument("--no-pos-weight", action="store_true",
                        help="disable pos_weight (focal already handles imbalance; pilot ± this)")
    parser.add_argument("--noise-sids", default=None,
                        help="file of confident-learning-flagged sids to DOWN-WEIGHT in training "
                             "(selective cleaning; targets the 121 scattered missed-spikes + DA00103D, "
                             "NOT a blanket robust loss). Only affects sids present in the train split.")
    parser.add_argument("--noise-weight", type=float, default=0.15,
                        help="per-sample loss weight for flagged sids (0=drop, 1=no-op)")
    parser.add_argument("--relabel-sids", default=None,
                        help="file of high-consensus missed-spike sids to RELABEL y=0->1 in the "
                             "TRAIN split only (val labels + OOF metric stay original/honest)")
    parser.add_argument("--soft-label-file", default=None,
                        help="file of `sid prob` lines: SOFT-distillation target (ensemble prob in [0,1]) "
                             "that REPLACES the hard label for those sids in TRAIN only (val/OOF stay real). "
                             "BCE/focal already accept soft targets. For 2-stage soft pseudo-distillation.")
    parser.add_argument("--drop-path", type=float, default=None,
                        help="stochastic-depth rate; default = dropout*2. Raise (~0.3) for big backbones")
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--exp-prefix", default=None,
                        help="override exp prefix; default = concat_{spec}_fold__{bb_safe}")
    # Heavy-aug knobs (defaults reproduce previous SpecAug behaviour)
    parser.add_argument("--time-mask-max", type=int, default=12, help="max time mask length")
    parser.add_argument("--freq-mask-max", type=int, default=8, help="max freq mask length")
    parser.add_argument("--channel-drop-max", type=int, default=0, help="max channels to drop (0=off)")
    parser.add_argument("--n-time-masks", type=int, default=1)
    parser.add_argument("--n-freq-masks", type=int, default=1)
    parser.add_argument("--n-channel-masks", type=int, default=1)
    parser.add_argument("--spec-aug-prob", type=float, default=0.5)
    parser.add_argument("--output-root", default="neuromm26_results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    spec_root = f"neuromm26_datasets/processed/features/{args.spec_type}"
    bb_safe = args.backbone.replace("/", "-").replace(".", "_")
    prefix = args.exp_prefix if args.exp_prefix else f"concat_{args.spec_type}_fold__{bb_safe}"
    exp_name = f"{prefix}__fold{args.fold_idx}__seed{args.seed}"
    out_root = Path(args.output_root)
    ckpt_dir = ensure_dir(out_root / "checkpoints" / exp_name)
    pred_dir = ensure_dir(out_root / "predictions")
    metrics_dir = ensure_dir(out_root / "metrics")
    log_path = out_root / "logs" / f"{exp_name}.log"
    logger = get_logger("neuromm26.concat_fold", str(log_path))
    logger.info(f"exp={exp_name} fold={args.fold_idx} seed={args.seed}")

    if args.debug:
        args.stage1_epochs = 1; args.stage2_epochs = 1

    base_train = FoldSpecDataset(spec_root=spec_root, fold_csv=args.fold_csv,
                                  fold_idx=args.fold_idx, split="train", preload_in_memory=True)
    base_val = FoldSpecDataset(spec_root=spec_root, fold_csv=args.fold_csv,
                                fold_idx=args.fold_idx, split="val", preload_in_memory=True)
    # RELABEL high-consensus missed-spikes (y=0->1) in the TRAIN split only (before pos_weight).
    if args.relabel_sids:
        relabel_set = {l.strip() for l in Path(args.relabel_sids).read_text().splitlines() if l.strip()}
        n_flip = sum(1 for i, sid in enumerate(base_train.sample_ids)
                     if sid in relabel_set and base_train.labels[i] == 0)
        for i, sid in enumerate(base_train.sample_ids):
            if sid in relabel_set and base_train.labels[i] == 0:
                base_train.labels[i] = 1
        logger.info(f"relabel: flipped {n_flip} train sids y=0->1 ({len(relabel_set)} in file); val untouched")
    transform = (SpecAug(
                    t=(0, args.time_mask_max), f=(0, args.freq_mask_max),
                    c=args.channel_drop_max,
                    n_t=args.n_time_masks, n_f=args.n_freq_masks, n_c=args.n_channel_masks,
                    prob=args.spec_aug_prob)
                 if not args.no_augment else None)
    logger.info(
        f"SpecAug: t_max={args.time_mask_max} (×{args.n_time_masks})  "
        f"f_max={args.freq_mask_max} (×{args.n_freq_masks})  "
        f"c_max={args.channel_drop_max} (×{args.n_channel_masks})  prob={args.spec_aug_prob}"
    )
    train_ds = AugWrap(base_train, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=False, collate_fn=neuromm_collate,
                              generator=build_torch_generator(args.seed))
    val_loader = DataLoader(base_val, batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
                            pin_memory=False, collate_fn=neuromm_collate)
    logger.info(f"train={len(train_ds)} val={len(base_val)}")

    labels = np.array(base_train.labels, dtype=np.float32)
    pos, neg = float((labels == 1).sum()), float((labels == 0).sum())
    pos_weight = None if args.no_pos_weight else torch.tensor([neg / max(pos, 1.0)], device=device)

    def criterion(logits, target):
        if args.loss == "focal":
            return focal_loss(logits, target, args.focal_gamma, args.smoothing, pos_weight)
        if args.loss == "gce":
            return gce_loss(logits, target, args.gce_q, pos_weight)
        if args.loss == "rank":
            return rank_loss(logits, target, args.rank_margin, 0.1, args.smoothing, pos_weight)
        return bce_smoothed(logits, target, args.smoothing, pos_weight)
    logger.info(f"loss={args.loss}"
                f"{f' gamma={args.focal_gamma}' if args.loss=='focal' else ''} "
                f"pos_weight={'off' if pos_weight is None else f'{pos_weight.item():.2f}'}")

    noise_set = set()
    if args.noise_sids:
        noise_set = {l.strip() for l in Path(args.noise_sids).read_text().splitlines() if l.strip()}
        n_in_train = sum(1 for s in base_train.sample_ids if s in noise_set)
        logger.info(f"selective cleaning: {len(noise_set)} flagged sids total, "
                    f"{n_in_train} in this train split, down-weight={args.noise_weight}")

    soft_map = {}
    if args.soft_label_file:
        for ln in Path(args.soft_label_file).read_text().splitlines():
            if ln.strip():
                sid, p = ln.split()[:2]; soft_map[sid] = float(p)
        n_soft = sum(1 for s in base_train.sample_ids if s in soft_map)
        logger.info(f"soft-distillation: {len(soft_map)} soft targets total, {n_soft} in this train split")

    model = ConcatSpecCNN(num_classes=1, backbone=args.backbone, target_size=args.target_size,
                          pretrained=True, drop_path=args.drop_path).to(device)
    logger.info(f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,} AMP={args.amp}")

    best_auprc = float("-inf"); best_rec = None; elog = []
    best_val_logits = None; best_val_sids = None; best_val_labels = None
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp) if args.amp else None

    for stage, (n_ep, lr) in enumerate([(args.stage1_epochs, args.stage1_lr),
                                         (args.stage2_epochs, args.stage2_lr)], start=1):
        logger.info(f"=== Stage {stage}: lr={lr} epochs={n_ep} ===")
        opt = Adan(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        sched = OneCycleLR(opt, max_lr=lr, epochs=n_ep, steps_per_epoch=len(train_loader),
                           pct_start=0.1, anneal_strategy="cos", final_div_factor=100)
        for ep in range(1, n_ep + 1):
            model.train()
            tloss = 0.0; n = 0
            for step, b in enumerate(train_loader):
                b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
                labels = b["label"].view(-1).float()
                if soft_map:  # soft-distillation: replace hard label with ensemble prob for candidate sids
                    labels = torch.tensor([soft_map.get(s, float(l)) for s, l in zip(b["sample_id"], labels.tolist())],
                                          device=device, dtype=torch.float32)
                spec = b["spec"]
                w = None
                if noise_set:
                    w = torch.tensor([args.noise_weight if s in noise_set else 1.0
                                      for s in b["sample_id"]], device=device, dtype=torch.float32)
                ps_args = (args.loss, args.focal_gamma, args.gce_q, args.smoothing, pos_weight)
                # mixup softens targets -> breaks the pos/neg split a ranking loss needs
                use_mix = args.mixup_alpha > 0 and np.random.random() < args.mixup_prob and args.loss != "rank"
                if use_mix:
                    lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                    idx = torch.randperm(spec.size(0), device=spec.device)
                    mspec = lam * spec + (1 - lam) * spec[idx]
                    with torch.amp.autocast("cuda", enabled=args.amp):
                        lg = model({"spec": mspec})["main"].view(-1)
                        if w is None:
                            loss = lam * criterion(lg, labels) + (1 - lam) * criterion(lg, labels[idx])
                        else:
                            la = per_sample_loss(lg, labels, *ps_args)
                            lb = per_sample_loss(lg, labels[idx], *ps_args)
                            denom = (lam * w + (1 - lam) * w[idx]).sum().clamp(min=1.0)
                            loss = (lam * w * la + (1 - lam) * w[idx] * lb).sum() / denom
                else:
                    with torch.amp.autocast("cuda", enabled=args.amp):
                        lg = model({"spec": spec})["main"].view(-1)
                        if w is None:
                            loss = criterion(lg, labels)
                        else:
                            loss = (per_sample_loss(lg, labels, *ps_args) * w).sum() / w.sum().clamp(min=1.0)
                opt.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                sched.step()
                tloss += float(loss.item()); n += 1
                if step % 80 == 0:
                    logger.info(f"ep={(stage-1)*args.stage1_epochs + ep} step={step}/{len(train_loader)} loss={loss.item():.4f}")
            tloss /= max(n, 1)
            metrics, vL, vY, vS = evaluate(model, val_loader, device)
            auprc = metrics["auprc"]
            elog.append({"stage": stage, "epoch": ep, "train_loss": tloss,
                         **{f"val_{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}})
            logger.info(f"[s{stage} ep={ep}] tloss={tloss:.4f} val_auprc={auprc:.4f} f1={metrics['binary_f1']:.4f}")
            if auprc > best_auprc:
                best_auprc = auprc
                best_rec = {"stage": stage, "epoch": ep, "metrics": dict(metrics)}
                best_val_logits = vL.numpy(); best_val_sids = vS; best_val_labels = vY.numpy()
                torch.save({"model_state_dict": model.state_dict(),
                            "best_metric_name": "auprc", "best_metric_value": best_auprc,
                            "best_metrics": dict(metrics), "stage": stage, "epoch": ep,
                            "spec_type": args.spec_type, "fold_idx": args.fold_idx,
                            "args": vars(args)},
                           ckpt_dir / "best.pt")

    np.savez(pred_dir / f"{exp_name}_oof.npz",
             sample_ids=np.array(best_val_sids), logits=best_val_logits, labels=best_val_labels)
    summary = {
        "experiment_name": exp_name,
        "model_name": f"concat_{args.spec_type}_fold/{args.backbone}",
        "task_type": "concat_fold", "spec_type": args.spec_type,
        "fold_idx": int(args.fold_idx), "seed": int(args.seed),
        "best_metric_name": "auprc", "best_metric_value": float(best_auprc),
        "best_metrics": best_rec["metrics"] if best_rec else {},
        "best_stage": best_rec["stage"] if best_rec else None,
        "best_epoch": best_rec["epoch"] if best_rec else None,
        "epoch_log": elog,
    }
    (metrics_dir / f"{exp_name}_train_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    logger.info(f"DONE best_auprc={best_auprc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
