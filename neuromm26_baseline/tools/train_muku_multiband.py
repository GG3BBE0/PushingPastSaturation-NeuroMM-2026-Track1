"""Muku trainer using multi-band cache (in_chans=130)."""

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
from neuromm26_baseline.datasets.multiband_dataset import MultiBandFeatureDataset
from neuromm26_baseline.datasets.muku_dataset import AugmentedDataset, EEGAugmentation
from neuromm26_baseline.models.muku_eegnet import MukuEEGNet
from neuromm26_baseline.utils.adan import Adan
from neuromm26_baseline.utils.io import ensure_dir
from neuromm26_baseline.utils.logger import get_logger
from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics
from neuromm26_baseline.utils.seed import build_torch_generator, set_seed


def bce_smoothed(logits, target, smoothing=0.05, pos_weight=None):
    smoothed = target * (1 - smoothing) + (1 - target) * smoothing
    return Fnn.binary_cross_entropy_with_logits(logits, smoothed, pos_weight=pos_weight)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    L, Y = [], []
    for b in loader:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        out = model({"eeg": b["eeg"]})
        L.append(out["main"].view(-1).cpu())
        Y.append(b["label"].view(-1).float().cpu())
    L = torch.cat(L); Y = torch.cat(Y)
    return compute_binary_classification_metrics(L, Y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage1-lr", type=float, default=5e-4)
    parser.add_argument("--stage2-lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--smoothing", type=float, default=0.05)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--n-freqs", type=int, default=8)
    parser.add_argument("--in-chans", type=int, default=130)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--manifest", default="neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv")
    parser.add_argument("--multiband-root", default="neuromm26_datasets/processed/features/multiband")
    parser.add_argument("--output-root", default="neuromm26_results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    backbone_safe = args.backbone.replace("/", "-").replace(".", "_")
    exp_name = f"muku_mb__{backbone_safe}__seed{args.seed}__lr{args.stage1_lr}_{args.stage2_lr}"
    out_root = Path(args.output_root)
    ckpt_dir = ensure_dir(out_root / "checkpoints" / exp_name)
    metrics_dir = ensure_dir(out_root / "metrics")
    log_path = out_root / "logs" / f"{exp_name}.log"
    logger = get_logger("neuromm26.muku_mb", str(log_path))
    logger.info(f"exp={exp_name} backbone={args.backbone} seed={args.seed} in_chans={args.in_chans}")

    if args.debug:
        args.stage1_epochs = 1; args.stage2_epochs = 1

    base_train = MultiBandFeatureDataset(
        multiband_root=args.multiband_root, manifest_csv=args.manifest,
        split="train", preload_in_memory=True, target_shape=(args.in_chans, 2000),
    )
    base_val = MultiBandFeatureDataset(
        multiband_root=args.multiband_root, manifest_csv=args.manifest,
        split="val", preload_in_memory=True, target_shape=(args.in_chans, 2000),
    )
    transform = EEGAugmentation() if not args.no_augment else None
    train_ds = AugmentedDataset(base_train, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=False, collate_fn=neuromm_collate,
                              generator=build_torch_generator(args.seed))
    val_loader = DataLoader(base_val, batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
                            pin_memory=False, collate_fn=neuromm_collate)
    logger.info(f"train={len(train_ds)} val={len(base_val)}")

    labels = np.array(base_train.labels, dtype=np.float32)
    pos, neg = float((labels == 1).sum()), float((labels == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    logger.info(f"pos={int(pos)} neg={int(neg)} pos_weight={pos_weight.item():.3f}")

    model = MukuEEGNet(
        num_classes=1, backbone=args.backbone, in_chans=args.in_chans,
        n_freqs=args.n_freqs, fs=500, pretrained=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"params={n_params:,}")

    best_auprc = float("-inf"); best_rec = None; elog = []

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
                out = model({"eeg": b["eeg"]})
                lm = bce_smoothed(out["main"].view(-1), labels, args.smoothing, pos_weight)
                la = bce_smoothed(out["aux_a"].view(-1), labels, args.smoothing, pos_weight)
                lb = bce_smoothed(out["aux_b"].view(-1), labels, args.smoothing, pos_weight)
                loss = (1 - 2 * args.aux_weight) * lm + args.aux_weight * (la + lb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                tloss += float(loss.item()); n += 1
                if step % 50 == 0:
                    logger.info(f"ep={(stage-1)*args.stage1_epochs + ep} step={step}/{len(train_loader)} loss={loss.item():.4f}")
            tloss /= max(n, 1)
            metrics = evaluate(model, val_loader, device)
            auprc = metrics["auprc"]
            elog.append({"stage": stage, "epoch": ep, "train_loss": tloss,
                         **{f"val_{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}})
            logger.info(f"[s{stage} ep={ep}] tloss={tloss:.4f} val_auprc={auprc:.4f} f1={metrics['binary_f1']:.4f}")
            if auprc > best_auprc:
                best_auprc = auprc
                best_rec = {"stage": stage, "epoch": ep, "metrics": dict(metrics)}
                torch.save({"model_state_dict": model.state_dict(),
                            "best_metric_name": "auprc", "best_metric_value": best_auprc,
                            "best_metrics": dict(metrics), "stage": stage, "epoch": ep,
                            "in_chans": args.in_chans, "args": vars(args)},
                           ckpt_dir / "best.pt")

    summary = {
        "experiment_name": exp_name, "model_name": f"muku_mb/{args.backbone}",
        "task_type": "muku_mb_binary", "seed": int(args.seed),
        "learning_rate": float(args.stage2_lr), "in_chans": args.in_chans,
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
