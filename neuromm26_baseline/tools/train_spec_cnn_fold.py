"""SpecCNN trainer with 5-fold CV from fold_df.csv (for CWT-Morlet)."""

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
from neuromm26_baseline.models.spec_cnn import SpecCNN
from neuromm26_baseline.utils.adan import Adan
from neuromm26_baseline.utils.io import ensure_dir
from neuromm26_baseline.utils.logger import get_logger
from neuromm26_baseline.utils.metrics import compute_binary_classification_metrics
from neuromm26_baseline.utils.seed import build_torch_generator, set_seed


def bce_smoothed(logits, target, smoothing=0.05, pos_weight=None):
    smoothed = target * (1 - smoothing) + (1 - target) * smoothing
    return Fnn.binary_cross_entropy_with_logits(logits, smoothed, pos_weight=pos_weight)


class SpecAug:
    def __init__(self, t=(0, 8), f=(0, 6), prob=0.5):
        self.t = t; self.f = f; self.p = prob

    def __call__(self, x):
        if np.random.random() < self.p and self.t[1] > 0:
            tl = np.random.randint(*self.t)
            if tl > 0:
                ts = np.random.randint(0, max(1, x.shape[-1] - tl))
                x = x.clone(); x[..., ts:ts + tl] = 0
        if np.random.random() < self.p and self.f[1] > 0:
            fl = np.random.randint(*self.f)
            if fl > 0:
                fs = np.random.randint(0, max(1, x.shape[-2] - fl))
                x = x.clone(); x[..., fs:fs + fl, :] = 0
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
    parser.add_argument("--spec-type", choices=["cwt", "cwt_paul", "stft", "superlet", "cwt_filtered"], default="cwt")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--fold-csv", default="fold_df_fixed.csv")
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage1-lr", type=float, default=8e-4)
    parser.add_argument("--stage2-lr", type=float, default=8e-5)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--smoothing", type=float, default=0.05)
    parser.add_argument("--no-augment", action="store_true")
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
    exp_name = f"spec_{args.spec_type}_fold__{bb_safe}__fold{args.fold_idx}__seed{args.seed}"
    out_root = Path(args.output_root)
    ckpt_dir = ensure_dir(out_root / "checkpoints" / exp_name)
    pred_dir = ensure_dir(out_root / "predictions")
    metrics_dir = ensure_dir(out_root / "metrics")
    log_path = out_root / "logs" / f"{exp_name}.log"
    logger = get_logger(f"neuromm26.spec_fold", str(log_path))
    logger.info(f"exp={exp_name} spec={args.spec_type} backbone={args.backbone} fold={args.fold_idx} seed={args.seed}")

    if args.debug:
        args.stage1_epochs = 1; args.stage2_epochs = 1

    base_train = FoldSpecDataset(spec_root=spec_root, fold_csv=args.fold_csv,
                                  fold_idx=args.fold_idx, split="train", preload_in_memory=True)
    base_val = FoldSpecDataset(spec_root=spec_root, fold_csv=args.fold_csv,
                                fold_idx=args.fold_idx, split="val", preload_in_memory=True)
    transform = SpecAug() if not args.no_augment else None
    train_ds = AugWrap(base_train, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=False, collate_fn=neuromm_collate,
                              generator=build_torch_generator(args.seed))
    val_loader = DataLoader(base_val, batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
                            pin_memory=False, collate_fn=neuromm_collate)
    logger.info(f"train={len(train_ds)} val={len(base_val)}")

    sample = base_train[0]["spec"]
    in_chans = sample.shape[0]
    logger.info(f"spec shape: {tuple(sample.shape)}  in_chans={in_chans}")

    labels = np.array(base_train.labels, dtype=np.float32)
    pos, neg = float((labels == 1).sum()), float((labels == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    model = SpecCNN(num_classes=1, backbone=args.backbone, in_chans=in_chans, pretrained=True).to(device)
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
                with torch.amp.autocast("cuda", enabled=args.amp):
                    out = model({"spec": b["spec"]})
                    loss = bce_smoothed(out["main"].view(-1), labels, args.smoothing, pos_weight)
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
                if step % 50 == 0:
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
                            "spec_type": args.spec_type, "in_chans": in_chans,
                            "fold_idx": args.fold_idx, "args": vars(args)},
                           ckpt_dir / "best.pt")

    np.savez(pred_dir / f"{exp_name}_oof.npz",
             sample_ids=np.array(best_val_sids), logits=best_val_logits, labels=best_val_labels)
    summary = {
        "experiment_name": exp_name, "model_name": f"spec_{args.spec_type}_fold/{args.backbone}",
        "task_type": "spec_fold", "spec_type": args.spec_type,
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
