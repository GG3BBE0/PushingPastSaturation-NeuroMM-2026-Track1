"""Muku V2 trainer with 5-fold CV — raw EEG + superlet/CWT spec input."""

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
from neuromm26_baseline.datasets.fold_datasets import FoldEEGSpecDataset
from neuromm26_baseline.models.muku_eegnet_v2 import MukuEEGNetV2
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
    L, Y, S = [], [], []
    for b in loader:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        out = model({"eeg": b["eeg"], "spec": b["spec"]})
        L.append(out["main"].view(-1).cpu())
        Y.append(b["label"].view(-1).float().cpu())
        S.extend(b["sample_id"])
    L = torch.cat(L); Y = torch.cat(Y)
    m = compute_binary_classification_metrics(L, Y)
    return m, L, Y, S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--backbone-spec", default=None,
                        help="Optional separate backbone for the spec branch.")
    parser.add_argument("--spec-type", choices=["superlet", "cwt"], default="superlet")
    parser.add_argument("--fold-csv", default="fold_df_fixed.csv")
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage1-lr", type=float, default=8e-4)
    parser.add_argument("--stage2-lr", type=float, default=8e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--smoothing", type=float, default=0.05)
    parser.add_argument("--aux-weight", type=float, default=0.1, help="weight for each of 3 aux heads")
    parser.add_argument("--n-freqs", type=int, default=8)
    parser.add_argument("--spec-use-1ddw", action="store_true")
    parser.add_argument("--eeg-feature-root", default="neuromm26_datasets/processed/features/eeg")
    parser.add_argument("--exp-prefix", default=None,
                        help="override the experiment prefix (defaults to 'muku_v2_{spec_tag}_fold__{backbone}')")
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
    spec_tag = args.spec_type + ("_1ddw" if args.spec_use_1ddw else "")
    prefix = args.exp_prefix if args.exp_prefix else f"muku_v2_{spec_tag}_fold__{bb_safe}"
    exp_name = f"{prefix}__fold{args.fold_idx}__seed{args.seed}"
    out_root = Path(args.output_root)
    ckpt_dir = ensure_dir(out_root / "checkpoints" / exp_name)
    pred_dir = ensure_dir(out_root / "predictions")
    metrics_dir = ensure_dir(out_root / "metrics")
    log_path = out_root / "logs" / f"{exp_name}.log"
    logger = get_logger("neuromm26.muku_v2_fold", str(log_path))
    logger.info(f"exp={exp_name} spec={args.spec_type} backbone={args.backbone} "
                f"backbone_spec={args.backbone_spec} 1ddw={args.spec_use_1ddw} "
                f"fold={args.fold_idx} seed={args.seed}")

    if args.debug:
        args.stage1_epochs = 1; args.stage2_epochs = 1

    base_train = FoldEEGSpecDataset(
        eeg_feature_root=args.eeg_feature_root, spec_root=spec_root,
        fold_csv=args.fold_csv, fold_idx=args.fold_idx, split="train",
        preload_in_memory=True, target_shape=(26, 2000),
    )
    base_val = FoldEEGSpecDataset(
        eeg_feature_root=args.eeg_feature_root, spec_root=spec_root,
        fold_csv=args.fold_csv, fold_idx=args.fold_idx, split="val",
        preload_in_memory=True, target_shape=(26, 2000),
    )
    train_loader = DataLoader(base_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=False, collate_fn=neuromm_collate,
                              generator=build_torch_generator(args.seed))
    val_loader = DataLoader(base_val, batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
                            pin_memory=False, collate_fn=neuromm_collate)
    logger.info(f"train={len(base_train)} val={len(base_val)}")

    labels = np.array(base_train.labels, dtype=np.float32)
    pos, neg = float((labels == 1).sum()), float((labels == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    logger.info(f"pos={int(pos)} neg={int(neg)} pos_weight={pos_weight.item():.3f}")

    model = MukuEEGNetV2(
        num_classes=1, backbone=args.backbone, backbone_spec=args.backbone_spec,
        in_chans=26, n_freqs=args.n_freqs, fs=500,
        spec_use_1ddw=args.spec_use_1ddw, pretrained=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"params={n_params:,}  AMP={args.amp}")

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
                lab = b["label"].view(-1).float()
                with torch.amp.autocast("cuda", enabled=args.amp):
                    out = model({"eeg": b["eeg"], "spec": b["spec"]})
                    lm = bce_smoothed(out["main"].view(-1), lab, args.smoothing, pos_weight)
                    la = bce_smoothed(out["aux_a"].view(-1), lab, args.smoothing, pos_weight)
                    lb = bce_smoothed(out["aux_b"].view(-1), lab, args.smoothing, pos_weight)
                    ls = bce_smoothed(out["aux_spec"].view(-1), lab, args.smoothing, pos_weight)
                    loss = (1 - 3 * args.aux_weight) * lm + args.aux_weight * (la + lb + ls)
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
                            "spec_type": args.spec_type, "fold_idx": args.fold_idx,
                            "args": vars(args)},
                           ckpt_dir / "best.pt")

    np.savez(pred_dir / f"{exp_name}_oof.npz",
             sample_ids=np.array(best_val_sids), logits=best_val_logits, labels=best_val_labels)
    summary = {
        "experiment_name": exp_name, "model_name": f"muku_v2_{spec_tag}/{args.backbone}",
        "task_type": "muku_v2_fold", "spec_type": args.spec_type,
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
