"""Multi-class classification metrics for Task3 (5-class)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


PRIMARY_SELECTION_METRIC = "weighted_f1"

METRIC_DISPLAY_ORDER = (
    "weighted_f1",
    "macro_f1",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "num_samples",
    "num_classes",
)

METRIC_HIGHLIGHT_ORDER = (
    "weighted_f1",
    "macro_f1",
    "accuracy",
    "macro_precision",
    "macro_recall",
)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_multiclass_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 5,
) -> dict[str, Any]:
    """Compute accuracy, per-class P/R/F1, macro/weighted F1.

    Args:
        logits: (N, C)
        labels: (N,) integer class index
        num_classes: number of classes
    """
    logits = logits.detach().cpu().view(-1, num_classes)
    labels = labels.detach().cpu().long().view(-1)
    if labels.numel() == 0:
        return {"num_samples": 0, "num_classes": num_classes}
    preds = logits.argmax(dim=-1).long()

    n = labels.numel()
    accuracy = float((preds == labels).float().mean().item())

    per_class_p: list[float] = []
    per_class_r: list[float] = []
    per_class_f: list[float] = []
    per_class_support: list[int] = []
    out: dict[str, Any] = {}

    for c in range(num_classes):
        tp = float(((preds == c) & (labels == c)).sum().item())
        fp = float(((preds == c) & (labels != c)).sum().item())
        fn = float(((preds != c) & (labels == c)).sum().item())
        support = tp + fn
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, support)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        per_class_p.append(prec)
        per_class_r.append(rec)
        per_class_f.append(f1)
        per_class_support.append(int(support))
        out[f"precision_class_{c}"] = prec
        out[f"recall_class_{c}"] = rec
        out[f"f1_class_{c}"] = f1
        out[f"support_class_{c}"] = int(support)

    classes_with_support = [i for i, s in enumerate(per_class_support) if s > 0]
    if classes_with_support:
        macro_f1 = float(np.mean([per_class_f[i] for i in classes_with_support]))
        macro_p = float(np.mean([per_class_p[i] for i in classes_with_support]))
        macro_r = float(np.mean([per_class_r[i] for i in classes_with_support]))
    else:
        macro_f1 = macro_p = macro_r = 0.0
    weighted_f1 = float(
        sum(per_class_f[i] * per_class_support[i] for i in range(num_classes)) / max(n, 1)
    )

    out.update({
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "weighted_f1": weighted_f1,
        "num_samples": int(n),
        "num_classes": int(num_classes),
    })
    return out


def order_multiclass_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in METRIC_DISPLAY_ORDER:
        if key in metrics:
            ordered[key] = metrics[key]
    for key, value in metrics.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def format_multiclass_summary(metrics: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in METRIC_HIGHLIGHT_ORDER:
        if key in metrics:
            parts.append(f"{key}={float(metrics[key]):.4f}")
    return " ".join(parts)
