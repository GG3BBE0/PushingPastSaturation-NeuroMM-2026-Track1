"""Metrics for binary classification baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


PRIMARY_SELECTION_METRIC = "auprc"
METRIC_DISPLAY_ORDER = (
    "auprc",
    "binary_f1",
    "macro_f1",
    "weighted_f1",
    "binary_precision",
    "binary_recall",
    "specificity",
    "accuracy",
    "balanced_accuracy",
    "num_samples",
    "positive_ratio",
)
METRIC_HIGHLIGHT_ORDER = (
    "auprc",
    "binary_f1",
    "macro_f1",
    "weighted_f1",
)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1_score(tp: float, fp: float, fn: float) -> float:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return _safe_divide(2 * precision * recall, precision + recall)


def _compute_auprc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    y_score = probs.detach().view(-1).cpu().numpy().astype(np.float64)
    y_true = targets.detach().view(-1).cpu().numpy().astype(np.int64)

    num_positive = int(y_true.sum())
    if num_positive == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    distinct = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct, y_true_sorted.size - 1]

    tp = tp[threshold_idxs]
    fp = fp[threshold_idxs]

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / num_positive

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def order_metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in METRIC_DISPLAY_ORDER:
        if key in metrics:
            ordered[key] = metrics[key]
    for key, value in metrics.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def format_metric_summary(metrics: dict[str, Any]) -> str:
    ordered = order_metric_dict(metrics)
    parts: list[str] = []
    for key in METRIC_HIGHLIGHT_ORDER:
        if key in ordered:
            parts.append(f"{key}={float(ordered[key]):.6f}")
    return " ".join(parts)


def compute_binary_classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    probs = torch.sigmoid(logits.detach()).view(-1).cpu()
    y_true = targets.detach().view(-1).cpu().float()
    y_pred = (probs >= 0.5).float()

    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())

    positive_support = tp + fn
    negative_support = tn + fp
    num_samples = positive_support + negative_support

    binary_precision = _safe_divide(tp, tp + fp)
    binary_recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    binary_f1 = _f1_score(tp, fp, fn)
    negative_f1 = _f1_score(tn, fn, fp)
    macro_f1 = 0.5 * (binary_f1 + negative_f1)
    weighted_f1 = _safe_divide(binary_f1 * positive_support + negative_f1 * negative_support, num_samples)
    accuracy = _safe_divide(tp + tn, num_samples)
    balanced_accuracy = 0.5 * (binary_recall + specificity)
    auprc = _compute_auprc(probs, y_true)

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "specificity": specificity,
        "binary_f1": binary_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "auprc": auprc,
        "num_samples": int(y_true.numel()),
        "positive_ratio": float(np.mean(y_true.numpy())) if y_true.numel() else 0.0,
    }
    return order_metric_dict(metrics)
