"""Aggregate baseline results: compute mean and std across seeds per model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HIGHLIGHT_METRICS = ("auprc", "binary_f1", "macro_f1", "weighted_f1", "accuracy", "balanced_accuracy")

SEEDS_DEFAULT = {0, 1, 2, 3}


def discover_summaries(metrics_dir: Path, seeds: set[int] | None = None) -> dict[tuple[str, float], list[dict]]:
    """Read all *_train_summary.json files and group by (model_name, learning_rate)."""
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for path in sorted(metrics_dir.glob("*_train_summary.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        model_name = data.get("model_name")
        seed = data.get("seed")
        lr = data.get("learning_rate", 0.0)
        if model_name is None:
            continue
        if seeds is not None and seed not in seeds:
            continue
        groups[(model_name, lr)].append(data)
    return dict(groups)


def aggregate(groups: dict[tuple[str, float], list[dict]]) -> list[dict]:
    """Compute mean and std for each (model, lr) group across seeds."""
    rows = []
    for (model_name, lr) in sorted(groups):
        entries = groups[(model_name, lr)]
        seeds_found = sorted(e["seed"] for e in entries)
        n = len(entries)

        row = {"model": model_name, "lr": lr, "n_seeds": n, "seeds": seeds_found}
        # Collect metric values across seeds
        metric_values: dict[str, list[float]] = defaultdict(list)
        for entry in entries:
            best = entry.get("best_metrics", {})
            for key in HIGHLIGHT_METRICS:
                if key in best:
                    metric_values[key].append(float(best[key]))

        for key in HIGHLIGHT_METRICS:
            vals = metric_values.get(key, [])
            if vals:
                row[f"{key}_mean"] = float(np.mean(vals))
                row[f"{key}_std"] = float(np.std(vals))
            else:
                row[f"{key}_mean"] = None
                row[f"{key}_std"] = None
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    """Print a formatted table to stdout."""
    if not rows:
        print("No results found.")
        return

    # Header
    header_model = "Model"
    header_n = "N"
    metric_headers = []
    for m in HIGHLIGHT_METRICS:
        metric_headers.append(m)

    # Compute column widths
    model_width = max(len(header_model), max(len(r["model"]) for r in rows))

    print()
    # Title line
    parts = [f"{'Model':<{model_width}}", f"{'LR':>8}", f"{'N':>2}"]
    for m in HIGHLIGHT_METRICS:
        parts.append(f"{m:>18}")
    print("  ".join(parts))
    print("-" * len("  ".join(parts)))

    # Data lines (sorted by auprc_mean descending)
    sorted_rows = sorted(rows, key=lambda r: r.get("auprc_mean") or 0, reverse=True)
    for row in sorted_rows:
        lr_str = f"{row['lr']:.0e}" if row["lr"] >= 1e-6 else "0"
        parts = [f"{row['model']:<{model_width}}", f"{lr_str:>8}", f"{row['n_seeds']:>2}"]
        for m in HIGHLIGHT_METRICS:
            mean = row.get(f"{m}_mean")
            std = row.get(f"{m}_std")
            if mean is not None and std is not None:
                parts.append(f"{mean:.4f}\u00b1{std:.4f}")
            else:
                parts.append(f"{'---':>18}")
        print("  ".join(parts))
    print()


def save_csv(rows: list[dict], output_path: Path) -> None:
    """Save aggregated results to CSV."""
    fieldnames = ["model", "lr", "n_seeds", "seeds"]
    for m in HIGHLIGHT_METRICS:
        fieldnames.extend([f"{m}_mean", f"{m}_std"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        sorted_rows = sorted(rows, key=lambda r: r.get("auprc_mean") or 0, reverse=True)
        for row in sorted_rows:
            csv_row = {k: row.get(k) for k in fieldnames}
            csv_row["seeds"] = str(row.get("seeds", []))
            writer.writerow(csv_row)
    print(f"CSV saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate baseline results across seeds.")
    parser.add_argument(
        "--metrics-dir",
        default="neuromm26_results/metrics",
        help="Directory containing *_train_summary.json files",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Filter to these seeds only (default: all seeds)",
    )
    parser.add_argument(
        "--output-csv",
        default="neuromm26_results/metrics/baseline_summary.csv",
        help="Path to save CSV summary",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    seeds = set(args.seeds) if args.seeds is not None else None

    groups = discover_summaries(metrics_dir, seeds=seeds)
    if not groups:
        print(f"No train_summary.json files found in {metrics_dir}")
        return 1

    rows = aggregate(groups)
    print_table(rows)
    save_csv(rows, Path(args.output_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
