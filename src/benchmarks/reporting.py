"""Metric history and completion markers for binary benchmark runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .state import BenchmarkTrainingState


def persist_epoch(
    *,
    run_dir: Path,
    history_path: Path,
    metrics: Mapping[str, Any],
    report: str,
) -> None:
    epoch = int(metrics["epoch"])
    (run_dir / "metrics" / f"epoch_{epoch:03d}.json").write_text(
        json.dumps(dict(metrics), indent=2, default=str),
        encoding="utf-8",
    )
    (run_dir / "metrics" / f"epoch_{epoch:03d}_classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )
    row = {
        "epoch": epoch,
        "stage": metrics["stage"],
        "train_total": metrics["train"].get("total"),
        "train_binary": metrics["train"].get("binary"),
        "val_accuracy": metrics["val"].get("accuracy"),
        "val_macro_f1": metrics["val"].get("macro_f1"),
        "val_f1": metrics["val"].get("f1"),
        "val_mcc": metrics["val"].get("mcc"),
        "val_roc_auc": metrics["val"].get("roc_auc"),
        "binary_threshold": metrics["binary_threshold"],
        "effective_batch_size": metrics["effective_batch_size"],
    }
    exists = history_path.exists()
    with history_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_training_complete(
    *,
    run_dir: Path,
    dataset_name: str,
    state: BenchmarkTrainingState,
    best: Path,
    stop_reason: str,
) -> None:
    path = run_dir / "training_complete.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset": dataset_name,
                "final_epoch": state.epoch,
                "stop_reason": stop_reason,
                "best_model": str(best),
                "best_metric": state.best_metric,
                "binary_threshold": state.threshold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
