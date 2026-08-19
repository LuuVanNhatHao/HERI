"""Checkpoint-bound evaluation for standalone binary HERA benchmarks."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from src.evaluation.evaluator import move_batch_to_device
from src.training.precision import PrecisionManager

from .artifacts import (
    assert_checkpoint_signature,
    load_checkpoint_payload,
    resolve_evaluation_checkpoint,
)
from .inference import run_binary_inference
from .loader import create_binary_benchmark_dataloader
from .loss import BinaryBenchmarkLoss
from .metrics import auxiliary_metrics, binary_metrics_and_report
from .model import BinaryBenchmarkHERA


class BinaryBenchmarkEvaluator:
    """Evaluate one saved model with its validation-tuned binary threshold."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.device = torch.device(str(config["runtime"]["device"]))
        self.checkpoint_path = resolve_evaluation_checkpoint(config)
        self.payload = load_checkpoint_payload(self.checkpoint_path)
        assert_checkpoint_signature(self.payload, config)
        self.model = BinaryBenchmarkHERA.from_config(config, load_backbones=True).to(
            self.device
        )
        if self.model.deberta is None or not hasattr(self.model.deberta, "tokenizer"):
            raise RuntimeError(
                "Binary benchmark evaluation requires an attached fast tokenizer"
            )
        split = str(config["evaluation"]["split"])
        limits = config.get("runtime", {}).get("dataset_max_samples", {})
        self.loader = create_binary_benchmark_dataloader(
            config,
            split=split,
            tokenizer=self.model.deberta.tokenizer,
            max_samples=limits.get(split),
        )
        self.precision = PrecisionManager(
            precision=str(config["runtime"]["precision"]),
            device=self.device,
        )
        self.loss_fn = BinaryBenchmarkLoss(config)
        self.output_dir = _evaluation_output_dir(
            config, self.checkpoint_path, split=split
        )

    def run(self) -> Path:
        self._materialize_and_load()
        split = str(self.config["evaluation"]["split"])
        threshold = self.payload.get("binary_threshold")
        if threshold is None:
            threshold = self.config["evaluation"].get("binary_threshold")
        if threshold is None:
            raise ValueError(
                "The checkpoint has no validation-tuned binary_threshold and no explicit "
                "evaluation.binary_threshold was provided"
            )
        threshold = float(threshold)
        bundle = run_binary_inference(
            model=self.model,
            loader=self.loader,
            loss_fn=self.loss_fn,
            device=self.device,
            precision=self.precision,
            max_batches=self.config["runtime"].get("max_test_batches"),
            desc=f"{self.config['dataset']['name']} {split} evaluation",
            progress=bool(self.config["progress"]["enabled"]),
        )
        metrics, report = binary_metrics_and_report(
            bundle,
            threshold=threshold,
            negative_name=str(self.config["task"]["negative_name"]),
            positive_name=str(self.config["task"]["positive_name"]),
        )
        metrics.update(auxiliary_metrics(bundle))
        metrics.update(
            {
                "dataset": str(self.config["dataset"]["name"]),
                "split": split,
                "samples": len(bundle.targets),
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_epoch": int(self.payload.get("epoch", -1)),
            }
        )
        self._write_artifacts(
            bundle=bundle, metrics=metrics, report=report, threshold=threshold
        )
        print(f"{split.capitalize()} binary classification_report:")
        print(report)
        print(f"Binary threshold: {threshold:.4f}")
        print(f"Saved benchmark evaluation: {self.output_dir}")
        return self.output_dir / "summary.csv"

    def _materialize_and_load(self) -> None:
        batch = move_batch_to_device(next(iter(self.loader)), self.device)
        self.model.eval()
        with torch.no_grad(), self.precision.autocast():
            self.model(batch).validate()
        result = self.model.load_state_dict(self.payload["model"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint state mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        self.model.eval()
        # Release the CPU-mapped tensor dictionary before full inference.
        self.payload["model"] = {}
        gc.collect()

    def _write_artifacts(
        self,
        *,
        bundle: Any,
        metrics: Mapping[str, Any],
        report: str,
        threshold: float,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "classification_report.txt").write_text(
            report, encoding="utf-8"
        )
        (self.output_dir / "metrics.json").write_text(
            json.dumps(dict(metrics), indent=2, default=str),
            encoding="utf-8",
        )
        with (self.output_dir / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)
        with (self.output_dir / "predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fields = [
                "id",
                "target",
                "score",
                "prediction",
                "subtype_target",
                "subtype_prediction",
                "text_label_target",
                "text_label_score",
                "metadata_json",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, sample_id in enumerate(bundle.ids):
                score = float(bundle.scores[index])
                writer.writerow(
                    {
                        "id": sample_id,
                        "target": int(bundle.targets[index]),
                        "score": score,
                        "prediction": int(score >= threshold),
                        "subtype_target": bundle.subtype_targets[index],
                        "subtype_prediction": bundle.subtype_predictions[index],
                        "text_label_target": bundle.text_label_targets[index],
                        "text_label_score": bundle.text_label_scores[index],
                        "metadata_json": json.dumps(
                            bundle.metadata[index], ensure_ascii=False, default=str
                        ),
                    }
                )


def _evaluation_output_dir(
    config: Mapping[str, Any],
    checkpoint: Path,
    *,
    split: str,
) -> Path:
    configured = config["evaluation"].get("output_dir")
    if configured not in (None, ""):
        return Path(str(configured))
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent / "evaluations" / split
    return Path(str(config["output"]["root"])) / "evaluations" / split
