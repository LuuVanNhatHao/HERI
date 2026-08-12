from __future__ import annotations

import importlib


def test_evaluator_import_does_not_trigger_training_package_cycle() -> None:
    evaluator = importlib.import_module("src.evaluation.evaluator")
    assert hasattr(evaluator, "move_batch_to_device")
    assert hasattr(evaluator, "compute_metrics_and_reports")


def test_training_package_exports_trainers_lazily() -> None:
    training = importlib.import_module("src.training")
    assert training.HERATrainer.__name__ == "HERATrainer"
    assert training.BaselineTrainer.__name__ == "BaselineTrainer"
