import numpy as np
import pytest
import torch

from heri.metrics.attribution import attribution_metrics
from heri.metrics.detection import binary_detection_metrics
from heri.metrics.consistency import evidence_decision_metrics
from heri.metrics.grounding import text_grounding_metrics


def test_detection_metrics_known_values() -> None:
    metrics = binary_detection_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        threshold=0.5,
    )
    assert metrics["acc"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["eer"] == 0.0


def test_attribution_metrics_known_values() -> None:
    y_true = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
    y_score = np.array([[0.9, 0.1, 0.2, 0.1], [0.2, 0.8, 0.7, 0.2], [0.1, 0.1, 0.2, 0.9]])
    metrics = attribution_metrics(y_true, y_score, thresholds=np.array([0.5, 0.5, 0.5, 0.5]))
    assert metrics["cf1"] == 1.0
    assert metrics["of1"] == 1.0
    assert metrics["map"] == 1.0


def test_text_grounding_metrics_reconstruct_word_level_from_subwords() -> None:
    metrics = text_grounding_metrics(
        targets=torch.tensor([[1, 1, 0, 0]]),
        logits=torch.tensor([[0.2, 3.0, 3.0, -1.0]]),
        mask=torch.tensor([[True, True, True, True]]),
        applies=torch.tensor([True]),
        word_ids=torch.tensor([[0, 0, 1, 2]]),
        threshold=0.5,
    )
    assert metrics["text_precision"] == 0.5
    assert metrics["text_recall"] == 1.0
    assert round(metrics["text_f1"], 4) == 0.6667


def test_evidence_decision_metrics_follow_equation_32() -> None:
    metrics = evidence_decision_metrics(
        atomic_scores=np.array([[0.8, 0.6], [0.4, 0.9]]),
        evidence_support=np.array([[0.6, 0.7], [0.0, 0.3]]),
        thresholds=np.array([0.5, 0.5]),
    )

    assert metrics["active_atomic_decision_count"] == 3
    assert metrics["unsupported_decision_count"] == 1
    assert np.isclose(metrics["unsupported_decision_rate"], 1.0 / 3.0)
    assert np.isclose(metrics["evidence_decision_consistency"], (0.6 / 0.8 + 1 + 0.3 / 0.9) / 3)
    assert metrics["udr"] == metrics["unsupported_decision_rate"]
    assert metrics["edc"] == metrics["evidence_decision_consistency"]


def test_evidence_decision_metrics_are_undefined_without_active_decisions() -> None:
    metrics = evidence_decision_metrics(
        atomic_scores=np.array([[0.1, 0.2]]),
        evidence_support=np.array([[0.0, 0.0]]),
        thresholds=np.array([0.5, 0.5]),
    )

    assert metrics["active_atomic_decision_count"] == 0
    assert metrics["unsupported_decision_count"] == 0
    assert metrics["unsupported_decision_rate"] is None
    assert metrics["evidence_decision_consistency"] is None


def test_evidence_decision_metrics_use_sample_hypothesis_modality_units() -> None:
    metrics = evidence_decision_metrics(
        atomic_scores=np.array([[0.9], [0.8]]),
        evidence_support=np.array([[[0.95, 0.05]], [[0.4, 0.7]]]),
        thresholds=np.array([0.5]),
        modality_applicability=np.array([[True, True]]),
        modality_availability=np.array([[True, True], [True, False]]),
        modality_names=("image", "text"),
    )

    assert metrics["active_atomic_decision_count"] == 2
    assert metrics["active_evidence_decision_unit_count"] == 3
    assert metrics["unsupported_decision_count"] == 2
    assert np.isclose(metrics["udr"], 2.0 / 3.0)
    assert np.isclose(metrics["edc"], (1 + 0.05 / 0.9 + 0.4 / 0.8) / 3)
    assert metrics["evidence_decision_unit"] == "sample_hypothesis_modality"
    assert metrics["edc_text"] == pytest.approx(0.05 / (0.9 + 1e-8))
