import numpy as np
import torch

from src.metrics.attribution import attribution_metrics
from src.metrics.detection import binary_detection_metrics
from src.metrics.grounding import text_grounding_metrics


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
