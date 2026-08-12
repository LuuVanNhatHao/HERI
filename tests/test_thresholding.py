import numpy as np

from src.thresholding.tuner import tune_thresholds


def test_threshold_tuning_uses_validation_arrays() -> None:
    thresholds = tune_thresholds(
        binary_targets=np.array([0, 0, 1, 1]),
        binary_scores=np.array([0.1, 0.4, 0.6, 0.9]),
        atomic_targets=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]]),
        atomic_scores=np.array([[0.8, 0.2, 0.2, 0.1], [0.2, 0.9, 0.1, 0.1], [0.1, 0.1, 0.8, 0.9]]),
        config={
            "enabled": True,
            "binary": {"objective": "macro_f1", "min": 0.1, "max": 0.9, "step": 0.1},
            "attribution": {"mode": "per_class", "objective": "f1", "min": 0.1, "max": 0.9, "step": 0.1},
        },
        progress=False,
    )
    assert 0.1 <= thresholds.binary <= 0.9
    assert len(thresholds.atomic) == 4
