import numpy as np

import pytest

from src.thresholding.tuner import Thresholds, default_thresholds_for_hypotheses, tune_thresholds


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


def test_threshold_tuning_calibrates_image_presence_on_validation_only() -> None:
    config = {
        "enabled": True,
        "binary": {"objective": "macro_f1", "min": .1, "max": .9, "step": .1},
        "attribution": {
            "mode": "per_class", "objective": "f1", "min": .1, "max": .9, "step": .1
        },
        "image_presence": {
            "enabled": True, "objective": "macro_f1", "min": .1, "max": .9, "step": .1
        },
    }
    thresholds = tune_thresholds(
        binary_targets=np.array([0, 0, 1, 1]),
        binary_scores=np.array([.1, .2, .8, .9]),
        atomic_targets=np.array([[0, 0], [0, 0], [1, 0], [0, 1]]),
        atomic_scores=np.array([[.1, .1], [.2, .1], [.8, .1], [.1, .9]]),
        image_presence_targets=np.array([0, 0, 1, 1]),
        image_presence_scores=np.array([.1, .2, .7, .8]),
        config=config,
        progress=False,
        hypothesis_names=("SWAP", "ATTRIBUTE"),
    )
    assert thresholds.image_presence is not None
    assert thresholds.image_presence_source == "validation_tuned_macro_f1"
    assert thresholds.as_dict()["image_presence_threshold"] == thresholds.image_presence


def test_thresholds_are_ontology_aligned() -> None:
    thresholds = default_thresholds_for_hypotheses(["SWAP", "ATTRIBUTE"])
    assert thresholds.atomic == (0.5, 0.5)
    assert thresholds.hypothesis_names == ("SWAP", "ATTRIBUTE")
    with pytest.raises(ValueError, match="equal length"):
        Thresholds(binary=0.5, atomic=(0.5,), hypothesis_names=("SWAP", "ATTRIBUTE"))
