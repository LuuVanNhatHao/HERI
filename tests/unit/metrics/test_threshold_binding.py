import json
from pathlib import Path

import pytest

from heri.evaluation.run import _load_thresholds, apply_decision_threshold_policy
from heri.thresholding.tuner import Thresholds


def test_threshold_file_must_match_checkpoint_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    mismatch = tmp_path / "other.pt"
    checkpoint.write_bytes(b"checkpoint")
    mismatch.write_bytes(b"checkpoint")
    thresholds = tmp_path / "best_thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "binary_threshold": 0.5,
                "fs_threshold": 0.4,
                "fa_threshold": 0.4,
                "ts_threshold": 0.4,
                "ta_threshold": 0.4,
                "checkpoint_path": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    loaded = _load_thresholds(thresholds, checkpoint_path=checkpoint)
    assert loaded.binary == 0.5
    with pytest.raises(ValueError, match="Threshold file does not belong"):
        _load_thresholds(thresholds, checkpoint_path=mismatch)


def test_fixed_policy_overrides_bound_legacy_values_after_loading() -> None:
    historical = Thresholds(
        binary=0.37,
        atomic=(0.42, 0.61),
        hypothesis_names=("SWAP", "ATTRIBUTE"),
        image_presence=0.44,
        image_presence_source="validation_tuned_macro_f1",
        decision_source="legacy_validation_tuned",
    )
    fixed = apply_decision_threshold_policy(
        historical,
        {"thresholding": {"enabled": False, "image_presence": {"enabled": True}}},
    )
    assert fixed.binary == 0.5
    assert fixed.atomic == (0.5, 0.5)
    assert fixed.image_presence == 0.5
    assert fixed.decision_source == "fixed_0.5"
