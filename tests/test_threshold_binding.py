import json
from pathlib import Path

import pytest

from src.evaluation.run import _load_thresholds


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
