"""Alignment objective and source identity remain bound across run artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from heri.experiments.ablation import validate_hera_checkpoint_variant
from heri.experiments.provenance import canonical_config_sha256, checkpoint_metadata_path
from heri.training.checkpoint import save_checkpoint
from heri.training.state import TrainingState
from heri.utils.config import ConfigError, load_yaml_config, validate_hera_config
from heri.utils.hashing import sha256_file, source_tree_sha256
from tests.support.paths import REPO_ROOT


@pytest.mark.parametrize("previous_objective", [None, "valid_visual_mean"])
def test_checkpoint_cannot_silently_change_alignment_objective(previous_objective):
    current = load_yaml_config(REPO_ROOT / "config/datasets/dgm4/heri/train.yaml")
    previous = copy.deepcopy(current)
    if previous_objective is None:
        del previous["loss"]["alignment"]
    else:
        previous["loss"]["alignment"]["aggregation"] = previous_objective
    with pytest.raises(ValueError, match="alignment-objective mismatch"):
        validate_hera_checkpoint_variant(current, previous, context="Resume")
    validate_hera_checkpoint_variant(current, current, context="Resume")


def test_unknown_alignment_reduction_is_rejected():
    config = load_yaml_config(REPO_ROOT / "config/datasets/dgm4/heri/train.yaml")
    config["loss"]["alignment"]["aggregation"] = "valid_visual_mean"
    with pytest.raises(ConfigError, match="valid_text_mean"):
        validate_hera_config(config, mode="train")


def test_checkpoint_sidecar_binds_actual_bytes_and_survives_interrupted_write(
    tmp_path, monkeypatch
):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "train/checkpoints/best.pt"
    path.parent.mkdir(parents=True)
    (path.parent.parent / "environment.json").write_text(
        json.dumps({"source_sha256": "fixture-source", "git_commit": "fixture-commit"})
    )
    config = {"loss": {"alignment": {"aggregation": "valid_text_mean"}}}
    kwargs = dict(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        state=TrainingState(),
        metrics={},
        config=config,
        precision="fp32",
    )
    checkpoint_id = save_checkpoint(**kwargs)
    metadata_path = checkpoint_metadata_path(path)
    previous_bytes, previous_metadata = path.read_bytes(), metadata_path.read_bytes()
    metadata = json.loads(previous_metadata)
    assert metadata["checkpoint_id"] == checkpoint_id
    assert metadata["checkpoint_sha256"] == sha256_file(path, progress=False)
    assert metadata["effective_config_sha256"] == canonical_config_sha256(config)
    assert metadata["effective_config"] == config
    assert metadata["alignment_objective"] == "valid_text_mean"
    assert metadata["source_sha256"] == "fixture-source"

    def interrupted_save(payload, destination):
        Path(destination).write_bytes(b"partial checkpoint")
        raise OSError("interrupted")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        save_checkpoint(**kwargs)
    assert path.read_bytes() == previous_bytes
    assert metadata_path.read_bytes() == previous_metadata
    assert not list(path.parent.glob("*.tmp"))


def test_source_identity_is_shared_and_independent_of_checkout_line_endings(tmp_path):
    from heri.experiments.scientific_study import source_digest

    path = tmp_path / "src/example.py"
    path.parent.mkdir()
    path.write_bytes(b"value = 1\n")
    expected = source_tree_sha256(tmp_path)
    path.write_bytes(b"value = 1\r\n")
    assert source_tree_sha256(tmp_path) == expected == source_digest(tmp_path)
    path.write_bytes(b"value = 2\n")
    assert source_digest(tmp_path) != expected
