from __future__ import annotations

import os
from pathlib import Path

from src.evaluation.run import resolve_baseline_evaluation_artifacts
from src.training.trainer import _resolve_baseline_resume_checkpoint
from src.utils.config import load_yaml_config, validate_baseline_config


def test_phase2_yaml_configs_validate() -> None:
    files = {
        "config/train_baseline_b0.yaml": "train",
        "config/train_baseline_b1.yaml": "train",
        "config/evaluate_baseline_b0.yaml": "evaluate",
        "config/evaluate_baseline_b1.yaml": "evaluate",
        "config/smoke_baseline_b0.yaml": "smoke",
        "config/smoke_baseline_b1.yaml": "smoke",
    }
    for path, mode in files.items():
        validate_baseline_config(load_yaml_config(path), mode=mode)


def test_controlled_baseline_training_matches_hera_optimization_policy() -> None:
    hera = load_yaml_config("config/train_hera.yaml")
    for path in ("config/train_baseline_b0.yaml", "config/train_baseline_b1.yaml"):
        baseline = load_yaml_config(path)
        assert baseline["training"]["epochs"] == hera["training"]["epochs"] == 20
        assert baseline["training"]["micro_batch_size"] == hera["training"]["micro_batch_size"] == 16
        assert baseline["training"]["gradient_accumulation_steps"] == 1
        assert baseline["training"]["stages"] == hera["training"]["stages"]
        assert baseline["training"]["early_stopping"] == hera["training"]["early_stopping"]
        assert baseline["optimizer"] == hera["optimizer"]
        assert baseline["scheduler"] == hera["scheduler"]
        assert baseline["runtime"]["gradient_checkpointing"] == hera["runtime"]["gradient_checkpointing"]
        assert baseline["checkpointing"] == hera["checkpointing"]
        assert baseline["loss"]["weights"] == {
            "binary": hera["loss"]["lambda_bin"],
            "attribution": hera["loss"]["lambda_attr"],
            "image_grounding": hera["loss"]["lambda_img"],
            "text_grounding": hera["loss"]["lambda_txt"],
        }
        assert "stage_b_unfreeze_last_blocks" not in baseline["training"]["stages"]


def _write_artifact_pair(run_dir: Path, *, timestamp: int) -> tuple[Path, Path]:
    checkpoint = run_dir / "checkpoints" / "best.pt"
    thresholds = run_dir / "thresholds" / "best_thresholds.json"
    checkpoint.parent.mkdir(parents=True)
    thresholds.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    thresholds.write_text("{}", encoding="utf-8")
    os.utime(run_dir, (timestamp, timestamp))
    return checkpoint, thresholds


def test_baseline_evaluation_latest_resolution_does_not_mix_b0_and_b1(tmp_path: Path) -> None:
    b0_checkpoint, b0_thresholds = _write_artifact_pair(
        tmp_path / "b0_global_seed42_older",
        timestamp=1,
    )
    _write_artifact_pair(
        tmp_path / "b1_cross_attention_seed42_newer",
        timestamp=2,
    )
    config = load_yaml_config("config/evaluate_baseline_b0.yaml")
    config["evaluation"]["checkpoint_resolution"]["output_root"] = str(tmp_path)

    assert resolve_baseline_evaluation_artifacts(config) == (b0_checkpoint, b0_thresholds)


def test_baseline_auto_resume_selects_only_matching_variant(tmp_path: Path) -> None:
    b0_last = tmp_path / "b0_global_seed42_older" / "checkpoints" / "last.pt"
    b1_last = tmp_path / "b1_cross_attention_seed42_newer" / "checkpoints" / "last.pt"
    b0_last.parent.mkdir(parents=True)
    b1_last.parent.mkdir(parents=True)
    b0_last.write_bytes(b"b0")
    b1_last.write_bytes(b"b1")
    os.utime(b0_last, (1, 1))
    os.utime(b1_last, (2, 2))
    config = load_yaml_config("config/train_baseline_b0.yaml")
    config["output"]["root"] = str(tmp_path)

    assert _resolve_baseline_resume_checkpoint(config) == b0_last
