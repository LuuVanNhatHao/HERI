from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.experiments.ablation import (
    HERA_ABLATION_VARIANTS,
    apply_hera_ablation_variant,
    validate_hera_checkpoint_variant,
    validate_hera_ablation_contract,
)
from src.experiments.ablation_runner import (
    aggregate_ablation_summaries,
    load_ablation_suite,
    materialize_ablation_jobs,
    run_ablation_jobs,
)
from src.utils.config import ConfigError, load_yaml_config, validate_hera_config


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "config" / "experiments" / "dgm4" / "ablation.yaml"


@pytest.mark.parametrize("variant", HERA_ABLATION_VARIANTS)
def test_each_registered_ablation_is_a_valid_single_change(variant: str) -> None:
    canonical = load_yaml_config(ROOT / "config/datasets/dgm4/hera/train.yaml")
    untouched = copy.deepcopy(canonical)

    effective = apply_hera_ablation_variant(canonical, variant)

    assert canonical == untouched
    assert effective["ablation"] == {"enabled": True, "variant": variant}
    validate_hera_ablation_contract(effective)
    validate_hera_config(effective, mode="train")


def test_ablation_registry_explicitly_excludes_b0_and_b1() -> None:
    suite = load_ablation_suite(SUITE_PATH)
    assert tuple(suite["variants"]) == HERA_ABLATION_VARIANTS
    assert not any(name in {"b0", "b1", "b0_global", "b1_cross_attention"} for name in suite["variants"])

    baseline = load_yaml_config(
        ROOT / "config" / "datasets" / "dgm4" / "baselines" / "b0" / "train.yaml"
    )
    with pytest.raises(ValueError, match="B0/B1"):
        apply_hera_ablation_variant(baseline, "no_router")


def test_suite_materializes_isolated_full_train_and_evaluation_configs(tmp_path: Path) -> None:
    suite = load_ablation_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "ablation")

    jobs = materialize_ablation_jobs(
        suite,
        repo_root=ROOT,
        selected_variants=["no_router", "independent_binary"],
    )

    assert [job.variant for job in jobs] == ["no_router", "independent_binary"]
    assert jobs[0].train_output_root != jobs[1].train_output_root
    for job in jobs:
        train_config = load_yaml_config(job.train_config_path)
        evaluate_config = load_yaml_config(job.evaluate_config_path)
        validate_hera_config(train_config, mode="train")
        validate_hera_config(evaluate_config, mode="evaluate")
        assert train_config["training"]["auto_resume"]["enabled"] is True
        assert train_config["training"]["epochs"] == 12
        assert evaluate_config["training"]["epochs"] == 12
        assert evaluate_config["evaluation"]["checkpoint_resolution"]["output_root"] == train_config["output"]["root"]
        assert evaluate_config["ablation"]["variant"] == job.variant
        assert train_config["ablation"]["variant"] == job.variant


def test_suite_rejects_variant_not_enabled_by_manifest(tmp_path: Path) -> None:
    suite = load_ablation_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "ablation")
    with pytest.raises(ConfigError, match="not enabled"):
        materialize_ablation_jobs(suite, repo_root=ROOT, selected_variants=["b0_global"])


def test_checkpoint_variant_mismatch_is_rejected() -> None:
    canonical = load_yaml_config(ROOT / "config/datasets/dgm4/hera/train.yaml")
    no_router = apply_hera_ablation_variant(canonical, "no_router")
    gatv2 = apply_hera_ablation_variant(canonical, "gatv2")

    validate_hera_checkpoint_variant(no_router, no_router, context="fixture")
    with pytest.raises(ValueError, match="variant mismatch"):
        validate_hera_checkpoint_variant(no_router, gatv2, context="fixture")

    legacy = copy.deepcopy(no_router)
    legacy["project"].pop("architecture_version")
    with pytest.raises(ValueError, match="architecture mismatch"):
        validate_hera_checkpoint_variant(no_router, legacy, context="fixture")


def test_summary_aggregation_preserves_variant_identity(tmp_path: Path) -> None:
    suite = load_ablation_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "ablation")
    jobs = materialize_ablation_jobs(
        suite,
        repo_root=ROOT,
        selected_variants=["no_router"],
    )
    summary = jobs[0].expected_summary_path
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("ablation_variant,acc,cf1\nno_router,0.91,0.88\n", encoding="utf-8")

    aggregate = aggregate_ablation_summaries(
        jobs,
        output_path=tmp_path / "ablation_summary.csv",
    )

    assert aggregate is not None
    assert aggregate.read_text(encoding="utf-8").splitlines() == [
        "ablation_variant,acc,cf1",
        "no_router,0.91,0.88",
    ]


def test_completed_phases_are_idempotently_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_ablation_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "ablation")
    job = materialize_ablation_jobs(
        suite,
        repo_root=ROOT,
        selected_variants=["no_router"],
    )[0]
    run_dir = job.train_output_root / "hera_seed42_fixture"
    best = run_dir / "checkpoints" / "best.pt"
    best.parent.mkdir(parents=True, exist_ok=True)
    best.write_bytes(b"fixture")
    (run_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "final_epoch": 8,
                "stop_reason": "early_stopping",
                "best_checkpoint": str(best),
                "ablation_variant": "no_router",
            }
        ),
        encoding="utf-8",
    )
    job.expected_summary_path.parent.mkdir(parents=True, exist_ok=True)
    job.expected_summary_path.write_text(
        "ablation_variant,acc\nno_router,0.91\n",
        encoding="utf-8",
    )

    def fail_if_launched(*args, **kwargs):
        raise AssertionError("Completed train/evaluate phases must not launch subprocesses")

    monkeypatch.setattr("src.experiments.ablation_runner.subprocess.run", fail_if_launched)
    summaries = run_ablation_jobs([job], repo_root=ROOT, phase="all")

    assert summaries == [job.expected_summary_path]
