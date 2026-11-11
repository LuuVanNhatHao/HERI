from __future__ import annotations

import copy
import csv
from pathlib import Path
from tests.support.paths import REPO_ROOT

import pytest

from heri.experiments.ablation import (
    apply_hera_ablation_variant,
    validate_hera_checkpoint_variant,
)
from heri.experiments.cycle_study import apply_hera_cycle_study_variant
from heri.experiments.scientific_study import METRICS, aggregate_study, materialize_study
from heri.evaluation.protocol import EVALUATION_PROTOCOL
from heri.experiments.samm_matched_multiseed import (
    load_samm_matched_suite,
    materialize_samm_matched_jobs,
)
from heri.utils.config import ConfigError, load_yaml_config, validate_hera_config

ROOT = REPO_ROOT


def test_masked_annotations_do_not_invalidate_complete_run_identity(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.csv"
    row = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "checkpoint_evaluation_protocol": EVALUATION_PROTOCOL,
        "scientific_study_id": "tiny-fixture",
        "scientific_variant": "full_hera",
        "checkpoint_id": "tiny-checkpoint",
        "seed": 7,
        "split": "test",
        "sample_count": 2,
        "text_coverage_complete": False,
        "text_original_word_metadata_complete": True,
        "text_grounding_unavailable_count": 1,
        **{metric: 0.5 for metric in METRICS},
    }
    manifest = {
        "protocol_id": "tiny-fixture",
        "expected_test_samples": 2,
        "seeds": [7],
        "jobs": [{"variant": "full_hera", "seed": 7, "summary": str(summary)}],
    }

    def write():
        with summary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    write()
    assert aggregate_study(manifest)["status"] == "complete"
    row["text_original_word_metadata_complete"] = False
    write()
    with pytest.raises(ValueError, match="original-word"):
        aggregate_study(manifest)


def test_refresh_controls_bind_distinct_mechanisms() -> None:
    canonical = load_yaml_config(ROOT / "config/datasets/dgm4/heri/train.yaml")
    expected = {
        "two_cycle_static": (False, False, "evolved"),
        "semantic_refresh_only": (True, False, "evolved"),
        "hypothesis_refresh_only": (False, True, "evolved"),
        "initial_hypothesis_control": (True, True, "initial"),
        "full_hera": (True, True, "evolved"),
    }
    for variant, flags in expected.items():
        config = apply_hera_cycle_study_variant(canonical, variant)
        validate_hera_config(config, mode="train")
        reasoning = config["graph"]["reasoning"]
        assert (
            tuple(
                reasoning[key]
                for key in (
                    "refresh_semantic",
                    "refresh_hypothesis",
                    "hypothesis_state",
                )
            )
            == flags
        )


def test_checkpoint_rejects_legacy_or_cross_seed() -> None:
    canonical = load_yaml_config(ROOT / "config/datasets/dgm4/heri/train.yaml")
    for key, value in (("architecture_version", "incompatible_architecture"), ("seed", 42)):
        previous = copy.deepcopy(canonical)
        previous["project"][key] = value
        with pytest.raises(ValueError, match="mismatch"):
            validate_hera_checkpoint_variant(canonical, previous, context="test")
    previous = apply_hera_cycle_study_variant(canonical, "semantic_refresh_only")
    current = apply_hera_cycle_study_variant(canonical, "hypothesis_refresh_only")
    with pytest.raises(ValueError, match="mismatch"):
        validate_hera_checkpoint_variant(current, previous, context="test")


def test_rejects_protocol_drift_and_registers_real_controls() -> None:
    canonical = load_yaml_config(ROOT / "config/datasets/dgm4/heri/train.yaml")
    canonical["thresholding"]["enabled"] = True
    with pytest.raises(ConfigError, match="fixed 0.5"):
        validate_hera_config(canonical, mode="train")
    canonical["thresholding"]["enabled"] = False
    assert (
        apply_hera_ablation_variant(canonical, "no_hypothesis_nodes")["graph"][
            "include_hypothesis_nodes"
        ]
        is False
    )
    assert (
        apply_hera_ablation_variant(canonical, "no_evidence_sufficiency")["loss"]["lambda_ev"] == 0
    )
    pooled = apply_hera_ablation_variant(canonical, "evidence_decomposition")
    assert pooled["graph"]["reasoning"]["operator"] == "pooled"
    assert pooled["ablation"]["operationalization"] == "table7_pooled_readout"


def test_paper_suites_materialize_without_results(tmp_path: Path) -> None:
    for name in ("progressive", "components", "matched_components"):
        suite = load_yaml_config(ROOT / f"config/experiments/dgm4/{name}.yaml")["suite"]
        suite["output_root"] = (tmp_path / name).as_posix()
        manifest = materialize_study(suite, ROOT)
        assert manifest["jobs"]
        assert all(not Path(job["summary"]).exists() for job in manifest["jobs"])
    suite = load_samm_matched_suite(ROOT / "config/experiments/samm/matched_multiseed.yaml")
    suite["output_root"] = (tmp_path / "samm").as_posix()
    jobs = materialize_samm_matched_jobs(suite, repo_root=ROOT)
    assert {job.seed for job in jobs} == {7, 21, 101}
    assert sum(job.variant == "two_cycle_static" for job in jobs) == 3
    assert suite["retention"]["delete_best_after_evaluation"] is False
