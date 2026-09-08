"""Matched SAMM multi-seed orchestration for HERA, B1, and key ablations."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.data.samm import SAMM_EXPECTED_SHA256
from src.evaluation.protocol import EVALUATION_PROTOCOL
from src.experiments.ablation import (
    apply_hera_ablation_variant,
    get_hera_ablation_variant,
)
from src.utils.config import (
    ConfigError,
    dump_yaml,
    load_yaml_config,
    validate_baseline_config,
    validate_hera_config,
)


SAMM_MATCHED_VARIANTS: tuple[str, ...] = (
    "hera",
    "b1",
    "no_router",
    "generic_grounding",
)

_MATCHED_TRAINING_SECTIONS: tuple[str, ...] = (
    "dataset",
    "dataset_hashes",
    "image",
    "tokenizer",
    "backbones",
    "optimizer",
    "scheduler",
    "thresholding",
    "dataloader",
)


@dataclass(frozen=True)
class SAMMMatchedJob:
    """One variant/seed job and its isolated production artifacts."""

    variant: str
    seed: int
    train_config_path: Path
    evaluate_config_path: Path
    train_output_root: Path
    train_run_dir: Path
    evaluation_output_root: Path
    evaluation_run_dir: Path
    expected_summary_path: Path
    reusable_summary_path: Path | None = None


def load_samm_matched_suite(path: str | Path) -> dict[str, Any]:
    """Load and validate the prespecified matched SAMM suite."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ConfigError(f"SAMM matched-multiseed suite does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("suite"), dict):
        raise ConfigError("SAMM matched-multiseed YAML requires a top-level suite mapping")
    suite = dict(payload["suite"])
    mandatory = (
        "name",
        "protocol_id",
        "hera_train_config",
        "hera_evaluate_config",
        "b1_train_config",
        "b1_evaluate_config",
        "output_root",
        "variants",
        "seeds",
        "aggregate_metrics",
    )
    for key in mandatory:
        if key not in suite:
            raise ConfigError(f"Missing mandatory SAMM multiseed key: suite.{key}")

    variants = [str(value) for value in _validated_sequence(suite["variants"], "suite.variants")]
    if tuple(variants) != SAMM_MATCHED_VARIANTS:
        raise ConfigError(
            "SAMM matched suite must contain exactly the prespecified variants in order: "
            f"{list(SAMM_MATCHED_VARIANTS)}"
        )
    seeds = _validated_seeds(suite["seeds"])
    if len(seeds) != 3:
        raise ConfigError("SAMM matched suite requires exactly three training seeds")
    metrics = [
        str(value) for value in _validated_sequence(
            suite["aggregate_metrics"], "suite.aggregate_metrics"
        )
    ]
    if len(metrics) != len(set(metrics)):
        raise ConfigError("suite.aggregate_metrics must not contain duplicates")
    if int(suite.get("expected_test_samples", 50_000)) != 50_000:
        raise ConfigError("Canonical SAMM matched evaluation requires exactly 50,000 test samples")

    retention = suite.get("retention", {})
    if not isinstance(retention, Mapping):
        raise ConfigError("suite.retention must be a mapping")
    minimum_free = float(retention.get("minimum_free_gib_before_train", 0.0))
    if minimum_free < 0:
        raise ConfigError("retention.minimum_free_gib_before_train must be non-negative")

    suite["variants"] = variants
    suite["seeds"] = seeds
    suite["aggregate_metrics"] = metrics
    suite["expected_test_samples"] = 50_000
    return suite


def materialize_samm_matched_jobs(
    suite: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> list[SAMMMatchedJob]:
    """Create complete isolated train/evaluation configs for all 12 jobs."""

    root = Path(repo_root).resolve()
    hera_train = load_yaml_config(_resolve_path(root, suite["hera_train_config"]))
    hera_evaluate = load_yaml_config(_resolve_path(root, suite["hera_evaluate_config"]))
    b1_train = load_yaml_config(_resolve_path(root, suite["b1_train_config"]))
    b1_evaluate = load_yaml_config(_resolve_path(root, suite["b1_evaluate_config"]))
    _validate_base_configs(hera_train, hera_evaluate, b1_train, b1_evaluate)

    output_root = _resolve_path(root, suite["output_root"])
    jobs: list[SAMMMatchedJob] = []
    for seed in suite["seeds"]:
        for variant in suite["variants"]:
            train, evaluate = _variant_configs(
                variant,
                hera_train=hera_train,
                hera_evaluate=hera_evaluate,
                b1_train=b1_train,
                b1_evaluate=b1_evaluate,
            )
            job_root = output_root / f"seed_{seed}" / variant
            config_root = job_root / "configs"
            train_output_root = job_root / "train"
            evaluation_output_root = job_root / "evaluation"
            train_run_id = _train_run_id(variant, seed)
            evaluation_run_id = f"samm_{variant}_seed{seed}_test"
            train_run_dir = train_output_root / train_run_id
            evaluation_run_dir = evaluation_output_root / evaluation_run_id
            _bind_job_protocol(
                train,
                evaluate,
                suite=suite,
                variant=variant,
                seed=seed,
                train_run_id=train_run_id,
                evaluation_run_id=evaluation_run_id,
                train_output_root=_portable_path(train_output_root, root),
                evaluation_output_root=_portable_path(evaluation_output_root, root),
                checkpoint_path=_portable_path(
                    train_run_dir / "checkpoints" / "best.pt", root
                ),
                thresholds_path=_portable_path(
                    train_run_dir / "thresholds" / "best_thresholds.json", root
                ),
            )
            _validate_variant_config(train, variant=variant, mode="train")
            _validate_variant_config(evaluate, variant=variant, mode="evaluate")
            config_root.mkdir(parents=True, exist_ok=True)
            train_config_path = config_root / "train.yaml"
            evaluate_config_path = config_root / "evaluate.yaml"
            dump_yaml(train, train_config_path)
            dump_yaml(evaluate, evaluate_config_path)
            jobs.append(
                SAMMMatchedJob(
                    variant=variant,
                    seed=seed,
                    train_config_path=train_config_path,
                    evaluate_config_path=evaluate_config_path,
                    train_output_root=train_output_root,
                    train_run_dir=train_run_dir,
                    evaluation_output_root=evaluation_output_root,
                    evaluation_run_dir=evaluation_run_dir,
                    expected_summary_path=evaluation_run_dir / "summary.csv",
                    reusable_summary_path=_reusable_summary_path(
                        suite, root=root, variant=variant, seed=seed
                    ),
                )
            )
    return jobs


def select_samm_matched_jobs(
    jobs: Sequence[SAMMMatchedJob],
    *,
    variants: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[SAMMMatchedJob]:
    """Select a debug/recovery subset while preserving suite order."""

    requested_variants = set(str(value) for value in variants or SAMM_MATCHED_VARIANTS)
    unknown_variants = requested_variants - set(SAMM_MATCHED_VARIANTS)
    if unknown_variants:
        raise ConfigError(f"Unknown SAMM matched variants: {sorted(unknown_variants)}")
    configured_seeds = {job.seed for job in jobs}
    requested_seeds = set(_validated_seeds(seeds)) if seeds else configured_seeds
    unknown_seeds = requested_seeds - configured_seeds
    if unknown_seeds:
        raise ConfigError(f"Requested seeds are absent from the suite: {sorted(unknown_seeds)}")
    return [
        job for job in jobs
        if job.variant in requested_variants and job.seed in requested_seeds
    ]


def run_samm_matched_jobs(
    jobs: Sequence[SAMMMatchedJob],
    *,
    suite: Mapping[str, Any],
    repo_root: str | Path,
    phase: str = "all",
    fail_fast: bool = True,
    keep_checkpoints: bool = False,
) -> list[Path]:
    """Run jobs sequentially with reuse, resume, and post-evaluation pruning."""

    if phase not in {"all", "train", "evaluate"}:
        raise ValueError("phase must be all, train, or evaluate")
    root = Path(repo_root).resolve()
    environment = os.environ.copy()
    for key, value in dict(suite.get("environment", {})).items():
        environment[str(key)] = str(value)

    failures: list[str] = []
    for job in jobs:
        label = f"variant={job.variant} seed={job.seed}"
        try:
            if _adopt_reusable_summary(job, suite=suite):
                print(f"[{label}] adopted validated existing test summary", flush=True)

            if job.expected_summary_path.is_file():
                _validate_summary(job.expected_summary_path, job=job, suite=suite)
                _validate_prediction_csv(
                    _prediction_path(job),
                    expected_samples=int(suite["expected_test_samples"]),
                )
                print(f"[{label}] complete summary exists; skipping", flush=True)
                if phase in {"all", "evaluate"} and not keep_checkpoints:
                    _prune_job_checkpoints(job, suite=suite)
                continue

            if phase in {"all", "train"}:
                if _training_is_complete(job, suite=suite):
                    print(f"[{label}] train already complete; skipping", flush=True)
                else:
                    _require_free_disk(root, suite=suite, label=label)
                    _run_subprocess(job, "train", root=root, environment=environment)
                    if not _training_is_complete(job, suite=suite):
                        raise RuntimeError("training exited without a valid completion artifact")

            if phase in {"all", "evaluate"}:
                if not _training_is_complete(job, suite=suite):
                    raise RuntimeError("evaluation requires a complete matching training run")
                _run_subprocess(job, "evaluate", root=root, environment=environment)
                _validate_summary(job.expected_summary_path, job=job, suite=suite)
                _validate_prediction_csv(
                    _prediction_path(job),
                    expected_samples=int(suite["expected_test_samples"]),
                )
                if not keep_checkpoints:
                    _prune_job_checkpoints(job, suite=suite)
        except Exception as exc:
            failure = f"{label}: {exc}"
            failures.append(failure)
            if fail_fast:
                raise RuntimeError(failure) from exc
            print(f"[{label}] FAILED: {exc}", file=sys.stderr, flush=True)

    if failures:
        raise RuntimeError("SAMM matched-multiseed failures: " + "; ".join(failures))
    return [job.expected_summary_path for job in jobs if job.expected_summary_path.is_file()]


def aggregate_samm_matched_results(
    jobs: Iterable[SAMMMatchedJob],
    *,
    suite: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Path] | None:
    """Write per-seed, per-variant, and paired HERA-minus-control tables."""

    job_list = list(jobs)
    result_rows: list[dict[str, Any]] = []
    field_order = ["multiseed_variant", "result_origin", "seed"]
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for job in job_list:
        if not job.expected_summary_path.is_file():
            continue
        row = _validate_summary(job.expected_summary_path, job=job, suite=suite)
        origin = "reused" if (job.evaluation_run_dir / "adopted_summary.json").is_file() else "generated"
        normalized = {
            **row,
            "multiseed_variant": job.variant,
            "result_origin": origin,
            "seed": job.seed,
        }
        result_rows.append(normalized)
        indexed[(job.variant, job.seed)] = normalized
        for key in normalized:
            if key not in field_order:
                field_order.append(key)
    if not result_rows:
        return None

    destination = Path(output_root)
    seed_results = destination / "seed_results.csv"
    _write_csv_atomic(seed_results, result_rows, field_order)

    configured_seeds = list(suite["seeds"])
    metrics = [str(value) for value in suite["aggregate_metrics"]]
    statistics_rows: list[dict[str, Any]] = []
    for variant in suite["variants"]:
        for metric in metrics:
            values = [
                _finite_float(indexed.get((variant, seed), {}).get(metric))
                for seed in configured_seeds
            ]
            numeric = [value for value in values if value is not None]
            if not numeric:
                continue
            statistics_rows.append(
                _statistics_row(
                    variant=variant,
                    metric=metric,
                    values=numeric,
                    expected_n=len(configured_seeds),
                )
            )
    statistics_path = destination / "variant_statistics.csv"
    _write_csv_atomic(
        statistics_path,
        statistics_rows,
        ("variant", "metric", "n", "expected_n", "complete", "mean", "std", "min", "max"),
    )

    delta_rows: list[dict[str, Any]] = []
    paired_statistics: list[dict[str, Any]] = []
    for control in [variant for variant in suite["variants"] if variant != "hera"]:
        for metric in metrics:
            deltas: list[float] = []
            for seed in configured_seeds:
                hera_value = _finite_float(indexed.get(("hera", seed), {}).get(metric))
                control_value = _finite_float(indexed.get((control, seed), {}).get(metric))
                if hera_value is None or control_value is None:
                    continue
                delta = hera_value - control_value
                deltas.append(delta)
                delta_rows.append(
                    {
                        "control_variant": control,
                        "seed": seed,
                        "metric": metric,
                        "hera_value": hera_value,
                        "control_value": control_value,
                        "hera_minus_control": delta,
                    }
                )
            if deltas:
                paired_statistics.append(
                    _statistics_row(
                        variant=control,
                        metric=metric,
                        values=deltas,
                        expected_n=len(configured_seeds),
                    )
                )
    delta_path = destination / "paired_seed_deltas.csv"
    _write_csv_atomic(
        delta_path,
        delta_rows,
        ("control_variant", "seed", "metric", "hera_value", "control_value", "hera_minus_control"),
    )
    paired_path = destination / "paired_statistics.csv"
    _write_csv_atomic(
        paired_path,
        paired_statistics,
        ("variant", "metric", "n", "expected_n", "complete", "mean", "std", "min", "max"),
    )
    return {
        "seed_results": seed_results,
        "variant_statistics": statistics_path,
        "paired_seed_deltas": delta_path,
        "paired_statistics": paired_path,
    }


def _variant_configs(
    variant: str,
    *,
    hera_train: Mapping[str, Any],
    hera_evaluate: Mapping[str, Any],
    b1_train: Mapping[str, Any],
    b1_evaluate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if variant == "b1":
        return copy.deepcopy(dict(b1_train)), copy.deepcopy(dict(b1_evaluate))
    train = copy.deepcopy(dict(hera_train))
    evaluate = copy.deepcopy(dict(hera_evaluate))
    if variant in {"no_router", "generic_grounding"}:
        train = apply_hera_ablation_variant(train, variant)
        evaluate = apply_hera_ablation_variant(evaluate, variant)
    return train, evaluate


def _bind_job_protocol(
    train: dict[str, Any],
    evaluate: dict[str, Any],
    *,
    suite: Mapping[str, Any],
    variant: str,
    seed: int,
    train_run_id: str,
    evaluation_run_id: str,
    train_output_root: str,
    evaluation_output_root: str,
    checkpoint_path: str,
    thresholds_path: str,
) -> None:
    study = {
        "protocol_id": str(suite["protocol_id"]),
        "variant": variant,
        "seed": seed,
    }
    for config, mode in ((train, "train"), (evaluate, "evaluate")):
        config["project"]["name"] = f"HERA-SAMM-multiseed-{variant}"
        config["project"]["seed"] = seed
        config["project"]["config_name"] = f"samm_multiseed/{variant}/seed_{seed}/{mode}"
        config["dataloader"]["seed"] = seed
        config["training"]["resume_from"] = None
        config["scientific_study"] = copy.deepcopy(study)
    train["project"]["run_id"] = train_run_id
    train["output"]["root"] = train_output_root
    train["training"]["auto_resume"]["enabled"] = True
    train["training"]["auto_resume"]["run_dir"] = None
    train["checkpointing"]["keep_best"] = True
    train["checkpointing"]["keep_last"] = True
    train["checkpointing"]["top_k"] = 0

    evaluate["project"]["run_id"] = evaluation_run_id
    evaluate["output"]["root"] = evaluation_output_root
    evaluate["training"]["epochs"] = int(train["training"]["epochs"])
    evaluate["evaluation"]["split"] = "test"
    evaluate["evaluation"]["checkpoint_path"] = checkpoint_path
    evaluate["evaluation"]["thresholds_path"] = thresholds_path


def _validate_base_configs(
    hera_train: Mapping[str, Any],
    hera_evaluate: Mapping[str, Any],
    b1_train: Mapping[str, Any],
    b1_evaluate: Mapping[str, Any],
) -> None:
    validate_hera_config(hera_train, mode="train")
    validate_hera_config(hera_evaluate, mode="evaluate")
    validate_baseline_config(b1_train, mode="train")
    validate_baseline_config(b1_evaluate, mode="evaluate")
    for label, config in (
        ("HERA train", hera_train),
        ("HERA evaluate", hera_evaluate),
        ("B1 train", b1_train),
        ("B1 evaluate", b1_evaluate),
    ):
        if str(config["dataset"]["name"]).upper() != "SAMM":
            raise ConfigError(f"{label} is not a SAMM config")
        if config.get("dataset_hashes", {}) != SAMM_EXPECTED_SHA256:
            raise ConfigError(f"{label} does not use the frozen SAMM split hashes")
    for section in _MATCHED_TRAINING_SECTIONS:
        if hera_train.get(section) != b1_train.get(section):
            raise ConfigError(f"HERA and B1 differ in matched training section {section!r}")
    for dotted in (
        "training.epochs",
        "training.micro_batch_size",
        "training.gradient_accumulation_steps",
        "training.early_stopping",
        "training.stages",
    ):
        if _get(hera_train, dotted) != _get(b1_train, dotted):
            raise ConfigError(f"HERA and B1 differ in matched setting {dotted!r}")


def _validate_variant_config(config: Mapping[str, Any], *, variant: str, mode: str) -> None:
    if variant == "b1":
        validate_baseline_config(config, mode=mode)
        if config["model"]["baseline_type"] != "b1_cross_attention":
            raise ConfigError("Matched B1 variant must use b1_cross_attention")
        return
    validate_hera_config(config, mode=mode)
    actual = get_hera_ablation_variant(config)
    expected = variant if variant in {"no_router", "generic_grounding"} else None
    if actual != expected:
        raise ConfigError(
            f"Matched variant identity mismatch: requested={variant!r}, configured={actual!r}"
        )


def _run_subprocess(
    job: SAMMMatchedJob,
    phase: str,
    *,
    root: Path,
    environment: Mapping[str, str],
) -> None:
    config = job.train_config_path if phase == "train" else job.evaluate_config_path
    command = [sys.executable, str(root / "scripts" / f"{phase}.py"), str(config)]
    print(
        f"[variant={job.variant} seed={job.seed}] {phase}: {' '.join(command)}",
        flush=True,
    )
    completed = subprocess.run(command, cwd=root, env=dict(environment), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{phase} subprocess exited with code {completed.returncode}")


def _training_is_complete(
    job: SAMMMatchedJob,
    *,
    suite: Mapping[str, Any],
) -> bool:
    marker = job.train_run_dir / "training_complete.json"
    best = job.train_run_dir / "checkpoints" / "best.pt"
    thresholds = job.train_run_dir / "thresholds" / "best_thresholds.json"
    effective_path = job.train_run_dir / "effective_config.yaml"
    if not all(path.is_file() for path in (marker, best, thresholds, effective_path)):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        effective = load_yaml_config(effective_path)
    except (OSError, json.JSONDecodeError, ConfigError):
        return False
    if payload.get("status") != "complete":
        return False
    if int(effective.get("project", {}).get("seed", -1)) != job.seed:
        return False
    if int(effective.get("dataloader", {}).get("seed", -1)) != job.seed:
        return False
    expected_study = {
        "protocol_id": str(suite["protocol_id"]),
        "variant": job.variant,
        "seed": job.seed,
    }
    if effective.get("scientific_study") != expected_study:
        return False
    try:
        _validate_variant_config(effective, variant=job.variant, mode="train")
    except (ConfigError, KeyError, TypeError, ValueError):
        return False
    return True


def _adopt_reusable_summary(job: SAMMMatchedJob, *, suite: Mapping[str, Any]) -> bool:
    if job.expected_summary_path.is_file() or job.reusable_summary_path is None:
        return False
    if not job.reusable_summary_path.is_file():
        return False
    _validate_summary(
        job.reusable_summary_path,
        job=job,
        suite=suite,
        allow_legacy_study=True,
    )
    source_prediction = (
        job.reusable_summary_path.parent / "predictions" / "test_predictions.csv"
    )
    _validate_prediction_csv(
        source_prediction,
        expected_samples=int(suite["expected_test_samples"]),
    )
    job.expected_summary_path.parent.mkdir(parents=True, exist_ok=True)
    destination_prediction = _prediction_path(job)
    destination_prediction.parent.mkdir(parents=True, exist_ok=True)
    temporary_prediction = destination_prediction.with_name(
        ".test_predictions.csv.tmp"
    )
    shutil.copyfile(source_prediction, temporary_prediction)
    temporary_prediction.replace(destination_prediction)
    marker = job.evaluation_run_dir / "adopted_summary.json"
    marker_temporary = marker.with_name(".adopted_summary.json.tmp")
    marker_temporary.write_text(
        json.dumps(
            {
                "source": str(job.reusable_summary_path),
                "source_prediction": str(source_prediction),
                "variant": job.variant,
                "seed": job.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    marker_temporary.replace(marker)
    temporary = job.expected_summary_path.with_name(".summary.csv.tmp")
    shutil.copyfile(job.reusable_summary_path, temporary)
    temporary.replace(job.expected_summary_path)
    return True


def _validate_summary(
    path: Path,
    *,
    job: SAMMMatchedJob,
    suite: Mapping[str, Any],
    allow_legacy_study: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing SAMM test summary: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {path}, got {len(rows)}")
    row = rows[0]
    if row.get("dataset_name", "").upper() != "SAMM":
        raise ValueError(f"Summary is not SAMM: {path}")
    if row.get("split") != "test":
        raise ValueError(f"Summary is not a test result: {path}")
    if int(row.get("sample_count", 0)) != int(suite["expected_test_samples"]):
        raise ValueError(f"Summary does not contain the full 50,000-sample SAMM test split: {path}")
    if int(row.get("seed", -1)) != job.seed:
        raise ValueError(f"Summary seed mismatch for {job.variant}: {path}")
    if row.get("checkpoint_evaluation_protocol") != EVALUATION_PROTOCOL:
        raise ValueError(f"Summary uses a non-current evaluation protocol: {path}")
    try:
        hashes = json.loads(row.get("dataset_hashes", "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Summary has malformed dataset_hashes: {path}") from exc
    if hashes != SAMM_EXPECTED_SHA256:
        raise ValueError(f"Summary SAMM split hashes do not match the frozen protocol: {path}")
    adopted = (job.evaluation_run_dir / "adopted_summary.json").is_file()
    if not (allow_legacy_study or adopted):
        if row.get("scientific_study_id") != str(suite["protocol_id"]):
            raise ValueError(f"Summary scientific-study protocol mismatch: {path}")
        if row.get("scientific_variant") != job.variant:
            raise ValueError(f"Summary scientific-study variant mismatch: {path}")
    if job.variant == "b1":
        if row.get("baseline_type") != "b1_cross_attention":
            raise ValueError(f"Summary is not the matched B1 baseline: {path}")
    elif job.variant == "hera":
        if row.get("model_variant") != "hera" or row.get("ablation_variant") not in {
            "",
            "canonical",
            None,
        }:
            raise ValueError(f"Summary is not canonical HERA: {path}")
    elif row.get("ablation_variant") != job.variant:
        raise ValueError(f"Summary ablation mismatch for {job.variant}: {path}")
    return row


def _prediction_path(job: SAMMMatchedJob) -> Path:
    return job.evaluation_run_dir / "predictions" / "test_predictions.csv"


def _validate_prediction_csv(path: Path, *, expected_samples: int) -> None:
    """Reject missing or truncated prediction artifacts before checkpoint pruning."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing SAMM test prediction artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or "id" not in header:
            raise ValueError(f"Malformed SAMM prediction CSV header: {path}")
        rows = sum(1 for _ in reader)
    if rows != expected_samples:
        raise ValueError(
            f"SAMM prediction CSV has {rows} rows; expected {expected_samples}: {path}"
        )


def _prune_job_checkpoints(job: SAMMMatchedJob, *, suite: Mapping[str, Any]) -> None:
    retention = dict(suite.get("retention", {}))
    targets: list[Path] = []
    if bool(retention.get("delete_last_after_evaluation", True)):
        targets.append(job.train_run_dir / "checkpoints" / "last.pt")
    if bool(retention.get("delete_best_after_evaluation", False)):
        targets.append(job.train_run_dir / "checkpoints" / "best.pt")
    freed = 0
    checkpoint_root = (job.train_run_dir / "checkpoints").resolve()
    for target in targets:
        resolved = target.resolve()
        try:
            resolved.relative_to(checkpoint_root)
        except ValueError as exc:
            raise RuntimeError(f"Refusing to prune checkpoint outside job root: {target}") from exc
        if resolved.is_file():
            freed += resolved.stat().st_size
            resolved.unlink()
            print(
                f"[variant={job.variant} seed={job.seed}] pruned {resolved}",
                flush=True,
            )
    if freed:
        print(
            f"[variant={job.variant} seed={job.seed}] freed {freed / 1024**3:.2f} GiB",
            flush=True,
        )


def _require_free_disk(root: Path, *, suite: Mapping[str, Any], label: str) -> None:
    minimum = float(
        dict(suite.get("retention", {})).get("minimum_free_gib_before_train", 0.0)
    )
    available = shutil.disk_usage(root).free / 1024**3
    if available < minimum:
        raise RuntimeError(
            f"only {available:.2f} GiB free; {minimum:.2f} GiB required before training"
        )
    print(f"[{label}] free disk before train: {available:.2f} GiB", flush=True)


def _statistics_row(
    *,
    variant: str,
    metric: str,
    values: Sequence[float],
    expected_n: int,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "metric": metric,
        "n": len(values),
        "expected_n": expected_n,
        "complete": len(values) == expected_n,
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else "",
        "min": min(values),
        "max": max(values),
    }


def _reusable_summary_path(
    suite: Mapping[str, Any],
    *,
    root: Path,
    variant: str,
    seed: int,
) -> Path | None:
    reusable = suite.get("reuse_summaries", {})
    if not isinstance(reusable, Mapping):
        raise ConfigError("suite.reuse_summaries must be a mapping")
    by_variant = reusable.get(variant, {})
    if not isinstance(by_variant, Mapping):
        raise ConfigError(f"suite.reuse_summaries.{variant} must be a mapping")
    value = by_variant.get(seed, by_variant.get(str(seed)))
    if value in (None, ""):
        return None
    return _resolve_path(root, value)


def _train_run_id(variant: str, seed: int) -> str:
    if variant == "b1":
        return f"b1_cross_attention_seed{seed}_train"
    return f"hera_{variant}_seed{seed}_train"


def _validated_sequence(values: Any, context: str) -> list[Any]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ConfigError(f"{context} must be a non-empty sequence")
    result = list(values)
    if len(result) != len(set(result)):
        raise ConfigError(f"{context} must not contain duplicates")
    return result


def _validated_seeds(values: Any) -> list[int]:
    seeds: list[int] = []
    for value in _validated_sequence(values, "suite.seeds"):
        if isinstance(value, bool):
            raise ConfigError("suite.seeds must contain integers")
        try:
            seed = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid training seed: {value!r}") from exc
        if seed < 0 or seed > 2**32 - 1:
            raise ConfigError(f"Training seed outside NumPy range: {seed}")
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise ConfigError("suite.seeds must not contain duplicates after integer conversion")
    return seeds


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _get(config: Mapping[str, Any], dotted: str) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
