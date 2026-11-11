"""Inspect experiment artifacts without deserializing model checkpoints.

An internally consistent record links the supplied files and identities. This
assessment does not establish that a training execution or a published result
was produced by those files.
"""

from __future__ import annotations

from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import yaml

from heri.utils.config import load_yaml_config


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible effective configuration deterministically."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_metadata_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    return path.with_suffix(path.suffix + ".metadata.json")


def write_evaluation_evidence(
    *,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    checkpoint_id: str,
    split: str = "test",
) -> Path:
    """Seal completed evaluation artifacts with file hashes; load no checkpoint."""

    run_dir = Path(run_dir).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    config = load_yaml_config(run_dir / "effective_config.yaml")
    annotation_path = Path(config["dataset"][f"{split}_annotations"]).resolve()
    paths = {
        "summary": run_dir / "summary.csv",
        "predictions": run_dir / "predictions" / f"{split}_predictions.csv",
        "effective_evaluate_config": run_dir / "effective_config.yaml",
        "environment": run_dir / "environment.json",
        "checkpoint": checkpoint_path,
        "checkpoint_metadata": checkpoint_metadata_path(checkpoint_path),
        "effective_train_config": checkpoint_path.parent.parent / "effective_config.yaml",
        "train_environment": checkpoint_path.parent.parent / "environment.json",
        "test_annotations": annotation_path,
    }
    records = {
        name: {"path": str(path), "sha256": _file_sha256(path)} for name, path in paths.items()
    }
    payload = {
        "schema": "heri_evaluation_evidence",
        "checkpoint_id": checkpoint_id,
        "split": split,
        "artifacts": records,
    }
    destination = run_dir / "run_evidence.json"
    temporary = run_dir / f".run_evidence.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _alignment_objective(config: Mapping[str, Any]) -> str | None:
    if config.get("model", {}).get("model_type") != "hera":
        return "not_applicable"
    return config.get("loss", {}).get("alignment", {}).get("aggregation")


def _execution_independent(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result.pop("observed_dataset_hashes", None)
    # Resuming from the same run changes this execution pointer, not the model.
    if isinstance(result.get("training"), dict):
        result["training"].pop("resume_from", None)
    return result


class _Audit:
    def __init__(self) -> None:
        self.missing: list[str] = []
        self.errors: list[str] = []
        self.artifacts: dict[str, dict[str, str]] = {}

    def file(self, label: str, path: Path) -> bool:
        if not path.is_file():
            self.missing.append(f"{label}: {path}")
            return False
        try:
            self.artifacts[label] = {"path": str(path), "sha256": _file_sha256(path)}
        except OSError as exc:
            self.errors.append(f"Cannot read {label}: {exc}")
            return False
        return True

    def config(self, label: str, path: Path) -> dict[str, Any] | None:
        if not self.file(label, path):
            return None
        try:
            return load_yaml_config(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self.errors.append(f"Invalid {label}: {exc}")
            return None

    def json(self, label: str, path: Path) -> Any:
        if not self.file(label, path):
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.errors.append(f"Invalid {label}: {exc}")
            return None


def inspect_study_evidence(
    manifest_path: str | Path, *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    """Inspect all jobs in a scientific-study manifest; never import scores."""

    root = Path(repository_root or Path.cwd()).resolve()
    manifest_path = _resolve(manifest_path, root)
    audit = _Audit()
    manifest = audit.json("manifest", manifest_path)
    jobs: list[dict[str, Any]] = []
    if manifest is not None:
        if not isinstance(manifest, dict) or not isinstance(manifest.get("jobs"), list):
            audit.errors.append("Manifest must contain a jobs list")
        elif not manifest["jobs"]:
            audit.errors.append("Manifest contains no jobs")
        elif not all(
            key in manifest
            for key in ("protocol_id", "evaluation_protocol", "expected_test_samples")
        ):
            audit.errors.append("Manifest is missing study, evaluation or sample-count identity")
        else:
            try:
                expected_count = int(manifest["expected_test_samples"])
                if expected_count < 1:
                    raise ValueError("expected_test_samples must be positive")
                seen = set()
                for job in manifest["jobs"]:
                    if not isinstance(job, Mapping):
                        audit.errors.append("Each manifest job must be a mapping")
                        continue
                    identity = (job.get("variant"), job.get("seed"))
                    if identity in seen:
                        audit.errors.append(f"Duplicate manifest job: {identity}")
                    seen.add(identity)
                    jobs.append(
                        inspect_run_evidence(
                            job,
                            protocol_id=str(manifest["protocol_id"]),
                            evaluation_protocol=str(manifest["evaluation_protocol"]),
                            expected_test_samples=expected_count,
                            expected_source_sha256=manifest.get("source_digest"),
                            base_dir=root,
                        )
                    )
            except (TypeError, ValueError) as exc:
                audit.errors.append(f"Invalid manifest identity: {exc}")
    counts = Counter(job["status"] for job in jobs)
    status = (
        "inconsistent"
        if audit.errors or counts["inconsistent"]
        else "missing"
        if audit.missing or counts["missing"]
        else "internally_consistent"
    )
    return {
        "schema": "heri_study_evidence",
        "scope": "file_integrity_and_run_identity",
        "manifest": str(manifest_path),
        "status": status,
        "counts": {
            "total": len(jobs),
            **{key: counts[key] for key in ("internally_consistent", "missing", "inconsistent")},
        },
        "missing": audit.missing,
        "errors": audit.errors,
        "jobs": jobs,
    }


def inspect_run_evidence(
    job: Mapping[str, Any],
    *,
    protocol_id: str,
    evaluation_protocol: str,
    expected_test_samples: int,
    base_dir: str | Path | None = None,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Check a planned job against effective configurations and actual artifacts."""

    root = Path(base_dir or Path.cwd()).resolve()
    audit = _Audit()
    result: dict[str, Any] = {"variant": job.get("variant"), "seed": job.get("seed")}
    configs: dict[str, Any] = {}
    for phase in ("train", "evaluate"):
        value = job.get(f"{phase}_config")
        if not value:
            audit.missing.append(f"{phase}_config path")
            continue
        config = audit.config(f"planned_{phase}_config", _resolve(value, root))
        if config is not None:
            configs[phase] = config
            expected_hash = job.get(f"{phase}_config_hash")
            if expected_hash and canonical_config_sha256(config) != expected_hash:
                audit.errors.append(f"Planned {phase} configuration hash mismatch")
    summary_value = job.get("summary")
    if not summary_value:
        audit.missing.append("summary path")
    evaluation = configs.get("evaluate")
    if summary_value and evaluation is not None:
        summary_path = _resolve(summary_value, root)
        evaluation_config = audit.config(
            "effective_evaluate_config", summary_path.parent / "effective_config.yaml"
        )
        _compare_config(audit, evaluation, evaluation_config, "evaluation")
        checkpoint_value = evaluation.get("evaluation", {}).get("checkpoint_path")
        if not checkpoint_value:
            audit.missing.append("evaluation.checkpoint_path")
        else:
            checkpoint_path = _resolve(checkpoint_value, root)
            checkpoint_exists = audit.file("checkpoint", checkpoint_path)
            metadata = audit.json("checkpoint_metadata", checkpoint_metadata_path(checkpoint_path))
            train_config = audit.config(
                "effective_train_config", checkpoint_path.parent.parent / "effective_config.yaml"
            )
            _compare_config(audit, configs.get("train"), train_config, "training")
            checkpoint_id = _check_checkpoint(
                audit, metadata, train_config, checkpoint_exists, evaluation_protocol
            )
            train_environment = audit.json(
                "train_environment", checkpoint_path.parent.parent / "environment.json"
            )
            evaluation_environment = audit.json(
                "environment", summary_path.parent / "environment.json"
            )
            _check_source_identity(
                audit, metadata, train_environment, evaluation_environment, expected_source_sha256
            )
            snapshot = metadata.get("effective_config") if isinstance(metadata, Mapping) else None
            alignment_config = (
                snapshot
                if isinstance(snapshot, Mapping)
                else train_config or configs.get("train", {})
            )
            alignment = _alignment_objective(alignment_config)
            identity = {
                "checkpoint_id": checkpoint_id,
                "scientific_study_id": protocol_id,
                "scientific_variant": job.get("variant"),
                "seed": job.get("seed"),
                "evaluation_protocol": evaluation_protocol,
                "alignment_objective": alignment,
            }
            for label, config in (("training", train_config), ("evaluation", evaluation_config)):
                _check_config_identity(audit, config, identity, label)
            summary = _summary_row(audit, summary_path)
            if summary is not None:
                _check_identity(audit, summary, identity, "summary")
                if summary.get("checkpoint_evaluation_protocol") != evaluation_protocol:
                    audit.errors.append("Summary checkpoint evaluation protocol mismatch")
                if summary.get("split") != "test":
                    audit.errors.append("Summary split must be test")
                try:
                    if int(summary.get("sample_count", "")) != expected_test_samples:
                        audit.errors.append("Summary sample count mismatch")
                except (TypeError, ValueError):
                    audit.errors.append("Summary has no valid sample count")
            annotation_ids = _annotation_ids(audit, evaluation_config or evaluation, summary, root)
            _check_predictions(
                audit,
                summary_path.parent / "predictions" / "test_predictions.csv",
                identity,
                annotation_ids,
                expected_test_samples,
            )
            _check_seal(audit, summary_path.parent, checkpoint_id)
    status = (
        "inconsistent" if audit.errors else "missing" if audit.missing else "internally_consistent"
    )
    result.update(
        status=status, missing=audit.missing, errors=audit.errors, artifacts=audit.artifacts
    )
    return result


def _compare_config(audit: _Audit, planned: Any, actual: Any, label: str) -> None:
    if (
        planned is not None
        and actual is not None
        and _execution_independent(planned) != _execution_independent(actual)
    ):
        audit.errors.append(f"Effective {label} configuration differs from planned configuration")


def _check_checkpoint(
    audit: _Audit, metadata: Any, config: Any, exists: bool, protocol: str
) -> str:
    if metadata is None:
        return ""
    if not isinstance(metadata, Mapping):
        audit.errors.append("Checkpoint metadata must be a mapping")
        return ""
    if metadata.get("schema") != "heri_checkpoint_metadata":
        audit.errors.append("Unknown checkpoint metadata schema")
    checkpoint_id = _text(metadata.get("checkpoint_id"))
    if not checkpoint_id:
        audit.errors.append("Checkpoint metadata has no checkpoint_id")
    if exists and metadata.get("checkpoint_sha256") != audit.artifacts["checkpoint"]["sha256"]:
        audit.errors.append("Checkpoint SHA-256 mismatch")
    snapshot = metadata.get("effective_config")
    if not isinstance(snapshot, Mapping):
        audit.missing.append("Checkpoint effective configuration snapshot")
    else:
        if metadata.get("effective_config_sha256") != canonical_config_sha256(snapshot):
            audit.errors.append("Checkpoint effective configuration SHA-256 mismatch")
        _compare_config(audit, snapshot, config, "checkpoint-bound training")
        _check_training_data_identity(audit, snapshot, config)
        objective = _alignment_objective(snapshot)
        if metadata.get("alignment_objective") != objective:
            audit.errors.append("Checkpoint alignment objective mismatch")
    if metadata.get("evaluation_protocol") != protocol:
        audit.errors.append("Checkpoint evaluation protocol mismatch")
    return checkpoint_id


def _check_training_data_identity(audit: _Audit, snapshot: Mapping[str, Any], config: Any) -> None:
    records = (("checkpoint", snapshot), ("run", config or {}))
    observed: dict[str, Mapping[str, Any]] = {}
    declared_splits: set[str] = set()
    for label, record in records:
        values = record.get("observed_dataset_hashes", {})
        if not isinstance(values, Mapping):
            audit.errors.append(f"Invalid {label} observed dataset hashes")
            values = {}
        observed[label] = values
        dataset = record.get("dataset", {})
        hashes = record.get("dataset_hashes", {})
        for split in ("train", "val"):
            if (isinstance(dataset, Mapping) and dataset.get(f"{split}_annotations")) or (
                isinstance(hashes, Mapping) and split in hashes
            ):
                declared_splits.add(split)
    for split in ("train", "val"):
        expected = observed["checkpoint"].get(split)
        actual = observed["run"].get(split)
        if not (expected or actual or split in declared_splits):
            continue
        for label, value in (("checkpoint", expected), ("run", actual)):
            if not value:
                audit.missing.append(f"{label} observed {split} annotation SHA-256")
            elif not isinstance(value, str):
                audit.errors.append(f"Invalid {label} observed {split} annotation SHA-256")
        if expected and actual and expected != actual:
            audit.errors.append(
                f"Observed {split} annotation SHA-256 mismatch between checkpoint and run"
            )


def _check_config_identity(
    audit: _Audit, config: Any, identity: Mapping[str, Any], label: str
) -> None:
    if config is None:
        return
    study = config.get("scientific_study", {})
    expected = {
        "protocol_id": identity["scientific_study_id"],
        "variant": identity["scientific_variant"],
        "seed": identity["seed"],
    }
    if study != expected or config.get("project", {}).get("seed") != identity["seed"]:
        audit.errors.append(f"Effective {label} study/variant/seed identity mismatch")


def _check_source_identity(
    audit: _Audit,
    metadata: Any,
    train_environment: Any,
    evaluation_environment: Any,
    expected: str | None,
) -> None:
    observed_sources: set[str] = set()
    for label, record in (
        ("checkpoint", metadata),
        ("training", train_environment),
        ("evaluation", evaluation_environment),
    ):
        if record is None:
            continue
        value = record.get("source_sha256") if isinstance(record, Mapping) else None
        if not value:
            audit.missing.append(f"{label} source SHA-256")
        elif (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            audit.errors.append(f"Invalid {label} source SHA-256")
        else:
            observed_sources.add(value)
            if expected and value != expected:
                audit.errors.append(f"{label} source differs from the study manifest")
    if len(observed_sources) > 1:
        audit.errors.append(
            "Source identity mismatch across checkpoint/training/evaluation artifacts"
        )


def _check_identity(
    audit: _Audit, row: Mapping[str, Any], identity: Mapping[str, Any], label: str
) -> None:
    for key, expected in identity.items():
        if key not in row:
            audit.missing.append(f"{label} identity column: {key}")
        elif _text(row[key]) != _text(expected):
            audit.errors.append(f"{label} {key} mismatch")


def _summary_row(audit: _Audit, path: Path) -> dict[str, str] | None:
    if not audit.file("summary", path):
        return None
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            audit.errors.append("Expected exactly one evaluation summary row")
            return None
        return rows[0]
    except (OSError, ValueError, csv.Error) as exc:
        audit.errors.append(f"Invalid summary CSV: {exc}")
        return None


def _annotation_ids(
    audit: _Audit, config: Mapping[str, Any], summary: Any, root: Path
) -> list[str] | None:
    value = config.get("dataset", {}).get("test_annotations")
    if not value:
        audit.missing.append("dataset.test_annotations path")
        return None
    rows = audit.json("test_annotations", _resolve(value, root))
    if rows is None:
        return None
    if not isinstance(rows, list) or any(
        not isinstance(row, Mapping) or "id" not in row for row in rows
    ):
        audit.errors.append("Test annotations require a list of sample IDs")
        return None
    actual = audit.artifacts["test_annotations"]["sha256"]
    if config.get("dataset_hashes", {}).get("test") != actual:
        audit.errors.append("Declared test annotation SHA-256 mismatch")
    if summary is not None:
        try:
            observed = json.loads(summary.get("dataset_hashes", ""))
            if not isinstance(observed, Mapping) or observed.get("test") != actual:
                audit.errors.append("Summary observed test annotation SHA-256 mismatch")
        except (TypeError, ValueError):
            audit.errors.append("Summary has no valid observed dataset hashes")
    return [_text(row["id"]) for row in rows]


def _check_predictions(
    audit: _Audit,
    path: Path,
    identity: Mapping[str, Any],
    annotation_ids: list[str] | None,
    expected_count: int,
) -> None:
    if not audit.file("predictions", path):
        return
    try:
        ids: list[str] = []
        identity_error_rows = 0
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            for key in ("id", *identity):
                if key not in columns:
                    audit.missing.append(f"prediction identity column: {key}")
            for row in reader:
                ids.append(row.get("id", ""))
                if any(
                    key in row and _text(row[key]) != _text(value)
                    for key, value in identity.items()
                ):
                    identity_error_rows += 1
        if identity_error_rows:
            audit.errors.append(f"Prediction run identity mismatch in {identity_error_rows} rows")
        if len(ids) != expected_count:
            audit.errors.append("Prediction row count mismatch")
        # IDs may legitimately repeat across annotation variants: compare multiplicities.
        if annotation_ids is not None and Counter(ids) != Counter(annotation_ids):
            audit.errors.append("Prediction sample IDs differ from the test annotations")
    except (OSError, ValueError, csv.Error) as exc:
        audit.errors.append(f"Invalid prediction CSV: {exc}")


def _check_seal(audit: _Audit, run_dir: Path, checkpoint_id: str) -> None:
    seal = audit.json("evaluation_evidence", run_dir / "run_evidence.json")
    if seal is None:
        return
    if not isinstance(seal, Mapping) or not isinstance(seal.get("artifacts"), Mapping):
        audit.errors.append("Evaluation evidence requires an artifact-hash mapping")
        return
    if seal.get("schema") != "heri_evaluation_evidence":
        audit.errors.append("Unknown evaluation evidence schema")
    if seal.get("checkpoint_id") != checkpoint_id or seal.get("split") != "test":
        audit.errors.append("Evaluation evidence checkpoint/split identity mismatch")
    labels = (
        "summary",
        "predictions",
        "effective_evaluate_config",
        "environment",
        "checkpoint",
        "checkpoint_metadata",
        "effective_train_config",
        "test_annotations",
        "train_environment",
    )
    for label in labels:
        sealed = seal["artifacts"].get(label)
        if not isinstance(sealed, Mapping) or not sealed.get("sha256"):
            audit.missing.append(f"Evaluation evidence hash: {label}")
        elif label in audit.artifacts and sealed["sha256"] != audit.artifacts[label]["sha256"]:
            audit.errors.append(f"Sealed {label} SHA-256 mismatch")
