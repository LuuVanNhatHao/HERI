"""Dataset-level evidence-routing analysis from exported prediction artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROUTER_FAMILIES: tuple[str, ...] = (
    "visual_semantic",
    "visual_forensic",
    "text_alignment",
    "text_context",
)


def compare_evidence_routing(
    *,
    inputs: Mapping[str, str | Path],
    output_dir: str | Path,
    reference: str,
    comparison: str,
    bootstrap_samples: int = 1_000,
    seed: int = 777,
) -> dict[str, Any]:
    """Compare learned router behavior across two evaluated datasets.

    Input CSVs are the ``test_predictions.csv`` artifacts emitted by HERA
    evaluation.  Both absolute reliability and within-sample family share are
    reported; the latter distinguishes routing strategy from a global score
    calibration shift.
    """

    if reference == comparison:
        raise ValueError("reference and comparison datasets must be different")
    if reference not in inputs or comparison not in inputs:
        raise ValueError("reference/comparison must name entries in inputs")
    if len(inputs) < 2:
        raise ValueError("At least two routing prediction artifacts are required")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")

    frames: list[pd.DataFrame] = []
    sources: dict[str, Any] = {}
    for dataset_name, raw_path in inputs.items():
        path = Path(raw_path)
        frame = load_routing_predictions(path, dataset_name=str(dataset_name))
        frames.append(frame)
        sources[str(dataset_name)] = {
            "path": str(path),
            "sha256": _sha256(path),
            "sample_count": int(len(frame)),
            "classes": sorted(str(value) for value in frame["class_name"].unique()),
        }
    samples = pd.concat(frames, ignore_index=True)
    by_group = summarize_routing(samples)
    modality = summarize_modality_balance(samples)
    contrast = routing_contrasts(
        samples,
        reference=reference,
        comparison=comparison,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    by_group_path = destination / "routing_by_dataset_and_class.csv"
    modality_path = destination / "routing_modality_balance.csv"
    contrast_path = destination / "routing_contrast.csv"
    by_group.to_csv(by_group_path, index=False)
    modality.to_csv(modality_path, index=False)
    contrast.to_csv(contrast_path, index=False)

    rankings: dict[str, list[str]] = {}
    overall = by_group[by_group["scope"] == "all"]
    for dataset_name, subset in overall.groupby("dataset", sort=True):
        rankings[str(dataset_name)] = list(
            subset.sort_values("mean_reliability", ascending=False)["family"]
        )
    manifest = {
        "analysis": "HERA evidence routing comparison",
        "reference": reference,
        "comparison": comparison,
        "router_families": list(ROUTER_FAMILIES),
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
        "sources": sources,
        "overall_family_rankings": rankings,
        "artifacts": {
            "by_dataset_and_class": str(by_group_path),
            "modality_balance": str(modality_path),
            "contrast": str(contrast_path),
        },
        "interpretation_guard": (
            "Router reliability is a learned gating diagnostic, not a causal attribution. "
            "Use matched HERA architecture/backbones and the real/fake-stratified contrasts "
            "before claiming a dataset-dependent reasoning-strategy change."
        ),
    }
    manifest_path = destination / "routing_analysis.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def load_routing_predictions(path: str | Path, *, dataset_name: str) -> pd.DataFrame:
    """Parse one HERA prediction CSV into a numeric per-sample routing table."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Routing prediction artifact does not exist: {source}")
    raw = pd.read_csv(source, keep_default_na=False)
    required = {"id", "binary_target", "router_diagnostics"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Routing artifact {source} is missing columns: {missing}")
    records: list[dict[str, Any]] = []
    for row_index, row in raw.iterrows():
        encoded = str(row["router_diagnostics"]).strip()
        if not encoded:
            raise ValueError(
                f"Routing artifact {source} row {row_index} has no router diagnostics; "
                "evaluate canonical HERA rather than a baseline/no-router ablation"
            )
        try:
            diagnostics = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Routing artifact {source} row {row_index} contains invalid JSON"
            ) from exc
        absent = [family for family in ROUTER_FAMILIES if family not in diagnostics]
        if absent:
            raise ValueError(
                f"Routing artifact {source} row {row_index} lacks families: {absent}"
            )
        values = np.asarray([diagnostics[family] for family in ROUTER_FAMILIES], dtype=float)
        if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(
                f"Routing artifact {source} row {row_index} has scores outside [0,1]"
            )
        class_name = _class_name(row)
        record: dict[str, Any] = {
            "dataset": dataset_name,
            "id": row["id"],
            "binary_target": int(float(row["binary_target"]) >= 0.5),
            "class_name": class_name,
        }
        total = float(values.sum())
        for family, value in zip(ROUTER_FAMILIES, values):
            record[family] = float(value)
            record[f"{family}_share"] = float(value / total) if total > 0 else 0.25
        record["visual_mean"] = float(values[:2].mean())
        record["text_mean"] = float(values[2:].mean())
        record["visual_minus_text"] = record["visual_mean"] - record["text_mean"]
        records.append(record)
    if not records:
        raise ValueError(f"Routing artifact is empty: {source}")
    frame = pd.DataFrame.from_records(records)
    if np.allclose(frame[list(ROUTER_FAMILIES)].to_numpy(dtype=float), 1.0, atol=1e-8):
        raise ValueError(
            f"Routing artifact {source} contains uniform bypass scores; "
            "use a canonical HERA run with routing.enabled=true"
        )
    return frame


def summarize_routing(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name in sorted(samples["dataset"].unique()):
        dataset = samples[samples["dataset"] == dataset_name]
        for scope, subset in _scopes(dataset):
            for family in ROUTER_FAMILIES:
                values = subset[family].to_numpy(dtype=float)
                shares = subset[f"{family}_share"].to_numpy(dtype=float)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "scope": scope,
                        "family": family,
                        "n": int(len(values)),
                        "mean_reliability": float(values.mean()),
                        "std_reliability": _sample_std(values),
                        "median_reliability": float(np.median(values)),
                        "q25_reliability": float(np.quantile(values, 0.25)),
                        "q75_reliability": float(np.quantile(values, 0.75)),
                        "mean_family_share": float(shares.mean()),
                    }
                )
    return pd.DataFrame.from_records(rows)


def summarize_modality_balance(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name in sorted(samples["dataset"].unique()):
        dataset = samples[samples["dataset"] == dataset_name]
        for scope, subset in _scopes(dataset):
            rows.append(
                {
                    "dataset": dataset_name,
                    "scope": scope,
                    "n": int(len(subset)),
                    "visual_mean": float(subset["visual_mean"].mean()),
                    "text_mean": float(subset["text_mean"].mean()),
                    "visual_minus_text_mean": float(subset["visual_minus_text"].mean()),
                    "visual_minus_text_std": _sample_std(
                        subset["visual_minus_text"].to_numpy(dtype=float)
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def routing_contrasts(
    samples: pd.DataFrame,
    *,
    reference: str,
    comparison: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Return comparison-minus-reference contrasts overall and by binary stratum."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    columns = list(ROUTER_FAMILIES) + [f"{family}_share" for family in ROUTER_FAMILIES]
    for scope, target in (("all", None), ("real", 0), ("fake", 1)):
        reference_rows = samples[samples["dataset"] == reference]
        comparison_rows = samples[samples["dataset"] == comparison]
        if target is not None:
            reference_rows = reference_rows[reference_rows["binary_target"] == target]
            comparison_rows = comparison_rows[comparison_rows["binary_target"] == target]
        if reference_rows.empty or comparison_rows.empty:
            raise ValueError(f"Both datasets require samples in routing scope {scope!r}")
        ref = reference_rows[columns].to_numpy(dtype=float)
        cmp = comparison_rows[columns].to_numpy(dtype=float)
        bootstrap = _bootstrap_mean_differences(
            ref, cmp, samples=bootstrap_samples, rng=rng
        )
        for index, family in enumerate(ROUTER_FAMILIES):
            raw_delta = float(cmp[:, index].mean() - ref[:, index].mean())
            share_index = index + len(ROUTER_FAMILIES)
            share_delta = float(cmp[:, share_index].mean() - ref[:, share_index].mean())
            raw_ci = _percentile_interval(bootstrap[:, index])
            share_ci = _percentile_interval(bootstrap[:, share_index])
            rows.append(
                {
                    "scope": scope,
                    "family": family,
                    "reference": reference,
                    "comparison": comparison,
                    "reference_n": int(len(ref)),
                    "comparison_n": int(len(cmp)),
                    "reference_mean": float(ref[:, index].mean()),
                    "comparison_mean": float(cmp[:, index].mean()),
                    "reliability_delta": raw_delta,
                    "reliability_ci95_low": raw_ci[0],
                    "reliability_ci95_high": raw_ci[1],
                    "reliability_cohens_d": _cohens_d(ref[:, index], cmp[:, index]),
                    "reference_mean_share": float(ref[:, share_index].mean()),
                    "comparison_mean_share": float(cmp[:, share_index].mean()),
                    "family_share_delta": share_delta,
                    "family_share_ci95_low": share_ci[0],
                    "family_share_ci95_high": share_ci[1],
                    "family_share_cohens_d": _cohens_d(
                        ref[:, share_index], cmp[:, share_index]
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _scopes(dataset: pd.DataFrame):
    yield "all", dataset
    real = dataset[dataset["binary_target"] == 0]
    fake = dataset[dataset["binary_target"] == 1]
    if not real.empty:
        yield "real", real
    if not fake.empty:
        yield "fake", fake
    for class_name in sorted(dataset["class_name"].unique()):
        yield f"class:{class_name}", dataset[dataset["class_name"] == class_name]


def _class_name(row: pd.Series) -> str:
    for column in ("class_name", "fake_cls"):
        if column in row.index and str(row[column]).strip():
            return str(row[column]).strip()
    return "real" if int(float(row["binary_target"]) >= 0.5) == 0 else "fake"


def _bootstrap_mean_differences(
    reference: np.ndarray,
    comparison: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if samples == 0:
        return np.empty((0, reference.shape[1]), dtype=float)
    result = np.empty((samples, reference.shape[1]), dtype=float)
    chunk_size = 16
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        count = stop - start
        ref_indices = rng.integers(0, len(reference), size=(count, len(reference)))
        cmp_indices = rng.integers(0, len(comparison), size=(count, len(comparison)))
        result[start:stop] = (
            comparison[cmp_indices].mean(axis=1) - reference[ref_indices].mean(axis=1)
        )
    return result


def _percentile_interval(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _cohens_d(reference: np.ndarray, comparison: np.ndarray) -> float | None:
    if len(reference) < 2 or len(comparison) < 2:
        return None
    pooled_variance = (
        (len(reference) - 1) * reference.var(ddof=1)
        + (len(comparison) - 1) * comparison.var(ddof=1)
    ) / (len(reference) + len(comparison) - 2)
    if pooled_variance <= 0:
        return 0.0
    return float((comparison.mean() - reference.mean()) / np.sqrt(pooled_variance))


def _sample_std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
