"""Target-free evidence--decision consistency diagnostics for HERA."""

from __future__ import annotations

from typing import Any

import numpy as np


def evidence_decision_metrics(
    atomic_scores: np.ndarray,
    evidence_support: np.ndarray,
    thresholds: np.ndarray,
    *,
    modality_applicability: np.ndarray | None = None,
    modality_availability: np.ndarray | None = None,
    modality_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Measure how well positive atomic decisions are covered by local evidence.

    A hypothesis is an active decision when its atomic probability reaches the
    fixed cutoff 0.5 (Eq.31). An active decision is unsupported when
    its evidence support is below 0.5. Eq.32 defines:

        EDC = mean min(1, s/(p+1e-8)); UDR = mean [s < 0.5].

    For per-modality support ``[N,K,M]``, the evaluation domain contains only
    active hypothesis--modality pairs that are ontology-applicable and whose
    annotation is available.  This definition remains target-free at
    evaluation time and mirrors the granularity of the training penalty.
    """

    scores = np.asarray(atomic_scores, dtype=float)
    support = np.asarray(evidence_support, dtype=float)
    cutoffs = np.asarray(thresholds, dtype=float)
    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError("atomic_scores must be [N,K]")
    if support.ndim not in {2, 3} or support.shape[:2] != scores.shape:
        raise ValueError("evidence_support must be [N,K] or [N,K,M]")
    if cutoffs.shape != (scores.shape[1],):
        raise ValueError("thresholds must align with atomic_scores hypotheses [K]")
    if not np.allclose(cutoffs, 0.5):
        raise ValueError("HERI EDC/UDR require the fixed atomic cutoff 0.5")
    if not np.isfinite(scores).all() or not np.isfinite(support).all():
        raise ValueError("atomic_scores and evidence_support must be finite")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("atomic_scores must be probabilities in [0,1]")
    if ((support < 0.0) | (support > 1.0)).any():
        raise ValueError("evidence_support must be probabilities in [0,1]")
    if ((cutoffs < 0.0) | (cutoffs > 1.0)).any():
        raise ValueError("thresholds must be in [0,1]")

    active = scores >= cutoffs.reshape(1, -1)
    if support.ndim == 2:
        valid_units = active
        probability_units = scores
        names: tuple[str, ...] = ()
        applicability = None
        availability = None
        metric_definition = "heri_eq32"
    else:
        modality_count = support.shape[2]
        applicability = np.asarray(modality_applicability, dtype=bool)
        availability = np.asarray(modality_availability, dtype=bool)
        if applicability.shape != (scores.shape[1], modality_count):
            raise ValueError("modality_applicability must be [K,M]")
        if availability.shape != (scores.shape[0], modality_count):
            raise ValueError("modality_availability must be [N,M]")
        names = tuple(
            str(value)
            for value in (
                modality_names
                if modality_names is not None
                else tuple(f"modality_{index}" for index in range(modality_count))
            )
        )
        if len(names) != modality_count or len(set(names)) != modality_count:
            raise ValueError("modality_names must contain one unique name per modality")
        valid_units = (
            active[:, :, None] & applicability[None, :, :] & availability[:, None, :]
        )
        probability_units = np.broadcast_to(scores[:, :, None], support.shape)
        metric_definition = "heri_eq32"

    active_unit_count = int(valid_units.sum())
    if support.ndim == 3:
        active_atomic_count = int(
            (
                active
                & (applicability[None, :, :] & availability[:, None, :]).any(axis=2)
            ).sum()
        )
    else:
        active_atomic_count = active_unit_count
    unsupported = valid_units & (support < 0.5)
    unsupported_count = int(unsupported.sum())
    if active_unit_count == 0:
        consistency = None
        unsupported_rate = None
        active_confidence = 0.0
        unsupported_confidence = 0.0
    else:
        active_confidence = float(probability_units[valid_units].sum())
        unsupported_confidence = float(
            np.maximum(probability_units - support, 0.0)[valid_units].sum()
        )
        consistency = float(
            np.minimum(
                1.0, support[valid_units] / (probability_units[valid_units] + 1e-8)
            ).mean()
        )
        unsupported_rate = float(unsupported_count / active_unit_count)

    per_modality: dict[str, Any] = {}
    if support.ndim == 3:
        for modality_index, name in enumerate(names):
            token = "".join(
                char.lower() if char.isalnum() else "_" for char in name
            ).strip("_")
            modality_valid = valid_units[:, :, modality_index]
            count = int(modality_valid.sum())
            modality_probability = probability_units[:, :, modality_index]
            modality_support = support[:, :, modality_index]
            modality_unsupported = modality_valid & (modality_support < 0.5)
            confidence = float(modality_probability[modality_valid].sum())
            per_modality[f"edc_{token}"] = (
                float(
                    np.minimum(
                        1.0,
                        modality_support[modality_valid]
                        / (modality_probability[modality_valid] + 1e-8),
                    ).mean()
                )
                if confidence > 0.0
                else None
            )
            per_modality[f"udr_{token}"] = (
                float(modality_unsupported.sum() / count) if count else None
            )
            per_modality[f"active_evidence_decision_unit_count_{token}"] = count

    return {
        "evidence_decision_metrics_available": True,
        "edc": consistency,
        "udr": unsupported_rate,
        "evidence_decision_consistency": consistency,
        "unsupported_decision_rate": unsupported_rate,
        "unsupported_decision_count": unsupported_count,
        "active_atomic_decision_count": active_atomic_count,
        "active_evidence_decision_unit_count": active_unit_count,
        "active_decision_confidence_sum": active_confidence,
        "unsupported_decision_confidence_sum": unsupported_confidence,
        "evidence_decision_threshold": (
            float(cutoffs[0]) if bool(np.allclose(cutoffs, cutoffs[0])) else None
        ),
        "evidence_decision_threshold_source": (
            "fixed_0.5" if bool(np.allclose(cutoffs, 0.5)) else "caller_supplied"
        ),
        "evidence_decision_metric_definition": metric_definition,
        "evidence_decision_unit": (
            "sample_hypothesis_modality" if support.ndim == 3 else "sample_hypothesis"
        ),
        **per_modality,
    }


def unavailable_evidence_decision_metrics() -> dict[str, Any]:
    """Return an explicit unavailable schema for models without evidence support."""

    return {
        "evidence_decision_metrics_available": False,
        "edc": None,
        "udr": None,
        "evidence_decision_consistency": None,
        "unsupported_decision_rate": None,
        "unsupported_decision_count": None,
        "active_atomic_decision_count": None,
        "active_evidence_decision_unit_count": None,
        "active_decision_confidence_sum": None,
        "unsupported_decision_confidence_sum": None,
        "evidence_decision_threshold": None,
        "evidence_decision_threshold_source": None,
        "evidence_decision_metric_definition": "heri_eq32",
        "evidence_decision_unit": "sample_hypothesis_modality",
    }
