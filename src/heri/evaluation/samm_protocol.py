"""SAMM released-protocol reporting adapters.

HERA reasons with two coupled SAMM hypotheses (SWAP and ATTRIBUTE).  The
released SAMM/RamDG evaluator instead reports a three-column multilabel target:
visual swap, visual attribute, and text attribute.  This module makes that
reporting projection explicit without changing HERA's dataset-independent
decision architecture.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from heri.metrics.attribution import attribution_metrics


SAMM_RELEASED_LABEL_ORDER: tuple[str, ...] = (
    "visual_swap",
    "visual_attribute",
    "text_attribute",
)
SAMM_RELEASED_REFERENCE_REPOSITORY = "https://github.com/shen8424/SAMM-RamDG-CAP"
SAMM_RELEASED_REFERENCE_COMMIT = "f685f0d6e64dbbbc5a4e787eab768f5f42099ce9"

_SAMM_RELEASED_TARGETS: dict[str, tuple[int, int, int]] = {
    "orig": (0, 0, 0),
    "swap_manipulation": (1, 0, 1),
    "attribute_manipulation": (0, 1, 1),
}


def is_samm_reporting_ontology(
    fake_cls: Sequence[str], hypothesis_names: Sequence[str]
) -> bool:
    """Return whether predictions carry the supported SAMM ontology."""

    return (
        tuple(str(name).upper() for name in hypothesis_names) == ("SWAP", "ATTRIBUTE")
        and bool(fake_cls)
        and set(map(str, fake_cls)).issubset(_SAMM_RELEASED_TARGETS)
    )


def samm_released_targets(fake_cls: Sequence[str]) -> np.ndarray:
    """Build the exact three-label targets used by the released evaluator."""

    try:
        rows = [_SAMM_RELEASED_TARGETS[str(label)] for label in fake_cls]
    except KeyError as exc:
        raise ValueError(f"Unknown SAMM class label: {exc.args[0]!r}") from exc
    return np.asarray(rows, dtype=int)


def samm_released_scores(
    atomic_scores: np.ndarray,
    binary_scores: np.ndarray,
    hypothesis_names: Sequence[str],
) -> np.ndarray:
    """Project HERA outputs to SAMM's three released reporting columns.

    The two subtype probabilities provide visual-swap and visual-attribute.
    The binary fake probability provides text-attribute because both SAMM
    manipulation classes alter text.  For canonical HERA this binary score is
    itself the Noisy-OR of the two subtype hypotheses; B1 uses its matched
    binary decision head.
    """

    scores = np.asarray(atomic_scores, dtype=float)
    binary = np.asarray(binary_scores, dtype=float)
    names = tuple(str(name).upper() for name in hypothesis_names)
    if scores.ndim != 2 or names != ("SWAP", "ATTRIBUTE"):
        raise ValueError("SAMM reporting requires [SWAP, ATTRIBUTE] scores")
    if binary.shape != (scores.shape[0],):
        raise ValueError("binary_scores must align with SAMM atomic scores")
    if not np.isfinite(scores).all() or not np.isfinite(binary).all():
        raise ValueError("SAMM reporting scores must be finite")
    if ((scores < 0.0) | (scores > 1.0)).any() or (
        (binary < 0.0) | (binary > 1.0)
    ).any():
        raise ValueError("SAMM reporting scores must be probabilities in [0,1]")
    return np.column_stack((scores[:, 0], scores[:, 1], binary))


def samm_released_metrics(
    *,
    fake_cls: Sequence[str],
    atomic_scores: np.ndarray,
    binary_scores: np.ndarray,
    hypothesis_names: Sequence[str],
) -> dict[str, Any]:
    """Compute released-compatible SAMM mAP/CF1/OF1 at fixed 0.5."""

    targets = samm_released_targets(fake_cls)
    scores = samm_released_scores(atomic_scores, binary_scores, hypothesis_names)
    base = attribution_metrics(
        targets,
        scores,
        thresholds=np.full(len(SAMM_RELEASED_LABEL_ORDER), 0.5, dtype=float),
    )
    predictions = (scores >= 0.5).astype(int)
    true_positives = ((targets == 1) & (predictions == 1)).sum(axis=0)
    predicted_positives = (predictions == 1).sum(axis=0)
    ground_truth_positives = (targets == 1).sum(axis=0)
    overall_precision = float(
        true_positives.sum() / max(int(predicted_positives.sum()), 1)
    )
    overall_recall = float(
        true_positives.sum() / max(int(ground_truth_positives.sum()), 1)
    )
    result: dict[str, Any] = {
        f"samm_official_{key}": value for key, value in base.items()
    }
    for index, label in enumerate(SAMM_RELEASED_LABEL_ORDER):
        result[f"samm_official_{label}_ap"] = (
            float(average_precision_score(targets[:, index], scores[:, index]))
            if bool(targets[:, index].any())
            else 0.0
        )
    result.update(
        {
            "samm_official_label_order": ",".join(SAMM_RELEASED_LABEL_ORDER),
            "samm_official_decision_threshold": 0.5,
            "samm_official_op": overall_precision,
            "samm_official_or": overall_recall,
            "samm_official_cp": base["class_precision"],
            "samm_official_cr": base["class_recall"],
            "samm_official_exact_match_ratio": float(
                np.all(predictions == targets, axis=1).mean()
            ),
            "samm_official_target_mapping": (
                "orig=000;swap_manipulation=101;attribute_manipulation=011"
            ),
            "samm_official_score_projection": (
                "visual_swap=p_swap;visual_attribute=p_attribute;text_attribute=p_fake"
            ),
            "samm_official_reference_repository": SAMM_RELEASED_REFERENCE_REPOSITORY,
            "samm_official_reference_commit": SAMM_RELEASED_REFERENCE_COMMIT,
        }
    )
    return result
