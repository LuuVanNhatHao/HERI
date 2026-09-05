"""Controlled production ablations for HERA-DGM.

Each registered variant changes exactly one scientific mechanism while keeping
the canonical data, backbone, optimization, and checkpoint protocols intact.
The contract is shared by the suite runner, configuration validator, model,
and checkpoint metadata so a run cannot silently be mislabeled.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from typing import Any


HERA_ABLATION_VARIANTS: tuple[str, ...] = (
    "no_router",
    "homogeneous_evidence",
    "gatv2",
    "no_hypothesis_feedback",
    "generic_grounding",
    "independent_binary",
)


def get_hera_ablation_variant(config: Mapping[str, Any]) -> str | None:
    """Return the controlled variant name, or ``None`` for canonical HERA."""

    ablation = config.get("ablation")
    if not isinstance(ablation, Mapping) or not bool(ablation.get("enabled", False)):
        return None
    variant = str(ablation.get("variant", "")).strip()
    return variant or None


def apply_hera_ablation_variant(config: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """Return a deep-copied canonical HERA config with one controlled change."""

    variant = str(variant).strip()
    if variant not in HERA_ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown HERA ablation variant {variant!r}; expected one of "
            f"{list(HERA_ABLATION_VARIANTS)}"
        )
    if _get(config, "model.model_type") != "hera" or _get(config, "model.baseline_type") is not None:
        raise ValueError("HERA ablations require a HERA base config; B0/B1 configs are forbidden")

    result = copy.deepcopy(dict(config))
    _set(result, "ablation.enabled", True)
    _set(result, "ablation.variant", variant)

    # Explicit canonical defaults make generated effective configs self-contained.
    _set(result, "project.architecture_version", "hera_final_v2")
    _set(result, "dataset.use_mtcnn_boxes", False)
    _set(result, "graph.type", "hgt")
    _set(result, "graph.evidence_mode", "heterogeneous")
    _set(result, "graph.reasoning.hypothesis_feedback", True)
    _set(result, "graph.reasoning.cycles", 2)
    _set(result, "graph.reasoning.dynamic_rebuild", True)
    _set(result, "graph.reasoning.hypothesis_edge_reweighting", "cosine")
    _set(result, "graph.edge_attention.enabled", True)
    _set(result, "graph.edge_attention.confidence_gate", "direct")
    _set(result, "graph.edges.feedback.enabled", True)
    _set(result, "routing.enabled", True)
    _set(result, "routing.regularization.enabled", True)
    _set(result, "model.grounding.mode", "hypothesis_conditioned")
    _set(result, "model.grounding.image.protocol", "dense_bbox")
    _set(result, "model.binary.aggregation", "noisy_or")

    if variant == "no_router":
        _set(result, "routing.enabled", False)
        _set(result, "routing.regularization.enabled", False)
        _set(result, "routing.regularization.weight", 0.0)
    elif variant == "homogeneous_evidence":
        _set(result, "graph.evidence_mode", "homogeneous")
    elif variant == "gatv2":
        _set(result, "graph.type", "gatv2")
    elif variant == "no_hypothesis_feedback":
        _set(result, "graph.reasoning.hypothesis_feedback", False)
        _set(result, "graph.edges.feedback.enabled", False)
    elif variant == "generic_grounding":
        _set(result, "model.grounding.mode", "generic")
    elif variant == "independent_binary":
        _set(result, "model.binary.aggregation", "independent_head")

    validate_hera_ablation_contract(result)
    return result


def validate_hera_ablation_contract(config: Mapping[str, Any]) -> None:
    """Fail if an ablation is mislabeled or combines multiple core changes."""

    if _get(config, "model.model_type") != "hera" or _get(config, "model.baseline_type") is not None:
        raise ValueError("Controlled ablation configs must be HERA-only and cannot contain baseline_type")
    variant = get_hera_ablation_variant(config)
    if variant not in HERA_ABLATION_VARIANTS:
        raise ValueError(
            "ablation.enabled=true requires ablation.variant in "
            f"{list(HERA_ABLATION_VARIANTS)}"
        )

    expected = {
        "project.architecture_version": "hera_final_v2",
        "dataset.use_mtcnn_boxes": False,
        "graph.type": "gatv2" if variant == "gatv2" else "hgt",
        "graph.evidence_mode": "homogeneous" if variant == "homogeneous_evidence" else "heterogeneous",
        "graph.reasoning.cycles": 2,
        "graph.reasoning.dynamic_rebuild": True,
        "graph.reasoning.hypothesis_edge_reweighting": "cosine",
        "graph.edge_attention.enabled": True,
        "graph.edge_attention.confidence_gate": "direct",
        "graph.reasoning.hypothesis_feedback": variant != "no_hypothesis_feedback",
        "graph.edges.feedback.enabled": variant != "no_hypothesis_feedback",
        "routing.enabled": variant != "no_router",
        "routing.regularization.enabled": variant != "no_router",
        "model.grounding.mode": "generic" if variant == "generic_grounding" else "hypothesis_conditioned",
        "model.grounding.image.protocol": "dense_bbox",
        "model.binary.aggregation": "independent_head" if variant == "independent_binary" else "noisy_or",
    }
    for dotted_key, expected_value in expected.items():
        actual = _get(config, dotted_key)
        if actual != expected_value:
            raise ValueError(
                f"Ablation {variant!r} requires {dotted_key}={expected_value!r}, got {actual!r}"
            )
    if variant == "no_router" and float(_get(config, "routing.regularization.weight", -1.0)) != 0.0:
        raise ValueError("no_router requires routing.regularization.weight=0.0")


def validate_hera_checkpoint_variant(
    current_config: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Prevent resume/evaluation from silently crossing experiment variants."""

    from .cycle_study import (
        get_hera_cycle_study_variant,
        validate_hera_cycle_study_contract,
    )

    current_variant = get_hera_ablation_variant(current_config)
    if current_config.get("scientific_study", {}) != checkpoint_config.get("scientific_study", {}):
        raise ValueError(f"{context} checkpoint scientific-study mismatch")
    if current_config.get("reasoning_control", {}) != checkpoint_config.get("reasoning_control", {}):
        raise ValueError(f"{context} checkpoint reasoning-control mismatch")
    checkpoint_variant = get_hera_ablation_variant(checkpoint_config)
    current_cycle_variant = get_hera_cycle_study_variant(current_config)
    checkpoint_cycle_variant = get_hera_cycle_study_variant(checkpoint_config)
    if checkpoint_config.get("model", {}).get("model_type") != "hera":
        raise ValueError(f"{context} checkpoint is not a HERA checkpoint")
    current_architecture = _get(current_config, "project.architecture_version")
    checkpoint_architecture = _get(checkpoint_config, "project.architecture_version")
    if checkpoint_architecture != current_architecture or current_architecture != "hera_final_v2":
        raise ValueError(
            f"{context} checkpoint architecture mismatch: current={current_architecture!r}, "
            f"checkpoint={checkpoint_architecture!r}; final HERA requires 'hera_final_v2'"
        )
    if checkpoint_variant != current_variant:
        raise ValueError(
            f"{context} checkpoint variant mismatch: current={current_variant!r}, "
            f"checkpoint={checkpoint_variant!r}"
        )
    if checkpoint_cycle_variant != current_cycle_variant:
        raise ValueError(
            f"{context} checkpoint cycle-study mismatch: "
            f"current={current_cycle_variant!r}, checkpoint={checkpoint_cycle_variant!r}"
        )
    if checkpoint_cycle_variant is not None:
        validate_hera_cycle_study_contract(checkpoint_config)
        current_protocol = _get(current_config, "cycle_study.protocol_id")
        checkpoint_protocol = _get(checkpoint_config, "cycle_study.protocol_id")
        if current_protocol != checkpoint_protocol:
            raise ValueError(
                f"{context} checkpoint cycle-study protocol mismatch: "
                f"current={current_protocol!r}, checkpoint={checkpoint_protocol!r}"
            )


def _get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _set(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value
