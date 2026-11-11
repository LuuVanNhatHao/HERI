"""Controlled cycle-depth and state-dependent graph-refinement study.

The study changes only graph pass depth and inter-pass rebuild semantics.  All
backbones, evidence families, router behavior, objectives, and optimization
settings remain bound to canonical HERA.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from typing import Any


HERA_CYCLE_STUDY_VARIANTS: tuple[str, ...] = (
    "single_pass",
    "two_cycle_static",
    "semantic_refresh_only",
    "hypothesis_refresh_only",
    "initial_hypothesis_control",
    "full_hera",
    "three_cycle_extension",
)

_CYCLE_PROTOCOLS: dict[str, tuple[int, bool, str]] = {
    "single_pass": (1, False, "none"),
    "two_cycle_static": (2, False, "static"),
    "semantic_refresh_only": (2, True, "semantic_only"),
    "hypothesis_refresh_only": (2, True, "hypothesis_only"),
    "initial_hypothesis_control": (2, True, "initial_hypothesis"),
    "full_hera": (2, True, "state_dependent"),
    "three_cycle_extension": (3, True, "state_dependent"),
}


def get_hera_cycle_study_variant(config: Mapping[str, Any]) -> str | None:
    """Return the controlled cycle-study variant, or ``None`` otherwise."""

    study = config.get("cycle_study")
    if not isinstance(study, Mapping) or not bool(study.get("enabled", False)):
        return None
    variant = str(study.get("variant", "")).strip()
    return variant or None


def cycle_study_protocol(variant: str) -> dict[str, Any]:
    """Return the immutable pass/rebuild protocol for a registered variant."""

    variant = str(variant).strip()
    if variant not in _CYCLE_PROTOCOLS:
        raise ValueError(
            f"Unknown HERA cycle-study variant {variant!r}; expected one of "
            f"{list(HERA_CYCLE_STUDY_VARIANTS)}"
        )
    cycles, dynamic_rebuild, rebuild_mode = _CYCLE_PROTOCOLS[variant]
    return {
        "cycles": cycles,
        "dynamic_rebuild": dynamic_rebuild,
        "rebuild_mode": rebuild_mode,
        "refresh_semantic": dynamic_rebuild and variant != "hypothesis_refresh_only",
        "refresh_hypothesis": dynamic_rebuild and variant != "semantic_refresh_only",
        "hypothesis_state": "initial" if variant == "initial_hypothesis_control" else "evolved",
    }


def apply_hera_cycle_study_variant(
    config: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    """Return a canonical config with exactly one registered cycle protocol."""

    if _get(config, "model.model_type") != "hera" or _get(
        config, "model.baseline_type"
    ) is not None:
        raise ValueError("Cycle studies require a HERA base config; B0/B1 are forbidden")
    if bool(_get(config, "ablation.enabled", False)):
        raise ValueError("Cycle studies cannot be combined with component ablations")
    if bool(_get(config, "cycle_study.enabled", False)):
        raise ValueError("Cycle-study variants must be derived from canonical HERA")

    protocol = cycle_study_protocol(variant)
    result = copy.deepcopy(dict(config))
    _set(result, "cycle_study.enabled", True)
    _set(result, "cycle_study.variant", str(variant))
    _set(result, "cycle_study.rebuild_mode", protocol["rebuild_mode"])
    _set(result, "graph.reasoning.cycles", protocol["cycles"])
    _set(result, "graph.reasoning.dynamic_rebuild", protocol["dynamic_rebuild"])
    for key in ("refresh_semantic", "refresh_hypothesis", "hypothesis_state"):
        _set(result, f"graph.reasoning.{key}", protocol[key])
    validate_hera_cycle_study_contract(result)
    return result


def validate_hera_cycle_study_contract(config: Mapping[str, Any]) -> None:
    """Reject mislabeled studies and any drift from the matched HERA protocol."""

    if _get(config, "model.model_type") != "hera" or _get(
        config, "model.baseline_type"
    ) is not None:
        raise ValueError("Cycle-study configs must be HERA-only")
    if bool(_get(config, "ablation.enabled", False)):
        raise ValueError("cycle_study and ablation are mutually exclusive")
    variant = get_hera_cycle_study_variant(config)
    if variant not in HERA_CYCLE_STUDY_VARIANTS:
        raise ValueError(
            "cycle_study.enabled=true requires cycle_study.variant in "
            f"{list(HERA_CYCLE_STUDY_VARIANTS)}"
        )
    protocol = cycle_study_protocol(variant)
    expected = {
        "project.architecture_version": "heri",
        "dataset.use_mtcnn_boxes": False,
        "graph.type": "hgt",
        "graph.evidence_mode": "heterogeneous",
        "graph.reasoning.cycles": protocol["cycles"],
        "graph.reasoning.dynamic_rebuild": protocol["dynamic_rebuild"],
        "graph.reasoning.refresh_semantic": protocol["refresh_semantic"],
        "graph.reasoning.refresh_hypothesis": protocol["refresh_hypothesis"],
        "graph.reasoning.hypothesis_state": protocol["hypothesis_state"],
        "graph.reasoning.hypothesis_feedback": True,
        "graph.reasoning.hypothesis_edge_reweighting": "cosine",
        "graph.edge_attention.enabled": True,
        "graph.edge_attention.confidence_gate": "direct",
        "graph.edges.feedback.enabled": True,
        "routing.enabled": True,
        "routing.regularization.enabled": True,
        "model.grounding.mode": "hypothesis_conditioned",
        "model.grounding.image.protocol": "dense_bbox",
        "model.binary.aggregation": "noisy_or",
        "cycle_study.rebuild_mode": protocol["rebuild_mode"],
    }
    for dotted_key, expected_value in expected.items():
        actual = _get(config, dotted_key)
        if actual != expected_value:
            raise ValueError(
                f"Cycle study {variant!r} requires {dotted_key}="
                f"{expected_value!r}, got {actual!r}"
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
