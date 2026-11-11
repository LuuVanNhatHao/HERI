"""Registered HERI controls and pooled readouts for progressive model stages."""

from __future__ import annotations
import copy
import math
from collections.abc import Mapping, MutableMapping
from typing import Any

from .design import describe_model_design

PROGRESSIVE_VARIANTS = (
    "evidence_decomposition",
    "reliability_router",
    "hypothesis_evidence_graph",
    "constrained_single_pass",
)
HERA_ABLATION_VARIANTS = (
    "no_router",
    "homogeneous_evidence",
    "gatv2",
    "no_hypothesis_feedback",
    "generic_grounding",
    "independent_binary",
    "any_modality_or",
    "no_hypothesis_nodes",
    "no_cross_modal_edges",
    "no_reliability_attention",
    "no_forensic_evidence",
    "no_evidence_sufficiency",
    "homogeneous_gat",
    "standard_hgt",
    "edge_residual",
    "similarity_conditioned_attention",
    *PROGRESSIVE_VARIANTS,
)


def get_hera_ablation_variant(config: Mapping[str, Any]) -> str | None:
    ablation = config.get("ablation")
    if not isinstance(ablation, Mapping) or not bool(ablation.get("enabled", False)):
        return None
    return str(ablation.get("variant", "")).strip() or None


def _variant_contract(variant: str) -> dict[str, Any]:
    """Specify only the mechanisms intentionally changed by an intervention."""
    updates = {
        "no_router": {
            "routing.enabled": False,
            "routing.regularization.enabled": False,
            "routing.regularization.weight": 0.0,
        },
        "homogeneous_evidence": {"graph.evidence_mode": "homogeneous"},
        "gatv2": {"graph.type": "gatv2"},
        "homogeneous_gat": {
            "graph.type": "gatv2",
            "graph.evidence_mode": "homogeneous",
        },
        "no_hypothesis_feedback": {
            "graph.reasoning.hypothesis_feedback": False,
            "graph.edges.feedback.enabled": False,
        },
        "generic_grounding": {"model.grounding.mode": "generic"},
        "independent_binary": {"model.binary.aggregation": "independent_head"},
        "no_hypothesis_nodes": {"graph.include_hypothesis_nodes": False},
        "no_cross_modal_edges": {"graph.edges.cross_modal.enabled": False},
        "no_forensic_evidence": {"evidence.visual_forensic.enabled": False},
        "no_evidence_sufficiency": {"loss.lambda_ev": 0.0},
        "no_reliability_attention": {"graph.edge_attention.variant": "similarity_conditioned"},
        "standard_hgt": {"graph.edge_attention.variant": "standard"},
        "edge_residual": {"graph.edge_attention.variant": "edge_residual"},
        "similarity_conditioned_attention": {
            "graph.edge_attention.variant": "similarity_conditioned"
        },
        "any_modality_or": {
            "loss.evidence_sufficiency.protocol": "any_modality_or_v1",
            "loss.evidence_sufficiency.aggregation": "any_modality_or",
            "loss.evidence_sufficiency.validity": "unmasked_legacy",
        },
    }
    if variant not in PROGRESSIVE_VARIANTS:
        return updates[variant]
    pooled_stage = variant in {"evidence_decomposition", "reliability_router"}
    expected = {
        "graph.reasoning.cycles": 1,
        "graph.reasoning.dynamic_rebuild": False,
        "graph.reasoning.refresh_semantic": False,
        "graph.reasoning.refresh_hypothesis": False,
        "graph.reasoning.operator": "pooled" if pooled_stage else "graph",
        "graph.include_hypothesis_nodes": not pooled_stage,
        "model.binary.aggregation": (
            "noisy_or" if variant == "constrained_single_pass" else "independent_head"
        ),
    }
    if variant != "constrained_single_pass":
        expected["loss.lambda_ev"] = 0.0
    if variant == "evidence_decomposition":
        expected.update(updates["no_router"])
    else:
        expected.update({"routing.enabled": True, "routing.regularization.enabled": True})
    return expected


def apply_hera_ablation_variant(config: Mapping[str, Any], variant: str) -> dict[str, Any]:
    variant = str(variant).strip()
    if variant not in HERA_ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown HERI ablation variant {variant!r}; expected {list(HERA_ABLATION_VARIANTS)}"
        )
    if (
        _get(config, "model.model_type") != "hera"
        or _get(config, "model.baseline_type") is not None
    ):
        raise ValueError("HERI ablations require a HERI base config; B0/B1 are forbidden")
    if bool(_get(config, "cycle_study.enabled", False)):
        raise ValueError("Component controls cannot combine with cycle studies")
    if (
        bool(_get(config, "ablation.enabled", False))
        and get_hera_ablation_variant(config) != variant
    ):
        raise ValueError(
            "Different ablations cannot be layered; derive each variant from its base experiment"
        )
    if (
        _get(config, "model.binary.aggregation") == "independent_head"
        and variant != "independent_binary"
        and variant not in PROGRESSIVE_VARIANTS
    ):
        raise ValueError(
            f"An independent-head base is not supported by {variant!r}; "
            "the direct-head output contract requires an independent_binary or progressive control"
        )
    result = copy.deepcopy(dict(config))
    _set(result, "ablation.enabled", True)
    _set(result, "ablation.variant", variant)
    for key, value in _variant_contract(variant).items():
        _set(result, key, value)
    if variant in PROGRESSIVE_VARIANTS:
        design = describe_model_design(result)
        _set(result, "ablation.readout", design["model"]["readout"])
        _set(result, "ablation.experimental_design", design)
    validate_hera_ablation_contract(result)
    return result


def validate_hera_ablation_contract(config: Mapping[str, Any]) -> None:
    if (
        _get(config, "model.model_type") != "hera"
        or _get(config, "model.baseline_type") is not None
    ):
        raise ValueError("Controlled ablations must be HERI-only")
    variant = get_hera_ablation_variant(config)
    if variant not in HERA_ABLATION_VARIANTS:
        raise ValueError(f"Unknown controlled ablation variant {variant!r}")
    for key, expected in _variant_contract(variant).items():
        actual = _get(config, key)
        if actual != expected:
            raise ValueError(f"Ablation {variant!r} requires {key}={expected!r}, got {actual!r}")
    if variant == "constrained_single_pass":
        try:
            evidence_weight = float(_get(config, "loss.lambda_ev"))
        except (TypeError, ValueError) as exc:
            raise ValueError("constrained_single_pass requires a positive loss.lambda_ev") from exc
        if not math.isfinite(evidence_weight) or evidence_weight <= 0:
            raise ValueError("constrained_single_pass requires a positive loss.lambda_ev")


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
    current_alignment = _get(current_config, "loss.alignment.aggregation")
    checkpoint_alignment = _get(checkpoint_config, "loss.alignment.aggregation")
    if current_alignment != "valid_text_mean" or checkpoint_alignment != current_alignment:
        raise ValueError(
            f"{context} checkpoint alignment-objective mismatch: "
            f"current={current_alignment!r}, checkpoint={checkpoint_alignment!r}"
        )
    for key in ("project.seed", "dataloader.seed"):
        if _get(current_config, key) != _get(checkpoint_config, key):
            raise ValueError(f"{context} checkpoint {key} mismatch")
    if current_config.get("dataset", {}).get("name") != checkpoint_config.get("dataset", {}).get(
        "name"
    ):
        raise ValueError(f"{context} checkpoint dataset mismatch")
    if current_config.get("dataset", {}).get("root") != checkpoint_config.get("dataset", {}).get(
        "root"
    ):
        raise ValueError(f"{context} checkpoint dataset root mismatch")
    if current_config.get("model", {}).get("hypotheses") != checkpoint_config.get("model", {}).get(
        "hypotheses"
    ):
        raise ValueError(f"{context} checkpoint hypothesis ontology mismatch")
    if current_config.get("dataset_hashes", {}) != checkpoint_config.get("dataset_hashes", {}):
        raise ValueError(f"{context} checkpoint dataset hash mismatch")
    if current_config.get("scientific_study", {}) != checkpoint_config.get("scientific_study", {}):
        raise ValueError(f"{context} checkpoint scientific-study mismatch")
    if current_config.get("reasoning_control", {}) != checkpoint_config.get(
        "reasoning_control", {}
    ):
        raise ValueError(f"{context} checkpoint reasoning-control mismatch")
    current_evidence_loss = current_config.get("loss", {}).get("evidence_sufficiency", {})
    checkpoint_evidence_loss = checkpoint_config.get("loss", {}).get("evidence_sufficiency", {})
    if current_evidence_loss != checkpoint_evidence_loss:
        raise ValueError(
            f"{context} checkpoint evidence-sufficiency protocol mismatch: "
            f"current={current_evidence_loss!r}, "
            f"checkpoint={checkpoint_evidence_loss!r}"
        )
    checkpoint_variant = get_hera_ablation_variant(checkpoint_config)
    current_cycle_variant = get_hera_cycle_study_variant(current_config)
    checkpoint_cycle_variant = get_hera_cycle_study_variant(checkpoint_config)
    if checkpoint_config.get("model", {}).get("model_type") != "hera":
        raise ValueError(f"{context} checkpoint is not a HERA checkpoint")
    current_architecture = _get(current_config, "project.architecture_version")
    checkpoint_architecture = _get(checkpoint_config, "project.architecture_version")
    if checkpoint_architecture != current_architecture or current_architecture != "heri":
        raise ValueError(
            f"{context} checkpoint architecture mismatch: current={current_architecture!r}, "
            f"checkpoint={checkpoint_architecture!r}; HERI requires 'heri'"
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
    for section in ("evidence", "routing", "graph", "model"):
        if current_config.get(section) != checkpoint_config.get(section):
            raise ValueError(f"{context} checkpoint {section} architecture mismatch")
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
