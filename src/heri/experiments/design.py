"""Configuration-derived model descriptions and controlled comparison deltas.

Descriptions report the operators a resolved configuration selects. They do
not establish that a configuration produced any particular experimental result.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import Any


_FAMILIES = ("visual_semantic", "visual_forensic", "text_alignment", "text_context")
_SCIENTIFIC_SECTIONS = (
    "dataset",
    "dataset_hashes",
    "image",
    "tokenizer",
    "backbones",
    "evidence",
    "routing",
    "graph",
    "model",
    "loss",
    "training",
    "optimizer",
    "scheduler",
    "thresholding",
    "reasoning_control",
)
_BOOKKEEPING_KEYS = {
    "seed",
    "run_id",
    "run_dir",
    "config_name",
    "root",
    "image_root",
    "train_annotations",
    "val_annotations",
    "test_annotations",
    "resume_from",
    "auto_resume",
    "local_files_only",
    "verify_image_exists",
    "verify_hashes",
}
_MISSING = object()


def describe_model_design(config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe effective readouts, reasoning and objectives without loading a model."""
    baseline = _get(config, "model.baseline_type")
    is_heri = baseline is None and _get(config, "model.model_type") == "hera"
    queries = is_heri and bool(_get(config, "reasoning_control.enabled", False))
    pooled = (
        is_heri and not queries and _get(config, "graph.reasoning.operator", "graph") == "pooled"
    )
    graph_active = is_heri and not queries and not pooled
    hypothesis_nodes = graph_active and bool(_get(config, "graph.include_hypothesis_nodes", True))
    cycles = int(_get(config, "graph.reasoning.cycles", 1)) if is_heri else 0
    dynamic = (
        graph_active and cycles > 1 and bool(_get(config, "graph.reasoning.dynamic_rebuild", False))
    )
    router = is_heri and bool(_get(config, "routing.enabled", True))
    graph_type = _get(config, "graph.type", "hgt") if graph_active else None
    attention = (
        "gatv2"
        if graph_type == "gatv2"
        else (
            _get(config, "graph.edge_attention.variant", "reliability_conditioned")
            if bool(_get(config, "graph.edge_attention.enabled", True))
            else "standard"
        )
        if graph_active
        else None
    )
    if baseline == "b0_global":
        readout = "global_fusion"
    elif baseline == "b1_cross_attention":
        readout = "cross_attention"
    elif baseline is not None:
        readout = str(baseline)
    elif queries:
        readout = "learned_class_queries"
    elif hypothesis_nodes:
        readout = "hypothesis_graph"
    elif is_heri:
        readout = "family_mean_projection"
    else:
        readout = "unspecified"
    families = [
        name
        for name in _FAMILIES
        if is_heri and bool(_get(config, f"evidence.{name}.enabled", True))
    ]
    sufficiency_weight = float(_get(config, "loss.lambda_ev", 0.0)) if is_heri else 0.0
    alignment_available = is_heri and bool(_get(config, "evidence.cross_modal.enabled", False))
    alignment_weight = float(_get(config, "loss.lambda_align", 0.0)) if is_heri else 0.0
    no_hypothesis_readout = graph_active and not hypothesis_nodes
    return {
        "model": {
            "kind": "heri" if is_heri else "baseline",
            "baseline_type": baseline,
            "architecture": _get(config, "project.architecture_version"),
            "readout": readout,
        },
        "evidence": {"families": families, "decomposed": is_heri},
        "routing": {
            "enabled": router,
            "regularization": router
            and bool(_get(config, "routing.regularization.enabled", False)),
            "regularization_weight": float(_get(config, "routing.regularization.weight", 0.0))
            if router
            else 0.0,
        },
        "graph": {
            "message_passing": graph_active,
            "type": graph_type,
            "evidence_mode": _get(config, "graph.evidence_mode", "heterogeneous")
            if graph_active
            else None,
            "hypothesis_nodes": hypothesis_nodes,
            "attention": attention,
            "reliability_conditioned_attention": graph_active
            and attention == "reliability_conditioned",
        },
        "reasoning": {
            "operator": "learned_queries"
            if queries
            else "pooled"
            if pooled
            else "graph"
            if graph_active
            else None,
            "cycles": cycles,
            "dynamic_rebuild": dynamic,
            "refresh_semantic": dynamic
            and bool(_get(config, "graph.reasoning.refresh_semantic", True)),
            "refresh_hypothesis": dynamic
            and hypothesis_nodes
            and bool(_get(config, "graph.reasoning.refresh_hypothesis", True)),
            "hypothesis_state": _get(config, "graph.reasoning.hypothesis_state", "evolved")
            if hypothesis_nodes
            else None,
        },
        "heads": {
            "binary_aggregation": (
                _get(config, "model.binary.aggregation", "noisy_or")
                if is_heri
                else "independent_head"
                if baseline == "b0_global"
                else _get(config, "model.binary_aggregation", "independent_head")
            ),
            "grounding": _get(config, "model.grounding.mode", "hypothesis_conditioned")
            if is_heri
            else "fused_context",
        },
        "objectives": {
            "evidence_sufficiency": {
                "available": is_heri,
                "weight": sufficiency_weight,
                "enabled": is_heri and sufficiency_weight != 0.0,
                "aggregation": _get(config, "loss.evidence_sufficiency.aggregation")
                if is_heri
                else None,
                "validity": _get(config, "loss.evidence_sufficiency.validity") if is_heri else None,
            },
            "alignment": {
                "available": alignment_available,
                "weight": alignment_weight,
                "enabled": alignment_available and alignment_weight != 0.0,
                "aggregation": _get(config, "loss.alignment.aggregation") if is_heri else None,
            },
        },
        "interpretation": {
            "coupled_interventions": ["hypothesis_graph_nodes", "prediction_readout"]
            if no_hypothesis_readout
            else [],
            "comparison_limit": (
                "Removing hypothesis nodes also replaces node-state readout with a learned "
                "projection of pooled evidence; the contrast does not isolate node count alone."
                if no_hypothesis_readout
                else None
            ),
        },
    }


def design_delta(
    before_config: Mapping[str, Any], after_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return stable dotted-key changes in scientific configuration fields.

    Data/source locations, seed identifiers, output directories, resume state and
    runtime placement are bookkeeping. Numerical precision, batch construction,
    checkpoint selection and all model/optimization settings remain comparable.
    """
    before, after = _scientific_fields(before_config), _scientific_fields(after_config)
    changed = {}
    for key in sorted(before.keys() | after.keys()):
        left, right = before.get(key, _MISSING), after.get(key, _MISSING)
        if left == right:
            continue
        changed[key] = {
            "before": None if left is _MISSING else copy.deepcopy(left),
            "after": None if right is _MISSING else copy.deepcopy(right),
            "change": "added"
            if left is _MISSING
            else "removed"
            if right is _MISSING
            else "changed",
        }
    return changed


def _scientific_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    selected = {key: config[key] for key in _SCIENTIFIC_SECTIONS if key in config}
    for section, keys in (
        ("project", ("architecture_version",)),
        ("dataloader", ("batch_size", "drop_last", "shuffle", "id_aware_sampler")),
        ("checkpointing", ("selection_metric", "tie_breakers")),
        ("runtime", ("precision", "model_dtype", "attn_implementation")),
        ("evaluation", ("split", "text_grounding_diagnostic")),
    ):
        values = config.get(section, {})
        if isinstance(values, Mapping):
            included = {key: values[key] for key in keys if key in values}
            if included:
                selected[section] = included
    flattened = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                if key in _BOOKKEEPING_KEYS or key.endswith(("_path", "_root", "_dir")):
                    continue
                visit(value[key], f"{prefix}.{key}" if prefix else str(key))
        else:
            flattened[prefix] = copy.deepcopy(value)

    visit(selected, "")
    return flattened


def _get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current = config
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current
