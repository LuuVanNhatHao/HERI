"""Target-independent routing of hypothesis-conditioned grounding predictions."""

from __future__ import annotations

import torch

from heri.models.heri.outputs import HERADecisionOutput


TEXT_GROUNDING_AGGREGATIONS = (
    "relative_soft",
    "presence_gated",
    "token_noisy_or",
)


def grounding_route_weights(
    atomic_probabilities: torch.Tensor,
    *,
    start: int | None = None,
    indices: tuple[int, ...] | None = None,
    mode: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return target-free weights for configured grounding hypotheses."""

    if atomic_probabilities.ndim != 2 or atomic_probabilities.shape[1] < 1:
        raise ValueError("atomic_probabilities must be [B,K]")
    if indices is None:
        if start not in (0, 2):
            raise ValueError("legacy grounding route start must be 0 or 2")
        indices = (start, start + 1)
    if not indices or min(indices) < 0 or max(indices) >= atomic_probabilities.shape[1]:
        raise ValueError("grounding hypothesis indices are invalid")
    pair = atomic_probabilities[:, list(indices)].float().clamp(0.0, 1.0)
    normalized_mode = str(mode).lower()
    if normalized_mode == "hard":
        selected = pair.argmax(dim=1)
        return torch.nn.functional.one_hot(selected, num_classes=pair.shape[1]).to(pair.dtype)
    if normalized_mode != "soft":
        raise ValueError("grounding routing mode must be soft or hard")
    if float(temperature) <= 0:
        raise ValueError("grounding routing temperature must be positive")
    mass = pair.pow(1.0 / float(temperature))
    return mass / (mass.sum(dim=1, keepdim=True) + 1e-8)


def route_image_grounding_boxes(
    output: HERADecisionOutput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate configured image boxes using predicted atomic probabilities only."""

    # Eq.22: only individually active hypotheses contribute. Two weak
    # hypotheses must never combine into an active visual prediction.
    indices = tuple(output.metadata.get("image_hypothesis_indices", (0, 1)))
    probabilities = output.atomic_probabilities[:, list(indices)].float()
    active = probabilities >= 0.5
    mass = probabilities * active
    weights = mass / (mass.sum(dim=1, keepdim=True) + 1e-8)
    boxes = torch.sum(weights.unsqueeze(-1) * output.image_grounding_boxes, dim=1)
    return boxes, weights


def image_grounding_presence_probability(output: HERADecisionOutput) -> torch.Tensor:
    """Return maximum image-hypothesis probability for fixed-0.5 abstention.

    The ontology is read from ``image_hypothesis_indices`` rather than inferred
    from the number or names of dataset classes.  This keeps the abstention
    mechanism identical for DGM4, SAMM, and future ontologies.
    """

    probabilities = output.atomic_probabilities
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("atomic_probabilities must be [B,K]")
    metadata = getattr(output, "metadata", {})
    indices = tuple(metadata.get("image_hypothesis_indices", (0, 1)))
    if not indices or min(indices) < 0 or max(indices) >= probabilities.shape[1]:
        raise ValueError("image_hypothesis_indices are invalid")
    selected = probabilities[:, list(indices)].clamp(0.0, 1.0)
    return selected.max(dim=1).values


def route_text_grounding_probabilities(
    output: HERADecisionOutput,
    *,
    aggregation: str = "relative_soft",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate configured token probabilities without consulting target labels.

    ``relative_soft`` is the released HERA rule and is retained as the default
    for exact backward compatibility.  It normalizes the relevant atomic
    probabilities to sum to one, so it selects *which* hypothesis explains a
    token but does not represent whether any text manipulation is present.

    ``presence_gated`` multiplies that relative mixture by the target-free
    Noisy-OR probability that at least one text hypothesis is present.

    ``token_noisy_or`` treats each hypothesis probability as an absolute gate
    on its token probabilities and combines the gated heads with Noisy-OR.
    """

    mode, temperature = _routing_config(output)
    normalized_aggregation = str(aggregation).strip().lower()
    if normalized_aggregation not in TEXT_GROUNDING_AGGREGATIONS:
        raise ValueError(
            "text grounding aggregation must be one of "
            f"{TEXT_GROUNDING_AGGREGATIONS}, got {aggregation!r}"
        )
    indices = tuple(output.metadata.get("text_hypothesis_indices", (2, 3)))
    weights = grounding_route_weights(
        output.atomic_probabilities,
        indices=indices,
        mode=mode,
        temperature=temperature,
    )
    head_probabilities = output.text_grounding_probabilities
    if head_probabilities.ndim != 3:
        raise ValueError("text_grounding_probabilities must be [B,H,L]")
    if head_probabilities.shape[:2] != weights.shape:
        raise ValueError(
            "text grounding heads must align with configured text hypotheses"
        )
    relative_mixture = torch.sum(
        weights.unsqueeze(-1) * head_probabilities, dim=1
    )
    if normalized_aggregation == "relative_soft":
        probabilities = relative_mixture
    else:
        atomic_presence = output.atomic_probabilities[:, list(indices)].clamp(0.0, 1.0)
        if normalized_aggregation == "presence_gated":
            presence = 1.0 - torch.prod(1.0 - atomic_presence, dim=1)
            probabilities = presence.unsqueeze(-1) * relative_mixture
        else:
            gated_heads = atomic_presence.unsqueeze(-1) * head_probabilities
            probabilities = 1.0 - torch.prod(1.0 - gated_heads, dim=1)
    probabilities = probabilities.clamp(0.0, 1.0)
    return probabilities, weights


def _routing_config(output: HERADecisionOutput) -> tuple[str, float]:
    return (
        str(output.metadata.get("grounding_routing", "soft")),
        float(output.metadata.get("grounding_routing_temperature", 1.0)),
    )
