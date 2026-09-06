"""Target-independent routing of hypothesis-conditioned grounding predictions."""

from __future__ import annotations

import torch

from src.models.hera.outputs import HERADecisionOutput


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
    pair = atomic_probabilities[:, list(indices)].clamp_min(1e-8)
    normalized_mode = str(mode).lower()
    if normalized_mode == "hard":
        selected = pair.argmax(dim=1)
        return torch.nn.functional.one_hot(selected, num_classes=pair.shape[1]).to(pair.dtype)
    if normalized_mode != "soft":
        raise ValueError("grounding routing mode must be soft or hard")
    if float(temperature) <= 0:
        raise ValueError("grounding routing temperature must be positive")
    return torch.softmax(torch.log(pair) / float(temperature), dim=1)


def route_image_grounding_boxes(
    output: HERADecisionOutput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate configured image boxes using predicted atomic probabilities only."""

    mode, temperature = _routing_config(output)
    weights = grounding_route_weights(
        output.atomic_probabilities,
        indices=tuple(output.metadata.get("image_hypothesis_indices", (0, 1))),
        mode=mode,
        temperature=temperature,
    )
    boxes = torch.sum(weights.unsqueeze(-1) * output.image_grounding_boxes, dim=1)
    return boxes, weights


def route_text_grounding_probabilities(
    output: HERADecisionOutput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate configured token probabilities without consulting target labels."""

    mode, temperature = _routing_config(output)
    weights = grounding_route_weights(
        output.atomic_probabilities,
        indices=tuple(output.metadata.get("text_hypothesis_indices", (2, 3))),
        mode=mode,
        temperature=temperature,
    )
    probabilities = torch.sum(
        weights.unsqueeze(-1) * output.text_grounding_probabilities,
        dim=1,
    )
    return probabilities, weights


def _routing_config(output: HERADecisionOutput) -> tuple[str, float]:
    return (
        str(output.metadata.get("grounding_routing", "soft")),
        float(output.metadata.get("grounding_routing_temperature", 1.0)),
    )
