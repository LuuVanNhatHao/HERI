"""Target-independent routing of hypothesis-conditioned grounding predictions."""

from __future__ import annotations

import torch

from src.models.hera.outputs import HERADecisionOutput


def grounding_route_weights(
    atomic_probabilities: torch.Tensor,
    *,
    start: int,
    mode: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return target-free weights for an adjacent pair of manipulation hypotheses."""

    if atomic_probabilities.ndim != 2 or atomic_probabilities.shape[1] != 4:
        raise ValueError("atomic_probabilities must be [B,4]")
    if start not in (0, 2):
        raise ValueError("grounding hypothesis pair must start at 0 (image) or 2 (text)")
    pair = atomic_probabilities[:, start : start + 2].clamp_min(1e-8)
    normalized_mode = str(mode).lower()
    if normalized_mode == "hard":
        indices = pair.argmax(dim=1)
        return torch.nn.functional.one_hot(indices, num_classes=2).to(pair.dtype)
    if normalized_mode != "soft":
        raise ValueError("grounding routing mode must be soft or hard")
    if float(temperature) <= 0:
        raise ValueError("grounding routing temperature must be positive")
    return torch.softmax(torch.log(pair) / float(temperature), dim=1)


def route_image_grounding_boxes(
    output: HERADecisionOutput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate FS/FA boxes using only the model's predicted atomic probabilities."""

    mode, temperature = _routing_config(output)
    weights = grounding_route_weights(
        output.atomic_probabilities,
        start=0,
        mode=mode,
        temperature=temperature,
    )
    boxes = torch.sum(weights.unsqueeze(-1) * output.image_grounding_boxes, dim=1)
    return boxes, weights


def route_text_grounding_probabilities(
    output: HERADecisionOutput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate TS/TA token probabilities without consulting target labels."""

    mode, temperature = _routing_config(output)
    weights = grounding_route_weights(
        output.atomic_probabilities,
        start=2,
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
