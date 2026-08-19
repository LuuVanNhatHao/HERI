"""Stage, CUDA display, and full-fine-tune audit helpers."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def stage_name(config: Mapping[str, Any], epoch: int) -> str:
    freeze = int(config["training"]["stages"]["stage_a_freeze_epochs"])
    return "A" if epoch <= freeze else "B"


def cuda_memory_postfix(device: torch.device) -> str | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return (
        f"{torch.cuda.memory_allocated(index) / 1024**3:.1f}/"
        f"{torch.cuda.memory_reserved(index) / 1024**3:.1f}GiB"
    )


def full_ft_gradient_coverage(
    model: torch.nn.Module,
    *,
    minimum_element_coverage: float,
) -> dict[str, dict[str, int | float]]:
    """Prove that full-backbone parameters are trainable and receive gradients."""

    report: dict[str, dict[str, int | float]] = {}
    expected = {"siglip2", "dinov3", "deberta"}
    backbones = model.backbone_modules()
    missing = expected - set(backbones)
    if missing:
        raise RuntimeError(
            f"Full-FT gradient audit is missing backbones: {sorted(missing)}"
        )
    for name, module in backbones.items():
        parameters = list(module.parameters())
        total_elements = sum(parameter.numel() for parameter in parameters)
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        trainable_elements = sum(parameter.numel() for parameter in trainable)
        with_gradient = [
            parameter for parameter in trainable if parameter.grad is not None
        ]
        gradient_elements = sum(parameter.numel() for parameter in with_gradient)
        if total_elements == 0 or trainable_elements != total_elements:
            raise RuntimeError(
                f"Backbone {name} is not fully trainable: {trainable_elements}/{total_elements}"
            )
        ratio = gradient_elements / total_elements
        if ratio < minimum_element_coverage:
            raise RuntimeError(
                f"Backbone {name} gradient coverage is too low: "
                f"{ratio:.6f} < {minimum_element_coverage:.6f}"
            )
        report[name] = {
            "total_tensors": len(parameters),
            "trainable_tensors": len(trainable),
            "gradient_tensors": len(with_gradient),
            "total_elements": total_elements,
            "trainable_elements": trainable_elements,
            "gradient_elements": gradient_elements,
            "gradient_element_coverage": ratio,
        }
    return report
