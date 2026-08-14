"""Freeze/unfreeze and optimizer parameter-group utilities."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from src.models.encoders.common import count_parameters, freeze_module


@dataclass(frozen=True)
class ParameterGroupSummary:
    """Optimizer group accounting for logs/tests."""

    name: str
    lr: float
    weight_decay: float
    parameters: int


def apply_training_stage(model: nn.Module, config: Mapping[str, Any], *, epoch: int) -> None:
    """Apply Stage A freeze or Stage B partial unfreeze from YAML."""

    stages = config["training"]["stages"]
    freeze_epochs = int(stages["stage_a_freeze_epochs"])
    backbones = model.backbone_modules() if hasattr(model, "backbone_modules") else {}
    for module in backbones.values():
        if hasattr(module, "freeze_all"):
            module.freeze_all()
        else:
            freeze_module(module)
    for module in model.new_modules() if hasattr(model, "new_modules") else []:
        for parameter in module.parameters():
            parameter.requires_grad = True
    if epoch <= freeze_epochs:
        return
    unfreeze = stages["stage_b_unfreeze_last_blocks"]
    for name, count in unfreeze.items():
        count_int = int(count)
        if count_int <= 0:
            continue
        if name not in backbones:
            raise ValueError(f"Cannot unfreeze missing backbone group {name!r}")
        module = backbones[name]
        if not hasattr(module, "unfreeze_last_blocks"):
            raise ValueError(f"Backbone {name!r} does not support partial unfreeze")
        module.unfreeze_last_blocks(count_int)


def create_optimizer(
    model: nn.Module,
    config: Mapping[str, Any],
) -> tuple[torch.optim.Optimizer, list[ParameterGroupSummary]]:
    """Create optimizer with separate new/backbone groups and no frozen params."""

    ensure_no_uninitialized_parameters(model, context="optimizer creation")
    opt_config = config["optimizer"]
    weight_decay = float(opt_config["weight_decay"])
    groups: list[dict[str, Any]] = []
    summaries: list[ParameterGroupSummary] = []

    def add_group(name: str, modules: list[nn.Module], lr: float) -> None:
        params = [
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        count = sum(parameter.numel() for parameter in params)
        if count == 0:
            return
        groups.append({"params": params, "lr": lr, "weight_decay": weight_decay, "name": name})
        summaries.append(
            ParameterGroupSummary(
                name=name,
                lr=lr,
                weight_decay=weight_decay,
                parameters=count,
            )
        )

    if hasattr(model, "new_modules"):
        new_group_name = "new_hera_modules" if getattr(model, "model_type", None) == "hera" else "new_modules"
        add_group(new_group_name, model.new_modules(), float(opt_config["lr_new"]))
    backbones = model.backbone_modules() if hasattr(model, "backbone_modules") else {}
    backbone_lrs = opt_config["lr_backbones"]
    for name, module in backbones.items():
        add_group(name, [module], float(backbone_lrs[name]))
    if not groups:
        raise ValueError("Optimizer received no trainable parameters")

    optimizer_type = str(opt_config["type"]).lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            groups,
            betas=tuple(float(value) for value in opt_config.get("betas", [0.9, 0.999])),
            eps=float(opt_config.get("eps", 1e-8)),
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type!r}")
    return optimizer, summaries


def migrate_optimizer_state(
    source: torch.optim.Optimizer,
    target: torch.optim.Optimizer,
) -> int:
    """Preserve Adam state for parameter objects shared across a stage rebuild."""

    migrated = 0
    target_parameters = {
        parameter
        for group in target.param_groups
        for parameter in group["params"]
    }
    for parameter, state in source.state.items():
        if parameter not in target_parameters:
            continue
        target.state[parameter] = copy.deepcopy(state)
        migrated += 1
    source_groups = {str(group.get("name")): group for group in source.param_groups}
    for group in target.param_groups:
        previous = source_groups.get(str(group.get("name")))
        if previous is not None:
            group["lr"] = float(previous["lr"])
    return migrated


def load_compatible_optimizer_state(
    optimizer: torch.optim.Optimizer,
    saved_state_dict: Mapping[str, Any],
) -> int:
    """Load matching named groups when a checkpoint crosses a stage boundary."""

    saved_groups = {
        str(group.get("name")): group
        for group in saved_state_dict.get("param_groups", [])
    }
    saved_states = saved_state_dict.get("state", {})
    loaded = 0
    for current_group in optimizer.param_groups:
        saved_group = saved_groups.get(str(current_group.get("name")))
        if saved_group is None:
            continue
        current_params = list(current_group["params"])
        saved_param_ids = list(saved_group.get("params", []))
        if len(current_params) != len(saved_param_ids):
            raise RuntimeError(
                f"Optimizer group {current_group.get('name')!r} changed parameter ordering/size"
            )
        for parameter, saved_id in zip(current_params, saved_param_ids, strict=True):
            if saved_id not in saved_states:
                continue
            optimizer.state[parameter] = _optimizer_state_to_parameter(
                copy.deepcopy(saved_states[saved_id]), parameter
            )
            loaded += 1
        for key, value in saved_group.items():
            if key not in {"params", "name"}:
                current_group[key] = copy.deepcopy(value)
    return loaded


def _optimizer_state_to_parameter(value: Any, parameter: torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor):
        dtype = parameter.dtype if value.is_floating_point() else value.dtype
        return value.to(device=parameter.device, dtype=dtype)
    if isinstance(value, dict):
        return {key: _optimizer_state_to_parameter(item, parameter) for key, item in value.items()}
    if isinstance(value, list):
        return [_optimizer_state_to_parameter(item, parameter) for item in value]
    if isinstance(value, tuple):
        return tuple(_optimizer_state_to_parameter(item, parameter) for item in value)
    return value


def parameter_accounting(model: nn.Module) -> dict[str, dict[str, int]]:
    """Return total/trainable parameter counts for all major groups."""

    ensure_no_uninitialized_parameters(model, context="parameter accounting")
    result = {"model": {}}
    total, trainable = count_parameters(model)
    result["model"] = {"total": total, "trainable": trainable}
    if hasattr(model, "backbone_modules"):
        for name, module in model.backbone_modules().items():
            total, trainable = count_parameters(module)
            result[name] = {"total": total, "trainable": trainable}
    if hasattr(model, "hera_component_modules"):
        for name, module in model.hera_component_modules().items():
            total, trainable = count_parameters(module)
            result[name] = {"total": total, "trainable": trainable}
    return result


def ensure_no_uninitialized_parameters(module: nn.Module, *, context: str) -> None:
    """Fail clearly if a lazy module has not been materialized."""

    uninitialized = [
        name
        for name, parameter in module.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ]
    if uninitialized:
        raise RuntimeError(
            f"Lazy parameters are uninitialized before {context}: {uninitialized[:10]}"
        )
