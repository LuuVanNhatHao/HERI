from __future__ import annotations

import torch
from torch import nn

from tests.support.decision import decision_config, graph_output_from_evidence
from tests.support.evidence import evidence_config
from tests.support.graph import graph_config
from tests.support.routing import routing_config


def hera_test_config(hidden_dim: int = 12, *, routing_enabled: bool = True) -> dict:
    model = decision_config(hidden_dim=hidden_dim, token_input_dim=19)["model"]
    model["model_type"] = "hera"
    loss = decision_config(hidden_dim=hidden_dim, token_input_dim=19)["loss"]
    loss["lambda_ev"] = 0.0
    return {
        "project": {"architecture_version": "heri"},
        "evidence": evidence_config(hidden_dim=hidden_dim),
        "routing": routing_config(hidden_dim=hidden_dim, enabled=routing_enabled)["routing"],
        "graph": graph_config(hidden_dim=hidden_dim),
        "model": model,
        "loss": loss,
        "training": {
            "micro_batch_size": 3,
            "gradient_accumulation_steps": 2,
            "stages": {
                "stage_a_freeze_epochs": 1,
                "stage_b_full_unfreeze": {"enabled": True, "start_epoch": 2},
            },
        },
        "optimizer": {
            "type": "adamw",
            "lr_new": 1e-4,
            "lr_backbones": {"siglip2": 1e-6, "dinov3": 1e-6, "deberta": 1e-6},
            "weight_decay": 0.01,
            "foreach": False,
        },
    }


class FixtureGraphReasoner(nn.Module):
    def __init__(self, hidden_dim: int = 12) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hypotheses = nn.Embedding(4, hidden_dim)
        self.hgt = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, evidence):
        output = graph_output_from_evidence(evidence, hidden_dim=self.hidden_dim)
        projected = self.hgt(output.hypothesis_features)
        output.hypothesis_features = projected
        output.intermediate_hypothesis_features = [projected]
        output.validate()
        return output


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze_last_blocks(self, count: int) -> int:
        changed = 0
        for layer in self.layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
                changed += parameter.numel()
        return changed


def perturb_forbidden_targets(batch: dict) -> dict:
    clone = {}
    for key, value in batch.items():
        clone[key] = (
            value.clone()
            if isinstance(value, torch.Tensor)
            else list(value)
            if isinstance(value, list)
            else value
        )
    clone["binary_target"] = 1.0 - clone["binary_target"]
    clone["atomic_target"] = 1.0 - clone["atomic_target"]
    clone["fake_cls"] = ["face_swap&text_attribute"] * len(clone["fake_cls"])
    clone["fake_image_box_target"] = clone["fake_image_box_target"] + 17.0
    clone["manipulated_face_target_index"] = torch.zeros_like(
        clone["manipulated_face_target_index"]
    )
    clone["fake_text_pos"] = torch.full_like(clone["fake_text_pos"], 0)
    return clone
