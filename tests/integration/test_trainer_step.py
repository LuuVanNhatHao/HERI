import pytest
from torch import nn

from heri.models.baselines.b0_global import B0GlobalFusionBaseline
from heri.training.optimizer import apply_training_stage, create_optimizer

from tests.support.baseline import synthetic_batch, synthetic_features


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


def config() -> dict:
    return {
        "training": {
            "stages": {
                "stage_a_freeze_epochs": 1,
                "stage_b_unfreeze_last_blocks": {"siglip2": 1, "dinov3": 1, "deberta": 1},
            }
        },
        "optimizer": {
            "type": "adamw",
            "lr_new": 1e-4,
            "lr_backbones": {"siglip2": 1e-6, "dinov3": 1e-6, "deberta": 1e-6},
            "weight_decay": 0.01,
        },
    }


def test_stage_freeze_unfreeze_and_optimizer_groups() -> None:
    model = B0GlobalFusionBaseline(
        fusion_dim=16,
        fusion_hidden_dim=32,
        head_hidden_dim=16,
        dropout=0.0,
        siglip2=TinyBackbone(),
        dinov3=TinyBackbone(),
        deberta=TinyBackbone(),
    )
    model(synthetic_batch(), features=synthetic_features())
    cfg = config()

    apply_training_stage(model, cfg, epoch=1)
    assert all(not p.requires_grad for p in model.siglip2.parameters())
    _, groups = create_optimizer(model, cfg)
    assert [group.name for group in groups] == ["new_modules"]

    apply_training_stage(model, cfg, epoch=2)
    _, groups = create_optimizer(model, cfg)
    names = {group.name for group in groups}
    assert {"new_modules", "siglip2", "dinov3", "deberta"} <= names


def test_controlled_baseline_stage_b_full_unfreezes_every_backbone_parameter() -> None:
    model = B0GlobalFusionBaseline(
        fusion_dim=16,
        fusion_hidden_dim=32,
        head_hidden_dim=16,
        dropout=0.0,
        siglip2=TinyBackbone(),
        dinov3=TinyBackbone(),
        deberta=TinyBackbone(),
    )
    model(synthetic_batch(), features=synthetic_features())
    cfg = config()
    cfg["training"]["stages"] = {
        "stage_a_freeze_epochs": 1,
        "stage_b_full_unfreeze": {"enabled": True, "start_epoch": 2},
    }

    apply_training_stage(model, cfg, epoch=1)
    assert all(
        not parameter.requires_grad
        for backbone in model.backbone_modules().values()
        for parameter in backbone.parameters()
    )

    apply_training_stage(model, cfg, epoch=2)
    assert all(
        parameter.requires_grad
        for backbone in model.backbone_modules().values()
        for parameter in backbone.parameters()
    )


def test_optimizer_rejects_unmaterialized_lazy_parameters() -> None:
    model = B0GlobalFusionBaseline(
        fusion_dim=16,
        fusion_hidden_dim=32,
        head_hidden_dim=16,
        dropout=0.0,
    )
    with pytest.raises(RuntimeError, match="Lazy parameters"):
        create_optimizer(model, config())
