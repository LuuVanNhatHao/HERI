from pathlib import Path

import torch

from heri.models.baselines.b0_global import B0GlobalFusionBaseline
from heri.thresholding.tuner import DEFAULT_THRESHOLDS
from heri.training.checkpoint import load_checkpoint, save_checkpoint
from heri.training.state import TrainingState

from tests.support.baseline import synthetic_batch, synthetic_features


def test_checkpoint_save_and_resume_model_optimizer(tmp_path: Path) -> None:
    model = B0GlobalFusionBaseline(
        fusion_dim=16,
        fusion_hidden_dim=32,
        head_hidden_dim=16,
        dropout=0.0,
    )
    model(synthetic_batch(), features=synthetic_features())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    state = TrainingState(
        epoch=2,
        global_step=5,
        best_metric=0.7,
        best_score=(0.7, 0.8),
        early_stopping_best_metric=0.72,
        early_stopping_bad_epochs=2,
        thresholds=DEFAULT_THRESHOLDS,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        state=state,
        metrics={"cf1": 0.7},
        config={"project": {"seed": 42}},
        precision="fp32",
        scheduler_plan={"type": "none", "total_steps": 10, "warmup_steps": 0},
    )
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    checkpoint = load_checkpoint(path=path, model=model, optimizer=optimizer, restore_rng=False)
    assert checkpoint["epoch"] == 2
    assert checkpoint["best_score"] == [0.7, 0.8]
    assert checkpoint["early_stopping_best_metric"] == 0.72
    assert checkpoint["early_stopping_bad_epochs"] == 2
    assert checkpoint["scheduler_plan"]["total_steps"] == 10
    for key, value in model.state_dict().items():
        assert torch.equal(value, saved[key])
