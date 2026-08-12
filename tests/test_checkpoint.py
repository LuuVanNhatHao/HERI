from pathlib import Path

import torch

from src.models.baselines.b0_global import B0GlobalFusionBaseline
from src.thresholding.tuner import DEFAULT_THRESHOLDS
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.state import TrainingState

from phase2_fixtures import synthetic_batch, synthetic_features


def test_checkpoint_save_and_resume_model_optimizer(tmp_path: Path) -> None:
    model = B0GlobalFusionBaseline(
        fusion_dim=16,
        fusion_hidden_dim=32,
        head_hidden_dim=16,
        dropout=0.0,
    )
    model(synthetic_batch(), features=synthetic_features())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    state = TrainingState(epoch=2, global_step=5, best_metric=0.7, thresholds=DEFAULT_THRESHOLDS)
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
    )
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    checkpoint = load_checkpoint(path=path, model=model, optimizer=optimizer, restore_rng=False)
    assert checkpoint["epoch"] == 2
    for key, value in model.state_dict().items():
        assert torch.equal(value, saved[key])
