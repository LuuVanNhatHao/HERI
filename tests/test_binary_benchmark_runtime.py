from __future__ import annotations

from pathlib import Path

import torch

from src.benchmarks.artifacts import load_checkpoint_payload, save_last_checkpoint
from src.benchmarks.state import BenchmarkTrainingState
from src.benchmarks.training_utils import full_ft_gradient_coverage
from src.training.precision import PrecisionManager
from src.utils.config import load_yaml_config


def test_atomic_last_checkpoint_contains_complete_resume_state(tmp_path: Path) -> None:
    config = load_yaml_config("config/benchmark_multibanfakedetect.yaml")
    config["checkpointing"]["minimum_free_gib"] = 0
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    precision = PrecisionManager(precision="fp32", device=torch.device("cpu"))
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()
    state = BenchmarkTrainingState(
        epoch=4,
        global_step=17,
        threshold=0.37,
        best_metric=0.81,
        best_model_path=tmp_path / "best_model.pt",
        early_stopping_best=0.81,
        early_stopping_bad_epochs=1,
    )
    path = tmp_path / "last.pt"

    save_last_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=precision.scaler,
        state=state.checkpoint_dict(),
        metrics={"macro_f1": 0.81},
        config=config,
        precision=precision.actual,
        scheduler_plan={"type": "none", "total_steps": 1, "warmup_steps": 0},
    )
    payload = load_checkpoint_payload(path)

    assert payload["checkpoint_type"] == "resume"
    assert payload["epoch"] == 4
    assert payload["global_step"] == 17
    assert payload["binary_threshold"] == 0.37
    assert payload["optimizer"]["state"]
    assert payload["rng_state"]["torch"] is not None
    assert not path.with_suffix(".pt.tmp").exists()


class _ThreeBackbones(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modules_by_name = torch.nn.ModuleDict(
            {
                "siglip2": torch.nn.Linear(2, 2),
                "dinov3": torch.nn.Linear(2, 2),
                "deberta": torch.nn.Linear(2, 2),
            }
        )

    def backbone_modules(self) -> dict[str, torch.nn.Module]:
        return dict(self.modules_by_name.items())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return sum(module(value).sum() for module in self.modules_by_name.values())


def test_full_ft_gradient_audit_proves_all_backbones_receive_gradients() -> None:
    model = _ThreeBackbones()
    model(torch.ones(2, 2)).backward()

    report = full_ft_gradient_coverage(model, minimum_element_coverage=1.0)

    assert set(report) == {"siglip2", "dinov3", "deberta"}
    assert all(values["gradient_element_coverage"] == 1.0 for values in report.values())
