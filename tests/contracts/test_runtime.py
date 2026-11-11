"""Small CPU counterexamples for run identity, resume, and the schedule."""

from pathlib import Path
from types import SimpleNamespace
import hashlib

import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from heri.data.loader import verify_annotation_identity
from heri.data.samm_reconstruction import reconstruct_annotation_bytes
from heri.training.checkpoint import (
    dataloader_states,
    restore_dataloader_states,
    save_checkpoint,
    validate_resume_identity,
)
from heri.training.heri_trainer import HERATrainer
from heri.training.heri_trainer import _latest_hera_resume_checkpoint
from heri.training.precision import PrecisionManager
from heri.training.scheduler import create_stage_scheduler
from heri.training.state import TrainingState
from heri.utils.config import load_yaml_config


def test_samm_reconstruction_preserves_frozen_sampling_order_and_utf8_recipe() -> None:
    import json

    rows = [
        {"id": value, "image": f"orig_output/{value}.jpg", "text": "đẹp"}
        for value in (50, 10, 40, 20, 30)
    ]
    payload, ordered_ids = reconstruct_annotation_bytes(rows, split="train", train_size=3)
    decoded = json.loads(payload)
    assert [row["id"] for row in decoded] == [10, 20, 30]
    assert ordered_ids == b"10\n20\n30\n"
    assert not payload.endswith(b"\n") and b"\\u" not in payload
    assert payload.startswith(b'[{"id":10,"image":"images/orig_output/10.jpg","text":')
    assert rows[0]["image"] == "orig_output/50.jpg"


def test_samm_reconstruction_retains_validation_order_and_rejects_wrong_sources() -> None:
    import json

    rows = [{"id": 2, "image": "b.jpg"}, {"id": 1, "image": "a.jpg"}]
    payload, _ = reconstruct_annotation_bytes(rows, split="val")
    assert [row["id"] for row in json.loads(payload)] == [2, 1]
    with pytest.raises(ValueError, match="unique integers"):
        reconstruct_annotation_bytes([rows[0], rows[0]], split="test")
    with pytest.raises(ValueError, match="unprefixed"):
        reconstruct_annotation_bytes([{"id": 1, "image": "images/a.jpg"}], split="val")


def _loader() -> DataLoader:
    dataset = TensorDataset(torch.arange(8))
    sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(7))
    return DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler,
        generator=torch.Generator().manual_seed(7),
    )


def test_resume_preserves_next_epoch_order_with_independent_worker_seed() -> None:
    original = _loader()
    next(iter(original))
    state = dataloader_states({"train": original})
    expected = next(iter(original))[0]
    resumed = _loader()
    restore_dataloader_states(state, {"train": resumed})
    assert torch.equal(next(iter(resumed))[0], expected)


def test_resume_rejects_different_model_or_shuffle_seed() -> None:
    reference = {"project": {"seed": 7}, "dataloader": {"seed": 7}}
    for changed in (
        {"project": {"seed": 21}, "dataloader": {"seed": 7}},
        {"project": {"seed": 7}, "dataloader": {"seed": 21}},
    ):
        with pytest.raises(ValueError, match="seed mismatch"):
            validate_resume_identity(reference, changed)


def test_auto_resume_ignores_newer_checkpoint_from_another_seed(tmp_path: Path) -> None:
    import copy
    import os
    import yaml

    canonical = load_yaml_config("config/datasets/dgm4/heri/train.yaml")
    candidates = {}
    for seed in (7, 21):
        config = copy.deepcopy(canonical)
        config["project"]["seed"] = config["dataloader"]["seed"] = seed
        directory = tmp_path / f"heri_seed{seed}_fixture"
        checkpoints = directory / "checkpoints"
        checkpoints.mkdir(parents=True)
        checkpoint = checkpoints / "last.pt"
        checkpoint.write_bytes(b"discovery reads config, not large model weights")
        (directory / "effective_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        os.utime(checkpoint, (seed, seed))
        candidates[seed] = checkpoint
    selected = _latest_hera_resume_checkpoint(tmp_path, checkpoint_name="last.pt", config=canonical)
    assert selected == candidates[7]


def test_observed_hash_checks_file_bytes_and_detects_tampering(tmp_path: Path) -> None:
    annotation = tmp_path / "train.json"
    annotation.write_bytes(b"[]")
    expected = hashlib.sha256(b"[]").hexdigest()
    config = {
        "dataset": {"train_annotations": str(annotation), "verify_hashes": True},
        "dataset_hashes": {"train": expected},
    }
    assert verify_annotation_identity(config, "train") == expected
    assert config["observed_dataset_hashes"]["train"] == expected
    annotation.write_bytes(b"[{}]")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_annotation_identity(config, "train")


def test_failed_atomic_checkpoint_write_preserves_previous_file(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "last.pt"
    path.write_bytes(b"previous checkpoint")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def interrupted_save(payload, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            state=TrainingState(),
            metrics={},
            config={},
            precision="fp32",
        )
    assert path.read_bytes() == b"previous checkpoint"
    assert not list(tmp_path.glob("*.tmp"))


def test_fp16_unscales_before_gradient_clipping() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    model = torch.nn.ParameterList([parameter])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    parameter.grad = torch.tensor(8.0)  # true gradient 1, scale factor 8

    class FakeScaler:
        def unscale_(self, optimizer):
            parameter.grad.div_(8)

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

    precision = object.__new__(PrecisionManager)
    precision.actual, precision.scaler = "fp16", FakeScaler()
    trainer = object.__new__(HERATrainer)
    trainer.model, trainer.precision = model, precision
    trainer.state = SimpleNamespace(global_step=0)
    trainer._optimizer_step(optimizer=optimizer, scheduler=None, max_grad_norm=0.5)
    assert parameter.item() == pytest.approx(0.95)
    assert trainer.state.global_step == 1


def test_full_finetuning_uses_global_linear_decay_for_new_encoder_groups() -> None:
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    optimizer = torch.optim.SGD(
        [
            {"params": [parameters[0]], "name": "new_hera_modules", "lr": 0.9},
            {"params": [parameters[1]], "name": "dinov3", "lr": 0.9},
        ]
    )
    config = {
        "optimizer": {"lr_new": 1e-4, "lr_backbones": {"dinov3": 2e-6}},
        "scheduler": {"type": "linear_warmup", "warmup_steps": 2},
    }
    scheduler, plan = create_stage_scheduler(optimizer, config, total_steps=10, global_step=5)
    assert plan["global_step_offset"] == 5
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 * 5 / 8)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(2e-6 * 5 / 8)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[1]["lr"] == pytest.approx(2e-6 * 4 / 8)
