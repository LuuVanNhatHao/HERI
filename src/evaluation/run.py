"""Final evaluation runner for Phase-2 baselines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from tqdm.auto import tqdm

from src.data.loader import create_dataloader
from src.evaluation.evaluator import compute_metrics_and_reports, move_batch_to_device, run_inference
from src.evaluation.summary import write_predictions_csv, write_summary_csv
from src.losses.baseline_losses import BaselineMultitaskLoss
from src.models.baselines import build_baseline_model
from src.thresholding.tuner import Thresholds
from src.training.checkpoint import load_checkpoint
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml
from src.utils.environment import collect_environment_metadata


def _load_thresholds(path: Path, *, checkpoint_path: Path) -> Thresholds:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected_checkpoint = data.get("checkpoint_path")
    if expected_checkpoint is not None:
        expected = Path(str(expected_checkpoint))
        if expected.exists() and checkpoint_path.exists():
            mismatch = expected.resolve() != checkpoint_path.resolve()
        else:
            mismatch = expected.as_posix() != checkpoint_path.as_posix()
        if mismatch:
            raise ValueError(
                "Threshold file does not belong to requested checkpoint: "
                f"thresholds checkpoint={expected}, evaluation checkpoint={checkpoint_path}"
            )
    return Thresholds(
        binary=float(data["binary_threshold"]),
        atomic=(
            float(data["fs_threshold"]),
            float(data["fa_threshold"]),
            float(data["ts_threshold"]),
            float(data["ta_threshold"]),
        ),
    )


def evaluate_baseline(config: Mapping[str, Any]) -> Path:
    """Run final test evaluation using frozen validation thresholds."""

    output_root = Path(str(config["output"]["root"]))
    run_id = str(config["project"].get("run_id", f"evaluation_{int(time.time())}"))
    run_dir = output_root / run_id
    for subdir in ("predictions", "metrics", "logs"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    dump_yaml(config, run_dir / "effective_config.yaml")
    environment = collect_environment_metadata(Path.cwd())
    environment["backbones"] = config["backbones"]
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2, default=str), encoding="utf-8")

    device = torch.device(str(config["runtime"]["device"]))
    model = build_baseline_model(config, load_backbones=True).to(device)
    tokenizer = model.deberta.tokenizer
    loader = create_dataloader(
        config,
        split=str(config["evaluation"]["split"]),
        tokenizer=tokenizer,
        max_samples=config.get("smoke", {}).get("test_max_samples"),
        samples_per_class=config.get("smoke", {}).get("samples_per_class"),
    )
    precision = PrecisionManager(precision=str(config["runtime"]["precision"]), device=device)
    raw_batch = next(iter(loader))
    with torch.no_grad():
        with precision.autocast():
            model(move_batch_to_device(raw_batch, device)).validate()
    checkpoint_path = Path(str(config["evaluation"]["checkpoint_path"]))
    load_checkpoint(path=checkpoint_path, model=model, restore_rng=False)
    thresholds = _load_thresholds(
        Path(str(config["evaluation"]["thresholds_path"])),
        checkpoint_path=checkpoint_path,
    )
    bundle = run_inference(
        model=model,
        loader=loader,
        loss_fn=BaselineMultitaskLoss(config["loss"]["weights"]),
        device=device,
        precision=precision,
        max_batches=config["runtime"].get("max_test_batches"),
        desc=f"{config['evaluation']['split']} inference",
        progress=bool(config["progress"]["enabled"]),
    )
    reports = compute_metrics_and_reports(bundle, thresholds)
    tqdm.write("Test binary classification_report:")
    tqdm.write(reports["binary_report"])
    tqdm.write("Test attribution classification_report:")
    tqdm.write(reports["attribution_report"])
    write_predictions_csv(
        bundle=bundle,
        thresholds=thresholds,
        path=run_dir / "predictions" / f"{config['evaluation']['split']}_predictions.csv",
    )
    run_info = {
        "run_id": run_id,
        "model_name": config["project"]["name"],
        "baseline_type": config["model"]["baseline_type"],
        "seed": config["project"]["seed"],
        "checkpoint": str(checkpoint_path),
        "split": config["evaluation"]["split"],
        "precision_mode": precision.actual,
        "micro_batch_size": config["dataloader"]["batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "effective_batch_size": int(config["dataloader"]["batch_size"])
        * int(config["training"]["gradient_accumulation_steps"]),
        "peak_vram": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "training_duration": None,
        "git_commit": environment.get("git_commit"),
        "dataset_hashes": json.dumps(config.get("dataset_hashes", {})),
        "transformers_version": environment["packages"].get("transformers"),
        "torch_version": environment["packages"].get("torch"),
        "backbone_identifiers": json.dumps(
            {key: value["name"] for key, value in config["backbones"].items()}
        ),
    }
    write_summary_csv(
        path=run_dir / "summary.csv",
        run_info=run_info,
        metrics=reports["metrics"],
        thresholds=thresholds,
        reports=reports,
    )
    return run_dir / "summary.csv"
