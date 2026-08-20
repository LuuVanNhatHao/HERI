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
from src.training.checkpoint import restore_checkpoint, validate_baseline_checkpoint_config
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml
from src.utils.environment import collect_environment_metadata


_MISSING_PATH_MARKERS = ("<run_id>", "RUN")


def resolve_baseline_evaluation_artifacts(config: Mapping[str, Any]) -> tuple[Path, Path]:
    """Resolve a matching checkpoint/threshold pair for the requested B0 or B1 baseline."""

    evaluation = config["evaluation"]
    checkpoint_path = evaluation.get("checkpoint_path")
    thresholds_path = evaluation.get("thresholds_path")
    if _is_explicit_artifact_path(checkpoint_path) and _is_explicit_artifact_path(thresholds_path):
        return Path(str(checkpoint_path)), Path(str(thresholds_path))

    resolution = evaluation.get("checkpoint_resolution")
    if not isinstance(resolution, Mapping) or not bool(resolution.get("enabled", False)):
        raise FileNotFoundError(
            "Baseline evaluation checkpoint_path/thresholds_path are not explicit. "
            "Set both paths in YAML, or enable evaluation.checkpoint_resolution."
        )

    checkpoint_name = str(resolution["checkpoint_name"])
    thresholds_name = str(resolution["thresholds_name"])
    mode = str(resolution["mode"])
    if mode == "run_dir":
        run_dir_value = resolution.get("run_dir")
        if run_dir_value in (None, ""):
            raise ValueError("evaluation.checkpoint_resolution.run_dir is required when mode=run_dir")
        run_dirs = [Path(str(run_dir_value))]
    elif mode == "latest":
        output_root = Path(str(resolution["output_root"]))
        baseline = str(config["model"]["baseline_type"])
        seed = int(config["project"]["seed"])
        prefix = f"{baseline}_seed{seed}_"
        run_dirs = (
            [path for path in output_root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
            if output_root.exists()
            else []
        )
    else:
        raise ValueError("evaluation.checkpoint_resolution.mode must be latest or run_dir")

    pairs = [
        (
            run_dir / "checkpoints" / checkpoint_name,
            run_dir / "thresholds" / thresholds_name,
        )
        for run_dir in run_dirs
    ]
    existing = [
        (checkpoint, thresholds)
        for checkpoint, thresholds in pairs
        if checkpoint.is_file() and thresholds.is_file()
    ]
    if mode == "latest":
        existing.sort(
            key=lambda pair: (
                pair[0].stat().st_mtime,
                pair[1].stat().st_mtime,
                pair[0].as_posix(),
            ),
            reverse=True,
        )
    if existing:
        return existing[0]
    raise FileNotFoundError(
        "No complete evaluation artifact pair found for "
        f"{config['model']['baseline_type']}. Expected checkpoints/{checkpoint_name} "
        f"and thresholds/{thresholds_name}."
    )


def _is_explicit_artifact_path(value: Any) -> bool:
    if value in (None, ""):
        return False
    return not any(marker in str(value) for marker in _MISSING_PATH_MARKERS)


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
    checkpoint_path, thresholds_path = resolve_baseline_evaluation_artifacts(config)
    tqdm.write(f"Resolved evaluation checkpoint: {checkpoint_path}")
    tqdm.write(f"Resolved validation thresholds: {thresholds_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise RuntimeError("Evaluation checkpoint does not contain its effective baseline config")
    validate_baseline_checkpoint_config(config, checkpoint_config, context="Evaluation")
    restore_checkpoint(checkpoint=checkpoint, model=model, restore_rng=False)
    thresholds = _load_thresholds(
        thresholds_path,
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
        "architecture_version": config["project"]["architecture_version"],
        "image_grounding_protocol": "dense_bbox",
        "baseline_type": config["model"]["baseline_type"],
        "seed": config["project"]["seed"],
        "config_name": str(config.get("project", {}).get("config_name", "evaluate_baseline")),
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
